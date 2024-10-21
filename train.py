import json
import os
import pickle
import random
import warnings
from pprint import pprint

import faiss

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl

# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from IPython.display import clear_output

# from peft import Loraconfig["retrieval"], get_peft_model
from pytorch_lightning.loggers import WandbLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging,
    pipeline,
)

from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from data_modules.data_sets import ContextDataset
from models.metric import compute_exact, compute_f1
from models.model import ReaderModel, RetrievalModel
from utils import load_config, set_seed
from utils.embedding import context_embedding


def main():
    # 하이퍼파라미터 로딩, config['seed']['value'] 형태로 사용
    config = load_config("./config.yaml")

    # 시드 고정
    set_seed(config["seed"]["value"], config["seed"]["deterministic"])

    # wandb 로깅
    wandb_logger = WandbLogger(project=config["wandb"]["project"], name=config["wandb"]["name"])

    # 데이터셋 로드
    dataset = load_from_disk(config["data"]["train_dataset_path"])
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    with open(config["data"]["contexts_path"], "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    # lr_scheduler 추가

    #################################################################################
    # 리트리버 학습 초기화 (dense embedding)
    retrieval_tokenizer = AutoTokenizer.from_pretrained(config["retrieval"]["model_name"])
    retrieval_dataloader = RetrievalDataLoader(
        tokenizer=retrieval_tokenizer,
        q_max_length=config["retrieval"]["question_max_length"],
        c_max_length=config["retrieval"]["context_max_length"],
        stride=config["retrieval"]["context_stride"],
        train_data=train_dataset,
        val_data=valid_dataset,
        predict_data=valid_dataset,
        contexts=contexts,
        batch_size=config["retrieval"]["batch_size"],
    )
    retrieval_model = RetrievalModel(config["retrieval"])

    # 리트리버 임베딩 학습
    if config["retrieval"]["TRAIN_RETRIEVAL"]:  # 학습 수행
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=config["retrieval"]["epoch"],
            logger=wandb_logger,
            val_check_interval=1.0,
        )
        trainer.fit(retrieval_model, datamodule=retrieval_dataloader)
        torch.save(retrieval_model.state_dict(), "best_retrieval_model.pth")
    else:  # 학습된 모델을 불러옴
        retrieval_model.load_state_dict(torch.load("best_retrieval_model.pth"))
        assert os.path.isfile("best_retrieval_model.pth"), "No index file exists"

    retrieval_model.eval()

    ################################################################################
    # 학습된 retrieval으로 context의 임베딩을 생성,
    # faiss 인덱스 생성 후 저장
    if config["retrieval"]["TRAIN_RETRIEVAL"]:
        context_dataset = ContextDataset(
            context=list(contexts.values()),
            document_id=list(contexts.keys()),
            tokenizer=retrieval_tokenizer,
            max_length=config["retrieval"]["context_max_length"],
            stride=config["retrieval"]["context_stride"],
        )
        contexts_emb = context_embedding(context_dataset, retrieval_model)

        c_emb = contexts_emb["contexts_embedding"].detach().numpy().astype("float32")
        emb_size = c_emb.shape[-1]

        quantizer = faiss.IndexFlatIP(emb_size)
        index = faiss.IndexIVFPQ(
            quantizer, emb_size, config["faiss"]["nlist"], config["faiss"]["m"], 8, faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.IndexIDMap(index)

        sgr = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(sgr, 0, index)

        faiss.normalize_L2(c_emb)
        index.train(c_emb)
        index.add_with_ids(c_emb, np.array(contexts_emb["document_id"]).astype("int64"))

        cpu_index = faiss.index_gpu_to_cpu(index)
        index_file_path = "./data/embedding/context_index.faiss"
        faiss.write_index(cpu_index, index_file_path)

    ################################################################################
    # 리더 학습 초기화
    reader_tokenizer = AutoTokenizer.from_pretrained(config["reader"]["model_name"])
    reader_dataloader = ReaderDataLoader(
        tokenizer=reader_tokenizer,
        max_len=config["reader"]["max_length"],
        stride=config["reader"]["context_stride"],
        train_data=train_dataset,
        val_data=valid_dataset,
        predict_data=valid_dataset,
        batch_size=config["reader"]["batch_size"],
    )
    reader_model = ReaderModel(config["reader"])

    # 리더 모델 학습
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=config["reader"]["epoch"],
        logger=wandb_logger,
        val_check_interval=1.0,
    )
    trainer.fit(reader_model, datamodule=reader_dataloader)

    reader_model.eval()
    #############################################################################
    # faiss 인덱스를 불러와 valid data의 query에 대한 top k passages 출력
    index_file_path = config["data"]["index_file_path"]
    assert os.path.isfile(index_file_path), "No index file exists"
    index = faiss.read_index(index_file_path)
    sgr = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(sgr, 0, index)

    retrieval_model.index = index
    retrieval_output = trainer.predict(retrieval_model, datamodule=retrieval_dataloader)

    # reader에 retrieval의 결과를 넣어 정답을 출력
    doc_id = [contexts[id.item()] for ids in retrieval_output for id in ids]
    reader_dataloader.predict_data.map(lambda x, idx: {"context": doc_id[idx]}, with_indices=True)
    # predict_dataset = valid_dataset.add_column("context", doc_id)

    # reader_dataloader = ReaderDataLoader(
    #     tokenizer=reader_tokenizer,
    #     max_len=256,
    #     stride=32,
    #     predict_data=predict_dataset,
    #     batch_size=8,
    # )

    reader_outputs = trainer.predict(reader_model, datamodule=reader_dataloader)
    # print(len(reader_outputs))

    predictions = reader_outputs
    ground_truths = [sample["answers"]["text"][0] for sample in reader_dataloader.predict_data]

    # for i in zip(predictions, ground_truths):
    #     print(i)

    # EM과 F1 점수 계산
    em_scores = [compute_exact(truth, pred) for truth, pred in zip(ground_truths, predictions)]
    f1_scores = [compute_f1(truth, pred) for truth, pred in zip(ground_truths, predictions)]

    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)

    print(f"EM: {em:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
