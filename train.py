import json
import os
import pickle
import random
import warnings

import faiss

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl

# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from IPython.display import clear_output
from peft import Loraretrieval_config, get_peft_model
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
from utils import set_seed
from utils.embedding import context_embedding


def main():
    # 리트리버 하이퍼파라미터 설정
    retrieval_config = {
        "model_name": "bert-base-multilingual-cased",
        "module_names": ["query", "key", "value"],
        "epoch": 3,
        "batch_size": 2,
        "lr": 0.00001,
        "question_max_length": 128,
        "context_max_length": 384,
        "context_stride": 16,
        "negative_length": 2,
        "LoRA_r": 8,
        "LoRA_alpha": 8,
        "LoRA_drop_out": 0.005,
    }

    # 시드 고정
    SEED = 123
    DETERMINISTIC = True
    set_seed(SEED, DETERMINISTIC)

    # wandb 로깅
    wandb_logger = WandbLogger(project="ODQA_project", name="retrieval_reader_training")

    # 데이터셋 로드
    dataset = load_from_disk("./data/train_dataset/")
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    with open("./data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    # lr_scheduler
    #################################################################################
    # 리트리버 학습 초기화 (dense embedding)
    retrieval_tokenizer = AutoTokenizer.from_pretrained(retrieval_config["model_name"])
    retrieval_dataloader = RetrievalDataLoader(
        tokenizer=retrieval_tokenizer,
        q_max_length=retrieval_config["question_max_length"],
        c_max_length=retrieval_config["context_max_length"],
        stride=retrieval_config["context_stride"],
        train_data=train_dataset,
        val_data=valid_dataset,
        predict_data=valid_dataset,
        contexts=contexts,
        batch_size=retrieval_config["batch_size"],
    )
    retrieval_model = RetrievalModel(retrieval_config)

    # 리트리버 임베딩 학습
    TRAIN_RETRIEVAL = True
    if TRAIN_RETRIEVAL:  # 학습 수행
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=retrieval_config["epoch"],
            logger=wandb_logger,
            val_check_interval=1.0,
        )
        trainer.fit(retrieval_model, datamodule=retrieval_dataloader)
    else:  # 학습된 모델을 불러옴
        retrieval_model.load_state_dict(torch.load("best_retrieval_model.pth"))
        assert os.path.isfile("best_retrieval_model.pth"), "No index file exists"

    retrieval_model.eval()

    ################################################################################
    # 학습된 retrieval으로 context의 임베딩을 생성,
    # faiss 인덱스 생성 후 저장
    if TRAIN_RETRIEVAL:
        context_dataset = ContextDataset(
            context=list(contexts.values()),
            document_id=list(contexts.keys()),
            tokenizer=retrieval_tokenizer,
            max_length=256,
            stride=16,
        )
        contexts_emb = context_embedding(context_dataset, retrieval_model)

        c_emb = contexts_emb["contexts_embedding"].detach().numpy().astype("float32")
        emb_size = c_emb.shape[-1]

        nlist = 100  # 클러스터 개수
        m = 8  # subquantizer 개수

        quantizer = faiss.IndexFlatIP(emb_size)
        index = faiss.IndexIVFPQ(quantizer, emb_size, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
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
    # 리더 하이퍼파라미터 설정
    reader_config = {
        "model_name": "klue/bert-base",
        "module_names": ["query", "key", "value"],
        "epoch": 3,
        "batch_size": 2,
        "lr": 0.00001,
        "max_length": 256,
        "context_stride": 16,
        # "negative_length": 2,
        # "LoRA_r": 8,
        # "LoRA_alpha": 8,
        # "LoRA_drop_out": 0.005,
    }

    # 리더 학습 초기화
    reader_tokenizer = AutoTokenizer.from_pretrained(reader_config["model_name"])
    reader_dataloader = ReaderDataLoader(
        tokenizer=reader_tokenizer,
        max_len=reader_config["max_length"],
        stride=reader_config["context_stride"],
        train_data=train_dataset,
        val_data=valid_dataset,
        predict_data=valid_dataset,
        batch_size=reader_config["batch_size"],
    )
    reader_model = ReaderModel(reader_config)

    # 리더 모델 학습
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=reader_config["epoch"],
        logger=wandb_logger,
        val_check_interval=1.0,
    )
    trainer.fit(reader_model, datamodule=reader_dataloader)

    reader_model.eval()
    #############################################################################
    # faiss 인덱스를 불러와 valid data의 query에 대한 top k passages 출력
    index_file_path = "./data/embedding/context_index.faiss"
    assert os.path.isfile(index_file_path), "No index file exists"
    index = faiss.read_index(index_file_path)
    sgr = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(sgr, 0, index)

    retrieval_model.index = index
    retrieval_output = trainer.predict(retrieval_model, datamodule=retrieval_dataloader)

    # reader에 retrieval의 결과를 넣어 정답 span을 출력
    doc_id = [contexts[id.item()] for ids in retrieval_output for id in ids]
    predict_dataset = predict_dataset.add_column("context", doc_id)

    reader_outputs = trainer.predict(reader_model, datamodule=reader_dataloader)
    reader_outputs = [answer for batch_outputs in reader_outputs for answer in batch_outputs]

    ground_truths = [sample["answer"] for sample in reader_dataloader.val_dataset]
    predictions = [output["predicted_answer"] for output in reader_outputs]

    # EM과 F1 점수 계산
    em_scores = [compute_exact(truth, pred) for truth, pred in zip(ground_truths, predictions)]
    f1_scores = [compute_f1(truth, pred) for truth, pred in zip(ground_truths, predictions)]

    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)

    print(f"EM: {em:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
