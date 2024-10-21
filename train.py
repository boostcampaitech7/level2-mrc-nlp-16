import json
import logging
import os
import pickle
import random
import warnings
from pprint import pprint

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from IPython.display import clear_output
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

import wandb
from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from data_modules.data_sets import ContextDataset, ReaderDataset
from models.metric import compute_exact, compute_f1
from models.model import ReaderModel, RetrievalModel
from utils import load_config, set_seed
from utils.embedding import context_embedding

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 하이퍼파라미터 로딩, config['seed']['value'] 형태로 사용
    config = load_config("./config.yaml")
    logger.info("Configuration loaded.")

    # 시드 고정
    set_seed(config["seed"]["value"], config["seed"]["DETERMINISTIC"])
    logger.info("Random seed set.")

    # wandb 로깅
    wandb_logger = WandbLogger(project=config["wandb"]["project"], name=config["wandb"]["name"])
    logger.info("Wandb logger initialized.")

    # 데이터셋 로드
    dataset = load_from_disk(config["data"]["train_dataset_path"])
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    logger.info("Datasets loaded.")

    with open(config["data"]["contexts_path"], "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}
    logger.info("Contexts loaded.")

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
        # test_data=valid_dataset,
        predict_data=valid_dataset,
        contexts=contexts,
        batch_size=config["retrieval"]["batch_size"],
        negative_length=config["retrieval"]["negative_length"],
    )
    retrieval_model = RetrievalModel(config["retrieval"])
    logger.info("Retrieval model and dataloader initialized.")

    # 리트리버 임베딩 학습
    if config["retrieval"]["TRAIN_RETRIEVAL"]:  # 학습 수행
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=config["retrieval"]["epoch"],
            logger=wandb_logger,
            val_check_interval=1.0,
        )
        logger.info("Starting Retrieval model training.")
        trainer.fit(retrieval_model, datamodule=retrieval_dataloader)
        torch.save(retrieval_model.state_dict(), "best_retrieval_model.pth")
        logger.info("Retrieval model training completed and saved.")
    else:  # 학습된 모델을 불러옴
        retrieval_model.load_state_dict(torch.load("best_retrieval_model.pth"))
        assert os.path.isfile("best_retrieval_model.pth"), "No retrieval model file exists."
        logger.info("Retrieval model loaded from saved state.")

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
        contexts_emb = context_embedding(
            contextdataset=context_dataset, retrieval=retrieval_model, batch_size=config["retrieval"]["batch_size"]
        )
        logger.info("Context embeddings generated.")

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
        index_file_path = config["data"]["index_file_path"]
        faiss.write_index(cpu_index, index_file_path)
        logger.info(f"FAISS index created and saved at {index_file_path}.")

    ################################################################################
    # 리더 학습 초기화
    if config["reader"]["TRAIN_READER"]:
        reader_tokenizer = AutoTokenizer.from_pretrained(config["reader"]["model_name"])
        reader_dataloader = ReaderDataLoader(
            tokenizer=reader_tokenizer,
            max_len=config["reader"]["max_length"],
            stride=config["reader"]["context_stride"],
            train_data=train_dataset,
            val_data=valid_dataset,
            batch_size=config["reader"]["batch_size"],
        )
        reader_model = ReaderModel(config["reader"])
        logger.info("Reader model and dataloader initialized.")

        MODEL_NAME = config["model_name"]
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"saved/{MODEL_NAME}/{wandb.run.id}",
            filename="{epoch:02d}",
            save_top_k=1,
            monitor="Exact Match",
            mode="max",
        )
        early_stop_callback = EarlyStopping(monitor="validation_loss", patience=4, mode="min")

        # 리더 모델 학습
        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=config["reader"]["epoch"],
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            val_check_interval=1.0,
        )
        logger.info("Starting Reader model training.")

        trainer.fit(reader_model, datamodule=reader_dataloader)

        ## best model & configuration uploading
        config_dict = dict(config)
        with open("config.json", "w") as f:
            json.dump(config_dict, f)

        artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
        artifact.add_file(checkpoint_callback.best_model_path)
        artifact.add_file("config.json")
        wandb.log_artifact(artifact)
        logger.info("Reader model training completed and saved.")

    #############################################################################
    # faiss 인덱스를 불러와 valid data의 query에 대한 top k passages 출력
    index_file_path = config["data"]["index_file_path"]
    assert os.path.isfile(index_file_path), "No FAISS index file exists."
    index = faiss.read_index(index_file_path)
    sgr = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(sgr, 0, index)
    logger.info("FAISS index loaded.")

    retrieval_model.index = index
    logger.info("Retrieval model index set.")

    retrieval_output = trainer.predict(retrieval_model, datamodule=retrieval_dataloader)
    logger.info("Retrieval model predictions completed.")

    # 리더 예측 준비
    doc_id = [contexts[idx.item()] for batch in retrieval_output for idx in batch]
    logger.info(f"Document IDs extracted from retrieval output. Total: {len(doc_id)}")

    assert len(doc_id) == len(valid_dataset), f"Expected {len(valid_dataset)} contexts, got {len(doc_id)}"
    logger.info("Document IDs match the number of ground truth samples.")

    # 새로운 ReaderDataset 생성
    updated_predict_dataset = ReaderDataset(
        question=valid_dataset["question"],
        context=doc_id,
        answer=valid_dataset["answers"],
        tokenizer=reader_tokenizer,
        max_len=config["reader"]["max_length"],
        stride=config["reader"]["context_stride"],
        stage="predict",
    )

    # ReaderDataLoader의 predict_dataset 교체
    reader_dataloader.predict_dataset = updated_predict_dataset
    logger.info("Reader dataloader predict dataset updated with retrieved contexts.")

    # 리더 예측 수행
    reader_outputs = trainer.predict(reader_model, datamodule=reader_dataloader)
    logger.info("Reader model predictions completed.")

    # reader_outputs는 리스트의 리스트 형태이므로 평탄화
    predictions = [answer for batch in reader_outputs for answer in batch]
    logger.info(f"Predictions flattened. Total: {len(predictions)}")
    print(len(predictions), predictions)

    # ground_truths 수집
    ground_truths = [answer["text"][0] for answer in updated_predict_dataset.answer]
    logger.info(f"Ground truths collected. Total: {len(ground_truths)}")
    print(len(ground_truths), ground_truths)

    # 예측과 정답의 수가 일치하는지 확인
    assert len(predictions) == len(
        ground_truths
    ), f"Predictions count {len(predictions)} does not match ground truths count {len(ground_truths)}."
    logger.info("Predictions and ground truths count match.")

    # EM과 F1 점수 계산
    em_scores = [compute_exact(truth, pred) for truth, pred in zip(ground_truths, predictions)]
    f1_scores = [compute_f1(truth, pred) for truth, pred in zip(ground_truths, predictions)]

    em = sum(em_scores) / len(em_scores) if em_scores else 0
    f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    logger.info(f"Evaluation Results - EM: {em:.4f}, F1: {f1:.4f}")
    # print(f"EM: {em:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
