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
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def main():
    # wandb 로깅
    wandb_logger = WandbLogger()
    config = wandb_logger.experiment.config
    # logger.info("Wandb logger initialized.")

    # Parameters
    SEED = config["SEED"]
    DETERMINISTIC = config["DETERMINISTIC"]
    DATA_PATH = config["DATA_PATH"]
    CONTEXTS_PATH = config["CONTEXTS_PATH"]
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    MAX_LEN = config["MAX_LEN"]
    STRIDE = config["STRIDE"]
    MODEL_NAME = config["MODEL_NAME"]
    MODULE_NAMES = config["MODULE_NAMES"]
    LORA_RANK = config["LORA_RANK"]
    LORA_ALPHA = config["LORA_ALPHA"]
    LORA_DROP_OUT = config["LORA_DROP_OUT"]

    # 하이퍼파라미터 로딩, config['seed']['value'] 형태로 사용
    # config = load_config("./config.yaml")
    # logger.info("Configuration loaded.")

    # 시드 고정
    set_seed(SEED, DETERMINISTIC)
    # logger.info("Random seed set.")

    # 데이터셋 로드
    dataset = load_from_disk(DATA_PATH)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    # logger.info("Datasets loaded.")

    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}
    # logger.info("Contexts loaded.")

    reader_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    reader_dataloader = ReaderDataLoader(
        tokenizer=reader_tokenizer,
        max_len=MAX_LEN,
        stride=STRIDE,
        train_data=train_dataset,
        val_data=valid_dataset,
        batch_size=BATCH_SIZE,
    )
    reader_model = ReaderModel(dict(config))
    # logger.info("Reader model and dataloader initialized.")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"saved/{MODEL_NAME}/{wandb.run.id}",
        filename="reader_{epoch:02d}",
        save_top_k=1,
        monitor="Exact Match",
        mode="max",
    )
    early_stop_callback = EarlyStopping(monitor="validation_loss", patience=4, mode="min")

    # 리더 모델 학습
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=1.0,
    )
    # logger.info("Starting Reader model training.")

    trainer.fit(reader_model, datamodule=reader_dataloader)

    ## best model & configuration uploading
    config_dict = dict(config)
    with open("config_reader.json", "w") as f:
        json.dump(config_dict, f)

    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(checkpoint_callback.best_model_path)
    artifact.add_file("config_reader.json")
    wandb.log_artifact(artifact)
    # logger.info("Reader model training completed and saved.")


if __name__ == "__main__":
    main()
