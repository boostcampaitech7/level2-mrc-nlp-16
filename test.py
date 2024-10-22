import argparse
import json
import os
import pickle

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

import wandb
from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from data_modules.data_sets import ReaderDataset
from models.model import ReaderModel, RetrievalModel
from utils.util import normalize_rows


def main(arg):
    contexts_embedding_path = arg.contexts_embedding_path
    model_path = arg.model_path
    data_path = arg.data_path
    contexts_path = arg.contexts_path
    k = arg.k

    # 데이터셋 로드
    dataset = load_from_disk(data_path)
    valid_dataset = dataset["validation"]
    # logger.info("Datasets loaded.")

    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}
    # logger.info("Contexts loaded.")

    with open(contexts_embedding_path, "rb") as f:
        contexts_embedding = pickle.load(f)
    contexts_embedding = np.array(contexts_embedding)
    contexts_embedding = normalize_rows(contexts_embedding)

    ## model/config loading
    wandb.login()

    run = wandb.init()
    artifact = run.use_artifact(model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataloader = RetrievalDataLoader(
        tokenizer=tokenizer,
        q_max_length=config["question_max_length"],
        c_max_length=config["context_max_length"],
        stride=config["context_stride"],
        predict_data=valid_dataset,
        contexts=contexts,
        batch_size=config["batch_size"],
        negative_length=config["negative_length"],
    )
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    retrieval.load_state_dict(checkpoint["model_state_dict"])

    retrieval.c_emb = contexts_embedding

    trainer = pl.Trainer(accelerator="gpu")
    sims_dense = trainer.predict(retrieval, datamodule=dataloader)
    # logger.info("Retrieval model predictions completed.")

    sims = sims_dense
    selected_doc_ids = np.argpartition(sims, -k, axis=1)[-k:]
    selected_contexts = [contexts[idx] for idx in selected_doc_ids]
    # logger.info(f"Document IDs extracted from retrieval output. Total: {len(doc_id)}")

    run = wandb.init()
    artifact = run.use_artifact(model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataloader = ReaderDataLoader(
        tokenizer=tokenizer,
        max_len=config["max_length"],
        stride=config["stride"],
        test_data=valid_dataset,
        selected_contexts=selected_contexts,
        batch_size=config["batch_size"],
    )
    reader = ReaderModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    reader.load_state_dict(checkpoint["model_state_dict"])

    reader_outputs = trainer.test(reader, datamodule=dataloader)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-ce",
        "--contexts_embedding_path",
        default=None,
        type=str,
        help="file path for context embedding (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: None)",
    )
    args.add_argument(
        "-d",
        "--data_path",
        default=None,
        type=str,
        help="directory path for datasets (default: None)",
    )
    args.add_argument(
        "-c",
        "--context_path",
        default=None,
        type=str,
        help="directory path for contexts (default: None)",
    )
    args.add_argument(
        "-k",
        "--k",
        default=10,
        type=int,
        help="number of selected contexts (default: 10)",
    )

    arg = args.parse_args()
    main(arg)
