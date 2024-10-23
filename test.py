import argparse
import json
import os
import pickle

import faiss
import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from data_modules.data_sets import ReaderDataset
from models.model import ReaderModel, RetrievalModel
from utils.util import normalize_rows


def main(arg):
    retrieval_model_path = arg.retrieval_model_path
    retrieval_file_path = arg.retrieval_file_path
    reader_model_path = arg.reader_model_path
    reader_file_path = arg.reader_file_path
    data_path = arg.data_path
    context_path = arg.context_path
    contexts_dense_embedding_path = arg.contexts_dense_embedding_path
    contexts_sparse_embedding_path = arg.contexts_sparse_embedding_path
    k = arg.k
    w = arg.w

    # 데이터셋 로드
    dataset = load_from_disk(data_path)
    valid_dataset = dataset["validation"]
    # logger.info("Datasets loaded.")

    with open(context_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}
    # logger.info("Contexts loaded.")

    with open(contexts_dense_embedding_path, "rb") as f:
        contexts_dense_embedding = pickle.load(f)
    contexts_dense_embedding = np.array(contexts_dense_embedding)
    contexts_dense_embedding = normalize_rows(contexts_dense_embedding)

    with open(contexts_sparse_embedding_path, "rb") as f:
        bm25 = pickle.load(f)

    ## model/config loading
    wandb.login()

    run = wandb.init()
    artifact = run.use_artifact(retrieval_model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    dataloader = RetrievalDataLoader(
        tokenizer=tokenizer,
        q_max_length=config["QUESTION_MAX_LEN"],
        c_max_length=config["CONTEXT_MAX_LEN"],
        stride=config["CONTEXT_STRIDE"],
        predict_data=valid_dataset,
        contexts=contexts,
        batch_size=config["BATCH_SIZE"],
        negative_length=config["NEGATIVE_LENGTH"],
    )
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{retrieval_file_path}")
    retrieval.load_state_dict(checkpoint["state_dict"])

    retrieval.c_emb = contexts_dense_embedding

    trainer = pl.Trainer(accelerator="gpu")
    sims_dense = trainer.predict(retrieval, datamodule=dataloader)
    # logger.info("Retrieval model predictions completed.")

    sims_sparse = []
    for question in tqdm(valid_dataset["question"], desc="sparse embedding similarity"):
        tokenized_question = tokenizer.tokenize(question)
        scores = bm25.get_scores(tokenized_question)
        sims_sparse.append(scores)
        del scores, tokenized_question
    sims_sparse = np.vstack(sims_sparse)

    sims = w * sims_dense + (1 - w) * sims_sparse
    selected_doc_ids = np.argpartition(sims, -k, axis=1)[-k:]
    selected_contexts = [contexts[idx] for idx in selected_doc_ids]
    # logger.info(f"Document IDs extracted from retrieval output. Total: {len(doc_id)}")

    run = wandb.init()
    artifact = run.use_artifact(reader_model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_reader.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    dataloader = ReaderDataLoader(
        tokenizer=tokenizer,
        max_len=config["MAX_LEN"],
        stride=config["STRIDE"],
        test_data=valid_dataset,
        selected_contexts=selected_contexts,
        batch_size=config["BATCH_SIZE"],
    )
    reader = ReaderModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{reader_file_path}")
    reader.load_state_dict(checkpoint["state_dict"])

    reader_outputs = trainer.test(reader, datamodule=dataloader)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-cdp",
        "--contexts_dense_embedding_path",
        default=None,
        type=str,
        help="file path for context dense embedding (default: None)",
    )
    args.add_argument(
        "-csp",
        "--contexts_sparse_embedding_path",
        default=None,
        type=str,
        help="file path for context sparse embedding (default: None)",
    )
    args.add_argument(
        "-tm",
        "--retrieval_model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: None)",
    )
    args.add_argument(
        "-dm",
        "--reader_model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: None)",
    )
    args.add_argument(
        "-tf",
        "--retrieval_file_path",
        default=None,
        type=str,
        help="checkpoint file path for a model (default: None)",
    )
    args.add_argument(
        "-df",
        "--reader_file_path",
        default=None,
        type=str,
        help="checkpoint file path for a model (default: None)",
    )
    args.add_argument(
        "-d",
        "--data_path",
        default="./data/train_dataset/",
        type=str,
        help="directory path for datasets (default: None)",
    )
    args.add_argument(
        "-c",
        "--context_path",
        default="./data/wikipedia_documents.json",
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
    args.add_argument(
        "-w",
        "--w",
        default=0.5,
        type=float,
        help="weight for dense embedding in hybrid model (default: 0.5)",
    )

    arg = args.parse_args()
    main(arg)
