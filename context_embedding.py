import argparse
import json
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer

import wandb
from data_modules.data_sets import ContextDataset
from models.model import RetrievalModel
from utils.embedding import context_embedding


def main(arg):
    context_path = arg.context_path
    model_path = arg.model_path
    batch_size = arg.batch_size
    contexts_embedding_path = arg.contexts_embedding_path

    with open(context_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    ## model/config loading
    wandb.login()

    run = wandb.init()
    artifact = run.use_artifact(model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    retrieval.load_state_dict(checkpoint["model_state_dict"])

    # 학습된 retrieval으로 context의 임베딩을 생성,
    # faiss 인덱스 생성 후 저장
    context_dataset = ContextDataset(
        context=list(contexts.values()),
        document_id=list(contexts.keys()),
        tokenizer=tokenizer,
        max_length=config["context_max_length"],
    )

    contexts_emb = context_embedding(contextdataset=context_dataset, retrieval=retrieval, batch_size=batch_size)
    c_emb = contexts_emb["contexts_embedding"].detach().numpy().astype("float32")
    with open(contexts_embedding_path, "wb") as f:
        pickle.dump(c_emb, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--context_path",
        default=None,
        type=str,
        help="directory path for contexts (default: None)",
    )
    args.add_argument(
        "-ce",
        "--contecontexts_embedding_pathxt_path",
        default=None,
        type=str,
        help="directory path for context embedding (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: None)",
    )
    args.add_argument(
        "-b",
        "--batch_size",
        default=2,
        type=int,
        help="batch size (default: 2)",
    )

    arg = args.parse_args()
    main(arg)
