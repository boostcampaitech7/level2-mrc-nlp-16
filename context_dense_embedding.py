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
    model_path = arg.model_path  ## wandb artifact 상에 load된 model path
    model_name = arg.model_name  ## wandb artifact 상에 load된 model name
    batch_size = arg.batch_size

    ## context dataset load
    context_path = "data/wikipedia_documents.json"
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

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{model_name}")
    retrieval.load_state_dict(checkpoint["state_dict"])

    ## dataset setting
    context_dataset = ContextDataset(
        context=list(contexts.values()),
        document_id=list(contexts.keys()),
        tokenizer=tokenizer,
        max_length=config["CONTEXT_MAX_LEN"],
    )

    ## embedding
    contexts_emb = context_embedding(contextdataset=context_dataset, retrieval=retrieval, batch_size=batch_size)
    c_emb = contexts_emb["contexts_embedding"].detach().numpy().astype("float32")
    contexts_embedding_path = "data/embedding/context_dense_embedding.bin"
    with open(contexts_embedding_path, "wb") as f:
        pickle.dump(c_emb, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-mp",
        "--model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: None)",
    )
    args.add_argument(
        "-mn",
        "--model_name",
        default=None,
        type=str,
        help="model name in artifact (default: None)",
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
