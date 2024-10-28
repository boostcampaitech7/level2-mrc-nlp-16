import argparse
import json
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from models.model import ReaderModel, RetrievalModel
from utils.util import normalize_rows, zero_one_normalize_rows


def main(arg):
    retrieval_model_path = arg.retrieval_model_path  ## wandb artifact 상에 load된 retrieval model path
    retrieval_model_name = arg.retrieval_model_name  ## wandb artifact 상에 load된 retrieval model name
    reader_model_path = arg.reader_model_path  ## wandb artifact 상에 load된 reader model path
    reader_model_name = arg.reader_model_name  ## wandb artifact 상에 load된 reader model name
    k = arg.k  ## retrieval에서 선택할 문서의 개수
    w = arg.w  ## hybrid model에서 context embedding similarity에 적용한 weight


    ## load dataset
    data_path = "data/test_dataset"
    dataset = load_from_disk(data_path)
    predict_dataset = dataset["validation"]

    contexts_path = "data/wikipedia_documents.json"
    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    ## load embeddings
    ### dense
    contexts_dense_embedding_path = "data/embedding/context_dense_embedding.bin"
    with open(contexts_dense_embedding_path, "rb") as f:
        contexts_dense_embedding = pickle.load(f)
    contexts_dense_embedding = np.array(contexts_dense_embedding)
    contexts_dense_embedding = normalize_rows(contexts_dense_embedding)

    ### sparse
    contexts_sparse_embedding_path = "data/embedding/context_sparse_embedding.bin"
    with open(contexts_sparse_embedding_path, "rb") as f:
        bm25 = pickle.load(f)

    ## retrieval model/config loading
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
        predict_data=predict_dataset,
        contexts=contexts,
        batch_size=config["BATCH_SIZE"],
        negative_length=config["NEGATIVE_LENGTH"],
    )
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{retrieval_model_name}")
    retrieval.load_state_dict(checkpoint["state_dict"])

    ## dense embedding setting for model predict
    retrieval.c_emb = contexts_dense_embedding

    ## test retrieval
    trainer = pl.Trainer(accelerator="gpu")
    sims_dense = trainer.predict(retrieval, datamodule=dataloader)
    sims_dense = np.concatenate(sims_dense, axis=0)

    ## output processing
    sims_sparse = []
    for question in tqdm(predict_dataset["question"], desc="sparse embedding similarity"):
        tokenized_question = tokenizer.tokenize(question)
        scores = bm25.get_scores(tokenized_question)
        sims_sparse.append(scores)
        del scores, tokenized_question
    sims_sparse = np.vstack(sims_sparse)
    sims_sparse = zero_one_normalize_rows(sims_sparse)

    ## hybrid model
    sims = w * sims_dense + (1 - w) * sims_sparse

    ## select contexts
    selected_doc_ids = np.argpartition(sims, -k, axis=1)[:, -k:]
    selected_contexts = [[contexts[idx] for idx in row] for row in selected_doc_ids]

    ## reader model/config loading
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
        predict_data=predict_dataset,
        selected_contexts=selected_contexts,
        batch_size=config["BATCH_SIZE"],
    )
    reader = ReaderModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{reader_model_name}")
    reader.load_state_dict(checkpoint["state_dict"])

    ## test reader
    reader_outputs = trainer.predict(reader, datamodule=dataloader)
    reader_outputs = {k: v for k, v in reader_outputs}

    ## generate submission file
    with open("data/submission.json", "w", encoding="utf-8") as f:
        json.dump(reader_outputs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-rtmp",
        "--retrieval_model_path",
        default=None,
        type=str,
        help="artifact path for a retrieval model (default: None)",
    )
    args.add_argument(
        "-rtmn",
        "--retrieval_model_name",
        default=None,
        type=str,
        help="retrieval model name in artifact (default: None)",
    )
    args.add_argument(
        "-rdmp",
        "--reader_model_path",
        default=None,
        type=str,
        help="artifact path for a reader model (default: None)",
    )
    args.add_argument(
        "-rdmn",
        "--reader_model_name",
        default=None,
        type=str,
        help="reader model name in artifact (default: None)",
    )
    args.add_argument(
        "-k",
        "--k",
        default=5,
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
