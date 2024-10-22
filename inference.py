import argparse
import json
import os

import faiss
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

import wandb
from data_modules.data_loaders import ReaderDataLoader, RetrievalDataLoader
from data_modules.data_sets import ReaderDataset
from models.model import ReaderModel, RetrievalModel


def main(arg):
    index_file_path = arg.index_file_path
    model_path = arg.model_path
    data_path = arg.data_path
    contexts_path = arg.contexts_path

    # 데이터셋 로드
    dataset = load_from_disk(data_path)
    predict_dataset = dataset["validation"]
    # logger.info("Datasets loaded.")

    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}
    # logger.info("Contexts loaded.")

    # faiss 인덱스를 불러와 valid data의 query에 대한 top k passages 출력
    assert os.path.isfile(index_file_path), "No FAISS index file exists."
    index = faiss.read_index(index_file_path)

    sgr = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(sgr, 0, index)
    # logger.info("FAISS index loaded.")

    ## model/config loading
    wandb.login()

    run = wandb.init()
    artifact = run.use_artifact(model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])
    dataloader = RetrievalDataLoader(
        tokenizer=tokenizer,
        q_max_length=config["QUESTION_MAX_LEN"],
        c_max_length=config["CONTEXT_MAX_LEN"],
        stride=config["conteCONTEXT_STRIDExt_stride"],
        predict_data=predict_dataset,
        contexts=contexts,
        batch_size=config["BATCH_SIZE"],
        negative_length=config["NEGATIVE_LENGTH"],
    )
    retrieval = RetrievalModel(dict(config))
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    retrieval.load_state_dict(checkpoint["model_state_dict"])

    retrieval.index = index
    # logger.info("Retrieval model index set.")

    trainer = pl.Trainer(accelerator="gpu")
    selected_doc_ids = trainer.predict(retrieval, datamodule=dataloader)
    # logger.info("Retrieval model predictions completed.")

    # 리더 예측 준비
    selected_contexts = [contexts[idx.item()] for batch in selected_doc_ids for idx in batch]
    # logger.info(f"Document IDs extracted from retrieval output. Total: {len(doc_id)}")

    run = wandb.init()
    artifact = run.use_artifact(model_path)
    model_dir = artifact.download()

    with open(f"{model_dir}/config_retrieval.json", "r") as f:
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
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    reader.load_state_dict(checkpoint["model_state_dict"])

    reader_outputs = trainer.predict(reader, datamodule=dataloader)
    reader_outputs = {k: v for k, v in reader_outputs}

    with open("data/submission.json", "w") as f:
        json.dump(reader_outputs, f, indent=4)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i",
        "--index_file_path",
        default=None,
        type=str,
        help="file path for faiss index (default: None)",
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

    arg = args.parse_args()
    main(arg)
