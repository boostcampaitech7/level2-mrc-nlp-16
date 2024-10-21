import argparse
import json

import faiss
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
    nlist = arg.nlist
    m = arg.m
    index_file_path = arg.index_file_path

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
    retrieval = RetrievalModel(config)
    checkpoint = torch.load(f"{model_dir}/{model_path}")
    retrieval.load_state_dict(checkpoint["model_state_dict"])

    # 학습된 retrieval으로 context의 임베딩을 생성,
    # faiss 인덱스 생성 후 저장
    context_dataset = ContextDataset(
        context=list(contexts.values()),
        document_id=list(contexts.keys()),
        tokenizer=tokenizer,
        max_length=config["context_max_length"],
        stride=config["context_stride"],
    )

    contexts_emb = context_embedding(contextdataset=context_dataset, retrieval=retrieval, batch_size=batch_size)

    c_emb = contexts_emb["contexts_embedding"].detach().numpy().astype("float32")
    emb_size = c_emb.shape[-1]

    quantizer = faiss.IndexFlatIP(emb_size)
    index = faiss.IndexIVFPQ(quantizer, emb_size, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap(index)

    sgr = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(sgr, 0, index)

    faiss.normalize_L2(c_emb)
    index.train(c_emb)
    index.add_with_ids(c_emb, np.array(contexts_emb["document_id"]).astype("int64"))

    cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, index_file_path)


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
        "-mp",
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
    args.add_argument(
        "-n",
        "--nlist",
        default=8,
        type=int,
        help="Inverted File에서 사용할 Voronoi 셀의 수, 일반적으로 데이터셋 크기의 제곱근에 가까운 값을 사용 (default: 8)",
    )
    args.add_argument(
        "-m",
        "--m",
        default=8,
        type=int,
        help="Product Quantizer에서 사용할 서브벡터의 수, 벡터 차원 d가 M의 배수여야 함 (default: 8)",
    )
    args.add_argument(
        "-i",
        "--index_file_path",
        default=None,
        type=str,
        help="file path for faiss index (default: None)",
    )

    arg = args.parse_args()
    main(arg)
