import argparse
import json
import pickle

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


def main(arg):
    context_path = arg.context_path
    model_name = arg.model_name
    contexts_embedding_path = arg.contexts_embedding_path
    k = arg.k
    b = arg.b
    epsilon = arg.epsilon

    with open(context_path, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_contexts = [tokenizer.tokenize(context) for context in contexts.values()]
    bm25 = BM25Okapi(
        tokenized_contexts,
        k1=k,
        b=b,
        epsilon=epsilon,
    )

    with open(contexts_embedding_path, "wb") as file:
        pickle.dump(bm25, file)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-c",
        "--context_path",
        default="./data/wikipedia_documents.json",
        type=str,
        help="directory path for contexts (default: None)",
    )
    args.add_argument(
        "-ce",
        "--contexts_embedding_path",
        default="./saved/embeddings/context_sparse_index.pickle",
        type=str,
        help="directory path for context embedding (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_name",
        default=None,
        type=str,
        help="get model name to call tokenizer (default: None)",
    )
    args.add_argument(
        "-k",
        "--k",
        default=1.5,
        type=float,
        help="bm25 parameter (default: 1.5)",
    )
    args.add_argument(
        "-b",
        "--b",
        default=0.75,
        type=float,
        help="bm25 parameter (default: 0.75)",
    )
    args.add_argument(
        "-e",
        "--epsilon",
        default=0.25,
        type=float,
        help="bm25 parameter (default: 0.25)",
    )

    arg = args.parse_args()
    main(arg)
