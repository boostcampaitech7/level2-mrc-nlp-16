import random

import torch
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    def __init__(
        self,
        question,
        tokenizer,
        q_max_length,
        c_max_length,
        stride,
        stage,
        contexts=None,
        document_id=None,
        truncation=True,
        negative_length=2,
    ):
        self.question = question
        self.contexts = contexts
        self.document_id = document_id
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.c_max_length = c_max_length
        self.stride = stride
        self.truncation = truncation
        self.stage = stage
        self.negative_length = negative_length

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        question = self.question[idx]

        res = {}
        ## question tokenization
        q_enc = self.tokenizer(
            question,
            max_length=self.q_max_length,
            padding="max_length",
            truncation=self.truncation,
            return_tensors="pt",
        )
        res_question = {
            "input_ids": q_enc["input_ids"],
            "attention_mask": q_enc["attention_mask"],
        }
        res["question"] = res_question

        if self.stage == "fit":
            ## positive context tokenization
            document_id = self.document_id[idx]
            context = self.contexts[document_id]

            indices = random.sample([i for i in range(len(self.contexts)) if document_id != i], self.negative_length)
            negative_context = [self.contexts[idx] for idx in indices]

            c_enc = self.tokenizer(
                context,
                max_length=self.c_max_length,
                # stride=self.stride,
                padding="max_length",
                truncation=self.truncation,
                # return_overflowing_tokens=True,
                # return_offsets_mapping=True,
                return_tensors="pt",
            )
            res_context = {
                "input_ids": c_enc["input_ids"],
                "attention_mask": c_enc["attention_mask"],
                # "overflow": c_enc["overflow_to_sample_mapping"],
            }
            res["positive_context"] = res_context

            ## negative context tokenization
            res_negative_context = {}
            for i in range(self.negative_length):
                nc_enc = self.tokenizer(
                    negative_context[i],
                    max_length=self.c_max_length,
                    # stride=self.stride,
                    padding="max_length",
                    truncation=self.truncation,
                    # return_overflowing_tokens=True,
                    # return_offsets_mapping=True,
                    return_tensors="pt",
                )

                res_negative_context[i] = {
                    "input_ids": nc_enc["input_ids"],
                    "attention_mask": nc_enc["attention_mask"],
                    # "overflow": nc_enc["overflow_to_sample_mapping"],
                }
            res["negative_context"] = res_negative_context

        if self.stage == "test":
            document_id = self.document_id[idx]
            res["document_id"] = document_id

        return res


class ContextDataset(Dataset):
    def __init__(self, context, document_id, tokenizer, max_length, stride, truncation=True):
        self.context = context
        self.document_id = document_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.truncation = truncation

    def __len__(self):
        return len(self.document_id)

    def __getitem__(self, idx):
        context = self.context[idx]
        document_id = self.document_id[idx]

        c_enc = self.tokenizer(
            context,
            max_length=self.max_length,
            stride=self.stride,
            padding="max_length",
            truncation=self.truncation,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        return {
            "input_ids": c_enc["input_ids"],
            "attention_mask": c_enc["attention_mask"],
            "overflow": c_enc["overflow_to_sample_mapping"],
            "document_id": document_id,
        }


class ReaderDataset(Dataset):
    def __init__(self, question, context, tokenizer, max_len, stride, stage, answer=None):
        self.question = question
        self.context = context
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.stage = stage

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        question = self.question[idx]
        context = self.context[idx]

        res = {}
        ## question tokenization
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_len,
            stride=self.stride,
            padding="max_length",
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        res["input_ids"] = encoding["input_ids"]
        res["attention_mask"] = encoding["attention_mask"]
        res["token_type_ids"] = encoding["token_type_ids"]

        if self.stage in ["fit", "test"]:
            answer = self.answer[idx]
            answer_start = answer["answer_start"][0]
            answer_end = answer_start + len(answer["text"][0]) - 1

            ## answer one-hot encoding
            start_tokens = torch.zeros_like(encoding["input_ids"])
            end_tokens = torch.zeros_like(encoding["input_ids"])
            for chunk_idx, chunk in enumerate(encoding["offset_mapping"]):
                for token_idx, (start, end) in enumerate(chunk):
                    if encoding["token_type_ids"][chunk_idx, token_idx] == 0:
                        continue
                    if answer_start in range(start, end + 1):
                        start_tokens[chunk_idx, token_idx] = 1
                    if answer_end in range(start, end + 1):
                        end_tokens[chunk_idx, token_idx] = 1
            res["start_tokens"] = start_tokens
            res["end_tokens"] = end_tokens

        return res
