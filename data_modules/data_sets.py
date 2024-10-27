import random

import torch
from torch.utils.data import Dataset


class RetrievalDataset(Dataset):
    """
    retrieval model 입력 데이터셋 클래스

    question과 context를 포함하고 있고,
    fit, test, predict 각 step에 맞춰 negative samplig 등을 통해
    적절한 형태의 데이터를 반환함
    Args:
        question (List[str]): 질문
        contexts (List[str]): 정답이 포함된 문맥
        q_max_length (int): 질문 text의 max length
        c_max_length (int): 문맥 text의 max length
        stage (str): 학습, 평가, 추론 중 어느 단계인지 표현
        tokenizer: 문장을 토큰화하기 위한 토크나이저
        truncation (bool, optional): 토큰화 시 문장 절단 여부
        negative_length (int): 학습 과정에서 활용할 negative sample 수
    """
    def __init__(
        self,
        question,
        tokenizer,
        q_max_length,
        c_max_length,
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
                padding="max_length",
                truncation=self.truncation,
                return_tensors="pt",
            )
            res_context = {
                "input_ids": c_enc["input_ids"],
                "attention_mask": c_enc["attention_mask"],
            }
            res["positive_context"] = res_context

            ## negative context tokenization
            res_negative_context = {}
            for i in range(self.negative_length):
                nc_enc = self.tokenizer(
                    negative_context[i],
                    max_length=self.c_max_length,
                    padding="max_length",
                    truncation=self.truncation,
                    return_tensors="pt",
                )

                res_negative_context[i] = {
                    "input_ids": nc_enc["input_ids"],
                    "attention_mask": nc_enc["attention_mask"],
                }
            res["negative_context"] = res_negative_context

        if self.stage == "test":
            document_id = self.document_id[idx]
            res["document_id"] = document_id

        return res


class ContextDataset(Dataset):
    """
    전체 context embedding 과정에서 활용될 데이터셋 클래스
    context를 batch 단위로 구분하여 반환

    Args:
        context (List[str]): 문맥
        max_length (int): 최대 토큰 길이
        tokenizer: 문장을 토큰화하기 위한 토크나이저
        truncation (bool, optional): 토큰화 시 문장 절단 여부
    """
    def __init__(self, context, document_id, tokenizer, max_length, truncation=True):
        self.context = context
        self.document_id = document_id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

    def __len__(self):
        return len(self.document_id)

    def __getitem__(self, idx):
        context = self.context[idx]
        document_id = self.document_id[idx]

        c_enc = self.tokenizer(
            context,
            max_length=self.max_length,
            padding="max_length",
            truncation=self.truncation,
            return_tensors="pt",
        )

        return {
            "input_ids": c_enc["input_ids"],
            "attention_mask": c_enc["attention_mask"],
            "document_id": document_id,
        }


class ReaderDataset(Dataset):
    """
    reader model 입력 데이터셋 클래스

    question과 context, answer를 포함하고 있고,
    question, context에 대한 토큰화와 더불어
    answer start index, answer text를 통해 적절한 레이블을 생성함
    Args:
        question (List[str]): 질문
        question_id (List[str]): 질문의 id (inference output에 대한 indexing 목적)
        contexts (List[str]): 정답이 포함된 문맥 (test, predict step에서는 retrieval에서 선택된 context가 입력됨)
        answer (List[Dict["start index": int, "answer text": str]]): 정답에 대한 정보
        max_len (int): 질문과 context의 concat에 대한 max length
        stride (int): text의 chunk를 나누는 경우, 이전 text와 다음 text간의 겹쳐지는 토큰 수
        stage (str): 학습, 평가, 추론 중 어느 단계인지 표현
        tokenizer: 문장을 토큰화하기 위한 토크나이저
    """
    def __init__(self, question, context, tokenizer, max_len, stride, stage, question_id=None, answer=None):
        self.question = question
        self.context = context
        self.answer = answer
        self.question_id = question_id
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.stage = stage

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        if self.stage == "fit":
            question = self.question[idx]
            context = self.context[idx]  ## from train or valid dataset

            answer = self.answer[idx]
            answer_start = answer["answer_start"][0]
            answer_end = answer_start + len(answer["text"][0]) - 1

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
            res = {}
            res["input_ids"] = encoding["input_ids"]
            res["attention_mask"] = encoding["attention_mask"]

            res["start_tokens"] = torch.empty(encoding["input_ids"].shape[0])
            res["end_tokens"] = torch.empty(encoding["input_ids"].shape[0])
            for chunk_idx, chunk in enumerate(encoding["offset_mapping"]):
                sequence_ids = encoding.sequence_ids(chunk_idx)
                chunk_start_token = sequence_ids.index(1)
                chunk_end_token = len(sequence_ids) - 1 - (sequence_ids[::-1].index(1))
                if not (answer_start >= chunk[chunk_start_token][0] and answer_end <= chunk[chunk_end_token][1]):
                    res["start_tokens"][chunk_idx] = 0
                    res["end_tokens"][chunk_idx] = 0
                else:
                    for token_idx, (start, end) in enumerate(chunk):
                        if encoding.sequence_ids(chunk_idx)[token_idx] != 1:
                            continue
                        if answer_start in range(start, end + 1):
                            res["start_tokens"][chunk_idx] = token_idx
                        if answer_end in range(start, end + 1):
                            res["end_tokens"][chunk_idx] = token_idx

        elif self.stage in ["predict", "test"]:
            question = self.question[idx]
            contexts = self.context[idx]  ## selected contexts
            if self.stage == "test":
                answer = self.answer[idx]
            if self.stage == "predict":
                question_id = self.question_id[idx]

            res = {}
            for doc_id, context in enumerate(contexts):
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
                doc_res = {}
                for chunk_idx, chunk in enumerate(encoding["input_ids"]):
                    chunk_res = {}
                    chunk_res["input_ids"] = chunk
                    chunk_res["attention_mask"] = encoding["attention_mask"][chunk_idx]
                    if self.stage == "test":
                        chunk_res["answer_text"] = answer["text"][0]
                    if self.stage == "predict":
                        chunk_res["question_id"] = question_id
                    doc_res[chunk_idx] = chunk_res
                res[doc_id] = doc_res

        return res
