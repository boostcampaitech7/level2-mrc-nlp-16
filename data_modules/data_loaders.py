import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data_modules.data_sets import ReaderDataset, RetrievalDataset


class RetrievalDataLoader(pl.LightningDataModule):
    """
    retrieval model의 input 형태로 데이터를 처리 및 로드

    train, validation, test, predict 데이터셋을 관리하고,
    분석의 각 단계에 맞는 데이터로더를 제공
    Args:
        tokenizer: 문장을 토큰화하는 토크나이저
        q_max_length (int): question에 대한 최대 시퀀스 길이
        c_max_length (int): context에 대한 최대 시퀀스 길이
        train_data (List[str]): 학습 데이터
        val_data (List[str]): 검증 데이터
        test_data (List[str]): 평가 데이터
        predict_data (List[str]): 추론 대상 데이터
        contexts (List[str]): 전체 context 데이터
        truncation (bool, optional): 문장 절단 여부
        batch_size (int): 배치 크기
        negative_length (int): 학습 과정에서 활용할 negative sample 수 
    """
    def __init__(
        self,
        tokenizer,
        q_max_length,
        c_max_length,
        stride,   ## 지워야 함
        train_data=None,
        val_data=None,
        test_data=None,
        predict_data=None,
        contexts=None,
        truncation=True,
        batch_size=8,
        negative_length=2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.c_max_length = c_max_length
        self.stride = stride
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data
        self.contexts = contexts
        self.truncation = truncation
        self.batch_size = batch_size
        self.negative_length = negative_length

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = RetrievalDataset(
                question=self.train_data["question"],
                contexts=self.contexts,
                document_id=self.train_data["document_id"],
                tokenizer=self.tokenizer,
                q_max_length=self.q_max_length,
                c_max_length=self.c_max_length,
                stage=stage,
                truncation=self.truncation,
                negative_length=self.negative_length,
            )
            self.val_dataset = RetrievalDataset(
                question=self.val_data["question"],
                contexts=self.contexts,
                document_id=self.val_data["document_id"],
                tokenizer=self.tokenizer,
                q_max_length=self.q_max_length,
                c_max_length=self.c_max_length,
                stage=stage,
                truncation=self.truncation,
                negative_length=self.negative_length,
            )
        elif stage == "test":
            self.test_dataset = RetrievalDataset(
                question=self.test_data["question"],
                document_id=self.test_data["document_id"],
                tokenizer=self.tokenizer,
                q_max_length=self.q_max_length,
                c_max_length=None,
                stage=stage,
                truncation=self.truncation,
            )
        elif stage == "predict":
            self.predict_dataset = RetrievalDataset(
                question=self.predict_data["question"],
                tokenizer=self.tokenizer,
                q_max_length=self.q_max_length,
                c_max_length=None,
                stage=stage,
                truncation=self.truncation,
            )

    def collate_fn(self, batch):
        q_inputs = torch.cat([item["question"]["input_ids"] for item in batch])
        q_masks = torch.cat([item["question"]["attention_mask"] for item in batch])

        c_inputs = torch.cat([item["positive_context"]["input_ids"] for item in batch])
        c_masks = torch.cat([item["positive_context"]["attention_mask"] for item in batch])

        nc_inputs = [
            torch.cat([item["negative_context"][i]["input_ids"] for item in batch]) for i in range(self.negative_length)
        ]
        nc_masks = [
            torch.cat([item["negative_context"][i]["attention_mask"] for item in batch])
            for i in range(self.negative_length)
        ]

        return {
            "question": {"input_ids": q_inputs, "attention_mask": q_masks},
            "positive_context": {
                "input_ids": c_inputs,
                "attention_mask": c_masks,
            },
            "negative_context": {
                i: {
                    "input_ids": nc_inputs[i],
                    "attention_mask": nc_masks[i],
                }
                for i in range(self.negative_length)
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)


class ReaderDataLoader(pl.LightningDataModule):
    """
    reader model의 input 형태로 데이터를 처리 및 로드

    train, validation, test, predict 데이터셋을 관리하고,
    분석의 각 단계에 맞는 데이터로더를 제공
    Args:
        tokenizer: 문장을 토큰화하는 토크나이저
        max_len (int): 최대 시퀀스 길이
        train_data (List[str]): 학습 데이터
        val_data (List[str]): 검증 데이터
        test_data (List[str]): 평가 데이터
        predict_data (List[str]): 추론 대상 데이터
        selected_contexts (List[str]): retrieval을 통해 선택된 contexts
        truncation (bool, optional): 문장 절단 여부
        batch_size (int): 배치 크기
    """
    def __init__(
        self,
        tokenizer,
        max_len,
        stride,   ## 지워야 함
        train_data=None,
        val_data=None,
        test_data=None,
        predict_data=None,
        selected_contexts=None,
        truncation=True,
        batch_size=8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.predict_data = predict_data
        self.selected_contexts = selected_contexts
        self.truncation = truncation
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ReaderDataset(
                question=self.train_data["question"],
                context=self.train_data["context"],
                answer=self.train_data["answers"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )
            self.val_dataset = ReaderDataset(
                question=self.val_data["question"],
                context=self.val_data["context"],
                answer=self.val_data["answers"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )
        elif stage == "test":
            self.test_dataset = ReaderDataset(
                question=self.test_data["question"],
                context=self.selected_contexts,
                answer=self.test_data["answers"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )
        elif stage == "predict":
            self.predict_dataset = ReaderDataset(
                question=self.predict_data["question"],
                context=self.selected_contexts,
                question_id=self.predict_data["id"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
