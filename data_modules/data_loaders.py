import pytorch_lightning as pl
import torch
from data_sets import ReaderDataset, RetrievalDataset
from torch.utils.data import DataLoader


class RetrievalDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        q_max_length,
        c_max_length,
        stride,
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
                stride=self.stride,
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
                stride=self.stride,
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
                stride=self.stride,
                stage=stage,
                truncation=self.truncation,
            )
        elif stage == "predict":
            self.predict_dataset = RetrievalDataset(
                question=self.predict_data["question"],
                tokenizer=self.tokenizer,
                q_max_length=self.q_max_length,
                c_max_length=None,
                stride=self.stride,
                stage=stage,
                truncation=self.truncation,
            )

    def collate_fn(self, batch):
        q_inputs = torch.cat([item["question"]["input_ids"] for item in batch])
        q_masks = torch.cat([item["question"]["attention_mask"] for item in batch])

        c_inputs = torch.cat([item["positive_context"]["input_ids"] for item in batch])
        c_masks = torch.cat([item["positive_context"]["attention_mask"] for item in batch])
        # c_overflow = torch.cat([item["positive_context"]["overflow"]+idx for idx, item in enumerate(batch)]) ## dataset에서 문장 별로 tokenization적용하기 때문에, overflow가 0이 되는 문제 발생함 ## batch내의 각 문장 별로 index를 다르게 더해줌으로써 overflow 구현

        nc_inputs = [
            torch.cat([item["negative_context"][i]["input_ids"] for item in batch]) for i in range(self.negative_length)
        ]
        nc_masks = [
            torch.cat([item["negative_context"][i]["attention_mask"] for item in batch])
            for i in range(self.negative_length)
        ]
        # nc_overflow = [torch.cat([item["negative_context"][i]["overflow"]+idx for idx, item in enumerate(batch)]) for i in range(self.negative_length)]

        return {
            "question": {"input_ids": q_inputs, "attention_mask": q_masks},
            "positive_context": {
                "input_ids": c_inputs,
                "attention_mask": c_masks,
                # "overflow": c_overflow
            },
            "negative_context": {
                i: {
                    "input_ids": nc_inputs[i],
                    "attention_mask": nc_masks[i],
                    # "overflow": nc_overflow[i]
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
    def __init__(
        self,
        tokenizer,
        max_len,
        stride,
        train_data=None,
        val_data=None,
        test_data=None,
        predict_data=None,
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
                context=self.test_data["context"],
                answer=self.test_data["answers"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )
        elif stage == "predict":
            self.predict_dataset = ReaderDataset(
                question=self.predict_data["question"],
                context=self.predict_data["context"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                stride=self.stride,
                stage=stage,
            )

    def collate_fn(self, batch):
        input_ids = torch.cat([item["input_ids"] for item in batch])
        attention_mask = torch.cat([item["attention_mask"] for item in batch])
        token_type_ids = torch.cat([item["token_type_ids"] for item in batch])
        start_tokens = torch.cat([item["start_tokens"] for item in batch])
        end_tokens = torch.cat([item["end_tokens"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_tokens": start_tokens,
            "end_tokens": end_tokens,
        }

    def collate_fn_predict(self, batch):
        input_ids = torch.cat([item["input_ids"] for item in batch])
        attention_mask = torch.cat([item["attention_mask"] for item in batch])
        token_type_ids = torch.cat([item["token_type_ids"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn_predict)
