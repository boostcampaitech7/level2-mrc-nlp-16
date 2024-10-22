import faiss
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging,
    pipeline,
)

from models.metric import compute_exact, compute_f1
from utils.util import normalize_rows


class RetrievalModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model_name = config["MODEL_NAME"]
        self.module_names = config["MODULE_NAMES"]
        self.lr = config["LEARNING_RATE"]
        self.negative_length = config["NEGATIVE_LENGTH"]
        self.LoRA_r = config["LORA_RANK"]
        self.LoRA_alpha = config["LORA_ALPHA"]
        self.LoRA_drop_out = config["LORA_DROP_OUT"]
        self.c_emb = None

        mod = AutoModel.from_pretrained(self.model_name)

        peft_config = LoraConfig(
            r=self.LoRA_r,
            lora_alpha=self.LoRA_alpha,
            target_modules=self.module_names,
            lora_dropout=self.LoRA_drop_out,
            bias="none",
        )
        self.mod_q = get_peft_model(mod, peft_config)
        self.mod_c = get_peft_model(mod, peft_config)

        self.criterion = nn.NLLLoss()

    def forward(self, question, context):
        q_emb = self.mod_q(
            input_ids=question["input_ids"], attention_mask=question["attention_mask"]
        ).last_hidden_state[:, 0, :]
        c_emb = self.mod_c(input_ids=context["input_ids"], attention_mask=context["attention_mask"]).last_hidden_state[
            :, 0, :
        ]

        return q_emb, c_emb

    def training_step(self, batch, batch_idx):
        q_emb, c_emb = self(batch["question"], batch["positive_context"])
        c_embs = [c_emb]
        for i in range(self.negative_length):
            _, c_emb = self(batch["question"], batch["negative_context"][i])
            c_embs.append(c_emb)
        c_embs = torch.stack(c_embs)
        sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=2).T
        sims_softmax = F.log_softmax(sims)

        targets = torch.zeros(c_embs.shape[1]).long()
        loss = self.criterion(sims_softmax, targets.to("cuda"))
        sim = sims.mean(dim=0)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_similarity", sim[0], on_step=False, on_epoch=True)
        for i in range(self.negative_length):
            self.log(f"train_negative_similarity_{i+1}", sim[i + 1], on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "similarity": sim,
        }

    def validation_step(self, batch, batch_idx):
        q_emb, c_emb = self(batch["question"], batch["positive_context"])
        c_embs = [c_emb]
        for i in range(self.negative_length):
            _, c_emb = self(batch["question"], batch["negative_context"][i])
            c_embs.append(c_emb)
        c_embs = torch.stack(c_embs)
        sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=2).T
        sims_softmax = F.log_softmax(sims)

        targets = torch.zeros(c_embs.shape[1]).long()
        loss = self.criterion(sims_softmax, targets.to("cuda"))
        sim = sims.mean(dim=0)
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        self.log("validation_similarity", sim[0], on_step=False, on_epoch=True)
        for i in range(self.negative_length):
            self.log(f"validation_negative_similarity_{i+1}", sim[i + 1], on_step=False, on_epoch=True)
        return {"loss": loss, "similarity": sim}

    def test_step(self, batch, batch_idx):
        q_emb = self.mod_q(
            input_ids=batch["question"]["input_ids"].squeeze(),
            attention_mask=batch["question"]["attention_mask"].squeeze(),
        ).last_hidden_state[:, 0, :]
        q_emb = q_emb.cpu().detach().numpy().astype("float32")
        q_emb = normalize_rows(q_emb)

        sims = np.dot(q_emb, self.c_emb.T).squeeze()
        return sims

    def predict_step(self, batch, batch_idx):
        q_emb = self.mod_q(
            input_ids=batch["question"]["input_ids"].squeeze(),
            attention_mask=batch["question"]["attention_mask"].squeeze(),
        ).last_hidden_state[:, 0, :]
        q_emb = q_emb.cpu().detach().numpy().astype("float32")
        q_emb = normalize_rows(q_emb)

        sims = np.dot(q_emb, self.c_emb.T).squeeze()
        return sims

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ReaderModel(pl.LightningModule):
    """
    질문에 대한 답변의 시작 및 끝 위치를 예측하는 모델.

    Args:
        config (dict): 모델 설정을 포함하는 딕셔너리.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model_name = config["MODEL_NAME"]
        self.module_names = config["MODULE_NAMES"]
        self.lr = config["LEARNING_RATE"]
        self.LoRA_r = config["LORA_RANK"]
        self.LoRA_alpha = config["LORA_ALPHA"]
        self.LoRA_drop_out = config["LORA_DROP_OUT"]

        # 사전 학습된 QA 모델 로드 및 LoRA 적용
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        peft_config = LoraConfig(
            r=self.LoRA_r,
            lora_alpha=self.LoRA_alpha,
            target_modules=self.module_names,
            lora_dropout=self.LoRA_drop_out,
            bias="none",
        )
        self.mod = get_peft_model(model, peft_config)

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, input_ids, attention_mask):
        """
        모델의 순전파 과정.

        Args:
            input_ids (Tensor): 입력 토큰 ID.
            attention_mask (Tensor): 어텐션 마스크.

        Returns:
            Tuple[Tensor, Tensor]: Start logits과 End logits.
        """
        output = self.mod(input_ids=input_ids, attention_mask=attention_mask)
        return output.start_logits, output.end_logits

    def training_step(self, batch, batch_idx):
        """
        학습 단계에서의 손실 계산.

        Args:
            batch (dict): 배치 데이터.
            batch_idx (int): 배치 인덱스.

        Returns:
            Tensor: 손실 값.
        """
        input_ids = batch["input_ids"].transpose(0, 1)
        attention_mask = batch["attention_mask"].transpose(0, 1)

        start_logits, end_logits = [], []
        for chunk_idx in range(batch["input_ids"].shape[1]):
            start_logits_for_chunk, end_logits_for_chunk = self(input_ids[chunk_idx], attention_mask[chunk_idx])
            start_logits.append(start_logits_for_chunk)
            end_logits.append(end_logits_for_chunk)

        loss_start = self.criterion(torch.cat(start_logits, dim=0), batch["start_tokens"].squeeze(0).long())
        loss_end = self.criterion(torch.cat(end_logits, dim=0), batch["end_tokens"].squeeze(0).long())
        loss = (loss_start + loss_end) / 2
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        검증 단계에서의 손실 계산.

        Args:
            batch (dict): 배치 데이터.
            batch_idx (int): 배치 인덱스.

        Returns:
            Tensor: 손실 값.
        """
        input_ids = batch["input_ids"].transpose(0, 1)
        attention_mask = batch["attention_mask"].transpose(0, 1)

        start_logits, end_logits = [], []
        match = 0
        for chunk_idx in range(batch["input_ids"].shape[1]):
            start_logits_for_chunk, end_logits_for_chunk = self(input_ids[chunk_idx], attention_mask[chunk_idx])
            start_logits.append(start_logits_for_chunk)
            end_logits.append(end_logits_for_chunk)

            answer_start_real = int(batch["start_tokens"].squeeze(0)[chunk_idx].item())
            answer_end_real = int(batch["end_tokens"].squeeze(0)[chunk_idx].item())

            if not (answer_start_real == 0 and answer_end_real == 0):
                answer_start = start_logits_for_chunk.argmax()
                answer_end = end_logits_for_chunk.argmax()
                match = int(answer_start_real == answer_start and answer_end_real == answer_end)

        loss_start = self.criterion(torch.cat(start_logits, dim=0), batch["start_tokens"].squeeze(0).long())
        loss_end = self.criterion(torch.cat(end_logits, dim=0), batch["end_tokens"].squeeze(0).long())
        loss = (loss_start + loss_end) / 2
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Exact Match", match, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        max_prob = 0
        answer = ""
        answer_real = batch[0][0]["answer_text"][0]
        for doc_id in range(len(batch)):
            for chunk_idx in range(len(batch[doc_id])):
                input_ids = batch[doc_id][chunk_idx]["input_ids"]
                attention_mask = batch[doc_id][chunk_idx]["attention_mask"]
                start_logits, end_logits = self(input_ids, attention_mask)
                start_logit, end_logit = start_logits.max(), end_logits.max()
                answer_start = start_logits.argmax()
                answer_end = end_logits.argmax()
                if start_logit * end_logit > max_prob and answer_start != 0 and answer_end != 0:
                    answer = self.tokenizer.decode(input_ids[0, answer_start : answer_end + 1])
        EM = compute_exact(answer_real, answer)
        F1 = compute_f1(answer_real, answer)
        self.log("Test_Exact_Match", EM, on_step=False, on_epoch=True)
        self.log("Test_F1_Socre", F1, on_step=False, on_epoch=True)
        return answer

    def predict_step(self, batch, batch_idx):
        """
        예측 단계에서 각 질문에 대한 답변을 생성합니다.

        Args:
            batch (dict): 배치 데이터로, 'input_ids'와 'attention_mask'를 포함합니다.
            batch_idx (int): 배치 인덱스.

        Returns:
            list: 각 질문에 대한 예측된 답변 리스트.
        """
        max_prob = 0
        answer = ""
        question_id = batch[0][0]["question_id"][0]
        for doc_id in range(len(batch)):
            for chunk_idx in range(len(batch[doc_id])):
                input_ids = batch[doc_id][chunk_idx]["input_ids"]
                attention_mask = batch[doc_id][chunk_idx]["attention_mask"]
                start_logits, end_logits = self(input_ids, attention_mask)
                start_logit, end_logit = start_logits.max(), end_logits.max()
                answer_start = start_logits.argmax()
                answer_end = end_logits.argmax()
                if start_logit * end_logit > max_prob and answer_start != 0 and answer_end != 0:
                    answer = self.tokenizer.decode(input_ids[0, answer_start : answer_end + 1])
        return question_id, answer

    def configure_optimizers(self):
        """
        옵티마이저 설정.

        Returns:
            Optimizer: Adam 옵티마이저.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
