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


class RetrievalModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model_name = config["model_name"]
        self.module_names = config["module_names"]
        self.lr = config["lr"]
        self.negative_length = config["negative_length"]
        self.LoRA_r = config["LoRA_r"]
        self.LoRA_alpha = config["LoRA_alpha"]
        self.LoRA_drop_out = config["LoRA_drop_out"]
        self.index = None

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

        ## mean method
        # c_emb_list = []
        # unique_overflow = context["overflow"].unique()
        # for idx in unique_overflow:
        #     row = c_emb[context["overflow"]==idx, :].mean(dim=0)
        #     c_emb_list.append(row)
        # c_emb = torch.stack(c_emb_list)

        # nc_emb_list = []
        # unique_overflow = negative_context["overflow"].unique()
        # for idx in unique_overflow:
        #     row = nc_emb[negative_context["overflow"]==idx, :].mean(dim=0)
        #     nc_emb_list.append(row)
        # nc_emb = torch.stack(nc_emb_list)

        ## max method
        # c_emb_list = []
        # unique_overflow = context["overflow"].unique()
        # for idx in unique_overflow:
        #     rows = c_emb[context["overflow"]==idx, :]
        #     sim_max_index = q_emb[idx, :].matmul(rows.T).argmax()
        #     c_emb_list.append(rows[sim_max_index, :])
        # c_emb = torch.stack(c_emb_list)

        # nc_emb_list = []
        # unique_overflow = negative_context["overflow"].unique()
        # for idx in unique_overflow:
        #     rows = nc_emb[negative_context["overflow"]==idx, :]
        #     sim_max_index = q_emb[idx, :].matmul(rows.T).argmax()
        #     nc_emb_list.append(rows[sim_max_index, :])
        # nc_emb = torch.stack(nc_emb_list)

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
        q_emb = np.ascontiguousarray(q_emb)
        faiss.normalize_L2(q_emb)
        self.index.nprobe = 5
        sim, idx = self.index.search(q_emb, 1)  ## 상위 k개 뽑는 방식 고려 -> k개 뽑아서 k개 모두에 대해 answer 구하려고?
        match_ratio = (batch["document_id"].cpu().detach().numpy() == idx.squeeze()).sum() / len(idx.squeeze())
        self.log("cosine_smilarity", sim.mean(), on_step=False, on_epoch=True)
        self.log("match_ratio", match_ratio, on_step=False, on_epoch=True)
        return sim, match_ratio

    def predict_step(self, batch, batch_idx):
        q_emb = self.mod_q(
            input_ids=batch["question"]["input_ids"].squeeze(),
            attention_mask=batch["question"]["attention_mask"].squeeze(),
        ).last_hidden_state[:, 0, :]
        _, idx = self.index.search(q_emb.cpu().detach().numpy(), 1)
        return idx

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

        self.lr = config["lr"]
        self.model_name = config["model_name"]

        # 사전 학습된 QA 모델 로드 및 LoRA 적용
        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        peft_config = LoraConfig(
            r=config["LoRA_r"],
            lora_alpha=config["LoRA_alpha"],
            target_modules=["query", "key", "value"],
            lora_dropout=config["LoRA_drop_out"],
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
        output = self.mod(input_ids, attention_mask=attention_mask)
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
        start_logits, end_logits = self(batch["input_ids"], batch["attention_mask"])
        loss = (
            self.criterion(start_logits, batch["start_tokens"].long())
            + self.criterion(end_logits, batch["end_tokens"].long())
        ) / 2
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
        start_logits, end_logits = self(batch["input_ids"], batch["attention_mask"])
        loss = (
            self.criterion(start_logits, batch["start_tokens"].long())
            + self.criterion(end_logits, batch["end_tokens"].long())
        ) / 2
        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        예측 단계에서 각 질문에 대한 답변을 생성합니다.

        Args:
            batch (dict): 배치 데이터로, 'input_ids'와 'attention_mask'를 포함합니다.
            batch_idx (int): 배치 인덱스.

        Returns:
            list: 각 질문에 대한 예측된 답변 리스트.
        """
        try:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Start와 End 위치 예측
            start_logits, end_logits = self(input_ids, attention_mask)

            # Start와 End 위치의 인덱스 찾기
            start_index = start_logits.argmax(dim=1)
            end_index = end_logits.argmax(dim=1)

            # 각 샘플에 대한 답변 추출
            best_answers = []
            for i in range(input_ids.size(0)):
                answer_start = start_index[i]
                answer_end = end_index[i]

                # 유효한 답변인지 확인
                if answer_start <= answer_end and answer_end - answer_start < 100:
                    # 답변 토큰 디코딩
                    answer_text = self.tokenizer.decode(
                        input_ids[i][answer_start : answer_end + 1], skip_special_tokens=True
                    )
                    best_answers.append(answer_text)
                else:
                    best_answers.append("")

            return best_answers
        except KeyError as e:
            raise KeyError(f"배치 데이터에 필요한 키가 누락되었습니다: {e}")
        except Exception as e:
            raise RuntimeError(f"예측 중 오류가 발생했습니다: {e}")

    def configure_optimizers(self):
        """
        옵티마이저 설정.

        Returns:
            Optimizer: Adam 옵티마이저.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
