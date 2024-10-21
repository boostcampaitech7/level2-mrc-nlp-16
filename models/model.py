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
    def __init__(self, config, index=None):
        super().__init__()

        self.lr = config["lr"]
        self.model_name = config["model_name"]

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

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def forward(self, input_ids, attention_mask):
        output = self.mod(input_ids, attention_mask=attention_mask)
        return output.start_logits, output.end_logits

    def training_step(self, batch, batch_idx):
        start_logits, end_logits = self(batch["input_ids"], batch["attention_mask"])
        loss = (
            self.criterion(start_logits, batch["start_tokens"].float())
            + self.criterion(end_logits, batch["end_tokens"].float())
        ) / 2
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        start_logits, end_logits = self(batch["input_ids"], batch["attention_mask"])
        loss = (
            self.criterion(start_logits, batch["start_tokens"].float())
            + self.criterion(end_logits, batch["end_tokens"].float())
        ) / 2
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 1. Start/end position prediction
        start_logits, end_logits = self(input_ids, attention_mask)

        # 2. 가장 높은 점수의 start/end 위치 찾기
        start_index = start_logits.argmax(dim=1)
        end_index = end_logits.argmax(dim=1)

        # 3. 최고 점수의 답변 추출
        best_answer = ""
        max_score = float("-inf")

        for i in range(input_ids.size(0)):  # 배치의 각 항목에 대해
            answer_start = start_index[i]
            answer_end = end_index[i]

            if answer_start <= answer_end and answer_end - answer_start < 100:  # 최대 길이 제한
                score = start_logits[i, answer_start] + end_logits[i, answer_end]
                if score > max_score:
                    max_score = score
                    answer_text = self.tokenizer.decode(input_ids[i][answer_start : answer_end + 1])
                    best_answer = answer_text

        return best_answer

    # def predict_step(self, batch, batch_idx):
    #     start_logits, end_logits = self(batch["input_ids"], batch["attention_mask"])
    #     answer_start = start_logits.argmax(dim=1)
    #     answer_end = end_logits.argmax(dim=1)
    #     answers = []
    #     for idx, row in enumerate(batch["input_ids"]):
    #         answer = self.tokenizer.convert_ids_to_tokens(row[answer_start[idx] : answer_end[idx] + 1])
    #         answers.append(self.tokenizer.convert_tokens_to_string(answer))
    #     return answers

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
