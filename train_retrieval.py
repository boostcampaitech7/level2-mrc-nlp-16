import json

import pytorch_lightning as pl
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

import wandb
from data_modules.data_loaders import RetrievalDataLoader
from models.model import RetrievalModel
from utils import set_seed


def main():
    # wandb 로깅
    wandb_logger = WandbLogger()
    config = wandb_logger.experiment.config

    # Parameters
    SEED = config["SEED"]
    DETERMINISTIC = config["DETERMINISTIC"]
    DATA_PATH = config["DATA_PATH"]
    CONTEXTS_PATH = config["CONTEXTS_PATH"]
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    QUESTION_MAX_LEN = config["QUESTION_MAX_LEN"]
    CONTEXT_MAX_LEN = config["CONTEXT_MAX_LEN"]
    CONTEXT_STRIDE = config["CONTEXT_STRIDE"]
    NEGATIVE_LENGTH = config["NEGATIVE_LENGTH"]
    MODEL_NAME = config["MODEL_NAME"]
    MODULE_NAMES = config["MODULE_NAMES"]
    LORA_RANK = config["LORA_RANK"]
    LORA_ALPHA = config["LORA_ALPHA"]
    LORA_DROP_OUT = config["LORA_DROP_OUT"]

    # 시드 고정
    set_seed(SEED, DETERMINISTIC)

    # 데이터셋 로드
    dataset = load_from_disk(DATA_PATH)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    contexts = {value["document_id"]: value["text"] for value in contexts.values()}

    retrieval_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    retrieval_dataloader = RetrievalDataLoader(
        tokenizer=retrieval_tokenizer,
        q_max_length=QUESTION_MAX_LEN,
        c_max_length=CONTEXT_MAX_LEN,
        stride=CONTEXT_STRIDE,
        train_data=train_dataset,
        val_data=valid_dataset,
        contexts=contexts,
        batch_size=BATCH_SIZE,
        negative_length=NEGATIVE_LENGTH,
    )
    retrieval_model = RetrievalModel(dict(config))

    MODEL_NAME = MODEL_NAME
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"saved/{MODEL_NAME}/{wandb.run.id}",
        filename="retrieval_{epoch:02d}",
        save_top_k=1,
        monitor="validation_similarity",
        mode="max",
    )
    early_stop_callback = EarlyStopping(monitor="validation_loss", patience=4, mode="min")

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=1.0,
    )

    trainer.fit(retrieval_model, datamodule=retrieval_dataloader)

    ## best model & configuration uploading
    config_dict = dict(config)
    with open("config_retrieval.json", "w") as f:
        json.dump(config_dict, f)

    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
    artifact.add_file(checkpoint_callback.best_model_path)
    artifact.add_file("config_retrieval.json")
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
