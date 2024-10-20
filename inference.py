# import argparse

# import torch
# from tqdm import tqdm

# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from parse_config import ConfigParser


# def main(config):
#     dataset = load_from_disk("./data/test_dataset/")
#     predict_dataset = dataset["validation"]

#     model_name = "jhgan/ko-sroberta-multitask"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     dataloader = RetrievalDataLoader(
#         tokenizer=tokenizer,
#         q_max_length=128,
#         c_max_length=None,
#         stride=32,
#         predict_data=predict_dataset,
#         contexts=contexts,
#         batch_size=8,
#     )

#     index_file_path = "./data/embedding/context_index.faiss"
#     assert os.path.isfile(index_file_path), "No index file exists"
#     index = faiss.read_index(index_file_path)
#     sgr = faiss.StandardGpuResources()
#     index = faiss.index_cpu_to_gpu(sgr, 0, index)

#     retrieval.index = index
#     trainer = pl.Trainer(accelerator="gpu")
#     retrieval_output = trainer.predict(retrieval, datamodule=dataloader) ## output 형태가 어떻게 되려나

#     doc_id = [contexts[id.item()] for ids in retrieval_output for id in ids]
#     predict_dataset = predict_dataset.add_column("context", doc_id)

#     dataloader = ReaderDataLoader(
#         tokenizer=tokenizer,
#         max_len=256,
#         stride=32,
#         predict_data=predict_dataset,
#         batch_size=8,
#     )

#     reader_outputs = trainer.predict(reader, datamodule=dataloader)
#     reader_outputs = [answer for batch_outputs in reader_outputs for answer in batch_outputs]


# if __name__ == "__main__":
#     args = argparse.ArgumentParser(description="PyTorch Template")
#     args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
#     args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
#     args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

#     config = ConfigParser.from_args(args)
#     main(config)
