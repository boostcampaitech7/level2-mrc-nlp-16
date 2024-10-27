import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def context_embedding(contextdataset, retrieval, batch_size=2):
    contextloader = DataLoader(contextdataset, batch_size)

    mod = retrieval.mod_c.to("cuda")
    mod.eval()

    embeddings = []
    doc_id = []
    for batch in tqdm(contextloader):
        with torch.no_grad():
            c_emb = mod(
                input_ids=batch["input_ids"].squeeze(1).to("cuda"),
                attention_mask=batch["attention_mask"].squeeze(1).to("cuda"),
            ).last_hidden_state[:, 0, :]

        embeddings.append(c_emb.to("cpu"))
        doc_id.extend(batch["document_id"])

        del c_emb

    return {
        "contexts_embedding": torch.cat(embeddings, dim=0),
        "document_id": doc_id,
    }
