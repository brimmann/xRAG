from src.model import SFR
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from src.eval.run_eval import prepare_retrieval_embeds

device = "cuda"

def get_model():
    model_id = 'salesforce/sfr-embedding-mistral'
    model = SFR.from_pretrained(model_id,torch_dtype = torch.bfloat16)
    _ = model.eval()
    _ = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return tokenizer, model

def load_data(dataset_id, split_name):
    ds = load_dataset(dataset_id, split=split_name)
    documents = ds.select(range(3))["text"]
    documents_list = [[s] for s in list(documents)]
    return documents_list



def get_documents_embeds(dataset_id, split_name, batch_size=1):

    print("loading embedder model and tokenizer...")
    tokenizer, model = get_model()
    print("embedder model loaded")
    
    documents_list = load_data(dataset_id, split_name)

    num_samples = len(documents_list)
    original_orders = []
    for idx,background in enumerate(documents_list):
        original_orders.extend(
                [idx] * len(background)
            )

    documents_as_strings_list = [x for y in documents_list for x in y]

    print("generating embeddings...")
    _embeds = prepare_retrieval_embeds(list(documents_as_strings_list), model, tokenizer, batch_size=batch_size)
    print("embeddings generated")

    retrieval_embeds = [[] for _ in range(num_samples)]
    assert len(_embeds) == len(original_orders)
    for id,embeds in zip(original_orders,_embeds):
        retrieval_embeds[id].append(embeds)

    return retrieval_embeds

        

