import torch
import torch.nn.functional as F
from src.eval.symantic.generation_utils import generate_answers
import json
from pathlib import Path
import random
import numpy as np
import argparse

# model_id = 'Hannibal046/xrag-7b'
# # model_id = 'brimmann2/xgemma3-1b-v1'
dataset_id = 'brimmann2/squad_qa1'
split_name = "train"

def get_cosine_similarity(original_docs_emb, generated_docs_emb):

    stacked_embeddings_original = torch.stack([tensor_list[0] for tensor_list in original_docs_emb])
    stacked_embeddings_generated = torch.stack([tensor_list[0] for tensor_list in generated_docs_emb])

    stacked_embeddings_original = stacked_embeddings_original.to("cuda")
    stacked_embeddings_generated = stacked_embeddings_generated.to("cuda")

    cosine_sim_tensor = torch.nn.functional.cosine_similarity(
        stacked_embeddings_generated, 
        stacked_embeddings_original, 
        dim=1
    )

    return cosine_sim_tensor

def save_results_to_json(cosine_sim_tensor, model_id):
    scores_list = cosine_sim_tensor.cpu().float().tolist()
    average_score = cosine_sim_tensor.mean().item()

    data_to_save = {
        'similarity_scores': scores_list,
        'average_score': average_score
    }

    #converting path to python friendly name
    results_folder_name = model_id.replace('/', '_').replace('-', '_')
    file_path = Path(f'results/pretrain/{results_folder_name}/similarity_scores.json')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    print("similarty score saved in: ", file_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Cosine similarity evaluation of generated docs")
    parser.add_argument("-m", "--model_id", default="Hannibal046/xrag-7b", help="xRag model id")
    parser.add_argument("-d", "--debug_samples", type=int, default=None, help="number of debug samples")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="number of debug samples")

    args = parser.parse_args()

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    #     # The following two lines are for deterministic operations on CUDA.
    #     # Note that this can have a performance impact.
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    generated_docs, generated_docs_embeddings, original_docs_embeddings = generate_answers(args.model_id, dataset_id, split_name, debug_samples=args.debug_samples,batch_size=args.batch_size)
    print("generated_docs[0]", generated_docs[0])
    cosine_sim_tensor = get_cosine_similarity(original_docs_embeddings, generated_docs_embeddings)

    save_results_to_json(cosine_sim_tensor, args.model_id)