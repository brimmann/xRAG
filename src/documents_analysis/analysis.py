from src.eval.symantic.embedding_utils import get_documents_embeds
from src.documents_analysis.utils import reduce_dimensionality, discover_clusters, visualize_clusters
import torch

OUTPUT_FILENAME = "discovered_document_clusters.png"



if __name__ == "__main__":

    fairy_tales_embeds = get_documents_embeds('brimmann2/squad_qa1', split_name='train', batch_size=128, column_name='content', debug_samples=8550)
    fairy_tales_embeds = torch.stack([t[0] for t in fairy_tales_embeds]).to(torch.float32).numpy()
    print("Shape of embeds: ", fairy_tales_embeds)
    reduced_embeddings = reduce_dimensionality(fairy_tales_embeds)
    discovered_labels = discover_clusters(reduced_embeddings)

    visualize_clusters(reduced_embeddings, discovered_labels, OUTPUT_FILENAME)