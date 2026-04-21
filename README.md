Here is the updated README tailored for your thesis work. I stripped out the old image links, swapped the model info to Gemma 3, and updated the training and evaluation sections to reflect your new cosine similarity metric and current progress.
```markdown
# xRAG (Gemma 3 1B Fork)

This is a modified version of the original [xRAG](https://arxiv.org/abs/2405.13792) repository, developed as part of my Master's thesis in Data Science. 

**Key changes in this fork:**
* **Model Swap:** Replaced the original backbones with the lighter 1-billion parameter Gemma 3 model.
* **Current Training Stage:** The model has only been trained on the **paraphrasing stage**. It has not been fine-tuned yet, so it cannot answer standard RAG evaluation tasks at this time.
* **New Evaluation Task:** Introduced a custom evaluation metric to measure paraphrasing comprehension. This task calculates the cosine similarity between the original document and the document regenerated from its vector form to see how well the model actually understands the vectorized format.

## Get Started
Refer to the `Dockerfile` for required packages.

Configure your environment:
```bash
wandb login
accelerate config

```
## Data
 * Download enwiki-dec2021 as pretraining data and the retrieval corpus.
 * Prepare your instruction tuning data using prepare_data.ipynb.
 * Use ColBERT-v2 to conduct retrieval.
## Training (Paraphrasing Stage)
Training scripts are in scripts/. To train the Gemma 3 (1B) model on the paraphrasing stage:
```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 29666 \
    -m \
    src.language_modeling.train \
    --config config/language_modeling/pretrain.yaml \

```
## Evaluation (Cosine Similarity)
Since the model is not yet fine-tuned for downstream generation, standard RAG evaluation is disabled.
To run the custom paraphrasing evaluation (measuring cosine similarity between the original document and the vector-regenerated document):
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eval.run_similarity_eval \
        --model_name_or_path your-local-path/xrag-gemma3-1b

```
```

```
