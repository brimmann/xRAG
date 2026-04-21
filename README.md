# xRAG: Extreme Context Compression with Gemma 3
This repository is Mohamed Rashid's implementation of xRAG for a Master’s degree thesis in Data Science. The project focuses on compressing context into a single token using the **Gemma 3 (1B)** model.
## Project Status
Current focus is on the **paraphrasing stage**. The model has not been fine-tuned yet, so standard RAG evaluation tasks are currently disabled.
## Get Started
Check the Dockerfile for dependencies.
Set up your environment:
```bash
wandb login
accelerate config

```
## Checkpoints
| Model | Backbone | Status |
|---|---|---|
| xRAG-Gemma-1b | google/gemma-3-1b | Paraphrasing Stage Complete |
## Data & Training
 * **Data:** Uses enwiki-dec2021 for pre-training.
 * **Training:** Focused on teaching the model to compress and reconstruct information.
To run the paraphrasing training:
```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    -m \
    src.language_modeling.train \
    --config config/language_modeling/pretrain.yaml \

```
## New Evaluation Task: Vector Comprehension
Since the model is in the paraphrasing stage, I introduced a new metric to test how well the model understands the compressed vector format.
**The Task:**
 1. Compress a document into the xRAG vector form.
 2. Regenerate the document from that vector.
 3. Calculate the **Cosine Similarity** between the original document and the regenerated document.
This measures how much information the model actually retains in that single token.
## Benchmark
Run the profiler to check performance:
```bash
python -m src.language_modeling.profiler --instruction_length 54 --generation_length 30 --use_xrag

```
