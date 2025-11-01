from transformers import AutoTokenizer
from transformers import AutoConfig
import torch
from src.model import XMistralForCausalLM, XGemma3ForCausalLM
import random
from src.eval.symantic.embedding_utils import get_documents_embeds
from src.eval.run_eval import llm_for_open_generation

device = "cuda"
XRAG_TOKEN = "<xRAG>" 
ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]

def get_model(modle_id):
    config = AutoConfig.from_pretrained(modle_id)
    MODEL_CLASS = eval(config.architectures[0])
    print("loading generator model...")
    model = MODEL_CLASS.from_pretrained(
        modle_id,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        device_map= device,
    )
    print("generator model loaded")

    tokenizer = AutoTokenizer.from_pretrained(
        modle_id,
        padding_side = 'left',
        add_eos_token=False, ## import to include this!
        use_fast=False,
    )

    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model

def prepare_prompts(tokenizer, sample_length):
    prmopts = [
        random.choice(ParaphraseInstructions).format_map(
                dict
                (xrag_token=XRAG_TOKEN)
            ) for _ in range(sample_length)
    ]

    all_messages = [
        [
            {"role": "user", "content": p} 
        ] for p in prmopts
    ]
    print("all messages", all_messages)

    prompts = tokenizer.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True)
    print("prompts", prompts)

    return prompts


def generate_answers(model_id, datatset_id, split, batch_size=4):

    tokenizer, model = get_model(model_id)
    model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
    _ = model.eval()

    embeddings = get_documents_embeds(datatset_id, split)

    sample_length = len(embeddings)

    prompts = prepare_prompts(tokenizer, sample_length)

    print("generatring documetns from embeddings...")
    results = llm_for_open_generation(model,tokenizer,prompts,embeddings,batch_size,True)
    print("documents generated")

    return results



