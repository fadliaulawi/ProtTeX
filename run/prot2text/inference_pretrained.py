import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="xiao-fei/Prot2Text-V2-11B-Instruct-hf", 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
llama_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", 
    pad_token='<|reserved_special_token_0|>'
)

test_data = pd.read_csv("data/csv/test.csv")
result = []

for i in tqdm(test_data.index):
    example_sequence = test_data.loc[i, "sequence"]

    system_message = (
        "You are a scientific assistant specialized in protein function "
        "predictions. Given the sequence embeddings and other information "
        "of a protein, describe its function clearly and concisely in "
        "professional language. "
    )
    placeholder = '<|reserved_special_token_1|>'
    user_message = "Sequence embeddings: " + placeholder * (len(example_sequence)+2)
    tokenized_prompt = llama_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ], 
        add_generation_prompt=True, 
        tokenize=True, 
        return_tensors="pt", 
        return_dict=True
    )
    tokenized_sequence = esm_tokenizer(
        example_sequence, 
        return_tensors="pt"
    )

    model.eval()
    generated = model.generate(
        inputs=tokenized_prompt["input_ids"].to(model.device),
        attention_mask=tokenized_prompt["attention_mask"].to(model.device),
        protein_input_ids=tokenized_sequence["input_ids"].to(model.device),
        protein_attention_mask=tokenized_sequence["attention_mask"].to(model.device),
        max_new_tokens=1024,
        eos_token_id=128009, 
        pad_token_id=128002,
        return_dict_in_generate=False,
        num_beams=4,
        do_sample=False,
    )
    answer = llama_tokenizer.decode(generated[0], skip_special_tokens=True)
    result.append({
        "id": test_data.loc[i, "accession"],
        "sequence": example_sequence,
        "predicted": answer,
        "label": test_data.loc[i, "function"]
    })

# Save to JSON
import json
with open("data/prot2text/inference_results.json", "w") as f:
    json.dump(result, f, indent=4)
