from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             dtype = torch.bfloat16,
                                             device_map=device)

input_prompt = [
    "Hello how are you doing tell me",
    "The capital of Sweden is"
]

#tokenized = tokenizer(input_prompt, padding=True, return_tensors ="pt").to(device)

#print(tokenized["input_ids"].shape)
#print(tokenized["input_ids"])
#tokenizer.batch_decode


prompt_template = [
    {
        "role": "system",
        "content": "You are a smart AI assistant who speaks like a pirate."
    },
    {
        "role": "user",
        "content": "Where does the sun rise"
    }
]

tokenizer.pad_token = tokenizer.eos_token

tokenized = tokenizer.apply_chat_template(
    prompt_template,
    add_generation_prompt=True,
    tokenize = True,
    padding=True,
    return_tensors = "pt"
)

out = model.generate(**tokenized.to(device), max_new_tokens=40)
decoded = tokenizer.batch_decode(out)
print(decoded[0])