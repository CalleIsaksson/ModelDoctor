from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             torch_dtype = torch.bfloat16,
                                             device_map=device)
