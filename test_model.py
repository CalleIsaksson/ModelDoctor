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



SYSTEM_PROMPT = """
You are ModelDoctor, an assistant that diagnoses machine-learning training problems.

Use only the supplied training metrics and configuration.

The diagnosis must be one of:
- healthy
- overfitting
- underfitting
- learning_rate_too_high
- learning_rate_too_low
- data_leakage_suspected
- insufficient_evidence

Answer using:
Diagnosis:
Evidence:
Next experiment:
"""

USER_TEMPLATE = """
Analyze this training run.

Model: {model_name}
Task: {task}
Train loss: {train_loss}
Validation loss: {validation_loss}
Train metric: {train_metric}
Validation metric: {validation_metric}
Learning rate: {learning_rate}
Epochs: {epochs}
Additional information: {additional_information}
"""

user_content = USER_TEMPLATE.format(
    model_name=row["model_name"],
    task=row["task"],
    train_loss=row["train_loss"],
    validation_loss=row["validation_loss"],
    train_metric=row["train_metric"],
    validation_metric=row["validation_metric"],
    learning_rate=row["learning_rate"],
    epochs=row["epochs"],
    additional_information=row["additional_information"]
)