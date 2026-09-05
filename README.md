# ModelDoctor

ModelDoctor is a project that uses an existing language model and fine-tunes it into an SLM specialized in diagnosing machine-learning training problems, such as overfitting, underfitting, data leakage, and learning rates that are too high or too low.

The current starting model is:

`meta-llama/Llama-3.2-1B-Instruct`

The training dataset will consist of structured examples derived from published machine-learning experiments and controlled training runs. Each example will connect training metrics and configurations to a diagnosis, explanation, and suggested next experiment.

## Current progress

- Loaded the pretrained Llama model from Hugging Face
- Implemented tokenization
- Applied Llama's chat template
- Generated a response from the model
- Have imported data from a large dataset of machine-learning training results
- Have created an initial labeling guide for the diagnoses the model will later learn to identify
- Have started defining the CSV structure for labeled training examples
