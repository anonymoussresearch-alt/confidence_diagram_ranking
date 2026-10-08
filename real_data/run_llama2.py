# from llama import Llama
import os
import torch.distributed as dist
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import json
from datasets import load_dataset

# task_id = os.getenv('SLURM_ARRAY_TASK_ID')
# task_id = int(task_id) 
# ss0 = ['anatomy','clinical_knowledge','college_biology','college_medicine','medical_genetics']
# ss = ss0[task_id]
ss = 'college_biology'
print(ss)
num = 100
model = 'llama2'
ckpt_dir = "PATH/TO/Llama-2-7b-chat-hf"
tokenizer_path = "PATH/TO/Llama-2-7b-chat-hf"
# model = llama1
# ckpt_dir = "PATH/TO/llama-7b-hf"
# tokenizer_path = "PATH/TO/llama-7b-hf"
# Set environment variables

# "RANK": The rank of the process in the distributed training setup. In this code, there's only one process, so its rank is 0.
os.environ["RANK"] = "0"

# "WORLD_SIZE": The total number of processes that will be used for distributed training. Here, it's set to 1, meaning only one process.
os.environ["WORLD_SIZE"] = "1"


# Additional setup for local backend:
# These are setup commands specific to running distributed training on the local machine (localhost).

# "MASTER_ADDR": The IP address of the machine where the master process will run. Since it's a localhost setup, the IP is set to 127.0.0.1.
os.environ['MASTER_ADDR'] = '127.0.0.1'

# "MASTER_PORT": The port on which the master process will communicate. 
os.environ['MASTER_PORT'] = '29521'

# Initialize the process group for distributed training. 
# This is a necessary step before starting any distributed operations in PyTorch.
# backend='gloo': Specifies the backend to use. "gloo" is a collective communication library.
# rank=0: Rank of the current process. Here, it's the only process with rank 0.
# world_size=1: Number of processes in the group. Only one process in this case.
dist.init_process_group(backend='gloo', rank=0, world_size=1)

def generate_responses(prompts, ckpt_dir, tokenizer_path, max_length=100):
    # Load the tokenizer
    model = LlamaForCausalLM.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    responses = []
    
    for prompt in prompts:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        # .to(device)
        
        # Generate the response
        # generate_ids = model.generate(inputs.input_ids, max_length=max_length)
        try:
            generate_ids = model.generate(inputs.input_ids, max_length=max_length)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('Out of memory error, trying with smaller max_length')
                generate_ids = model.generate(inputs.input_ids, max_length=max_length/2)
            else:
                raise e

        # Decode the generated tokens to text
        generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        responses.append({
            'prompt': prompt,
            'response':  generated_text
        })
        
        # print(f"Prompt: {prompt}\nResponse: {generated_text}\n")
    
    return responses

### choose subject and collect questions
dataset = load_dataset('cais/mmlu', ss)
qt = []
for i in range(num):
    example = dataset['test'][i]
    question = example['question']
    choices = example['choices']
    answer = example['answer']
    # subject = example['subject']
    qt0 = f"Question: {question}. Please choose amony the following(note that it's multiple choices):{choices}"
    qt.append(qt0)

prompts = qt

responses = generate_responses(prompts, ckpt_dir, tokenizer_path, max_length=400)
for res in responses:
    print(f"Prompt: {res['prompt']}\nResponse: {res['response']}\n")

file_path = f"data/response/{ss}_{model}_num={num}.json"

# Saving the data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(responses, json_file, indent=4)

print(f"Data has been saved to {file_path}")