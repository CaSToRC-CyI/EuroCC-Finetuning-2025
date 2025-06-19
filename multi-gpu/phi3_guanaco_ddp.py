# Import libraries
import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import pynvml
import psutil

def set_cpu_affinity(local_rank):
    # Cyclone has two NUMA nodes, CPUs 0-19 and 20-39.
    # All four GPUs are connected to the first NUMA node.
    # To find out which GPU belongs to which NUMA node, use the following command:
    # `nvidia-smi topo -mp`
    Cyclone_GPU_CPU_map = {
        0: list(range(0,40)),
        1: list(range(0,40)),
        2: list(range(0,40)),
        3: list(range(0,40)),
    }

    allowed_cpus = psutil.Process().cpu_affinity()
    candidate_cpus = Cyclone_GPU_CPU_map[local_rank]

    # Filter the candidate CPUs to only include those we are allowed to use
    eligible_cpus = [cpu for cpu in candidate_cpus if cpu in allowed_cpus]

    if not eligible_cpus:
        raise ValueError(f"No eligible CPUs for rank {local_rank}. Allowed: {allowed_cpus}, Requested: {candidate_cpus}")

    print(f"Local rank {local_rank} binding to eligible CPUs: {eligible_cpus}")
    psutil.Process().cpu_affinity(eligible_cpus)

def print_gpu_utilization():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_used = []
    for device_index in range(device_count):
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        memory_used.append(device_info.used/1024**3)
    print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')
print("Current allowed CPUs:", psutil.Process().cpu_affinity())

# Choose a model and load tokenizer and model (using 4bit quantization):
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/nvme/scratch/edu28/models')
tokenizer.padding_side = 'right'

# For multi-GPU training, find out how many GPUs there are and which one we should use:
ps = PartialState()
num_processes = ps.num_processes
process_index = ps.process_index
local_process_index = ps.local_process_index
set_cpu_affinity(local_process_index)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map={'':local_process_index},  # Changed for DDP
    attn_implementation='eager',  # 'eager', 'sdpa', or "flash_attention_2"
    trust_remote_code=True,
    torch_dtype=torch.float16,
    cache_dir='/nvme/scratch/edu28/models'
)

# Load the guanaco dataset
guanaco_train = load_dataset("timdettmers/openassistant-guanaco", cache_dir='/nvme/scratch/edu28/data', split='train')
guanaco_test = load_dataset("timdettmers/openassistant-guanaco", cache_dir='/nvme/scratch/edu28/data', split='test')

guanaco_train = guanaco_train.map(lambda entry: {
    'question1': entry['text'].split('###')[1].removeprefix(' Human: '),
    'answer1': entry['text'].split('###')[2].removeprefix(' Assistant: ')
})
guanaco_test = guanaco_test.map(lambda entry: {
    'question1': entry['text'].split('###')[1].removeprefix(' Human: '),
    'answer1': entry['text'].split('###')[2].removeprefix(' Assistant: ')
})
guanaco_train = guanaco_train.map(lambda entry: {'messages': [
    {'role': 'user', 'content': entry['question1']},
    {'role': 'assistant', 'content': entry['answer1']}
]})
guanaco_test = guanaco_test.map(lambda entry: {'messages': [
    {'role': 'user', 'content': entry['question1']},
    {'role': 'assistant', 'content': entry['answer1']}
]})

model.config.use_cache = False  # KV cache can only speed up inference, but we are doing training.

# Add low-rank adapters (LORA) to the model:
peft_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=16,
    lora_alpha=32,  # thumb rule: lora_alpha should be 2*r
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
)
model = get_peft_model(model, peft_config)

import os

home_directory = os.path.expanduser("~")
output_dir = os.path.join(home_directory,"output/phi-3.5-mini-instruct-guanaco")

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=8//num_processes,  # Adjust per-device batch size for DDP
    gradient_accumulation_steps=1,
    gradient_checkpointing=True, # Gradient checkpointing improves memory efficiency, but slows down training,
        # e.g. Mistral 7B with PEFT using bitsandbytes:
        # - enabled: 11 GB GPU RAM and 8 samples/second
        # - disabled: 40 GB GPU RAM and 12 samples/second
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Use newer implementation that will become the default.
    ddp_find_unused_parameters=False,  # Set to False when using gradient checkpointing to suppress warning message.
    log_level_replica='error',  # Disable warnings in all but the first process.
    optim='adamw_torch',
    learning_rate=2e-4,  # QLoRA suggestions: 2e-4 for 7B or 13B, 1e-4 for 33B or 65B
    logging_strategy='no',
    # logging_strategy='steps',  # 'no', 'epoch' or 'steps'
    # logging_steps=10,
    save_strategy='no',  # 'no', 'epoch' or 'steps'
    # save_steps=2000,
    # num_train_epochs=5,
    max_steps=100,
    bf16=False,  # mixed precision training
    report_to='none',  # disable wandb
    max_seq_length=1024,
)

def formatting_func(entry):
    return tokenizer.apply_chat_template(entry['messages'], tokenize=False)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=guanaco_train,
    eval_dataset=guanaco_test,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

if process_index == 0:  # Only print in first process.
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

eval_result = trainer.evaluate()
if process_index == 0:
    print("Evaluation on test dataset before finetuning:")
    print(eval_result)

train_result = trainer.train()
if process_index == 0:
    print("Training result:")
    print(train_result)

eval_result = trainer.evaluate()
if process_index == 0:
    print("Evaluation on test dataset after finetuning:")
    print(eval_result)

# Print memory usage once per node:
if local_process_index == 0:
    print_gpu_utilization()

# # Save model in first process only:
# if process_index == 0:
#     trainer.save_model()
