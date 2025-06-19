import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import pynvml

def print_gpu_utilization():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_used = []
    for device_index in range(device_count):
        device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
        memory_used.append(device_info.used / 1024**3)
    print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')

def main():
    # Initialize Accelerator; it will auto-detect the distributed environment from SLURM
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Running on device: {device}")

    # Define model name and load tokenizer
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/nvme/scratch/edu28/models')
    tokenizer.padding_side = 'right'

    # Load the model with 4-bit quantization. Note that we do not specify a device map manually.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        ),
        attn_implementation='eager',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir='/nvme/scratch/edu28/models'
    )
    # Move the model to the device specified by Accelerator
    model.to(device)

    # Disable caching (only beneficial for inference)
    model.config.use_cache = False

    # Add LoRA adapters
    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=16,
        lora_alpha=32,       # rule of thumb: lora_alpha should be about 2 * r
        lora_dropout=0.05,
        bias='none',
        target_modules='all-linear',
    )
    model = get_peft_model(model, peft_config)

    # Load and preprocess the dataset
    guanaco_train = load_dataset("timdettmers/openassistant-guanaco", cache_dir='/nvme/scratch/edu28/data', split='train')
    guanaco_test = load_dataset("timdettmers/openassistant-guanaco", cache_dir='/nvme/scratch/edu28/data', split='test')

    # Process each example to extract the user prompt and assistant response
    guanaco_train = guanaco_train.map(lambda entry: {
        'question1': entry['text'].split('###')[1].removeprefix(' Human: '),
        'answer1': entry['text'].split('###')[2].removeprefix(' Assistant: ')
    })
    guanaco_test = guanaco_test.map(lambda entry: {
        'question1': entry['text'].split('###')[1].removeprefix(' Human: '),
        'answer1': entry['text'].split('###')[2].removeprefix(' Assistant: ')
    })
    # Restructure to a chat format expected by our formatting function
    guanaco_train = guanaco_train.map(lambda entry: {'messages': [
        {'role': 'user', 'content': entry['question1']},
        {'role': 'assistant', 'content': entry['answer1']}
    ]})
    guanaco_test = guanaco_test.map(lambda entry: {'messages': [
        {'role': 'user', 'content': entry['question1']},
        {'role': 'assistant', 'content': entry['answer1']}
    ]})

    import os

    home_directory = os.path.expanduser("~")
    output_dir = os.path.join(home_directory,"output/phi-3.5-mini-instruct-guanaco-ddp")

    # Define training arguments with SFTConfig.
    # Note: We use accelerator.num_processes to adjust the per-device batch size.
    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=8 // accelerator.num_processes,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False,
        log_level_replica='error',
        optim='adamw_torch',
        learning_rate=2e-4,
        logging_strategy='no',
        save_strategy='no',
        max_steps=100,
        bf16=False,
        report_to='none',
        max_seq_length=1024,
    )

    def formatting_func(entry):
        return tokenizer.apply_chat_template(entry['messages'], tokenize=False)

    # Create the SFTTrainer.
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=guanaco_train,
        eval_dataset=guanaco_test,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # Optionally print trainable parameters on the main process only.
    if accelerator.is_main_process and hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # Evaluate before training
    eval_result = trainer.evaluate()
    if accelerator.is_main_process:
        print("Evaluation on test dataset before finetuning:")
        print(eval_result)

    # Train the model
    train_result = trainer.train()
    if accelerator.is_main_process:
        print("Training result:")
        print(train_result)

    # Evaluate after training
    eval_result = trainer.evaluate()
    if accelerator.is_main_process:
        print("Evaluation on test dataset after finetuning:")
        print(eval_result)

    # Print GPU memory usage (only once per node)
    if accelerator.local_process_index == 0:
        print_gpu_utilization()

if __name__ == "__main__":
    main()
