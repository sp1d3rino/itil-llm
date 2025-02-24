# Import necessary libraries for fine tuning and handling datasets
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model  # Import PEFT and LoRA
from datasets import Dataset
from huggingface_hub import login
import PyPDF2
import pandas as pd
import json

# Verify GPU and CUDA version
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    assert torch.version.cuda == "12.4", f"Expected CUDA 12.4, but got {torch.version.cuda}"
    print(f"Free GPU memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GiB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

# Function to extract text from different file types
def extract_text_from_files(file_paths):
    texts = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                texts.append(text)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            text = ' '.join(df.astype(str).values.flatten())
            texts.append(text)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                text = ' '.join(str(value) for value in data.values())
                texts.append(text)
    return texts

# Function to prepare dataset with prompts for summarization and reasoning
def prepare_dataset(texts, tokenizer, max_length=256):
    training_texts = []
    for text in texts:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        for chunk in chunks:
            prompt = f"Provide a concise summary of the following content:\n{chunk.strip()}\nSummary: "
            training_texts.append(prompt)
    
    # Tokenize the training texts
    tokenized_data = tokenizer(
        training_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset_dict = {
        'input_ids': tokenized_data['input_ids'].tolist(),
        'attention_mask': tokenized_data['attention_mask'].tolist(),
        'labels': tokenized_data['input_ids'].tolist()  # Labels for autoregressive training
    }
    
    return Dataset.from_dict(dataset_dict)

# Main fine tuning function with GPU, Hugging Face token, and LoRA
def fine_tune_model(file_paths, hf_token, output_dir='fine_tuned_model', push_to_hub=False, hub_model_name=None):
    # Authenticate with Hugging Face
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face")

    # Load tokenizer and base model (using Phi-2)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS
    model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', token=hf_token, torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Base model loaded on: {device}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers in Phi-2
        lora_dropout=0.05,  # Dropout for regularization
        bias="none",  # No bias in LoRA adapters
        task_type="CAUSAL_LM"  # Task type for causal language modeling
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Show trainable parameters for verification
    print("LoRA adapters applied to the model")

    # Move the PEFT model to GPU (re-apply after LoRA)
    model.to(device)

    # Extract and prepare dataset
    texts = extract_text_from_files(file_paths)
    dataset = prepare_dataset(texts, tokenizer)

    # Define training arguments with GPU settings and memory optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for memory
        gradient_accumulation_steps=2,  # Accumulate gradients
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5,
        fp16=True,  # Enable native mixed precision training
        dataloader_num_workers=0,
    )

    # Initialize standard Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Perform fine tuning
    trainer.train()

    # Save the fine-tuned LoRA adapters and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA fine-tuned model saved locally to {output_dir}")

    # Optionally push to Hugging Face Hub
    if push_to_hub and hub_model_name:
        model.push_to_hub(hub_model_name, use_auth_token=hf_token)
        tokenizer.push_to_hub(hub_model_name, use_auth_token=hf_token)
        print(f"LoRA fine-tuned model pushed to Hugging Face Hub as {hub_model_name}")

# Example usage
if __name__ == "__main__":
    hf_token = input("Enter your Hugging Face token: ")
    file_paths = ['ITIL.pdf']
    fine_tune_model(
        file_paths=file_paths,
        hf_token=hf_token,
        output_dir='my_fine_tuned_phi2_lora',
        push_to_hub=True,
        hub_model_name='fabras/my-fine-tuned-phi2-lora'  # Replace with your details
    )