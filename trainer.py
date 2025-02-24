# Import necessary libraries for fine tuning and handling datasets
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import PyPDF2
import pandas as pd
import json

# Verify GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")

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
def prepare_dataset(texts, tokenizer, max_length=512):
    training_texts = []
    for text in texts:
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        for chunk in chunks:
            # Prompt optimized for summarization and reasoning
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

# Main fine tuning function with GPU support
def fine_tune_model(file_paths, output_dir='fine_tuned_model', push_to_hub=False, hub_model_name=None):
    # Load tokenizer and model (using Phi-2)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS
    model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', torch_dtype=torch.float16)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")

    # Extract and prepare dataset
    texts = extract_text_from_files(file_paths)
    dataset = prepare_dataset(texts, tokenizer)

    # Define training arguments with GPU settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # More epochs for better learning
        per_device_train_batch_size=2,  # Smaller batch size due to larger model
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-5,  # Lower learning rate for stability with larger model
        fp16=True,  # Mixed precision training for GPU
        dataloader_num_workers=0,  # 0 for Windows compatibility
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Perform fine tuning
    trainer.train()

    # Save the fine-tuned model locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved locally to {output_dir}")

    # Optionally push to Hugging Face Hub
    if push_to_hub and hub_model_name:
        model.push_to_hub(hub_model_name)
        tokenizer.push_to_hub(hub_model_name)
        print(f"Model pushed to Hugging Face Hub as {hub_model_name}")

# Example usage
if __name__ == "__main__":
    file_paths = ['ITIL.pdf']
    fine_tune_model(
        file_paths=file_paths,
        output_dir='my_fine_tuned_phi2',
        push_to_hub=True,
        hub_model_name='your-username/my-fine-tuned-phi2'  # Replace with your details
    )