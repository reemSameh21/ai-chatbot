import os
import shutil
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Set cache directories before importing any other libraries
os.environ["HF_HOME"] = r"D:\intelligent_chatbot\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\intelligent_chatbot\huggingface_cache"

def check_disk_space(required_space_gb=5):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage("/")
    free_space_gb = free / (1024 ** 3)  # Convert to GB
    print(f"Free disk space: {free_space_gb:.2f} GB")
    return free_space_gb >= required_space_gb

def load_and_tokenize_data(dataset_path: str, tokenizer, max_length=256):
    """Load and tokenize the dataset for model training."""
    df = pd.read_csv(dataset_path)

    questions = df['cleaned_question'].tolist()
    answers = df['cleaned_answer'].tolist()

    # Combine each question and answer into one text sequence
    data = [f"Question: {q} Answer: {a}" for q, a in zip(questions, answers)]

    # Convert to a Hugging Face Dataset object
    dataset = Dataset.from_dict({"text": data})

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # For language modeling

        return tokenized_inputs

    tokenized_data = dataset.map(tokenize_function, batched=True)
    return tokenized_data

def compute_metrics(eval_pred):
    """Define metrics to evaluate during training."""
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy.item()}

def train_chatbot_with_trainer(dataset_path: str, model_output_dir: str, epochs: int = 3, batch_size: int = 4):
    """Train the chatbot model using the Trainer API from Hugging Face."""

    # Check for available disk space before proceeding
    if not check_disk_space():
        print("Not enough disk space available. Please free up some space before proceeding.")
        return

    print(f"Using output directory: {model_output_dir}")

    # Load the GPT-Neo model and tokenizer
    model = GPTNeoForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M",
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M",
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

    # Set pad token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize the data
    tokenized_data = load_and_tokenize_data(dataset_path, tokenizer)

    # Split the data into train and test sets
    train_test_split = tokenized_data.train_test_split(test_size=0.1)
    tokenized_train = train_test_split["train"]
    tokenized_test = train_test_split["test"]

    # Training settings
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,  # Save model every 500 steps
        save_total_limit=2,  # Keep last 2 checkpoints
        fp16=torch.cuda.is_available(),  # Mixed precision if using GPU
    )

    # Trainer setup
    print("Setting up the trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting model training...")
    trainer.train()

    # Save the final model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")

if __name__ == "__main__":
    dataset_path = "data/cleaned_data.csv"  # Update this to your dataset path
    model_output_dir = "models/chatbot_model"  # Path to save the model
    os.makedirs(model_output_dir, exist_ok=True)

    # Train the model
    train_chatbot_with_trainer(dataset_path, model_output_dir, epochs=1, batch_size=4)