import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator

# Load the Indic-BERT tokenizer and model
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
dataset = load_dataset('csv', data_files={'train': '/home/aswin/Documents/Hackathons/ancient-language-preservation-/processed_data.csv'})

# Tokenization function
def tokenize_function(examples):
    encoding = tokenizer(examples['Preprocessed_Sloka'], truncation=True, padding='max_length', max_length=128)
    
    # Create labels (for masked LM)
    labels = []
    for i in range(len(encoding['input_ids'])):
        label = encoding['input_ids'][i].copy()
        for j, token_id in enumerate(encoding['input_ids'][i]):
            if token_id == tokenizer.mask_token_id:
                label[j] = -100  # Ignore loss for masked tokens
        labels.append(label)
    
    encoding['labels'] = labels
    return encoding

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments (optimized for low GPU memory)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Small batch size
    gradient_accumulation_steps=8,  # Simulates batch size 8
    num_train_epochs=3,
    report_to="none",  # Disable logging to WandB
    fp16=True,  # Mixed precision training (saves GPU memory)
)

# Enable Accelerate for memory efficiency
accelerator = Accelerator()
model, tokenized_datasets['train'] = accelerator.prepare(model, tokenized_datasets['train'])

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Clear GPU cache before training
torch.cuda.empty_cache()

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_indic_bert")
tokenizer.save_pretrained("fine_tuned_indic_bert")
