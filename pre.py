from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the Indic-BERT tokenizer and model
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Load your dataset
dataset = load_dataset('csv', data_files={'train': '/Users/symprks/ancient-language-preservation-/processed_data.csv'})

# Tokenization function with labels
def tokenize_function(examples):
    encoding = tokenizer(examples['Preprocessed_Sloka'], truncation=True, padding='max_length', max_length=128)
    
    # Create labels
    labels = []
    for i in range(len(encoding['input_ids'])):
        label = encoding['input_ids'][i].copy()
        for j, token_id in enumerate(encoding['input_ids'][i]):
            if token_id == tokenizer.mask_token_id:
                label[j] = -100  # Mask token should not contribute to loss
        labels.append(label)
    
    encoding['labels'] = labels
    return encoding

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # Change this to "no" if you don't have an evaluation dataset
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Start pretraining
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_indic_bert")
tokenizer.save_pretrained("fine_tuned_indic_bert")
