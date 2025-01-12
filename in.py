import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the fine-tuned model and tokenizer
model_name = "fine_tuned_indic_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Function for making predictions
def predict_masked_words(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Perform inference without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the indices of all [MASK] tokens
    mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Prepare to store predictions
    all_predictions = {}

    # Extract logits for each [MASK] token and get predictions
    for mask_index in mask_token_indices:
        mask_token_logits = logits[0, mask_index, :].squeeze()
        top_k = 5  # Adjust for desired number of predictions
        top_k_indices = torch.topk(mask_token_logits, top_k).indices
        predicted_words = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())

        # Store predictions for the current [MASK] token
        all_predictions[mask_index.item()] = predicted_words

    return all_predictions

# Example input with masked tokens
print("enter the sentence:")
sentence = input()  # Replace [MASK] with actual text
predictions = predict_masked_words(sentence)

# Print the predictions
print(f"Original Sentence: {sentence}")
for mask_index, words in predictions.items():
    print(f"Predictions for [MASK] at position {mask_index}:")
    for i, word in enumerate(words, start=1):
        print(f"{i}. {word}")
