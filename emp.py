from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load the model and tokenizer
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Input sentence with multiple masked words
sentence = input("Enter a sentence with multiple [MASK]: ")

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

# Display input IDs and the corresponding tokens for debugging
print("Input IDs:", inputs.input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))

# Perform inference without tracking gradients
with torch.no_grad():
    logits = model(**inputs).logits

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

# Print the results
print(f"Original Sentence: {sentence}")
for mask_index, words in all_predictions.items():
    print(f"Predictions for [MASK] at position {mask_index}:")
    for i, word in enumerate(words, start=1):
        # Strip special tokens if present
        if word.startswith('▁'):
            word = word[1:]  # Remove the subword prefix if needed
        print(f"{i}. {word}")
