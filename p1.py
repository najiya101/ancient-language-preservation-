from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load the Indic BERT model and tokenizer
model_name = "ai4bharat/indic-bert"  # Pretrained Indic BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example sentence with a masked word
sentence = "यह एक [MASK] वाक्य है।"  # Hindi for "This is a [MASK] sentence."

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted token ID for the [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = logits[0, mask_token_index].argmax(dim=-1).item()

# Decode the predicted token
predicted_word = tokenizer.decode([predicted_token_id])

print(f"Original Sentence: {sentence}")
print(f"Predicted Word for [MASK]: {predicted_word}")
