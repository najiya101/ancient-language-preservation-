from flask import Flask, request, render_template, redirect, url_for
import os
import io
from google.cloud import vision
from deep_translator import GoogleTranslator
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging

app = Flask(__name__)

# Configure upload folder for images
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credential_path'

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

client = vision.ImageAnnotatorClient()

model_name = "/home/aswin/Documents/Hackathons/ancient-language-preservation-/fine_tuned_indic_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def detect_text(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description  # Return the first detected text
    else:
        return "No text detected."

def translate_text(text):
    try:
        # Specify Sanskrit ('sa') as the source language and English ('en') as the target language
        translated = GoogleTranslator(source='sa', target='en').translate(text)
        return translated
    except Exception as e:
        return f"Error: {e}"

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
        top_k = 10  # Adjust for desired number of predictions
        top_k_indices = torch.topk(mask_token_logits, top_k).indices
        predicted_words = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())

        # Store predictions for the current [MASK] token
        all_predictions[mask_index.item()] = predicted_words 
        all_predictions
    return all_predictions

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    translation = ""
    predicted_text = {}  # Initialize as an empty dictionary
    image_url = None

    if request.method == "POST":
        # Handle text form submission
        if 'text_submit' in request.form:  
            text = request.form.get('text')
            translation = translate_text(text)
            predicted_text = predict_masked_words(text)  # Update predicted_text
            return render_template("index.html", translation=translation, text=text, prediction=predicted_text)

        # Handle image file upload
        elif 'image_submit' in request.form:  
            uploaded_image = request.files.get("image")
            if uploaded_image and allowed_file(uploaded_image.filename):
                image_filename = uploaded_image.filename
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
                try:
                    uploaded_image.save(image_path)
                    logging.debug(f"Image saved to {image_path}")

                    # Generate the relative URL for the uploaded image
                    image_url = url_for("static", filename=f"uploads/{image_filename}")

                    # Detect text from the uploaded image
                    img_text = detect_text(image_path)
                    if img_text:
                        img_translation = translate_text(img_text)
                    else:
                        img_translation = "No text detected or error in text detection."

                    # Render the template with text and image URL
                    return render_template("index.html", text=img_text, translation=img_translation, image_path=image_url, prediction=predicted_text)
                except Exception as e:
                    logging.error(f"Error handling image upload: {e}")
                    return render_template("index.html", error="Error handling image upload.", prediction=predicted_text)
            else:
                logging.error("Invalid file or file type not allowed.")
                return render_template("index.html", error="Invalid file or file type not allowed.", prediction=predicted_text)

    # For GET requests or initial page load
    return render_template("index.html", text=text, translation=translation, image_path=image_url, prediction=predicted_text)

if __name__ == "__main__":
    app.run(debug=True)
