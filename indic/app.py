from flask import Flask, request, render_template, redirect, url_for
import os
import io
from google.cloud import vision
import pandas as pd

app = Flask(__name__)
# Configure upload folder for images
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gen-lang-client-0845107144-6a7f58a13071.json' 
# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])
client = vision.ImageAnnotatorClient()
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
@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    image_url = None

    if request.method == "POST":
        # Get text from the textbox
        text = request.form.get("text")

        # Handle image file upload
        uploaded_image = request.files.get("image")
        if uploaded_image and allowed_file(uploaded_image.filename):
            image_filename = uploaded_image.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            uploaded_image.save(image_path)

            # Generate the relative URL for the uploaded image
            image_url = url_for("static", filename=f"uploads/{image_filename}")
            
            # Detect text from the uploaded image
            text = detect_text(image_path)

        # Render the template with text and image URL
        return render_template("index.html", text=text, image_path=image_url)

    return render_template("index.html", text=text, image_path=image_url)

if __name__ == "__main__":
    app.run(debug=True)