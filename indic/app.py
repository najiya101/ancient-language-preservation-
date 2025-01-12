from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)
# Configure upload folder for images
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    image_path = None

    if request.method == "POST":
        # Get text from the textbox
        text = request.form.get("text")

        # Handle image file upload
        uploaded_image = request.files.get("image")
        if uploaded_image and allowed_file(uploaded_image.filename):
            image_filename = uploaded_image.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            uploaded_image.save(image_path)

        # After handling the POST, render the template
        return render_template("index.html", text=text, image_path=image_path)

    return render_template("index.html", text=text, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
