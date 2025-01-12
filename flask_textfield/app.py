from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Save images in the static folder

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_input = ''
    image_path = ''
    
    if request.method == 'POST':
        user_input = request.form.get('textfield')

        # Handle image upload
        if 'imagefile' in request.files:
            image_file = request.files['imagefile']
            if image_file and image_file.filename != '':
                # Secure the filename and save it
                filename = secure_filename(image_file.filename)
                image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_path = f'uploads/{filename}'  # Correct path for rendering

    return render_template('index.html', user_input=user_input, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
