import os
from os import listdir
from os.path import isfile, join
from google.cloud import vision
import os, io
import pandas as pd
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
client = vision.ImageAnnotatorClient() 
image_path = r'image.png'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Annotate Image
response = client.text_detection(image=image)
texts = response.text_annotations

# Save results to DataFrame
df = pd.DataFrame(
    [{'locale': text.locale, 'description': text.description} for text in texts]
)

if not df.empty:
    print(df['description'][0])  # Print detected text
else:
    print("No text detected.")