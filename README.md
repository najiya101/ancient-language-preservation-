# Ancient Text Preservation with IndicBERT

This project aims to restore and preserve ancient Sanskrit texts by predicting missing words using a fine-tuned IndicBERT model. Users can upload images of ancient Sanskrit texts, extract text from them, and fill in missing words using BERT-based masked language modeling.

![339_1x_shots_so](https://github.com/user-attachments/assets/c42bb0c9-b560-43c2-8393-fa797858e8d3)

### Features
- __Upload Images__ – Extract text from scanned Sanskrit documents using Google Vision OCR
- __Masked Word Prediction__ – Replace missing words with [MASK] and predict possible words using Fine-tuned IndicBERT
- __Direct Text Input__ – Users can also manually enter Sanskrit text instead of uploading an image
- __Real-time Predictions__ – Get the top predicted words for each masked token

### How It Works

1️⃣ Upload an image of an ancient Sanskrit text
2️⃣ The site extracts text from the image
3️⃣ Add [MASK] in places where words are missing
4️⃣ The model predicts the top possible words
5️⃣ View the results and use them for text restoration

### Technologies Used
- __Flask__ – Web framework
- __Google Vision API__ – OCR for Sanskrit text extraction
- __Fine-tuned IndicBERT__ – Masked word prediction
- __Deep Translator__ – Sanskrit to English translation

### Citation

This project utilizes IndicBERT, a multilingual language model for Indian languages. If you use this work, please cite the original work:

```
@inproceedings{kakwani2020indicnlpsuite,
    title={{IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages}},
    author={Divyanshu Kakwani and Anoop Kunchukuttan and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
    year={2020},
    booktitle={Findings of EMNLP},
}
```

🚀 __Contributions Welcome!__ Fork, improve, and contribute!
