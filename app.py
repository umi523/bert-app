from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.eval()

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    text = request.json['text']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    
    # Assuming Class 0 is negative sentiment and Class 1 is positive sentiment
    sentiment_label = "Positive" if probabilities[1] > probabilities[0] else "Negative"
    
    return jsonify({'sentiment': sentiment_label, 'probabilities': probabilities})

@app.route('/spam-detection', methods=['POST'])
def spam_detection():
    text = request.json['text']
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]
    
    # Assuming Class 0 is not spam and Class 1 is spam
    spam_label = "Spam" if probabilities[1] > probabilities[0] else "Not Spam"
    
    return jsonify({'spam': spam_label, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True)
