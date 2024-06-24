from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Enable debug mode for detailed error messages
app.config["DEBUG"] = True

# Load the pre-trained DistilBERT model and tokenizer
# Deployment Comment
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.eval()

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    try:
        # Check if JSON data is present
        if not request.is_json:
            return jsonify({'error': 'Request payload must be in JSON format'}), 400
        
        # Parse JSON payload
        data = request.get_json()
        
        # Check if 'text' field is present
        if 'text' not in data:
            return jsonify({'error': "'text' field is required in the JSON payload"}), 400
        
        text = data['text']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        
        # Assuming Class 0 is negative sentiment and Class 1 is positive sentiment
        sentiment_label = "Positive" if probabilities[1] > probabilities[0] else "Negative"
        
        return jsonify({'sentiment': sentiment_label, 'probabilities': probabilities})

    except Exception as e:
        # Log the error and return a 500 Internal Server Error response
        print(f"Error during sentiment analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/spam-detection', methods=['POST'])
def spam_detection():
    try:
        # Check if JSON data is present
        if not request.is_json:
            return jsonify({'error': 'Request payload must be in JSON format'}), 400
        
        # Parse JSON payload
        data = request.get_json()
        
        # Check if 'text' field is present
        if 'text' not in data:
            return jsonify({'error': "'text' field is required in the JSON payload"}), 400
        
        text = data['text']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        
        # Assuming Class 0 is not spam and Class 1 is spam
        spam_label = "Spam" if probabilities[1] > probabilities[0] else "Not Spam"
        
        return jsonify({'spam': spam_label, 'probabilities': probabilities})

    except Exception as e:
        # Log the error and return a 500 Internal Server Error response
        print(f"Error during spam detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
