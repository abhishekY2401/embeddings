from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    data = request.json
    texts = data['texts']
    embeddings = model.encode(texts).tolist()
    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(debug=True, port=7000, host='0.0.0.0')