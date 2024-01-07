import random
import json

import torch
from flask import Flask, request, jsonify

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Get the JSON data from the POST request body
    if 'message' not in data:
        return jsonify({'error': 'No message found in the request'}), 400

    message = data['message']
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    return jsonify({'tag': tag})

if __name__ == '__main__':
    # Configuración de Gunicorn directamente en el código
    host = '0.0.0.0'
    port = 8000  # Puerto que desees utilizar
    workers = 4   # Número de workers

    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items()
                      if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': f'{host}:{port}',
        'workers': workers,
        'reload': True,  # Puedes cambiar a False en producción
    }

    StandaloneApplication(app, options).run()
