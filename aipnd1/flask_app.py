from flask import Flask, request, jsonify
import torch
from model_utils import load_checkpoint, process_image, predict
app = Flask(__name__)

model = load_checkpoint('checkpoint.pth')
model.eval()

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    image_path = file.filename
    file.save(image_path)
    img_tensor = process_image(image_path)
    top_classes, top_probabilities = predict(model, img_tensor)
    return jsonify({
        'top_classes': top_classes,
        'top_probabilities': top_probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)


