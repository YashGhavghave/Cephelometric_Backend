from flask import Flask, request, jsonify
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('------Model Path------')

@app.route('/api/detect', methods=['POST'])
def Handle_Image():
    image = request.files.get('image')
    if image is None:
        return jsonify({'error': 'No image provided'}), 400
    
    results = model.predict(source = image, save=False, conf=0.5)
    output = []
    for r in results:
        for box in r.boxes:
            output.append({
                'xyxy': box.xyxy.tolist(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0])
            })
    return jsonify({'message': 'Image processed successfully', 'results': output}), 200

if __name__ == '__main__':
    app.run(debug=True)