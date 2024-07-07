from flask import Flask, render_template, request, jsonify
import torch
import base64
import numpy as np
import cv2

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK", 200

@app.route('/detect_objects', methods=['POST'])
def detect_objects_api():
    try:
        image_data = request.form['image_data']

        if not image_data.startswith('data:image/jpeg;base64,'):
            raise ValueError("Invalid image data format")

        _, encoded_data = image_data.split(',', 1)

        decoded_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(decoded_data, np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(frame)

        detected_objects = []
        bboxes = results.xyxy[0].cpu().numpy() 
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, conf, cls = bbox
            object_name = model.names[int(cls)]
            detected_objects.append(object_name)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            cv2.putText(frame, f'{object_name} {conf:.2f}', (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        _, encoded_detected_image = cv2.imencode('.jpg', frame)
        base64_detected_image = base64.b64encode(encoded_detected_image).decode('utf-8')

        return jsonify({"image_data": base64_detected_image, "detected_objects": detected_objects})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
