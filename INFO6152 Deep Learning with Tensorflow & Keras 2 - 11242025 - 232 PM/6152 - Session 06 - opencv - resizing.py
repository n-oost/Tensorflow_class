import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import base64, io
from PIL import Image

app = Flask(__name__)

class ResizePlayground:
    def __init__(self):
        # Default 3x3 grayscale image with values 1–9
        self.original_img = np.arange(1, 10, dtype=np.uint8).reshape((3, 3))

    def set_image(self, img_array):
        self.original_img = img_array

    def resize_all_methods(self, img):
        results = {}
        target_size = (6, 6) if img.shape[0] == 3 else (200, 200)

        results['nearest'] = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        results['linear'] = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        results['cubic'] = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        results['lanczos'] = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

        # INTER_AREA better for shrinking
        if img.shape[0] == 3:
            results['area'] = cv2.resize(img, (2, 2), interpolation=cv2.INTER_AREA)
        else:
            results['area'] = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

        return results

playground = ResizePlayground()

def array_to_base64(img_array):
    # Scale up tiny images
    if img_array.shape[0] < 50:
        scaled_img = cv2.resize(img_array, (150, 150), interpolation=cv2.INTER_NEAREST)
    else:
        scaled_img = img_array

    # Convert grayscale to RGB if needed
    if len(scaled_img.shape) == 2:
        scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(scaled_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/process', methods=['POST'])
def process():
    results = playground.resize_all_methods(playground.original_img)
    web_results = {}
    for name, arr in results.items():
        web_results[name] = {
            'image': array_to_base64(arr),
            'shape': arr.shape,
            'values': arr.tolist() if arr.size <= 100 else "Too large"
        }
    return jsonify(web_results)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        img = Image.open(file.stream).convert("L")  # grayscale
        arr = np.array(img, dtype=np.uint8)
        playground.set_image(arr)
        return jsonify({"message": "Image uploaded", "shape": arr.shape})
    return jsonify({"message": "No file uploaded"}), 400

# Inline HTML (instead of templates folder)
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Resize Methods Playground</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; }
        h1 { text-align: center; }
        .upload { margin-bottom: 20px; }
        .process-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .process-btn:hover { background: #0056b3; }
        .result-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
        .result-item { border: 1px solid #ccc; padding: 10px; border-radius: 6px; background: #fafafa; }
        .result-title { font-weight: bold; margin-bottom: 10px; }
        .result-image img { border: 1px solid #ddd; border-radius: 4px; }
        .result-info { font-size: 12px; margin-top: 6px; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Resize Methods Playground</h1>

        <div class="upload">
            <h3>Upload Image:</h3>
            <input type="file" id="upload" accept="image/*" onchange="uploadImage()">
        </div>

        <button class="process-btn" onclick="processResize()">⚡ Run Resizing</button>

        <div id="results"></div>
    </div>

    <script>
        function uploadImage() {
            const file = document.getElementById('upload').files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            fetch('/upload', { method: 'POST', body: formData })
              .then(res => res.json())
              .then(data => alert(data.message + " " + JSON.stringify(data.shape)));
        }

        function processResize() {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = "Processing...";
            fetch('/process', { method: 'POST' })
              .then(res => res.json())
              .then(data => displayResults(data));
        }

        function displayResults(results) {
            let html = '<div class="result-grid">';
            for (const [name, data] of Object.entries(results)) {
                html += `
                    <div class="result-item">
                        <div class="result-title">${name}</div>
                        <div class="result-image"><img src="data:image/png;base64,${data.image}" /></div>
                        <div class="result-info">
                            Shape: ${JSON.stringify(data.shape)}<br>
                            Values: ${typeof data.values === 'string' ? data.values : JSON.stringify(data.values)}
                        </div>
                    </div>`;
            }
            html += '</div>';
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Resize Playground running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
