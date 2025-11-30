from flask import Flask, render_template_string, request, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
import io
import base64
from PIL import Image
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

# HTML template embedded in the code
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TensorFlow Data Augmentation Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            padding: 30px;
        }
        
        .controls {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            height: fit-content;
        }
        
        .control-group {
            margin-bottom: 20px;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            margin-bottom: 5px;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #4facfe;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        
        .control-group input[type="checkbox"] {
            margin-right: 8px;
            transform: scale(1.2);
        }
        
        .value-display {
            font-size: 0.9em;
            color: #666;
            text-align: right;
        }
        
        .file-upload {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }
        
        .file-upload:hover {
            border-color: #4facfe;
            background: #f0f8ff;
        }
        
        .file-upload.dragover {
            border-color: #4facfe;
            background: #f0f8ff;
        }
        
        .btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            background: white;
            border-radius: 15px;
            border: 1px solid #e0e0e0;
        }
        
        .results-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            border-radius: 15px 15px 0 0;
        }
        
        .results-content {
            padding: 20px;
            min-height: 400px;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .image-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
        }
        
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .image-card .caption {
            padding: 10px;
            text-align: center;
            font-size: 0.9em;
            color: #666;
            font-weight: 500;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffe6e6;
            color: #d63031;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #fab1a0;
        }
        
        .success {
            background: #e8f5e8;
            color: #00b894;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #a8e6cf;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TensorFlow Data Augmentation Dashboard</h1>
            <p>Apply various augmentation techniques to your images using TensorFlow</p>
        </div>
        
        <div class="content">
            <div class="controls">
                <h2>📸 Upload Image</h2>
                <div class="file-upload" id="fileUpload">
                    <p>📁 Drop an image here or click to select</p>
                    <input type="file" id="imageFile" accept="image/*" style="display: none;">
                </div>
                
                <h3>🎛️ Augmentation Parameters</h3>
                
                <div class="control-group">
                    <label>Rotation Range (degrees): <span class="value-display" id="rotationValue">40</span></label>
                    <input type="range" id="rotation" min="0" max="180" value="40">
                </div>
                
                <div class="control-group">
                    <label>Width Shift Range: <span class="value-display" id="widthValue">0.2</span></label>
                    <input type="range" id="widthShift" min="0" max="1" step="0.1" value="0.2">
                </div>
                
                <div class="control-group">
                    <label>Height Shift Range: <span class="value-display" id="heightValue">0.2</span></label>
                    <input type="range" id="heightShift" min="0" max="1" step="0.1" value="0.2">
                </div>
                
                <div class="control-group">
                    <label>Zoom Range: <span class="value-display" id="zoomValue">0.2</span></label>
                    <input type="range" id="zoom" min="0" max="1" step="0.1" value="0.2">
                </div>
                
                <div class="control-group">
                    <label>Shear Range: <span class="value-display" id="shearValue">0.2</span></label>
                    <input type="range" id="shear" min="0" max="1" step="0.1" value="0.2">
                </div>
                
                <div class="control-group">
                    <label>Brightness Range: <span class="value-display" id="brightnessValue">0.3</span></label>
                    <input type="range" id="brightness" min="0" max="1" step="0.1" value="0.3">
                </div>
                
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="horizontalFlip" checked>
                        Horizontal Flip
                    </label>
                </div>
                
                <div class="control-group">
                    <label>
                        <input type="checkbox" id="verticalFlip">
                        Vertical Flip
                    </label>
                </div>
                
                <div class="control-group">
                    <label>Number of Augmented Images: <span class="value-display" id="numImagesValue">8</span></label>
                    <input type="range" id="numImages" min="1" max="16" value="8">
                </div>
                
                <button class="btn" id="augmentBtn" disabled>🎨 Generate Augmentations</button>
            </div>
            
            <div class="results">
                <div class="results-header">
                    <h3>📊 Augmentation Results</h3>
                </div>
                <div class="results-content" id="resultsContent">
                    <p style="text-align: center; color: #666; padding: 40px;">
                        Upload an image and adjust parameters to see augmentation results
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update value displays
        const sliders = ['rotation', 'widthShift', 'heightShift', 'zoom', 'shear', 'brightness', 'numImages'];
        sliders.forEach(slider => {
            const element = document.getElementById(slider);
            const valueDisplay = document.getElementById(slider + 'Value');
            element.addEventListener('input', () => {
                valueDisplay.textContent = element.value;
            });
        });
        
        // File upload handling
        const fileUpload = document.getElementById('fileUpload');
        const imageFile = document.getElementById('imageFile');
        const augmentBtn = document.getElementById('augmentBtn');
        
        fileUpload.addEventListener('click', () => imageFile.click());
        
        fileUpload.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUpload.classList.add('dragover');
        });
        
        fileUpload.addEventListener('dragleave', () => {
            fileUpload.classList.remove('dragover');
        });
        
        fileUpload.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUpload.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                imageFile.files = files;
                handleFileSelect();
            }
        });
        
        imageFile.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            const file = imageFile.files[0];
            if (file) {
                fileUpload.innerHTML = `<p>✅ Selected: ${file.name}</p>`;
                augmentBtn.disabled = false;
            }
        }
        
        // Augmentation
        augmentBtn.addEventListener('click', performAugmentation);
        
        async function performAugmentation() {
            const file = imageFile.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            formData.append('rotation_range', document.getElementById('rotation').value);
            formData.append('width_shift_range', document.getElementById('widthShift').value);
            formData.append('height_shift_range', document.getElementById('heightShift').value);
            formData.append('zoom_range', document.getElementById('zoom').value);
            formData.append('shear_range', document.getElementById('shear').value);
            formData.append('brightness_range', document.getElementById('brightness').value);
            formData.append('horizontal_flip', document.getElementById('horizontalFlip').checked);
            formData.append('vertical_flip', document.getElementById('verticalFlip').checked);
            formData.append('num_images', document.getElementById('numImages').value);
            
            // Show loading
            const resultsContent = document.getElementById('resultsContent');
            resultsContent.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Generating augmented images...</p>
                </div>
            `;
            
            try {
                const response = await fetch('/augment', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.images);
                } else {
                    showError(result.error);
                }
            } catch (error) {
                showError('Failed to process image: ' + error.message);
            }
        }
        
        function displayResults(images) {
            const resultsContent = document.getElementById('resultsContent');
            const imageGrid = images.map((img, index) => `
                <div class="image-card">
                    <img src="data:image/png;base64,${img}" alt="Augmented ${index + 1}">
                    <div class="caption">Augmentation ${index + 1}</div>
                </div>
            `).join('');
            
            resultsContent.innerHTML = `
                <div class="success">✅ Successfully generated ${images.length} augmented images!</div>
                <div class="image-grid">${imageGrid}</div>
            `;
        }
        
        function showError(message) {
            const resultsContent = document.getElementById('resultsContent');
            resultsContent.innerHTML = `<div class="error">❌ ${message}</div>`;
        }
    </script>
</body>
</html>
"""

class DataAugmentationDashboard:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def create_sample_image(self):
        """Create a sample image for testing"""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add some patterns to make augmentations more visible
        img[50:150, 50:150] = [255, 0, 0]  # Red square
        img[100:200, 100:200] = [0, 255, 0]  # Green square
        return img
    
    def augment_image(self, image_array, params):
        """Apply data augmentation to an image"""
        # Ensure image is in correct format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # RGB image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
        else:
            raise ValueError("Image must be RGB with shape (height, width, 3)")
        
        # Create ImageDataGenerator with specified parameters
        # Note: ImageDataGenerator expects pixel values in [0, 255] range
        datagen_params = {
            'rotation_range': params.get('rotation_range', 0),
            'width_shift_range': params.get('width_shift_range', 0),
            'height_shift_range': params.get('height_shift_range', 0),
            'shear_range': params.get('shear_range', 0),
            'zoom_range': params.get('zoom_range', 0),
            'horizontal_flip': params.get('horizontal_flip', False),
            'vertical_flip': params.get('vertical_flip', False),
            'fill_mode': 'nearest'
        }
        
        # Add brightness range if specified
        brightness_range = params.get('brightness_range', None)
        if brightness_range and len(brightness_range) == 2:
            datagen_params['brightness_range'] = brightness_range
        
        datagen = ImageDataGenerator(**datagen_params)
        
        # Reshape image for the generator (add batch dimension)
        image_batch = np.expand_dims(image_array, axis=0)
        
        augmented_images = []
        num_images = params.get('num_images', 4)
        
        # Generate augmented images
        iterator = datagen.flow(image_batch, batch_size=1)
        
        for i in range(num_images):
            try:
                augmented_batch = next(iterator)
                augmented_image = augmented_batch[0]
                
                # Ensure the image is in the correct format
                if augmented_image.dtype == np.float32:
                    # If it's float, assume it's normalized [0,1] and convert to [0,255]
                    if augmented_image.max() <= 1.0:
                        augmented_image = (augmented_image * 255).astype(np.uint8)
                    else:
                        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
                elif augmented_image.dtype == np.uint8:
                    # Already in correct format
                    augmented_image = augmented_image.astype(np.uint8)
                else:
                    # Convert to uint8
                    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
                
                augmented_images.append(augmented_image)
                
            except Exception as e:
                print(f"Error generating augmented image {i}: {e}")
                # Add original image as fallback
                augmented_images.append(image_array)
        
        return augmented_images
    
    def image_to_base64(self, image_array):
        """Convert numpy array to base64 string"""
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        img = Image.fromarray(image_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

dashboard = DataAugmentationDashboard()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/augment', methods=['POST'])
def augment():
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        image_array = np.array(image)
        
        # Resize if too large
        if image_array.shape[0] > 512 or image_array.shape[1] > 512:
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            image_array = np.array(image)
        
        # Get augmentation parameters
        params = {
            'rotation_range': float(request.form.get('rotation_range', 0)),
            'width_shift_range': float(request.form.get('width_shift_range', 0)),
            'height_shift_range': float(request.form.get('height_shift_range', 0)),
            'zoom_range': float(request.form.get('zoom_range', 0)),
            'shear_range': float(request.form.get('shear_range', 0)),
            'horizontal_flip': request.form.get('horizontal_flip') == 'true',
            'vertical_flip': request.form.get('vertical_flip') == 'true',
            'num_images': int(request.form.get('num_images', 4))
        }
        
        # Handle brightness range
        brightness_range = float(request.form.get('brightness_range', 0))
        if brightness_range > 0:
            params['brightness_range'] = [1.0 - brightness_range, 1.0 + brightness_range]
        
        # Generate augmented images
        augmented_images = dashboard.augment_image(image_array, params)
        
        # Convert to base64 strings
        image_data = []
        for img in augmented_images:
            img_b64 = dashboard.image_to_base64(img)
            image_data.append(img_b64)
        
        return jsonify({
            'success': True,
            'images': image_data,
            'count': len(image_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sample')
def sample():
    """Generate a sample augmentation for testing"""
    try:
        # Create sample image
        sample_image = dashboard.create_sample_image()
        
        # Apply various augmentations
        params = {
            'rotation_range': 45,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'zoom_range': 0.3,
            'shear_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': [0.7, 1.3],
            'num_images': 6
        }
        
        augmented_images = dashboard.augment_image(sample_image, params)
        
        # Convert to base64
        image_data = []
        for img in augmented_images:
            img_b64 = dashboard.image_to_base64(img)
            image_data.append(img_b64)
        
        return jsonify({
            'success': True,
            'images': image_data,
            'count': len(image_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting TensorFlow Data Augmentation Dashboard...")
    print("📱 Access the dashboard at: http://localhost:5000")
    print("🔬 You can also test with sample data at: http://localhost:5000/sample")
    print("\n🎯 Features included:")
    print("   • Image upload (drag & drop or click)")
    print("   • Real-time parameter adjustment")
    print("   • Multiple augmentation types:")
    print("     - Rotation")
    print("     - Width/Height shifts")
    print("     - Zoom")
    print("     - Shear")
    print("     - Brightness adjustment")
    print("     - Horizontal/Vertical flips")
    print("   • Generate up to 16 augmented versions")
    print("   • Responsive web interface")
    print("\n✨ Simply upload an image and experiment with the parameters!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)