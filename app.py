"""
Handwritten Digits Recognition - Assignment 2
Flask Backend API
Astana IT University - Computer Vision
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import pickle
import time
import os
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from scipy import ndimage

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'model_path': 'models/best_baseline_LightGBM.pkl',
    'scaler_path': 'models/feature_scaler.pkl',
    'image_size': (64, 64),
    'display_size': (224, 224),
}

# ============================================================================
# Load Models
# ============================================================================

print("Loading models...")
try:
    with open(CONFIG['model_path'], 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {CONFIG['model_path']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open(CONFIG['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded: {CONFIG['scaler_path']}")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None


# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_image(img_array):
    """Preprocess image: resize, denoise, CLAHE, threshold"""
    # Resize to processing size
    img = cv2.resize(img_array, CONFIG['image_size'])

    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # white_ratio = np.mean(img > 127)
    # if white_ratio < 0.5:
    #     img = cv2.bitwise_not(img)

    return img


def extract_hog_features(image):
    """Extract HOG features"""
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features


def extract_lbp_features(image, num_points=24, radius=3):
    """Extract LBP features"""
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_statistical_features(image):
    """Extract statistical features"""
    features = []

    # Mean, std, min, max
    features.extend([
        np.mean(image),
        np.std(image),
        np.min(image),
        np.max(image)
    ])

    # Horizontal and vertical projections
    h_proj = np.sum(image, axis=1)
    v_proj = np.sum(image, axis=0)

    features.extend([
        np.mean(h_proj),
        np.std(h_proj),
        np.mean(v_proj),
        np.std(v_proj)
    ])

    return np.array(features)


def extract_combined_features(image):
    """Combine all features"""
    features = []

    # HOG
    hog_feat = extract_hog_features(image)
    features.extend(hog_feat)

    # LBP
    lbp_feat = extract_lbp_features(image)
    features.extend(lbp_feat)

    # Statistical
    stat_feat = extract_statistical_features(image)
    features.extend(stat_feat)

    return np.array(features)


# ============================================================================
# ArtAug Transformation Functions
# ============================================================================

class ArtAugTransforms:
    """12 ArtAug variants"""

    @staticmethod
    def lighting_dark(image):
        factor = 0.6
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def lighting_bright(image):
        factor = 1.4
        return np.clip(image * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def contrast_low(image):
        img_float = image.astype(float)
        mean = np.mean(img_float)
        adjusted = mean + (img_float - mean) * 0.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def contrast_high(image):
        img_float = image.astype(float)
        mean = np.mean(img_float)
        adjusted = mean + (img_float - mean) * 1.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def rotation_small(image):
        angle = np.random.uniform(-15, 15)
        return ndimage.rotate(image, angle, reshape=False, cval=255, order=1).astype(np.uint8)

    @staticmethod
    def rotation_large(image):
        angle = np.random.uniform(-30, 30)
        return ndimage.rotate(image, angle, reshape=False, cval=255, order=1).astype(np.uint8)

    @staticmethod
    def scale_small(image):
        h, w = image.shape
        scale = 0.75
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(image, (new_w, new_h))
            result = np.ones((h, w), dtype=np.uint8) * 255
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            result[start_h:start_h + new_h, start_w:start_w + new_w] = resized
            return result
        return image

    @staticmethod
    def scale_large(image):
        h, w = image.shape
        scale = 1.25
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > h and new_w > w:
            resized = cv2.resize(image, (new_w, new_h))
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h:start_h + h, start_w:start_w + w]
        return image

    @staticmethod
    def thin_brush(image):
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    @staticmethod
    def thick_marker(image):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        return cv2.GaussianBlur(dilated, (3, 3), 0)

    @staticmethod
    def noise(image):
        noise = np.random.normal(0, 12, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def motion_blur(image):
        size = 5
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(image, -1, kernel).astype(np.uint8)


ARTAUG_VARIANTS = {
    'lighting_dark': ('Poor lighting (dark)', ArtAugTransforms.lighting_dark),
    'lighting_bright': ('Overexposed', ArtAugTransforms.lighting_bright),
    'contrast_low': ('Low contrast', ArtAugTransforms.contrast_low),
    'contrast_high': ('High contrast', ArtAugTransforms.contrast_high),
    'rotation_small': ('Rotation ±15°', ArtAugTransforms.rotation_small),
    'rotation_large': ('Rotation ±30°', ArtAugTransforms.rotation_large),
    'scale_small': ('Smaller scale', ArtAugTransforms.scale_small),
    'scale_large': ('Larger scale', ArtAugTransforms.scale_large),
    'thin_brush': ('Thin strokes', ArtAugTransforms.thin_brush),
    'thick_marker': ('Thick marker', ArtAugTransforms.thick_marker),
    'noise': ('Noisy', ArtAugTransforms.noise),
    'motion_blur': ('Motion blur', ArtAugTransforms.motion_blur),
}


# ============================================================================
# Helper Functions
# ============================================================================

def image_to_base64(img_array):
    """Convert numpy array to base64 string"""
    img_pil = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def base64_to_image(base64_string):
    """Convert base64 string to numpy array"""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data)).convert('L')
    return np.array(img)


# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image"""
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode image
        img_array = base64_to_image(data['image'])

        # Start timing
        start_time = time.time()

        # Preprocess
        preprocessed = preprocess_image(img_array)

        # Extract features
        features = extract_combined_features(preprocessed)
        features = features.reshape(1, -1)

        # Scale
        if scaler:
            features = scaler.transform(features)

        # Predict
        if model:

            # print("FEATURES SHAPE:", features.shape)
            # print("FEATURES sample (first 10):", features[0][:10])
            # if np.isnan(features).any() or np.isinf(features).any():
            #     print("WARNING: NaN/Inf in features!")
            #
            # if scaler:
            #     features_scaled = scaler.transform(features)
            #     print("SCALED FEATURES sample (first 10):", features_scaled[0][:10])
            # else:
            #     features_scaled = features

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            # print("PRED:", prediction, "PROBS:", probabilities[:10])
        else:
            return jsonify({'error': 'Model not loaded'}), 500

        # End timing
        inference_time = (time.time() - start_time) * 1000  # ms

        # Prepare response
        response = {
            'prediction': int(prediction),
            'confidence': float(probabilities[prediction]),
            'probabilities': {str(i): float(p) for i, p in enumerate(probabilities)},
            'inference_time_ms': round(inference_time, 2),
            'preprocessed_image': image_to_base64(preprocessed)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_variants', methods=['POST'])
def generate_variants():
    """Generate 12 ArtAug variants of the input image"""
    try:
        data = request.get_json()

        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode image
        img_array = base64_to_image(data['image'])

        # Preprocess original
        preprocessed = preprocess_image(img_array)

        # Resize to display size
        display_original = cv2.resize(preprocessed, CONFIG['display_size'])

        # Generate variants
        variants = []

        for variant_name, (description, transform_func) in ARTAUG_VARIANTS.items():
            try:
                # Apply transformation
                transformed = transform_func(preprocessed.copy())

                # Resize to display size
                display_transformed = cv2.resize(transformed, CONFIG['display_size'])

                # Predict on variant
                features = extract_combined_features(transformed)
                features = features.reshape(1, -1)
                if scaler:
                    features = scaler.transform(features)

                if model:
                    pred = model.predict(features)[0]
                    prob = model.predict_proba(features)[0]
                    confidence = float(prob[pred])
                else:
                    pred = -1
                    confidence = 0.0

                variants.append({
                    'name': variant_name,
                    'description': description,
                    'image': image_to_base64(display_transformed),
                    'prediction': int(pred),
                    'confidence': confidence
                })

            except Exception as e:
                print(f"Error generating variant {variant_name}: {e}")
                continue

        response = {
            'original': image_to_base64(display_original),
            'variants': variants
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    stats = {
        'model_name': 'LightGBM + ArtAug',
        'baseline_accuracy': 0.5723,
        'improved_accuracy': 0.8409,
        'improvement': '+46.9%',
        'test_samples': 484,
        'training_samples': {
            'original': 1495,
            'synthetic': 157,
            'total': 1652
        },
        'artaug_variants': 12,
        'features': {
            'HOG': True,
            'LBP': True,
            'Statistical': True
        }
    }

    return jsonify(stats)


# Serve React App
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HANDWRITTEN DIGITS RECOGNITION - WEB APP")
    print("=" * 70)
    print("\n🚀 Starting Flask server...")
    print("📊 Model: LightGBM + ArtAug (84.09% accuracy)")
    print("🎨 ArtAug: 12 synthesis variants")
    print("\n💡 Open browser: http://localhost:5000")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)