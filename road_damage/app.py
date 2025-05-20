# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import os
import uuid
import base64
from datetime import datetime

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def detect_road_damage(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read the image"}
    
    # Create a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], analysis_id)
    os.makedirs(result_dir, exist_ok=True)
    
    # Store original image
    original_path = os.path.join(result_dir, "original.jpg")
    cv2.imwrite(original_path, image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Preprocessing
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced_path = os.path.join(result_dir, "enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    
    # Bilateral filtering
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    bilateral_path = os.path.join(result_dir, "bilateral.jpg")
    cv2.imwrite(bilateral_path, bilateral)
    
    # 2. Edge Detection Methods
    # Canny Edge Detection with multiple thresholds
    edges_low = cv2.Canny(bilateral, 50, 150)
    edges_high = cv2.Canny(bilateral, 100, 200)
    edges_combined = cv2.bitwise_or(edges_low, edges_high)
    edges_path = os.path.join(result_dir, "edges.jpg")
    cv2.imwrite(edges_path, edges_combined)
    
    # Sobel operators
    sobelx = cv2.Sobel(bilateral, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(bilateral, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    sobel_path = os.path.join(result_dir, "sobel.jpg")
    cv2.imwrite(sobel_path, sobel_magnitude)
    
    # Laplacian of Gaussian
    log = cv2.GaussianBlur(bilateral, (5, 5), 0)
    log = cv2.Laplacian(log, cv2.CV_64F)
    log = np.uint8(np.absolute(log))
    log_path = os.path.join(result_dir, "log.jpg")
    cv2.imwrite(log_path, log)
    
    # 3. Texture Analysis
    # Local Binary Patterns
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(bilateral, n_points, radius, method='uniform')
    lbp = np.uint8((lbp / lbp.max()) * 255)
    lbp_path = os.path.join(result_dir, "lbp.jpg")
    cv2.imwrite(lbp_path, lbp)
    
    # 4. Morphological Operations
    # Top-hat transform for bright features
    kernel = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(bilateral, cv2.MORPH_TOPHAT, kernel)
    tophat_path = os.path.join(result_dir, "tophat.jpg")
    cv2.imwrite(tophat_path, tophat)
    
    # Bottom-hat transform for dark features (potholes)
    bottomhat = cv2.morphologyEx(bilateral, cv2.MORPH_BLACKHAT, kernel)
    bottomhat_path = os.path.join(result_dir, "bottomhat.jpg")
    cv2.imwrite(bottomhat_path, bottomhat)
    
    # 5. Thresholding and Binary Processing
    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_path = os.path.join(result_dir, "adaptive_thresh.jpg")
    cv2.imwrite(adaptive_path, adaptive_thresh)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(bottomhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_path = os.path.join(result_dir, "otsu.jpg")
    cv2.imwrite(otsu_path, otsu)
    
    # 6. Feature Combination for Crack Detection
    crack_features = cv2.bitwise_or(edges_combined, otsu)
    crack_features = cv2.bitwise_or(crack_features, adaptive_thresh)
    
    # Apply morphological operations to connect discontinuous parts
    kernel_line = np.ones((3, 3), np.uint8)
    crack_features = cv2.morphologyEx(crack_features, cv2.MORPH_CLOSE, kernel_line)
    crack_path = os.path.join(result_dir, "crack_features.jpg")
    cv2.imwrite(crack_path, crack_features)
    
    # 7. Pothole Detection
    # Use bottom-hat + threshold for potential potholes
    _, pothole_candidates = cv2.threshold(bottomhat, 30, 255, cv2.THRESH_BINARY)
    
    # Use morphological operations to clean up
    kernel_pothole = np.ones((15, 15), np.uint8)
    pothole_candidates = cv2.morphologyEx(pothole_candidates, cv2.MORPH_CLOSE, kernel_pothole)
    pothole_candidates = cv2.morphologyEx(pothole_candidates, cv2.MORPH_OPEN, kernel_pothole)
    pothole_path = os.path.join(result_dir, "pothole_candidates.jpg")
    cv2.imwrite(pothole_path, pothole_candidates)
    
    # 8. Combine Results and Filter by Area
    # Label connected components
    combined_damage = cv2.bitwise_or(crack_features, pothole_candidates)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined_damage, connectivity=8)
    
    # Filter small objects (noise)
    min_size = 100  # Minimum area size
    filtered_mask = np.zeros_like(crack_features)
    
    # Skip label 0 (background)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255
    
    filtered_path = os.path.join(result_dir, "filtered_mask.jpg")
    cv2.imwrite(filtered_path, filtered_mask)
    
    # 9. Final Results
    overlay = image.copy()
    # Create a colored mask for visualization
    damage_vis = np.zeros_like(image)
    # Red for cracks
    damage_vis[crack_features > 0] = [0, 0, 255]
    # Blue for potholes
    damage_vis[pothole_candidates > 0] = [255, 0, 0]
    
    # Overlay with transparency
    result = cv2.addWeighted(image, 0.7, damage_vis, 0.3, 0)
    result_path = os.path.join(result_dir, "result.jpg")
    cv2.imwrite(result_path, result)
    
    # Calculate damage statistics
    total_pixels = gray.shape[0] * gray.shape[1]
    damaged_pixels = np.sum(filtered_mask > 0)
    damage_percentage = (damaged_pixels / total_pixels) * 100
    
    # Count distinct damage areas
    damage_areas = num_labels - 1  # Exclude background
    
    return {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "damage_percentage": f"{damage_percentage:.2f}%",
        "damage_areas": damage_areas,
        "image_paths": {
            "original": f"results/{analysis_id}/original.jpg",
            "enhanced": f"results/{analysis_id}/enhanced.jpg",
            "bilateral": f"results/{analysis_id}/bilateral.jpg",
            "edges": f"results/{analysis_id}/edges.jpg",
            "sobel": f"results/{analysis_id}/sobel.jpg",
            "log": f"results/{analysis_id}/log.jpg",
            "lbp": f"results/{analysis_id}/lbp.jpg",
            "tophat": f"results/{analysis_id}/tophat.jpg",
            "bottomhat": f"results/{analysis_id}/bottomhat.jpg",
            "adaptive_thresh": f"results/{analysis_id}/adaptive_thresh.jpg",
            "otsu": f"results/{analysis_id}/otsu.jpg",
            "crack_features": f"results/{analysis_id}/crack_features.jpg",
            "pothole_candidates": f"results/{analysis_id}/pothole_candidates.jpg",
            "filtered_mask": f"results/{analysis_id}/filtered_mask.jpg",
            "result": f"results/{analysis_id}/result.jpg"
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the file
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        try:
            results = detect_road_damage(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)