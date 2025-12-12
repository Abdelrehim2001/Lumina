from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
import base64
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models
plate_detector = YOLO('C://Users//rambo//depi project//runs//detect//license_plate_detector4//weights//weights for plate detection.pt') # or put the path of the weights file
char_detector = YOLO('C://Users//rambo//depi project//runs//detect//license_plate_detector5//weights//weights for chars detection.pt') # or put the path of the weights file

# Configure save directory
SAVE_DIR = "predictions_output"
UPLOAD_DIR = "uploads"

# Create directories if they don't exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Character decoding mapping (English to Arabic)
CHAR_MAPPING = {
    'a': 'أ',
    'aa': 'ع',
    'b': 'ب',
    'd': 'د',
    'f': 'ف',
    'g': 'ج',
    'h': 'ه',
    'k': 'ك',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'r': 'ر',
    's': 'س',
    'ss': 'ص',
    't': 'ط',
    'w': 'و',
    'y': 'ي',
    # Numbers stay the same
    '1': '١',
    '2': '٢',
    '3': '٣',
    '4': '٤',
    '5': '٥',
    '6': '٦',
    '7': '٧',
    '8': '٨',
    '9': '٩',
    '0': '٠'
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_to_arabic(detected_text):
    """
    Convert detected English characters to Arabic characters.
    
    Args:
        detected_text: String of detected characters in English format
    
    Returns:
        Arabic text string
    """
    # Split by common separators or process character by character
    result = []
    i = 0
    text_lower = detected_text.lower()
    
    while i < len(text_lower):
        # Check for two-character combinations first (aa, ss)
        if i < len(text_lower) - 1:
            two_char = text_lower[i:i+2]
            if two_char in CHAR_MAPPING:
                result.append(CHAR_MAPPING[two_char])
                i += 2
                continue
        
        # Check for single character
        single_char = text_lower[i]
        if single_char in CHAR_MAPPING:
            result.append(CHAR_MAPPING[single_char])
        else:
            # Keep original character if not in mapping (for special cases)
            result.append(detected_text[i])
        i += 1
    
    return ''.join(result)

def detect_license_plate(image_path, output_path=None, conf_threshold=0.25):
    """
    Detect license plates in an image using Model 2.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image
        conf_threshold: Confidence threshold for detection
    
    Returns:
        results: YOLO results object
        plates: List of dictionaries containing plate info (bbox, confidence)
    """
    # Run inference
    results = plate_detector.predict(source=image_path, conf=conf_threshold)
    
    plates = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            
            # Only process if it's a license plate
            if class_name == 'License Plate':
                plates.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class': class_name
                })
    
    # Save annotated image if requested
    if output_path or len(plates) > 0:
        if not output_path:
            output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
        
        # Get the annotated image from results
        annotated_img = results[0].plot()  # This returns the image with bounding boxes
        cv2.imwrite(str(output_path), annotated_img)
        print(f"Saved result to: {output_path}")
        
    return results, plates


def detect_characters(image_or_path, conf_threshold=0.25):
    """
    Detect Arabic characters and numbers on a license plate using Model 1.
    
    Args:
        image_or_path: Path to image or numpy array (cropped plate region)
        conf_threshold: Confidence threshold for detection
    
    Returns:
        results: YOLO results object
        characters: List of dictionaries containing character info (bbox, class, confidence)
    """
    # Run inference
    results = char_detector.predict(source=image_or_path, conf=conf_threshold)
    
    characters = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            
            characters.append({
                'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                'confidence': float(box.conf[0]),
                'class': class_name,
                'center_x': float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            })
    
    # Sort characters by x-coordinate (right to left for Arabic plates)
    characters.sort(key=lambda x: x['center_x'], reverse=True)
    
    return results, characters


def process_full_image(image_path, plate_conf=0.25, char_conf=0.25, save_results=True, output_dir=None):
    """
    Complete pipeline: Detect plates, then detect characters on each plate.
    
    Args:
        image_path: Path to the input image
        plate_conf: Confidence threshold for plate detection
        char_conf: Confidence threshold for character detection
        save_results: Whether to save visualized results
        output_dir: Directory to save results (uses SAVE_DIR if None)
    
    Returns:
        List of dictionaries, each containing plate info and detected characters
    """
    # Set output directory
    if output_dir is None:
        output_dir = SAVE_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()
    
    # Step 1: Detect license plates
    _, plates = detect_license_plate(image_path, output_path=None, conf_threshold=plate_conf)
    
    results = []
    
    # Step 2: For each detected plate, detect characters
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = map(int, plate['bbox'])
        
        # Crop the plate region
        plate_crop = image[y1:y2, x1:x2]
        
        # Detect characters in the cropped plate
        _, characters = detect_characters(plate_crop, conf_threshold=char_conf)
        
        # Extract text (reading order - right to left for Arabic)
        detected_text = ''.join([char['class'] for char in characters])
        arabic_text = decode_to_arabic(detected_text)
        
        results.append({
            'plate_number': i + 1,
            'plate_bbox': plate['bbox'],
            'plate_confidence': plate['confidence'],
            'characters': characters,
            'detected_text': detected_text,
            'arabic_text': arabic_text
        })
        
        # Save cropped plate if requested
        if save_results:
            # Use Arabic text for filename (or both)
            plate_filename = f"plate_{i+1}_{arabic_text}_{detected_text}.jpg"
            plate_path = os.path.join(output_dir, plate_filename)
            cv2.imwrite(plate_path, plate_crop)
    
    # Save annotated full image if requested
    annotated_image_path = None
    if save_results and len(plates) > 0:
        annotated_image = draw_results(original_image, results)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_annotated_{timestamp}.jpg"
        annotated_image_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"✓ Saved annotated image to: {annotated_image_path}")
    
    return results, annotated_image_path


def draw_results(image, results):
    """
    Draw bounding boxes and text on the image.
    
    Args:
        image: Input image (numpy array)
        results: List of detection results
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for plate_result in results:
        # Draw plate bounding box
        x1, y1, x2, y2 = map(int, plate_result['plate_bbox'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Put detected text above the plate (use Arabic text)
        arabic_text = plate_result['arabic_text']
        confidence = plate_result['plate_confidence']
        label = f"{arabic_text} ({confidence:.2f})"
        
        # Calculate text position
        text_y = y1 - 10 if y1 > 30 else y2 + 30
        
        # Draw text background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (x1, text_y - text_height - 5), (x1 + text_width, text_y + 5), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    return annotated


def save_results_to_file(results, output_dir=None):
    """
    Save detection results to a text file.
    
    Args:
        results: List of detection results
        output_dir: Directory to save results (uses SAVE_DIR if None)
    
    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = SAVE_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_results_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"License Plate Detection Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for plate_result in results:
            f.write(f"Plate {plate_result['plate_number']}:\n")
            f.write(f"  Detected Text (English): {plate_result['detected_text']}\n")
            f.write(f"  Arabic Text: {plate_result['arabic_text']}\n")
            f.write(f"  Confidence: {plate_result['plate_confidence']:.4f}\n")
            f.write(f"  Bounding Box: {plate_result['plate_bbox']}\n")
            f.write(f"  Number of Characters: {len(plate_result['characters'])}\n")
            f.write(f"  Character Details:\n")
            
            for j, char in enumerate(plate_result['characters'], 1):
                f.write(f"    {j}. '{char['class']}' - Confidence: {char['confidence']:.4f}\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
    
    print(f"✓ Saved results to: {filepath}")
    return filepath


# ==================== Flask API Endpoints ====================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'License Plate Detection API',
        'version': '1.0',
        'endpoints': {
            '/detect-plates': 'POST - Detect license plates only',
            '/detect-full': 'POST - Detect plates and characters (full pipeline)',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/detect-plates', methods=['POST'])
def api_detect_plates():
    """
    API endpoint to detect license plates only.
    
    Expected: multipart/form-data with 'image' file
    Optional query params: conf (confidence threshold, default 0.25)
    
    Returns: JSON with detected plates
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400
        
        # Get confidence threshold from query params
        conf_threshold = float(request.args.get('conf', 0.25))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(filepath)
        
        # Detect plates
        _, plates = detect_license_plate(filepath, conf_threshold=conf_threshold)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'plates_found': len(plates),
            'plates': plates
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-full', methods=['POST'])
def api_detect_full():
    """
    API endpoint for full detection pipeline (plates + characters).
    
    Expected: multipart/form-data with 'image' file
    Optional query params: 
        - plate_conf (default 0.25)
        - char_conf (default 0.15)
        - return_image (true/false, default false) - returns base64 encoded annotated image
    
    Returns: JSON with full detection results
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400
        
        # Get parameters
        plate_conf = float(request.args.get('plate_conf', 0.25))
        char_conf = float(request.args.get('char_conf', 0.15))
        return_image = request.args.get('return_image', 'false').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(filepath)
        
        # Process image
        results, annotated_image_path = process_full_image(
            filepath, 
            plate_conf=plate_conf, 
            char_conf=char_conf, 
            save_results=True,
            output_dir=SAVE_DIR
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'plates_found': len(results),
            'results': results
        }
        
        # Add base64 encoded image if requested
        if return_image and annotated_image_path and os.path.exists(annotated_image_path):
            with open(annotated_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                response_data['annotated_image'] = img_base64
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-annotated-image/<path:filename>', methods=['GET'])
def get_annotated_image(filename):
    """
    Retrieve an annotated image from the output directory.
    
    Args:
        filename: Name of the file to retrieve
    
    Returns: Image file
    """
    try:
        filepath = os.path.join(SAVE_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Starting License Plate Detection API")
    print("=" * 60)
    print(f"Save directory: {SAVE_DIR}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print("\nAvailable endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - Health check")
    print("  POST /detect-plates  - Detect plates only")
    print("  POST /detect-full    - Full detection pipeline")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
