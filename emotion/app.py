from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import base64
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from flask_cors import CORS
import json
import logging
from dotenv import load_dotenv
import copy
import random

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o755)

# Google Drive Configuration
GOOGLE_CREDENTIALS = json.loads(os.getenv('GOOGLE_CREDENTIALS'))
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '12CKEe1qAXXcUgQj0Xj2IRZbvbysE00aU'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path, return_landmarks=False):
    """Analyze facial landmarks with enhanced visualization."""
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        # Key points selection
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 
                      94, 0, 24, 130, 359, 288, 378]
        
        landmarks = results.multi_face_landmarks[0].landmark
        height, width = gray_image.shape

        # Create plots for different transformations
        transformations = {
            'Original': image,
            'Horizontal Flip': cv2.flip(image, 1),
            'Vertical Flip': cv2.flip(image, 0),
            'Brightness Increased': cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        }

        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        axs = axs.ravel()

        for idx, (name, transformed_image) in enumerate(transformations.items()):
            # Convert to grayscale for visualization
            transformed_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
            axs[idx].imshow(transformed_gray, cmap='gray')
            axs[idx].set_title(name)
            
            # Plot landmarks
            for point_idx in key_points:
                try:
                    landmark = landmarks[point_idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    axs[idx].plot(x, y, 'rx')
                except IndexError as e:
                    logger.warning(f"Processing error for point {point_idx}: {str(e)}")
                    continue
            
            axs[idx].axis('off')

        plt.tight_layout()

        # Save plot to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        if return_landmarks:
            # Extract landmark coordinates
            landmark_data = []
            for point_idx in key_points:
                landmark = landmarks[point_idx]
                landmark_data.append({
                    'index': point_idx,
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            return image_base64, landmark_data
        
        return image_base64

    except Exception as e:
        logger.error(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    """Home route - display uploaded images"""
    try:
        images = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                images.append(filename)
        return render_template('index.html', images=images)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image analysis and Drive upload"""
    try:
        filepath = None
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            # Save file locally
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Analyze the image
        result_image = analyze_face(filepath)

        return jsonify({
            'success': True,
            'image': result_image
        })

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({
            'error': str(e),
            'details': 'Could not process image. Ensure the image contains a clearly visible face.'
        }), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
