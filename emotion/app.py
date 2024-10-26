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

app = Flask(__name__)
CORS(app)

# Cargar las variables del archivo .env
load_dotenv()

# Configuración de rutas y directorios
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuración de Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o755)  # Establecer permisos de lectura/escritura

# Obtener las credenciales de Google desde el archivo .env
GOOGLE_CREDENTIALS = json.loads(os.getenv('GOOGLE_CREDENTIALS'))

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '12CKEe1qAXXcUgQj0Xj2IRZbvbysE00aU'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_info(GOOGLE_CREDENTIALS, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def check_file_exists_in_drive(service, filename, folder_id):
    """Comprueba si el archivo ya existe en la carpeta de Google Drive"""
    try:
        query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        return files[0]['id'] if files else None
        
    except Exception as e:
        logger.error(f"Error al comprobar la existencia de un archivo en Drive: {str(e)}")
        return None

def analyze_face(image_path):
    """Analizar puntos de referencia faciales en la imagen."""
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
            raise Exception("No se pudo cargar la imagen")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No se detecta ningún rostro en la imagen")

        # Select key points
        key_points = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]
        
        # Verify that all key points exist in the detection
        landmarks = results.multi_face_landmarks[0].landmark
        if max(key_points) >= len(landmarks):
            raise Exception("No se pudieron detectar algunos puntos de referencia faciales")

        height, width = gray_image.shape
        
        # Create a new figure for each analysis
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Plot facial landmarks with better visibility
        for point_idx in key_points:
            try:
                landmark = landmarks[point_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                plt.plot(x, y, 'r*', markersize=10)
            except IndexError as e:
                logger.warning(f"Punto de procesamiento de error {point_idx}: {str(e)}")
                continue

        # Improve plot aesthetics
        plt.title('Detección de puntos de referencia faciales')
        plt.axis('off')

        # Save plot to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
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
        return jsonify({'error': 'Error Interno del Servidor'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image analysis and Drive upload"""
    try:
        filepath = None
        
        # Handle new file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No hay ningún archivo seleccionado'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
            
            # Save file locally
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if file exists in Drive
            service = obtener_servicio_drive()
            existing_file_id = check_file_exists_in_drive(service, filename, FOLDER_ID)
            
            if existing_file_id:
                drive_id = existing_file_id
            else:
                # Upload only if it doesn't exist
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                
                archivo_drive = MediaIoBaseUpload(BytesIO(file_data), mimetype='image/png')
                archivo_metadata = {
                    'name': filename,
                    'mimeType': 'image/png',
                    'parents': [FOLDER_ID]
                }
                archivo_drive_subido = service.files().create(
                    body=archivo_metadata, 
                    media_body=archivo_drive
                ).execute()
                drive_id = archivo_drive_subido.get('id')
            
        # Handle existing file analysis
        elif 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': 'File not found'}), 404
            drive_id = None
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Analyze the image
        result_image = analyze_face(filepath)

        return jsonify({
            'success': True,
            'image': result_image,
            'drive_id': drive_id
        })

    except Exception as e:
        logger.error(f"Error in /analyze: {str(e)}")
        return jsonify({
            'error': str(e),
            'details': 'No se pudo procesar la imagen. Asegúrese de que la imagen contenga un rostro claramente visible.'
        }), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error al servir el archivo: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
