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
GOOGLE_CREDENTIALS = json.loads(os.getenv('GOOGLE_CREDENTIALS', '{}'))

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '12CKEe1qAXXcUgQj0Xj2IRZbvbysE00aU'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def obtener_servicio_drive():
    creds = service_account.Credentials.from_service_account_info(GOOGLE_CREDENTIALS, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def check_file_exists_in_drive(service, filename, folder_id):
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
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y Grayscale
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Procesar la imagen con MediaPipe
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detecta ningún rostro en la imagen")

        # Puntos de referencia que nos interesan
        key_points = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]
        landmarks = results.multi_face_landmarks[0].landmark
        height, width = gray_image.shape

        # Crear gráfico
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for point_idx in key_points:
            try:
                landmark = landmarks[point_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                plt.plot(x, y, 'r*', markersize=10)
            except IndexError as e:
                logger.warning(f"Punto de procesamiento de error {point_idx}: {str(e)}")
                continue
        
        # Título y ajustes visuales
        plt.title('Detección de puntos de referencia faciales')
        plt.axis('off')

        # Guardar la imagen en memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close(fig)

        # Convertir la imagen a base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        logger.error(f"Error en analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

def adjust_image_effects(image_path, effect_type):
    """Aplicar efectos a la imagen: 'cabeza', 'horizontal', 'brillo'"""
    image = cv2.imread(image_path)
    
    if effect_type == 'cabeza':
        # Ajuste de orientación a cabeza arriba (rotación de 180 grados)
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif effect_type == 'horizontal':
        # Ajuste horizontal (espejo)
        image = cv2.flip(image, 1)
    elif effect_type == 'brillo':
        # Ajuste de brillo (aumentar brillo)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    
    # Guardar la imagen modificada
    modified_image_path = os.path.join(basedir, 'static', 'modified', f"modified_{effect_type}.png")
    os.makedirs(os.path.dirname(modified_image_path), exist_ok=True)
    cv2.imwrite(modified_image_path, image)
    
    return modified_image_path

@app.route('/analyze', methods=['POST'])
def analyze():
    """Ruta que maneja el análisis de la imagen y subida a Google Drive"""
    try:
        filepath = None
        effect_type = request.form.get('effect_type', 'original')  # Tipo de efecto: cabeza, horizontal, brillo
        
        # Manejar la carga de la imagen
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No hay ningún archivo seleccionado'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
            
            # Guardar archivo localmente
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Aplicar efecto si es necesario
            modified_image_path = adjust_image_effects(filepath, effect_type)
            
            # Analizar la imagen (puntos faciales)
            result_image = analyze_face(modified_image_path)
            
            return jsonify({
                'success': True,
                'image': result_image,
                'filepath': modified_image_path
            })

        return jsonify({'error': 'No file provided'}), 400

    except Exception as e:
        logger.error(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e), 'details': 'No se pudo procesar la imagen'}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Servir los archivos subidos"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error al servir el archivo: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
