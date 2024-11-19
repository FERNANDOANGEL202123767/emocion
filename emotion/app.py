import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Configurar backend antes de importar pyplot
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import base64
from pyngrok import ngrok
from io import BytesIO

# Crear la aplicación Flask
app = Flask(__name__)

# Configurar carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB de tamaño máximo de archivo

# Asegurar que exista la carpeta de subida
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, operation):
    """Procesar la imagen según la operación seleccionada."""
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó ningún rostro en la imagen")

        # Selección de puntos clave principales
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
        height, width = gray_image.shape

        # Crear figura
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))

        if operation == "original":
            # Mostrar imagen original
            ax.imshow(gray_image, cmap='gray')
            ax.set_title("Imagen Original")
        elif operation == "flip":
            # Mostrar imagen girada horizontalmente
            flipped_image = cv2.flip(gray_image, 1)
            ax.imshow(flipped_image, cmap='gray')
            ax.set_title("Imagen Girada Horizontalmente")
        elif operation == "brightness":
            # Mostrar imagen con brillo aumentado
            bright_image = cv2.convertScaleAbs(gray_image, alpha=1.2, beta=30)
            ax.imshow(bright_image, cmap='gray')
            ax.set_title("Imagen con Brillo Aumentado")
        elif operation == "flip_vertical":
            # Mostrar imagen girada verticalmente
            flipped_vertical_image = cv2.flip(gray_image, 0)
            ax.imshow(flipped_vertical_image, cmap='gray')
            ax.set_title("Imagen Girada Verticalmente")

        # Dibujar puntos clave
        for point_idx in key_points:
            landmark = results.multi_face_landmarks[0].landmark[point_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if operation == "flip":
                x = int((1 - landmark.x) * width)
            elif operation == "flip_vertical":
                y = int((1 - landmark.y) * height)
            ax.plot(x, y, 'rx')

        # Guardar la imagen generada en memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convertir a base64 para enviar como respuesta
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error en process_image: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    """Página principal para subir imágenes."""
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Ruta para analizar imágenes con una operación específica."""
    try:
        # Obtener la operación seleccionada
        operation = request.form.get('operation')
        if operation not in ["original", "flip", "brightness", "flip_vertical"]:
            return jsonify({'error': 'Operación no válida'}), 400

        # Verificar si es un archivo existente o nuevo
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'Archivo no encontrado: {filename}'}), 404
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no permitido'}), 400
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

        # Procesar la imagen
        result_image = process_image(filepath, operation)
        return jsonify({'success': True, 'image': result_image})

    except Exception as e:
        print(f"Error en /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Ruta para servir archivos subidos."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    try:
        # Iniciar ngrok y Flask
        ngrok_tunnel = ngrok.connect(5001)
        public_url = ngrok_tunnel.public_url
        print(f" * ngrok URL: {public_url}")
        app.run(port=5001)
    except Exception as e:
        print(f"Error al iniciar ngrok o Flask: {e}")
