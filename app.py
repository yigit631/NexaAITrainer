from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import threading
from train_engine import LLMTrainer
from model_manager import ModelManager
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('trained_models', exist_ok=True)

trainer = None
training_thread = None
training_progress = {
    'status': 'idle',
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0,
    'log': []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Dosya yükleme endpoint'i"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    # Uploads klasörünü temizle
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
    
    return jsonify({
        'message': f'{len(uploaded_files)} files uploaded successfully',
        'files': uploaded_files
    })

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx', 'json', 'jsonl'}

@app.route('/models')
def get_models():
    """Mevcut modelleri listele"""
    models = ModelManager.get_available_models()
    return jsonify({'models': models})

@app.route('/start_training', methods=['POST'])
def start_training():
    """Eğitimi başlat"""
    global trainer, training_thread, training_progress
    
    data = request.json
    model_name = data.get('model_name', 'microsoft/DialoGPT-medium')
    epochs = int(data.get('epochs', 3))
    learning_rate = float(data.get('learning_rate', 2e-4))
    
    # Eğitim durumunu sıfırla
    training_progress = {
        'status': 'initializing',
        'progress': 0,
        'current_epoch': 0,
        'total_epochs': epochs,
        'current_loss': 0,
        'log': ['Training initializing...']
    }
    
    # Eğitimi thread'de başlat
    training_thread = threading.Thread(
        target=run_training,
        args=(model_name, epochs, learning_rate)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'message': 'Training started'})

def run_training(model_name, epochs, learning_rate):
    """Eğitimi çalıştır (background thread)"""
    global trainer, training_progress
    
    try:
        trainer = LLMTrainer(model_name)
        training_progress['status'] = 'loading_data'
        
        # Yüklenen dosyalardan veri hazırla
        data_files = [
            os.path.join(app.config['UPLOAD_FOLDER'], f) 
            for f in os.listdir(app.config['UPLOAD_FOLDER'])
        ]
        
        if not data_files:
            training_progress['status'] = 'error'
            training_progress['log'].append('ERROR: No training files found')
            return
        
        dataset = trainer.prepare_data(data_files)
        training_progress['status'] = 'training'
        
        # Eğitimi başlat
        def progress_callback(epoch, batch, total_batches, loss):
            progress = ((epoch * total_batches) + batch) / (epochs * total_batches) * 100
            training_progress.update({
                'progress': progress,
                'current_epoch': epoch + 1,
                'current_loss': loss,
                'log': [f'Epoch {epoch+1}/{epochs}, Batch {batch}/{total_batches}, Loss: {loss:.4f}']
            })
        
        final_loss = trainer.train(
            dataset, 
            epochs=epochs, 
            learning_rate=learning_rate,
            progress_callback=progress_callback
        )
        
        # Modeli kaydet
        model_path = f"trained_models/model_{len(os.listdir('trained_models'))}"
        trainer.save_model(model_path)
        
        training_progress.update({
            'status': 'completed',
            'progress': 100,
            'log': [f'Training completed! Final loss: {final_loss:.4f}']
        })
        
    except Exception as e:
        training_progress.update({
            'status': 'error',
            'log': [f'ERROR: {str(e)}']
        })

@app.route('/training_status')
def training_status():
    """Eğitim durumunu getir"""
    return jsonify(training_progress)

@app.route('/generate', methods=['POST'])
def generate_text():
    """Metin üret"""
    global trainer
    
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    
    if trainer is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    try:
        generated_text = trainer.generate(prompt, max_length=max_length)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_model/<model_name>')
def download_model(model_name):
    """Modeli indir"""
    model_path = os.path.join('trained_models', model_name)
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)