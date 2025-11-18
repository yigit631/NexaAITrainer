class LLMTrainerUI {
    constructor() {
        this.initializeEventListeners();
        this.loadModels();
        this.startStatusPolling();
    }

    initializeEventListeners() {
        // Dosya yükleme
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Eğitim butonu
        document.getElementById('startTraining').addEventListener('click', this.startTraining.bind(this));
        
        // Metin üretme butonu
        document.getElementById('generateBtn').addEventListener('click', this.generateText.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.style.background = '#f0f4ff';
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.style.background = '';
        const files = e.dataTransfer.files;
        this.uploadFiles(files);
    }

    handleFileSelect(e) {
        this.uploadFiles(e.target.files);
    }

    async uploadFiles(files) {
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showMessage('success', result.message);
                this.updateFileList(result.files);
            } else {
                this.showMessage('error', result.error);
            }
        } catch (error) {
            this.showMessage('error', 'Dosya yükleme hatası: ' + error.message);
        }
    }

    updateFileList(files) {
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = files.map(file => `
            <div class="file-item">
                <span>📄 ${file}</span>
                <span class="file-size">Yüklendi</span>
            </div>
        `).join('');
    }

    async startTraining() {
        const modelName = document.getElementById('modelSelect').value;
        const epochs = document.getElementById('epochs').value;
        const learningRate = document.getElementById('learningRate').value;

        try {
            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    epochs: parseInt(epochs),
                    learning_rate: parseFloat(learningRate)
                })
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showMessage('success', 'Eğitim başlatıldı!');
            } else {
                this.showMessage('error', result.error);
            }
        } catch (error) {
            this.showMessage('error', 'Eğitim başlatma hatası: ' + error.message);
        }
    }

    async generateText() {
        const prompt = document.getElementById('promptInput').value;
        
        if (!prompt.trim()) {
            this.showMessage('error', 'Lütfen bir metin girin');
            return;
        }

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_length: 100
                })
            });

            const result = await response.json();
            
            if (response.ok) {
                document.getElementById('generatedOutput').textContent = result.generated_text;
            } else {
                this.showMessage('error', result.error);
            }
        } catch (error) {
            this.showMessage('error', 'Metin üretme hatası: ' + error.message);
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/models');
            const result = await response.json();
            
            if (response.ok) {
                this.updateModelList(result.models);
            }
        } catch (error) {
            console.error('Model listeleme hatası:', error);
        }
    }

    updateModelList(models) {
        const modelList = document.getElementById('modelList');
        
        if (models.length === 0) {
            modelList.innerHTML = '<p>Henüz eğitilmiş model yok</p>';
            return;
        }

        modelList.innerHTML = models.map(model => `
            <div class="model-item">
                <h4>${model.name}</h4>
                <p>Boyut: ${model.size} MB</p>
                <button onclick="downloadModel('${model.name}')" class="btn btn-secondary">
                    📥 İndir
                </button>
            </div>
        `).join('');
    }

    startStatusPolling() {
        setInterval(async () => {
            try {
                const response = await fetch('/training_status');
                const status = await response.json();
                
                if (response.ok) {
                    this.updateTrainingStatus(status);
                }
            } catch (error) {
                console.error('Status polling error:', error);
            }
        }, 1000);
    }

    updateTrainingStatus(status) {
        // Progress bar
        const progressFill = document.querySelector('.progress-fill');
        progressFill.style.width = status.progress + '%';
        
        // Text bilgileri
        document.getElementById('progressText').textContent = status.progress.toFixed(1) + '%';
        document.getElementById('epochText').textContent = `${status.current_epoch}/${status.total_epochs}`;
        document.getElementById('lossText').textContent = status.current_loss.toFixed(4);
        
        // Status mesajı
        const statusMessage = document.querySelector('.status-message');
        statusMessage.textContent = this.getStatusText(status.status);
        statusMessage.style.color = this.getStatusColor(status.status);
        
        // Log
        const logContainer = document.getElementById('trainingLog');
        logContainer.innerHTML = status.log.map(log => 
            `<div class="log-entry">${log}</div>`
        ).join('');
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    getStatusText(status) {
        const statusMap = {
            'idle': 'Eğitim başlatılmadı',
            'initializing': 'Başlatılıyor...',
            'loading_data': 'Veriler yükleniyor...',
            'training': 'Eğitim devam ediyor...',
            'completed': 'Eğitim tamamlandı!',
            'error': 'Hata oluştu!'
        };
        return statusMap[status] || status;
    }

    getStatusColor(status) {
        const colorMap = {
            'idle': '#666',
            'initializing': '#ffc107',
            'loading_data': '#17a2b8',
            'training': '#007bff',
            'completed': '#28a745',
            'error': '#dc3545'
        };
        return colorMap[status] || '#666';
    }

    showMessage(type, message) {
        // Basit notification sistemi
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        alert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            z-index: 1000;
            background: ${type === 'success' ? '#28a745' : '#dc3545'};
        `;
        
        document.body.appendChild(alert);
        
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
}

// Global fonksiyonlar
async function downloadModel(modelName) {
    window.open(`/download_model/${modelName}`, '_blank');
}

// Uygulamayı başlat
document.addEventListener('DOMContentLoaded', () => {
    new LLMTrainerUI();
});