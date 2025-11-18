import os
import json

class ModelManager:
    @staticmethod
    def get_available_models():
        """Mevcut modelleri listele"""
        models = []
        models_dir = 'trained_models'
        
        if os.path.exists(models_dir):
            for model_name in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_name)
                if os.path.isdir(model_path):
                    models.append({
                        'name': model_name,
                        'path': model_path,
                        'size': ModelManager.get_folder_size(model_path)
                    })
        
        return models
    
    @staticmethod
    def get_folder_size(path):
        """Klasör boyutunu hesapla"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # MB'ye çevir
        return round(total_size / (1024 * 1024), 2)