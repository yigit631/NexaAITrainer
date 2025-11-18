import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import os
import json

class LLMTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Trainable parametreleri ayarla
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Sadece son katmanları eğit
        for param in self.model.transformer.h[-4:].parameters():
            param.requires_grad = True
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
    
    def prepare_data(self, file_paths):
        """Dosyalardan veri seti hazırla"""
        texts = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.endswith('.jsonl'):
                            for line in f:
                                data = json.loads(line)
                                texts.append(data.get('text', ''))
                        else:
                            data = json.load(f)
                            # JSON structure'ını işle
                            if isinstance(data, list):
                                for item in data:
                                    texts.append(str(item))
                            elif isinstance(data, dict):
                                texts.append(str(data))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, dataset, epochs=3, learning_rate=2e-4, progress_callback=None):
        """Modeli eğit"""
        from transformers import DataCollatorForLanguageModeling
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=data_collator,
            shuffle=True
        )
        
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        
        self.model.train()
        total_batches = len(dataloader)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if progress_callback:
                    progress_callback(epoch, batch_idx, total_batches, loss.item())
            
            avg_loss = epoch_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def generate(self, prompt, max_length=100):
        """Metin üret"""
        self.model.eval()
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_model(self, path):
        """Modeli kaydet"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)