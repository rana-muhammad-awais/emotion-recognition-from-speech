import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import numpy as np
import tensorflow as tf

try:
    from src.preprocessing.audio_utils import extract_features_from_file, normalize_features
    from src.preprocessing.data_loader import EMOTION_LABELS
    from src.training.train import predict_emotion
except ImportError:
    from preprocessing.audio_utils import extract_features_from_file, normalize_features
    from preprocessing.data_loader import EMOTION_LABELS
    from training.train import predict_emotion


class EmotionRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition from Speech")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        self.model = None
        self.model_path = None
        
        self.setup_styles()
        self.create_widgets()
        self.create_layout()
    
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Body.TLabel', font=('Helvetica', 10))
        style.configure('Emotion.TLabel', font=('Helvetica', 10))
        style.configure('Accent.TButton', font=('Helvetica', 10, 'bold'))
    
    def create_widgets(self):
        self.title_label = ttk.Label(
            self.root, 
            text="Emotion Recognition from Speech",
            style='Title.TLabel'
        )
        
        self.instruction_label = ttk.Label(
            self.root,
            text="Load a model and audio file to predict emotions",
            style='Body.TLabel'
        )
        
        self.model_frame = ttk.LabelFrame(self.root, text="Model", padding=10)
        self.model_path_label = ttk.Label(self.model_frame, text="No model loaded")
        self.load_model_btn = ttk.Button(
            self.model_frame,
            text="Load Model",
            command=self.load_model
        )
        
        self.audio_frame = ttk.LabelFrame(self.root, text="Audio File", padding=10)
        self.audio_path_label = ttk.Label(self.audio_frame, text="No file selected")
        self.select_audio_btn = ttk.Button(
            self.audio_frame,
            text="Select Audio",
            command=self.select_audio
        )
        
        self.predict_btn = ttk.Button(
            self.root,
            text="Predict Emotion",
            command=self.predict_emotion_threaded,
            state='disabled'
        )
        
        self.result_frame = ttk.LabelFrame(self.root, text="Result", padding=10)
        self.predicted_label = ttk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 14, 'bold')
        )
        
        self.prob_canvas = tk.Canvas(self.result_frame, height=150, bg='white')
        self.prob_scrollbar = ttk.Scrollbar(
            self.result_frame, 
            orient='vertical',
            command=self.prob_canvas.yview
        )
        self.prob_frame_inner = ttk.Frame(self.prob_canvas)
        self.prob_canvas.create_window(
            (0, 0), 
            window=self.prob_frame_inner, 
            anchor='nw'
        )
        
        self.status_label = ttk.Label(self.root, text="Ready")
        
        self.progress = ttk.Progressbar(
            self.root, 
            mode='indeterminate',
            visible=False
        )
    
    def create_layout(self):
        self.title_label.pack(pady=15)
        self.instruction_label.pack(pady=(0, 15))
        
        self.model_frame.pack(fill='x', padx=20, pady=5)
        self.model_path_label.pack(side='left', padx=5)
        self.load_model_btn.pack(side='right', padx=5)
        
        self.audio_frame.pack(fill='x', padx=20, pady=5)
        self.audio_path_label.pack(side='left', padx=5)
        self.select_audio_btn.pack(side='right', padx=5)
        
        self.predict_btn.pack(pady=15)
        
        self.result_frame.pack(fill='both', expand=True, padx=20, pady=5)
        self.predicted_label.pack(pady=5)
        
        self.prob_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        self.prob_scrollbar.pack(side='right', fill='y')
        self.prob_canvas.configure(yscrollcommand=self.prob_scrollbar.set)
        self.prob_frame_inner.bind(
            "<Configure>",
            lambda e: self.prob_canvas.configure(scrollregion=self.prob_canvas.bbox("all"))
        )
        
        self.status_label.pack(pady=5)
        self.progress.pack(fill='x', padx=20, pady=5)
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model files", "*.keras *.h5"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.status_label.config(text="Loading model...")
                self.progress.start()
                self.root.update()
                
                self.model = tf.keras.models.load_model(file_path)
                self.model_path = file_path
                self.model_path_label.config(
                    text=os.path.basename(file_path)
                )
                
                self.check_ready()
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            finally:
                self.progress.stop()
                self.status_label.config(text="Ready")
    
    def select_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.mp4 *.flac"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.audio_path = file_path
            self.audio_path_label.config(text=os.path.basename(file_path))
            self.check_ready()
    
    def check_ready(self):
        if self.model is not None and hasattr(self, 'audio_path'):
            self.predict_btn.config(state='normal')
    
    def predict_emotion_threaded(self):
        thread = threading.Thread(target=self.predict_emotion)
        thread.start()
    
    def predict_emotion(self):
        self.predict_btn.config(state='disabled')
        self.status_label.config(text="Processing...")
        self.progress.start()
        self.root.update()
        
        try:
            result = self.predict_emotion_core()
            
            self.predicted_label.config(
                text=f"Predicted: {result['predicted_emotion'].upper()}"
            )
            
            for widget in self.prob_frame_inner.winfo_children():
                widget.destroy()
            
            sorted_probs = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for emotion, prob in sorted_probs:
                color = '#4CAF50' if prob > 0.5 else '#2196F3' if prob > 0.2 else '#9E9E9E'
                
                frame = ttk.Frame(self.prob_frame_inner)
                frame.pack(fill='x', padx=5, pady=2)
                
                label = tk.Label(
                    frame,
                    text=f"{emotion:12s}",
                    bg=color,
                    fg='white',
                    font=('Helvetica', 9),
                    width=12,
                    anchor='w'
                )
                label.pack(side='left')
                
                prob_label = tk.Label(
                    frame,
                    text=f"{prob:.2%}",
                    bg='white',
                    fg='black',
                    font=('Helvetica', 9),
                    width=8
                )
                prob_label.pack(side='left')
                
                bar_width = int(prob * 100)
                bar = tk.Canvas(frame, height=18, width=bar_width, bg=color)
                bar.pack(side='left', padx=2)
            
            self.status_label.config(text="Prediction complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_label.config(text="Error occurred")
        finally:
            self.progress.stop()
            self.predict_btn.config(state='normal')
    
    def predict_emotion_core(self):
        features = extract_features_from_file(self.audio_path)
        features = normalize_features(features)
        features = features.reshape(1, features.shape[1], features.shape[0])
        
        predictions = self.model.predict(features, verbose=0)[0]
        emotion_probs = {EMOTION_LABELS[i]: float(predictions[i]) 
                      for i in range(len(EMOTION_LABELS))}
        predicted_emotion = EMOTION_LABELS[np.argmax(predictions)]
        
        return {
            'predicted_emotion': predicted_emotion,
            'probabilities': emotion_probs
        }


class TrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Train Emotion Model")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        self.create_widgets()
    
    def create_widgets(self):
        title = ttk.Label(
            self.root,
            text="Train Emotion Recognition Model",
            font=('Helvetica', 14, 'bold')
        )
        title.pack(pady=15)
        
        input_frame = ttk.LabelFrame(self.root, text="Training Settings", padding=15)
        input_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        ttk.Label(input_frame, text="Data Directory:").grid(
            row=0, column=0, sticky='w', pady=5
        )
        self.data_dir_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.data_dir_var, width=30).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(
            input_frame, text="Browse", 
            command=lambda: self.browse_dir(self.data_dir_var)
        ).grid(row=0, column=2, padx=5)
        
        ttk.Label(input_frame, text="Model Type:").grid(
            row=1, column=0, sticky='w', pady=5
        )
        self.model_type_var = tk.StringVar(value='cnn_lstm')
        model_combo = ttk.Combobox(
            input_frame,
            textvariable=self.model_type_var,
            values=['cnn_lstm', 'cnn', 'lstm'],
            state='readonly',
            width=10
        )
        model_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Epochs:").grid(
            row=2, column=0, sticky='w', pady=5
        )
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(
            input_frame, from_=10, to=500, textvariable=self.epochs_var, width=10
        ).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Batch Size:").grid(
            row=3, column=0, sticky='w', pady=5
        )
        self.batch_var = tk.IntVar(value=32)
        ttk.Spinbox(
            input_frame, from_=8, to=128, textvariable=self.batch_var, width=10
        ).grid(row=3, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(input_frame, text="Learning Rate:").grid(
            row=4, column=0, sticky='w', pady=5
        )
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(
            input_frame, textvariable=self.lr_var, width=10
        ).grid(row=4, column=1, sticky='w', padx=5, pady=5)
        
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        ttk.Button(
            btn_frame, text="Start Training", command=self.start_training
        ).pack(side='left', padx=5)
        
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x', padx=20, pady=5)
    
    def browse_dir(self, var):
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            var.set(directory)
    
    def start_training(self):
        from src.training.train import train_model
        
        data_dir = self.data_dir_var.get()
        if not data_dir or not os.path.exists(data_dir):
            messagebox.showerror("Error", "Please select a valid data directory")
            return
        
        self.status_label.config(text="Training... (this may take a while)")
        self.progress.start()
        self.root.update()
        
        try:
            model, history = train_model(
                data_dir=data_dir,
                model_type=self.model_type_var.get(),
                epochs=self.epochs_var.get(),
                batch_size=self.batch_var.get(),
                learning_rate=self.lr_var.get()
            )
            
            messagebox.showinfo("Success", "Training completed!")
            self.status_label.config(text="Training complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.status_label.config(text="Training failed")
        finally:
            self.progress.stop()


def main():
    root = tk.Tk()
    app = EmotionRecognitionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()