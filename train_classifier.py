import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime

class EmotionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Classifier - Training UI")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.model = None
        self.X_test = None
        self.y_test = None
        self.vectorizer = None
        self.training = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Emotion Classifier Training",
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Control Frame
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Train button
        self.train_button = ttk.Button(control_frame, text="Start Training",
                                       command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)

        # Test button
        self.test_button = ttk.Button(control_frame, text="Show Test Results",
                                      command=self.show_results, state=tk.DISABLED)
        self.test_button.grid(row=0, column=1, padx=5)

        # Clear button
        clear_button = ttk.Button(control_frame, text="Clear Log",
                                 command=self.clear_log)
        clear_button.grid(row=0, column=2, padx=5)

        # Dataset Info Frame
        info_frame = ttk.LabelFrame(main_frame, text="Dataset Information", padding="10")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.info_text = tk.Text(info_frame, height=3, width=80, state=tk.DISABLED)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.progress_label = ttk.Label(progress_frame, text="Ready to train")
        self.progress_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Log Frame
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=100,
                                                  yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.log_text.yview)

        # Prediction Frame
        pred_frame = ttk.LabelFrame(main_frame, text="Test Prediction", padding="10")
        pred_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(pred_frame, text="Enter text:").grid(row=0, column=0, sticky=tk.W)
        self.pred_input = ttk.Entry(pred_frame, width=70)
        self.pred_input.grid(row=0, column=1, padx=5)

        self.pred_button = ttk.Button(pred_frame, text="Predict",
                                     command=self.predict_text, state=tk.DISABLED)
        self.pred_button.grid(row=0, column=2, padx=5)

        ttk.Label(pred_frame, text="Prediction:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pred_output = ttk.Label(pred_frame, text="(No prediction yet)",
                                    font=("Arial", 10, "bold"))
        self.pred_output.grid(row=1, column=1, columnspan=2, sticky=tk.W)

        self.configure_grid_weights(main_frame)

    def configure_grid_weights(self, frame):
        """Configure grid weights for responsive layout"""
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)

    def log_message(self, message):
        """Add a message to the log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()

    def update_progress(self, value, message=""):
        """Update progress bar"""
        self.progress['value'] = value
        if message:
            self.progress_label.config(text=message)
        self.root.update()

    def update_info(self, text):
        """Update dataset info"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        self.info_text.config(state=tk.DISABLED)

    def start_training(self):
        """Start training in a separate thread"""
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.training = True
        self.train_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.DISABLED)
        self.pred_button.config(state=tk.DISABLED)

        # Run training in background thread
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()

    def train_model(self):
        """Train the emotion classifier model"""
        try:
            self.log_message("Loading dataset from train.txt...")
            self.update_progress(10, "Loading data...")

            # Load data
            data = []
            with open('train.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ';' in line:
                        # Format is: TEXT;EMOTION
                        parts = line.rsplit(';', 1)
                        if len(parts) == 2:
                            text = parts[0].strip()
                            emotion = parts[1].strip()
                            if text and emotion:
                                data.append({'text': text, 'emotion': emotion})

            df = pd.DataFrame(data)
            self.log_message(f"Loaded {len(df)} samples")

            # Display dataset info
            emotion_counts = df['emotion'].value_counts()
            info_text = f"Total samples: {len(df)}\n"
            info_text += f"Unique emotions: {df['emotion'].nunique()}\n"
            info_text += f"Emotions: {', '.join(emotion_counts.index.tolist())}"
            self.update_info(info_text)

            self.log_message(f"Emotion distribution:\n{emotion_counts.to_string()}")

            self.update_progress(20, "Splitting data...")
            self.log_message("Splitting data into train/test (80/20)...")

            # Split data
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                df['text'], df['emotion'], test_size=0.2, random_state=42,
                stratify=df['emotion']
            )

            self.log_message(f"Training samples: {len(X_train)}")
            self.log_message(f"Testing samples: {len(self.X_test)}")

            self.update_progress(40, "Creating pipeline...")
            self.log_message("Creating TF-IDF + Naive Bayes pipeline...")

            # Create pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True,
                                         stop_words='english')),
                ('clf', MultinomialNB())
            ])

            self.update_progress(60, "Training model...")
            self.log_message("Training Naive Bayes classifier...")
            self.model.fit(X_train, y_train)

            self.update_progress(80, "Evaluating model...")
            self.log_message("Evaluating on test set...")

            # Get predictions
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            self.log_message(f"Accuracy: {accuracy:.4f}")
            self.log_message("\nClassification Report:")
            report = classification_report(self.y_test, y_pred)
            self.log_message(report)

            self.update_progress(100, "Training complete!")
            self.log_message("✓ Training completed successfully!")

            # Enable testing
            self.test_button.config(state=tk.NORMAL)
            self.pred_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.4f}")

        except Exception as e:
            self.log_message(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"Training failed:\n{str(e)}")

        finally:
            self.training = False
            self.train_button.config(state=tk.NORMAL)

    def show_results(self):
        """Display detailed test results"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        y_pred = self.model.predict(self.X_test)

        # Create a new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("Test Results")
        results_window.geometry("700x600")

        # Display confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        emotion_labels = self.model.classes_

        # Create a text widget with results
        results_text = scrolledtext.ScrolledText(results_window, height=30, width=80)
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        results_text.insert(tk.END, "=== DETAILED TEST RESULTS ===\n\n")
        results_text.insert(tk.END, "Confusion Matrix:\n")
        results_text.insert(tk.END, f"Emotions: {', '.join(emotion_labels)}\n\n")

        # Format confusion matrix
        results_text.insert(tk.END, "     Predicted →\n")
        results_text.insert(tk.END, "Actual ↓\n")
        for i, label in enumerate(emotion_labels):
            results_text.insert(tk.END, f"{label:12} {cm[i]}\n")

        results_text.insert(tk.END, f"\nAccuracy: {accuracy_score(self.y_test, y_pred):.4f}\n")

        results_text.config(state=tk.DISABLED)

    def predict_text(self):
        """Predict emotion for input text"""
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        text = self.pred_input.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text")
            return

        try:
            emotion = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]

            # Get probability for predicted class
            pred_idx = list(self.model.classes_).index(emotion)
            confidence = probabilities[pred_idx]

            result_text = f"{emotion} (Confidence: {confidence:.2%})"
            self.pred_output.config(text=result_text, foreground="green")
            self.log_message(f"Prediction: '{text}' → {result_text}")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionClassifierGUI(root)
    root.mainloop()
