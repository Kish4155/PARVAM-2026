import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import threading

class TrainingTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Classifier - Test Results")
        self.root.geometry("950x700")
        self.root.resizable(True, True)

        self.model = None
        self.y_test = None
        self.y_pred = None
        self.emotion_labels = None
        self.accuracy = None

        self.setup_ui()
        self.run_training()

    def setup_ui(self):
        """Setup the GUI components"""
        # Main frame with dark background
        self.main_frame = ttk.Frame(self.root, padding="15")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(self.main_frame, text="Emotion Classifier - Training Test Results",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=15)

        # Progress section
        progress_frame = ttk.LabelFrame(self.main_frame, text="Training Progress", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        self.progress = ttk.Progressbar(progress_frame, mode='determinate', length=500)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        self.status_label = ttk.Label(progress_frame, text="Loading dataset...",
                                     font=("Arial", 10))
        self.status_label.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Results section - will be populated after training
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="15")
        self.results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)

    def update_status(self, message, progress=None):
        """Update status label and progress bar"""
        self.status_label.config(text=message)
        if progress is not None:
            self.progress['value'] = progress
        self.root.update()

    def run_training(self):
        """Run training in a separate thread"""
        thread = threading.Thread(target=self.train_and_evaluate)
        thread.daemon = True
        thread.start()

    def train_and_evaluate(self):
        """Train the model and show results"""
        try:
            # Step 1: Load data
            self.update_status("Loading dataset...", 10)
            data = []
            with open('train.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ';' in line:
                        parts = line.rsplit(';', 1)
                        if len(parts) == 2:
                            text = parts[0].strip()
                            emotion = parts[1].strip()
                            if text and emotion:
                                data.append({'text': text, 'emotion': emotion})

            df = pd.DataFrame(data)
            self.update_status(f"Loaded {len(df)} samples", 20)

            # Step 2: Split data
            self.update_status("Splitting data (80/20)...", 30)
            X_train, X_test, y_train, y_test = train_test_split(
                df['text'], df['emotion'], test_size=0.2, random_state=42,
                stratify=df['emotion']
            )

            # Step 3: Create pipeline
            self.update_status("Creating model pipeline...", 40)
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, lowercase=True,
                                         stop_words='english')),
                ('clf', MultinomialNB())
            ])

            # Step 4: Train model
            self.update_status("Training classifier...", 50)
            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            # Step 5: Evaluate
            self.update_status("Evaluating model...", 80)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            self.model = model
            self.y_test = y_test
            self.y_pred = y_pred
            self.emotion_labels = model.classes_
            self.accuracy = accuracy
            self.df = df
            self.train_time = train_time
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train

            self.update_status("Complete!", 100)

            # Display results
            self.display_results()

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.update_status(f"Error: {str(e)}", 0)

    def display_results(self):
        """Display results in the GUI"""
        # Clear previous content
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Create notebook (tabs) for different results
        notebook = ttk.Notebook(self.results_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Overview
        overview_tab = ttk.Frame(notebook, padding="15")
        notebook.add(overview_tab, text="Overview")
        self.create_overview_tab(overview_tab)

        # Tab 2: Classification Report
        report_tab = ttk.Frame(notebook, padding="15")
        notebook.add(report_tab, text="Classification Report")
        self.create_report_tab(report_tab)

        # Tab 3: Confusion Matrix
        matrix_tab = ttk.Frame(notebook, padding="15")
        notebook.add(matrix_tab, text="Confusion Matrix")
        self.create_matrix_tab(matrix_tab)

        # Tab 4: Sample Predictions
        samples_tab = ttk.Frame(notebook, padding="15")
        notebook.add(samples_tab, text="Sample Predictions")
        self.create_samples_tab(samples_tab)

    def create_overview_tab(self, parent):
        """Create overview tab with key metrics"""
        # Title
        title = ttk.Label(parent, text="Dataset & Training Summary",
                         font=("Arial", 12, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=10)

        # Dataset info
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Information", padding="10")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        info_text = f"""
Total Samples: {len(self.df):,}
Training Samples: {len(self.X_train):,}
Testing Samples: {len(self.X_test):,}

Emotions: {', '.join(self.emotion_labels)}
Number of Emotions: {len(self.emotion_labels)}
        """
        info_label = ttk.Label(dataset_frame, text=info_text, justify=tk.LEFT,
                              font=("Courier", 10))
        info_label.pack(fill=tk.X)

        # Metrics frame
        metrics_frame = ttk.LabelFrame(parent, text="Model Performance", padding="15")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Accuracy box (highlighted)
        accuracy_box = tk.Frame(metrics_frame, bg="#2ecc71", height=80)
        accuracy_box.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        accuracy_label = tk.Label(accuracy_box, text="ACCURACY", bg="#2ecc71",
                                 fg="white", font=("Arial", 10, "bold"))
        accuracy_label.pack(pady=(5, 0))

        accuracy_value = tk.Label(accuracy_box, text=f"{self.accuracy:.2%}",
                                 bg="#2ecc71", fg="white", font=("Arial", 28, "bold"))
        accuracy_value.pack(pady=(0, 5))

        # Other metrics
        metrics_text = f"Training Time: {self.train_time:.3f}s\nModel Type: TF-IDF + Naive Bayes"
        metrics_label = ttk.Label(metrics_frame, text=metrics_text, justify=tk.LEFT)
        metrics_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # Emotion distribution
        dist_frame = ttk.LabelFrame(parent, text="Emotion Distribution", padding="10")
        dist_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        emotion_counts = self.df['emotion'].value_counts()
        dist_text = ""
        for emotion, count in emotion_counts.items():
            percentage = (count / len(self.df)) * 100
            bar = "=" * max(1, int(percentage / 3))
            dist_text += f"{emotion:12} {count:5} ({percentage:5.1f}%) {bar}\n"

        dist_label = ttk.Label(dist_frame, text=dist_text, justify=tk.LEFT,
                              font=("Courier", 9))
        dist_label.pack(fill=tk.X)

    def create_report_tab(self, parent):
        """Create classification report tab"""
        title = ttk.Label(parent, text="Detailed Classification Report",
                         font=("Arial", 12, "bold"))
        title.pack(pady=10)

        report = classification_report(self.y_test, self.y_pred)
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget
        text_widget = tk.Text(text_frame, font=("Courier", 10),
                             yscrollcommand=scrollbar.set, height=20, width=80)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(1.0, report)
        text_widget.config(state=tk.DISABLED)

    def create_matrix_tab(self, parent):
        """Create confusion matrix tab"""
        title = ttk.Label(parent, text="Confusion Matrix",
                         font=("Arial", 12, "bold"))
        title.pack(pady=10)

        cm = confusion_matrix(self.y_test, self.y_pred)

        # Create matrix display
        matrix_frame = ttk.Frame(parent)
        matrix_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)

        # Header
        header_frame = ttk.Frame(matrix_frame)
        header_frame.pack(fill=tk.X, pady=5)

        ttk.Label(header_frame, text="Actual \\ Predicted", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        for emotion in self.emotion_labels:
            ttk.Label(header_frame, text=emotion[:3].upper(), font=("Arial", 9, "bold"),
                     width=8).pack(side=tk.LEFT, padx=2)

        # Rows
        for i, emotion in enumerate(self.emotion_labels):
            row_frame = ttk.Frame(matrix_frame)
            row_frame.pack(fill=tk.X, pady=2)

            ttk.Label(row_frame, text=emotion[:3].upper(), font=("Arial", 9, "bold"),
                     width=8).pack(side=tk.LEFT, padx=5)

            for value in cm[i]:
                # Color code: higher values = darker
                bg_color = self.get_color_for_value(value, cm.max())
                label = tk.Label(row_frame, text=str(value), font=("Courier", 9, "bold"),
                               bg=bg_color, fg="white", width=8, relief=tk.RAISED)
                label.pack(side=tk.LEFT, padx=2)

        # Summary
        summary_text = f"Total predictions: {cm.sum()}\nCorrect predictions: {np.trace(cm)}"
        summary_label = ttk.Label(parent, text=summary_text, justify=tk.LEFT)
        summary_label.pack(pady=10)

    def get_color_for_value(self, value, max_value):
        """Get color based on value intensity"""
        if max_value == 0:
            return "#cccccc"
        intensity = value / max_value
        if intensity > 0.7:
            return "#27ae60"  # Dark green
        elif intensity > 0.4:
            return "#f39c12"  # Orange
        elif intensity > 0:
            return "#e74c3c"  # Red
        else:
            return "#95a5a6"  # Gray

    def create_samples_tab(self, parent):
        """Create sample predictions tab"""
        title = ttk.Label(parent, text="Sample Emotion Predictions",
                         font=("Arial", 12, "bold"))
        title.pack(pady=10)

        samples = [
            "I feel so happy and excited right now",
            "This makes me so angry and frustrated",
            "I'm deeply sad about what happened",
            "I love you so much",
            "I'm afraid and scared",
            "I'm surprised by this news"
        ]

        samples_frame = ttk.Frame(parent)
        samples_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)

        for i, phrase in enumerate(samples):
            emotion = self.model.predict([phrase])[0]
            proba = self.model.predict_proba([phrase])[0]
            emotion_idx = list(self.emotion_labels).index(emotion)
            confidence = proba[emotion_idx]

            # Sample box
            box = tk.Frame(samples_frame, bg="#ecf0f1", relief=tk.RAISED, bd=2)
            box.pack(fill=tk.X, pady=8, padx=5)

            # Text
            text_label = tk.Label(box, text=f"Sample {i+1}: \"{phrase}\"",
                                 bg="#ecf0f1", fg="#2c3e50",
                                 font=("Arial", 9), justify=tk.LEFT, wraplength=700)
            text_label.pack(anchor=tk.W, padx=10, pady=5)

            # Prediction
            pred_frame = tk.Frame(box, bg="#ecf0f1")
            pred_frame.pack(anchor=tk.W, padx=10, pady=(0, 5), fill=tk.X)

            # Emotion color
            emotion_colors = {
                'joy': '#f1c40f',
                'sadness': '#3498db',
                'anger': '#e74c3c',
                'fear': '#9b59b6',
                'love': '#e91e63',
                'surprise': '#ff9800'
            }
            emotion_bg = emotion_colors.get(emotion, '#95a5a6')

            emotion_label = tk.Label(pred_frame, text=f"{emotion.upper()}",
                                    bg=emotion_bg, fg="white",
                                    font=("Arial", 10, "bold"), padx=10, pady=2)
            emotion_label.pack(side=tk.LEFT, padx=(0, 10))

            conf_label = tk.Label(pred_frame, text=f"Confidence: {confidence:.1%}",
                                 bg="#ecf0f1", fg="#2c3e50", font=("Arial", 9))
            conf_label.pack(side=tk.LEFT)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingTestGUI(root)
    root.mainloop()
