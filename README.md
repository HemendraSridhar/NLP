# NLP
NLP+Finance Project 
Welcome to my NLP for Finance project. This repository features a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model designed to classify financial news and statements into three categories: Positive, Negative, or Neutral.

Note: This project was originally completed previously and has been recently migrated to GitHub for portfolio documentation.

Financial language is unique, so words like "volatile" or "spread" have different meanings in finance than in casual conversation. This project uses the bert-base-uncased transformer model to capture the nuances of financial sentiment with high accuracy.

Technical Stack
Model: BERT (HuggingFace Transformers)
Framework: PyTorch & TensorFlow
Environment: Google Colab (GPU Accelerated)
Visualization: Seaborn, Matplotlib, and Plotly
Data Processing: Scikit-learn, Pandas, NumPy

Key Features & Workflow
1. Preprocessing: Full BERT pipeline including [CLS] and [SEP] token injection, attention masking, and sequence padding (max length 128).

2. Training: Fine-tuned over 4 epochs with an AdamW optimizer and a linear learning rate scheduler.

3. Evaluation: Includes a Confusion Matrix to track model performance across all three sentiment classes.

Results
The model demonstrates strong performance in distinguishing between subtle financial nuances.

Performance on Test Data:
Accuracy: 97.80%

Confusion Matrix: 
<img width="2010" height="1092" alt="image" src="https://github.com/user-attachments/assets/cdf0f4b8-cd95-4d70-a21e-6f1070a438f3" />

Outputs(Images):
<img width="1008" height="40" alt="Screenshot 2026-01-14 at 11 32 54 PM" src="https://github.com/user-attachments/assets/b6a9c1ba-f8e7-4d52-8d83-1bffb1ea3b3d" />

<img width="995" height="140" alt="Screenshot 2026-01-14 at 11 32 44 PM" src="https://github.com/user-attachments/assets/a62e2c80-b8f6-4f56-b4f8-7f590f0ea92a" />

<img width="990" height="471" alt="Screenshot 2026-01-14 at 11 27 19 PM" src="https://github.com/user-attachments/assets/67c13213-291a-4209-8eca-c8c66ddb4b3a" />

<img width="990" height="456" alt="Screenshot 2026-01-14 at 11 27 29 PM" src="https://github.com/user-attachments/assets/a49f2968-4f83-4332-97c5-a9ce2e857729" />

INTERFACE
app.py: Streamlit interface for real-time predictions. (with ngrok authoken)
Here are some demo videos of the sentiment analysis model in action!

Sentiment Analysis: Positive
https://github.com/user-attachments/assets/ac01bdbe-135b-48ec-afa3-ada90f0c64ee

Sentiment Analysis: Negative
https://github.com/user-attachments/assets/113b0525-a805-4703-ba5b-aef90eba47ef

Sentiment Analysis: Neutral
https://github.com/user-attachments/assets/6d3e7d7a-863e-4daa-959e-cf3577b386f0

BONUS: Longer text with nuanced words
https://github.com/user-attachments/assets/a9061d13-590d-4f80-b557-5c26cb500454




