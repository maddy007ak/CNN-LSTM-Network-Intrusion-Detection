# HybridGuard - Intrusion Detection System (IDS)

**HybridGuard** is a deep learning-based Intrusion Detection System that leverages CNN and LSTM to detect and classify network traffic as safe or intrusive. This project helps improve cybersecurity by identifying potential threats in network activity.

## 📁 Project Structure

```
intrusion_detection/
│
├── data/               # Dataset files (e.g., NSL-KDD)
├── model/              # Trained CNN-LSTM model
├── static/             # Static files (CSS, images)
├── templates/          # HTML templates for frontend
├── tf_env/             # TensorFlow environment-related setup
├── venv/               # Python virtual environment
│
├── app.py              # Main Flask app (connects frontend to backend)
├── train.py            # Model training script
├── requirements.txt    # List of dependencies
```

## 🚀 How It Works

1. User enters **41 network features** manually on the frontend.
2. On clicking **Analyze**, the features are passed to the backend.
3. The trained **CNN + LSTM model** evaluates the input.
4. The system outputs whether the traffic is **Normal** or **Intrusion**.

## 🛠️ Technologies Used

- **Frontend:** HTML, CSS (via `templates/` and `static/`)
- **Backend:** Flask (`app.py`)
- **Model:** CNN + LSTM using TensorFlow/Keras (`train.py`)
- **Language:** Python
- **Dataset:** NSL-KDD (or similar 41-feature intrusion dataset)

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/intrusion_detection.git
cd intrusion_detection
pip install -r requirements.txt
python app.py
```

## ✅ Features

- Real-time intrusion detection from user input
- Combines spatial and temporal pattern recognition
- Lightweight Flask-based web interface

## 📊 Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- False Positive Rate

## 📌 Future Enhancements

- Real-time packet sniffing integration
- Semi-supervised learning for evolving threats
- Dashboard for visualizing traffic patterns

## 🔐 Disclaimer

This project is for educational purposes only. Please ensure ethical use in compliance with your institution's and country's cybersecurity regulations.

## 📬 Contact

For questions or collaboration, feel free to contact me by email me at `pgtvmaddy@gmail.com.com`."# CNN-LSTM-Network-Intrusion-Detection" 
