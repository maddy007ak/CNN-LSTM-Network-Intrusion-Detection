from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load CNN-LSTM model
model = tf.keras.models.load_model('model/cnn_lstm_model.h5')
scaler = joblib.load('model/scaler.pkl')

# Load individual encoders
protocol_encoder = joblib.load('model/protocol_type_encoder.pkl')
service_encoder = joblib.load('model/service_encoder.pkl')
flag_encoder = joblib.load('model/flag_encoder.pkl')

def process_input(content):
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    processed = []
    
    for line in lines:
        parts = re.sub(r'\s+', ' ', line).split(' ')
        if len(parts) != 41:
            raise ValueError(f"Expected 41 features, got {len(parts)}")
        
        # Process categorical features
        parts[1] = protocol_encoder.transform([parts[1] if parts[1] in protocol_encoder.classes_ else 'unknown'])[0]
        parts[2] = service_encoder.transform([parts[2] if parts[2] in service_encoder.classes_ else 'unknown'])[0]
        parts[3] = flag_encoder.transform([parts[3] if parts[3] in flag_encoder.classes_ else 'unknown'])[0]
        
        # Convert to numeric
        processed.append([float(x) for x in parts])
    
    # Scale the data
    X_scaled = scaler.transform(np.array(processed))
    
    # Reshape for CNN-LSTM: (samples, 41 timesteps, 1 feature)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return X_reshaped

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.json.get('data', '')
        if not data.strip():
            return jsonify({'error': 'Empty input'}), 400
        
        X = process_input(data)
        predictions = model.predict(X)
        
        results = []
        for i, pred in enumerate(predictions):
            is_intrusion = pred[0] > 0.5  # Assuming binary classification (sigmoid output)
            results.append({
                "line": i+1, 
                "status": "INTRUSION" if is_intrusion else "SAFE"
            })
        
        return jsonify({'results': results})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Processing error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)