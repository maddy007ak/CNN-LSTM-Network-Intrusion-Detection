import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Create model directory
os.makedirs('model', exist_ok=True)

# Full 41 feature columns
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

def train_model():
    # Load data
    df = pd.read_csv('data/kddcup.data', names=columns)
    
    # Preprocessing
    df['label'] = df['label'].str.replace(r'\.$', '', regex=True)
    df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'intrusion')
    
    # Encode categoricals
    categoricals = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in categoricals:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        joblib.dump(le, f'model/{col}_encoder.pkl')
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].map({'normal': 0, 'intrusion': 1}).values
    
    # Scale all 41 features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # Reshape for CNN input
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define CNN-LSTM model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Save artifacts
    model.save('model/model.h5')
    joblib.dump(scaler, 'model/scaler.pkl')  # Save scaler for future use
    
    print("Training completed. Model saved as CNN-LSTM with 41-feature scaler.")

if __name__ == "__main__":
    train_model()
