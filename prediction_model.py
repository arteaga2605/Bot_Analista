# prediction_model.py (Versión con Persistencia)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # Librería para guardar y cargar modelos
import os # Para verificar si el archivo existe

# Variables globales para simular el aprendizaje
model = None
MODEL_FILENAME = 'crypto_model.joblib'
feature_cols = ['RSI', 'SMA_50', 'EMA_20', 'Polaridad_Sentimiento']

def load_model():
    """
    Intenta cargar un modelo pre-entrenado.
    """
    global model
    if os.path.exists(MODEL_FILENAME):
        try:
            model = joblib.load(MODEL_FILENAME)
            print(f"🧠 Modelo cargado exitosamente desde {MODEL_FILENAME}.")
            return True
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}. Entrenando uno nuevo.")
            return False
    return False

def save_model():
    """
    Guarda el modelo entrenado en disco.
    """
    global model
    if model is not None:
        try:
            joblib.dump(model, MODEL_FILENAME)
            print(f"💾 Modelo guardado exitosamente en {MODEL_FILENAME}.")
        except Exception as e:
            print(f"❌ Error al guardar el modelo: {e}")

def prepare_data_for_training(df_analyzed, sentiment_data):
    # (El resto de la función prepare_data_for_training es idéntica)
    df_features = df_analyzed[['RSI', 'SMA_50', 'EMA_20', 'close']].copy()
    df_features = df_features.dropna()
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)
    polaridad_promedio = sentiment_data[0]
    df_features['Polaridad_Sentimiento'] = polaridad_promedio
    df_features = df_features.dropna()
    return df_features

def train_or_update_model(data_df):
    """
    Entrena o actualiza el modelo y luego lo guarda.
    """
    global model
    
    # Si el modelo no se pudo cargar, lo inicializamos.
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    X = data_df[feature_cols]
    y = data_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar (o re-entrenar) el clasificador
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🧠 Modelo Random Forest entrenado/actualizado. Precisión: {round(accuracy * 100, 2)}%")
    
    # Guardar el modelo después de entrenar
    save_model()
    
    return accuracy

def predict_next_move(current_features):
    # (El resto de la función predict_next_move es idéntica)
    if model is None:
        return 0.0, "NEUTRAL", "Modelo no entrenado"

    input_df = pd.DataFrame([current_features], columns=feature_cols)
    probabilities = model.predict_proba(input_df)[0]
    prob_up = probabilities[1]
    prob_down = probabilities[0]
    
    if prob_up > 0.6:
        prediction_text = "ALTA PROBABILIDAD DE SUBIDA"
    elif prob_down > 0.6:
        prediction_text = "ALTA PROBABILIDAD DE BAJADA"
    else:
        prediction_text = "PROBABILIDAD MIXTA"

    return prob_up, prediction_text

def get_current_features(df_analyzed, sentiment_data):
    # (Función idéntica)
    last_row = df_analyzed.iloc[-1]
    current_features = {
        'RSI': last_row['RSI'],
        'SMA_50': last_row['SMA_50'],
        'EMA_20': last_row['EMA_20'],
        'Polaridad_Sentimiento': sentiment_data[0]
    }
    return current_features

# Intentar cargar el modelo al inicio del script
load_model()