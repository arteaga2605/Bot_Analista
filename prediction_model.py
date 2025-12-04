# prediction_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Variables globales para simular el aprendizaje
model = None
feature_cols = ['RSI', 'SMA_50', 'EMA_20', 'Polaridad_Sentimiento']

def prepare_data_for_training(df_analyzed, sentiment_data):
    """
    Combina datos técnicos y de sentimiento y crea la columna objetivo (Target).
    """
    
    # 1. Limpieza y preparación de datos técnicos
    # Eliminamos las columnas de Bollinger para simplificar el modelo inicial
    df_features = df_analyzed[['RSI', 'SMA_50', 'EMA_20', 'close']].copy()
    
    # Aseguramos que solo usamos filas completas
    df_features = df_features.dropna()
    
    # 2. Ingeniería de la Característica Objetivo (Target)
    # Target: 1 si el precio 'close' subió en la siguiente vela, 0 si bajó o se mantuvo.
    # El método .shift(-1) mueve el precio una fila hacia arriba (el precio futuro)
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)

    # 3. Integrar Sentimiento
    # Por ahora, solo tenemos una polaridad promedio, la aplicamos a todas las filas.
    polaridad_promedio = sentiment_data[0] # El primer elemento es la polaridad
    df_features['Polaridad_Sentimiento'] = polaridad_promedio
    
    # 4. Eliminar la última fila que tiene NaN en Precio_Futuro
    df_features = df_features.dropna()
    
    return df_features

def train_or_update_model(data_df):
    """
    Entrena un nuevo modelo Random Forest.
    """
    global model
    
    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    X = data_df[feature_cols]
    y = data_df['Target']
    
    # Dividir datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar y entrenar el clasificador
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🧠 Modelo Random Forest entrenado. Precisión (Accuracy): {round(accuracy * 100, 2)}%")
    return accuracy

def predict_next_move(current_features):
    """
    Hace una predicción basada en los datos más recientes.
    """
    if model is None:
        return 0.0, "NEUTRAL", "Modelo no entrenado"

    # Convertir la fila de características en un formato que el modelo pueda leer
    input_df = pd.DataFrame([current_features], columns=feature_cols)
    
    # Predecir probabilidades: [Prob. de Bajada (0), Prob. de Subida (1)]
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
    """
    Extrae la última fila de datos para hacer la predicción en tiempo real.
    """
    last_row = df_analyzed.iloc[-1]
    
    # Creamos el vector de entrada para el modelo
    current_features = {
        'RSI': last_row['RSI'],
        'SMA_50': last_row['SMA_50'],
        'EMA_20': last_row['EMA_20'],
        'Polaridad_Sentimiento': sentiment_data[0] # Polaridad promedio
    }
    return current_features