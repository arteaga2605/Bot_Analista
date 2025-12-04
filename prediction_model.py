# prediction_model.py (Versión con Persistencia y Nuevos Indicadores - CORREGIDA)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 
import os 

# Variables globales para simular el aprendizaje
model = None
MODEL_FILENAME = 'crypto_model.joblib'
# Lista de características que el modelo usará
# NOTA: La columna 'Polaridad_Sentimiento' se añadirá al DataFrame por separado.
feature_cols = ['RSI', 'SMA_50', 'EMA_20', 'ADX', 'CCI'] 
# La lista de features final que incluye sentimiento, usada para la predicción
final_feature_cols = feature_cols + ['Polaridad_Sentimiento']


def load_model():
    """
    Intenta cargar un modelo pre-entrenado.
    """
    global model
    if os.path.exists(MODEL_FILENAME):
        try:
            model = joblib.load(MODEL_FILENAME)
            print(f"🧠 Modelo cargado exitosamente desde {MODEL_FILENAME}.")
            # Es crucial verificar si el modelo antiguo es compatible con las nuevas features
            if model.n_features_in_ != len(final_feature_cols):
                 print(f"⚠️ El modelo cargado tiene {model.n_features_in_} features, pero se esperan {len(final_feature_cols)}. Se forzará re-entrenamiento.")
                 model = None # Forzar el re-entrenamiento
                 return False
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
    """
    Combina datos técnicos y de sentimiento y crea la columna objetivo (Target).
    """
    # 1. Creamos el DataFrame de características SÓLO con las columnas que vienen de df_analyzed.
    required_cols_from_df = feature_cols + ['close'] # RSI, SMA, EMA, ADX, CCI, close
    
    # **AQUÍ ESTÁ LA CORRECCIÓN:** Solo seleccionamos las columnas que existen en df_analyzed.
    df_features = df_analyzed[required_cols_from_df].copy() 
    
    # 2. Ingeniería de la Característica Objetivo (Target)
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)
    
    # 3. Integrar Sentimiento (SE AÑADE LA COLUMNA AQUÍ DESPUÉS DE LA SELECCIÓN)
    polaridad_promedio = sentiment_data[0]
    # Asignamos el mismo valor de polaridad (promedio de las últimas noticias) a todas las filas históricas
    df_features['Polaridad_Sentimiento'] = polaridad_promedio
    
    # 4. Limpieza final (eliminar la última fila que tiene NaN en Precio_Futuro)
    df_features = df_features.dropna()
    
    return df_features

def train_or_update_model(data_df):
    """
    Entrena un nuevo modelo Random Forest.
    """
    global model
    
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    X = data_df[final_feature_cols] # Usa la lista final (incluye sentimiento)
    y = data_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🧠 Modelo Random Forest entrenado/actualizado. Precisión: {round(accuracy * 100, 2)}%")
    
    save_model()
    
    return accuracy

def predict_next_move(current_features):
    """
    Hace una predicción basada en los datos más recientes.
    """
    if model is None:
        return 0.0, "NEUTRAL", "Modelo no entrenado"

    input_df = pd.DataFrame([current_features], columns=final_feature_cols) # Usa la lista final
    
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
    
    # Creamos el diccionario de características basado en final_feature_cols
    current_features = {
        'RSI': last_row['RSI'],
        'SMA_50': last_row['SMA_50'],
        'EMA_20': last_row['EMA_20'],
        'ADX': last_row['ADX'],
        'CCI': last_row['CCI'],
        'Polaridad_Sentimiento': sentiment_data[0] # Se extrae directamente del sentimiento
    }
    return current_features

# Intentar cargar el modelo al inicio del script
load_model()