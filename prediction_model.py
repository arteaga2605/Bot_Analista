# prediction_model.py (ACTUALIZADO con MinMaxScaler)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# >>> NUEVA IMPORTACIÓN PARA NORMALIZACIÓN
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
import joblib 
import os 

# Variables globales para simular el aprendizaje
model = None
# >>> NUEVA VARIABLE GLOBAL PARA ALMACENAR EL SCALER (ESCALADOR)
scaler = None 
MODEL_FILENAME = 'crypto_model_xgb.joblib' 
SCALER_FILENAME = 'crypto_scaler.joblib' # Nuevo archivo para guardar el scaler
feature_cols = ['RSI', 'SMA_50', 'EMA_20', 'ADX', 'CCI', 'MACD', 'ATR'] 
final_feature_cols = feature_cols + ['Polaridad_Sentimiento']

def load_model():
    """
    Intenta cargar un modelo pre-entrenado y el scaler.
    """
    global model, scaler
    loaded_ok = True

    if os.path.exists(MODEL_FILENAME):
        try:
            model = joblib.load(MODEL_FILENAME)
            print(f"🧠 Modelo cargado exitosamente desde {MODEL_FILENAME}.")
            
            if model.n_features_in_ != len(final_feature_cols):
                 print(f"⚠️ El modelo cargado tiene {model.n_features_in_} features, pero se esperan {len(final_feature_cols)}. Se forzará re-entrenamiento.")
                 model = None 
                 loaded_ok = False
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}. Entrenando uno nuevo.")
            model = None
            loaded_ok = False
            
    # --- Cargar Scaler ---
    if os.path.exists(SCALER_FILENAME):
        try:
            scaler = joblib.load(SCALER_FILENAME)
            print(f"📏 Scaler cargado exitosamente desde {SCALER_FILENAME}.")
        except Exception as e:
            print(f"⚠️ Error al cargar el scaler: {e}. Se necesitará re-entrenamiento.")
            scaler = None
            loaded_ok = False
    else:
        scaler = None
        loaded_ok = False
    
    return loaded_ok

def save_model():
    """
    Guarda el modelo entrenado y el scaler en disco.
    """
    global model, scaler
    if model is not None:
        try:
            joblib.dump(model, MODEL_FILENAME)
            print(f"💾 Modelo guardado exitosamente en {MODEL_FILENAME}.")
        except Exception as e:
            print(f"❌ Error al guardar el modelo: {e}")

    if scaler is not None:
        try:
            joblib.dump(scaler, SCALER_FILENAME)
            print(f"💾 Scaler guardado exitosamente en {SCALER_FILENAME}.")
        except Exception as e:
            print(f"❌ Error al guardar el scaler: {e}")


def prepare_data_for_training(df_analyzed, sentiment_data):
    """
    Combina datos técnicos y de sentimiento y crea la columna objetivo (Target).
    """
    required_cols_from_df = feature_cols + ['close'] 
    df_features = df_analyzed[required_cols_from_df].copy() 
    
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)
    
    polaridad_promedio = sentiment_data[0]
    df_features['Polaridad_Sentimiento'] = polaridad_promedio
    
    df_features = df_features.dropna()
    
    return df_features

def train_or_update_model(data_df):
    """
    Entrena un nuevo modelo XGBoost y ajusta el MinMaxScaler.
    """
    global model, scaler
    
    if model is None:
        model = XGBClassifier(
            objective='binary:logistic', 
            n_estimators=100,            
            learning_rate=0.1,           
            use_label_encoder=False,     
            eval_metric='logloss',       
            random_state=42
        )

    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    X = data_df[final_feature_cols] 
    y = data_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ----------------------------------------------------
    # >>> PASO DE NORMALIZACIÓN: FIT y TRANSFORM en el entrenamiento
    # ----------------------------------------------------
    # Creamos un nuevo scaler y lo ajustamos SOLO con los datos de entrenamiento
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🧠 Modelo XGBoost entrenado/actualizado. Precisión: {round(accuracy * 100, 2)}%")
    
    save_model()
    
    return accuracy

def predict_next_move(current_features):
    """
    Hace una predicción basada en los datos más recientes, aplicando el scaler.
    """
    global model, scaler
    
    if model is None or scaler is None:
        return 0.0, "NEUTRAL", "Modelo o Scaler no entrenados"

    input_df = pd.DataFrame([current_features], columns=final_feature_cols)
    
    # ----------------------------------------------------
    # >>> PASO DE NORMALIZACIÓN: TRANSFORM en la predicción en vivo
    # ----------------------------------------------------
    # Usamos el scaler previamente ajustado (fit) para transformar los nuevos datos.
    input_scaled = scaler.transform(input_df)

    probabilities = model.predict_proba(input_scaled)[0]
    
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
    
    current_features = {
        'RSI': last_row['RSI'],
        'SMA_50': last_row['SMA_50'],
        'EMA_20': last_row['EMA_20'],
        'ADX': last_row['ADX'],
        'CCI': last_row['CCI'],
        'MACD': last_row['MACD'], 
        'ATR': last_row['ATR'],   
        'Polaridad_Sentimiento': sentiment_data[0] 
    }
    return current_features

# Intentar cargar el modelo y el scaler al inicio del script
load_model()