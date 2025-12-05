# prediction_model.py (CÓDIGO COMPLETO CORREGIDO)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError # Añadido para manejo de errores
import joblib 
import os 
import ta 

# Variables globales para simular el aprendizaje
model = None
scaler = None 
MODEL_FILENAME = 'crypto_model_xgb.joblib' 
SCALER_FILENAME = 'crypto_scaler.joblib' 

# >>> LISTA DE FEATURES ACTUALIZADA con Patrones de Velas
feature_cols = [
    # 10 Features Técnicas Originales + 3 Features de MACD
    'RSI', 'SMA_50', 'EMA_20', 'ADX', 'CCI', 'MACD', 'ATR',
    'CLOSE_VS_SMA', 
    'CLOSE_VS_EMA', 
    'MACD_SIGNAL_DIFF',
    # 6 Nuevas Features de Patrones de Velas
    'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDOJI', 
    'CDLENGULFING', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR'
] 
final_feature_cols = feature_cols + ['Polaridad_Sentimiento']
# El número total de features ahora es 10 + 6 + 1 = 17

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
            
            # Verificación del número de features del modelo cargado
            # Usamos un try-except para manejar el caso de modelos antiguos o corruptos
            try:
                if model.n_features_in_ != len(final_feature_cols):
                     print(f"⚠️ El modelo cargado tiene {model.n_features_in_} features, pero se esperan {len(final_feature_cols)} (por los nuevos patrones). Se forzará re-entrenamiento.")
                     model = None 
                     loaded_ok = False
            except NotFittedError:
                 # Esto puede pasar si el modelo está cargado pero no tiene metadata de fitting
                 print("⚠️ Metadata de features no encontrada en el modelo. Se forzará re-entrenamiento.")
                 model = None 
                 loaded_ok = False
                 
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}. Entrenando uno nuevo.")
            model = None
            loaded_ok = False
            
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
    Combina datos técnicos y de sentimiento, crea variables relativas y el Target.
    """
    # Se añade MACD_Signal y los Patrones a las columnas requeridas
    required_cols_from_df = [
        'close', 'SMA_50', 'EMA_20', 
        'RSI', 'ADX', 'CCI', 'MACD', 'ATR', 'MACD_Signal',
        'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDOJI', 
        'CDLENGULFING', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR'
    ] 
    df_features = df_analyzed[required_cols_from_df].copy() 

    # --- CREACIÓN DE LAS FEATURES RELATIVAS ---
    df_features['CLOSE_VS_SMA'] = (df_features['close'] - df_features['SMA_50']) / df_features['close']
    df_features['CLOSE_VS_EMA'] = (df_features['close'] - df_features['EMA_20']) / df_features['close']
    df_features['MACD_SIGNAL_DIFF'] = df_features['MACD'] - df_features['MACD_Signal']
    # ------------------------------------------------

    # Preparación del Target
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)
    
    polaridad_promedio = sentiment_data[0]
    df_features['Polaridad_Sentimiento'] = polaridad_promedio
    
    COLS_TO_KEEP = final_feature_cols + ['Target', 'close'] 
    df_features = df_features[COLS_TO_KEEP].dropna()
    
    return df_features

def train_or_update_model(data_df):
    """
    Entrena un nuevo modelo XGBoost con hiperparámetros ajustados.
    """
    global model, scaler
    
    model_was_initialized = False # <<< FLAG PARA EVITAR NotFittedError

    if model is None:
        model = XGBClassifier(
            objective='binary:logistic', 
            n_estimators=300,            
            learning_rate=0.05,          
            max_depth=3,                 
            eval_metric='logloss',       
            random_state=42
        )
        model_was_initialized = True # <<< Se inicializó, forzamos re-entrenamiento del scaler
        print("Model has been re-initialized to None, proceeding with full fit.")

    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    X = data_df[final_feature_cols] 
    y = data_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Creamos un nuevo scaler si no existe O si el modelo fue reinicializado (por feature mismatch)
    if scaler is None or model_was_initialized: # <<< LÓGICA CORREGIDA: Usamos el flag
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        # Usamos el scaler existente
        X_train_scaled = scaler.transform(X_train)
        
    X_test_scaled = scaler.transform(X_test)
    
    # Ignoramos la advertencia del deprecated use_label_encoder (ya la corregimos previamente)
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

    # Verificación de la cantidad de features antes de transformar
    if len(current_features) != len(final_feature_cols):
         print(f"❌ Error de features: Se esperaban {len(final_feature_cols)} features, pero se recibieron {len(current_features)}.")
         return 0.0, "ERROR", "Conteo de features incorrecto"


    input_df = pd.DataFrame([current_features], columns=final_feature_cols)
    
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
    Extrae la última fila de datos y calcula las features relativas en tiempo real.
    """
    last_row = df_analyzed.iloc[-1]
    
    # Recálculo de la diferencia MACD (ya que TA-Lib no la incluye directamente en la librería 'ta')
    macd_signal_diff_val = last_row['MACD'] - last_row['MACD_Signal']
    
    current_features = {
        # 10 Características Técnicas Originales
        'RSI': last_row['RSI'],
        'SMA_50': last_row['SMA_50'],
        'EMA_20': last_row['EMA_20'],
        'ADX': last_row['ADX'],
        'CCI': last_row['CCI'],
        'MACD': last_row['MACD'], 
        'ATR': last_row['ATR'],   
        
        'CLOSE_VS_SMA': (last_row['close'] - last_row['SMA_50']) / last_row['close'], 
        'CLOSE_VS_EMA': (last_row['close'] - last_row['EMA_20']) / last_row['close'], 
        'MACD_SIGNAL_DIFF': macd_signal_diff_val,

        # 6 Características de Patrones de Velas (Directamente de la fila analizada)
        'CDLHAMMER': last_row['CDLHAMMER'],
        'CDLINVERTEDHAMMER': last_row['CDLINVERTEDHAMMER'],
        'CDLDOJI': last_row['CDLDOJI'],
        'CDLENGULFING': last_row['CDLENGULFING'],
        'CDLSHOOTINGSTAR': last_row['CDLSHOOTINGSTAR'],
        'CDLMORNINGSTAR': last_row['CDLMORNINGSTAR'],
        
        # 1 Característica de Sentimiento
        'Polaridad_Sentimiento': sentiment_data[0] 
    }
    
    return {k: current_features[k] for k in final_feature_cols}

load_model()