# prediction_model.py (CÓDIGO COMPLETO CORREGIDO PARA ROBUSTEZ Y CONSISTENCIA)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError 
import joblib 
import os 
import ta 

# Variables globales para simular el aprendizaje
model = None
scaler = None 
MODEL_FILENAME = 'crypto_model_xgb.joblib' 
SCALER_FILENAME = 'crypto_scaler.joblib' 

# --- DEFINICIÓN CANÓNICA Y EXPLICITA DE LOS 20 FEATURES EN ORDEN ESTRICTO ---

# 20 Features que van al modelo. Este orden es el que DEBE coincidir con el Scaler/Modelo.
final_feature_cols = [
    'RSI', 'SMA_50', 'EMA_20', 'ADX', 'CCI', 'MACD', 'ATR',
    'SMA_100', 'ADX_50', 'REL_VOLUME', # 10 Features Base
    'CLOSE_VS_SMA', 'CLOSE_VS_EMA', 'MACD_SIGNAL_DIFF', # 3 Features Derivadas
    'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDOJI', 'CDLENGULFING', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR', # 6 Features Candlestick
    'Polaridad_Sentimiento' # 1 Feature Sentimiento
] # Total: 20

# ------------------------------------------------------------------
# Funciones load_model, save_model y prepare_data_for_training se mantienen igual.
# ------------------------------------------------------------------

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
            
            # Verificación del número de features
            expected_features_count = len(final_feature_cols)
            try:
                if model.n_features_in_ != expected_features_count:
                     print(f"⚠️ El modelo cargado tiene {model.n_features_in_} features, pero se esperan {expected_features_count}. Eliminando modelo y scaler antiguos.")
                     model = None 
                     scaler = None 
                     if os.path.exists(SCALER_FILENAME): os.remove(SCALER_FILENAME)
                     if os.path.exists(MODEL_FILENAME): os.remove(MODEL_FILENAME)
                     loaded_ok = False
            except AttributeError:
                 # Algunos modelos no tienen n_features_in_
                 pass
            except NotFittedError:
                 print("⚠️ Metadata de features no encontrada en el modelo. Se forzará re-entrenamiento.")
                 model = None 
                 scaler = None 
                 loaded_ok = False
                 
        except Exception as e:
            print(f"⚠️ Error al cargar el modelo: {e}. Entrenando uno nuevo.")
            model = None
            scaler = None 
            loaded_ok = False
            
    if scaler is None and os.path.exists(SCALER_FILENAME):
        try:
            scaler = joblib.load(SCALER_FILENAME)
            print(f"📏 Scaler cargado exitosamente desde {SCALER_FILENAME}.")
            # Es vital que el scaler tenga la misma cantidad de features que el modelo
            try:
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != len(final_feature_cols) and model is not None:
                    print(f"⚠️ El Scaler cargado tiene {scaler.n_features_in_} features, pero se esperan {len(final_feature_cols)}. Se forzará re-entrenamiento.")
                    scaler = None
                    model = None
                    loaded_ok = False
            except AttributeError:
                 # No todos los scalers tienen n_features_in_
                 pass
                 
        except Exception as e:
            print(f"⚠️ Error al cargar el scaler: {e}. Se necesitará re-entrenamiento.")
            scaler = None
            loaded_ok = False
    elif scaler is None and model is not None:
        # Si el modelo está cargado, el scaler también debería estarlo. Si falta, forzamos re-entrenamiento.
        model = None
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
            # print(f"💾 Modelo guardado exitosamente en {MODEL_FILENAME}.") # Comentado para no repetir el mensaje
        except Exception as e:
            print(f"❌ Error al guardar el modelo: {e}")

    if scaler is not None:
        try:
            joblib.dump(scaler, SCALER_FILENAME)
            # print(f"💾 Scaler guardado exitosamente en {SCALER_FILENAME}.") # Comentado para no repetir el mensaje
        except Exception as e:
            print(f"❌ Error al guardar el scaler: {e}")


def prepare_data_for_training(df_analyzed, sentiment_data):
    """
    Combina datos técnicos y de sentimiento, crea variables relativas y el Target.
    """
    
    # Copiamos solo las columnas de precio y volumen para empezar
    df_features = df_analyzed.copy()

    # 1. CREACIÓN DE LAS FEATURES RELATIVAS/DERIVADAS
    # Usamos .get() para mayor robustez, asumiendo que technical_analysis.py ya calculó MACD_Signal
    df_features['CLOSE_VS_SMA'] = (df_features.get('close', 0.0) - df_features.get('SMA_50', 0.0)) / df_features.get('close', 1.0)
    df_features['CLOSE_VS_EMA'] = (df_features.get('close', 0.0) - df_features.get('EMA_20', 0.0)) / df_features.get('close', 1.0)
    # MACD_Signal ahora debería estar en df_features gracias a technical_analysis.py
    df_features['MACD_SIGNAL_DIFF'] = df_features.get('MACD', 0.0) - df_features.get('MACD_Signal', 0.0)

    # 2. Preparación del Target y Sentimiento
    df_features['Precio_Futuro'] = df_features['close'].shift(-1)
    df_features['Target'] = np.where(df_features['Precio_Futuro'] > df_features['close'], 1, 0)
    polaridad_promedio = sentiment_data[0]
    df_features['Polaridad_Sentimiento'] = polaridad_promedio 
    
    # 3. Seleccionar columnas finales y reordenarlas estrictamente
    # Estas son las columnas que necesitamos para el backtester y la predicción, además de las 20 features
    COLS_TO_KEEP = final_feature_cols + ['Target', 'close', 'high', 'low', 'ATR'] 
    
    # FIX CRÍTICO: Filtramos el DataFrame para que SOLO contenga las 20 features + las columnas auxiliares
    # MACD_Signal y otras columnas temporales se eliminan automáticamente si no están en COLS_TO_KEEP
    df_features = df_features[[col for col in COLS_TO_KEEP if col in df_features.columns]].dropna()
    
    return df_features

def train_or_update_model(data_df):
    """
    Entrena o actualiza un modelo XGBoost.
    """
    global model, scaler
    
    model_was_initialized = False 

    if model is None:
        model = XGBClassifier(
            objective='binary:logistic', n_estimators=300,            
            learning_rate=0.05, max_depth=3,                 
            eval_metric='logloss', random_state=42
        )
        model_was_initialized = True 
        print("Model has been re-initialized to None, proceeding with full fit.")

    if data_df.shape[0] < 50:
        print("⚠️ Datos insuficientes para entrenamiento. Se necesitan al menos 50 filas.")
        return 0.0

    # Extraer X asegurando el ORDEN DE LAS COLUMNAS según final_feature_cols (20 columnas)
    X = data_df[final_feature_cols] 
    y = data_df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_was_initialized or scaler is None:
        scaler = MinMaxScaler()
        # El scaler se ajusta aquí a 20 features con nombres de columna.
        X_train_scaled = scaler.fit_transform(X_train) 
    else:
        # Transformamos, esperando el mismo orden de columnas que en el fit (20)
        X_train_scaled = scaler.transform(X_train) 
        
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train) 
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🧠 Modelo XGBoost entrenado/actualizado. Precisión: {round(accuracy * 100, 2)}%")
    
    save_model()
    
    return accuracy

def predict_next_move(current_features_df):
    """
    Hace una predicción basada en los datos más recientes, aplicando el scaler.
    Recibe un DataFrame de 1 fila con TODAS las columnas necesarias.
    Devuelve 2 valores (prob_up, prediction_text).
    """
    global model, scaler
    
    if model is None or scaler is None:
        return 0.0, "NEUTRAL (Modelo/Scaler no entrenados)" 

    try:
        # FIX CRÍTICO: Aseguramos que solo pasamos las 20 features canónicas en el ORDEN CORRECTO
        input_df = current_features_df[final_feature_cols]
        
        # Transformación con el scaler (punto crítico de fallo si hay inconsistencia)
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
    
    except KeyError as e:
        # Esto captura si falta una columna, lo que causaría la inconsistencia
        print(f"❌ Error: La feature '{e.args[0]}' falta en el DataFrame de predicción.")
        return 0.0, "ERROR CRÍTICO EN PREDICCIÓN: Feature faltante"
    except Exception as e:
        # Captura errores internos del scaler/modelo (como el de feature names/count)
        print(f"❌ Error crítico en predicción: {e}")
        return 0.0, "ERROR CRÍTICO EN PREDICCIÓN"


def get_current_features(df_analyzed, sentiment_data):
    """
    Extrae la última fila de datos, calcula las features relativas en tiempo real, 
    y devuelve un DataFrame de 1 fila con las features canónicas más 'close' y 'ATR'.
    """
    if df_analyzed.empty:
        # Devolvemos un DataFrame vacío para manejar el error
        return pd.DataFrame() 
        
    last_row = df_analyzed.iloc[-1].copy() # Usamos una copia para evitar SettingWithCopyWarning
    
    close_val = last_row.get('close', 0.0)
    
    # 1. CÁLCULOS DERIVADOS EN TIEMPO REAL
    # Estas son series de Pandas, las calculamos en la fila para convertir a escalar al final
    last_row['CLOSE_VS_SMA'] = (close_val - last_row.get('SMA_50', 0.0)) / close_val if close_val != 0.0 else 0.0
    last_row['CLOSE_VS_EMA'] = (close_val - last_row.get('EMA_20', 0.0)) / close_val if close_val != 0.0 else 0.0
    last_row['MACD_SIGNAL_DIFF'] = last_row.get('MACD', 0.0) - last_row.get('MACD_Signal', 0.0)
    
    # 2. Agregar el Sentimiento
    last_row['Polaridad_Sentimiento'] = sentiment_data[0]
    
    # 3. Construir el DataFrame final (1 fila)
    # Incluimos ATR y close porque son necesarios para el cálculo de SL/TP en main.py
    required_cols = final_feature_cols + ['close', 'ATR']
    
    # Creamos un DF de una fila asegurando que solo tenga las columnas requeridas
    # y las reordenamos estrictamente
    df_live_features = pd.DataFrame([last_row])
    
    # Seleccionamos las columnas requeridas en el orden correcto.
    df_live_features = df_live_features[[col for col in required_cols if col in df_live_features.columns]]
    
    return df_live_features
    
load_model()