# pattern_detector.py (NUEVO ARCHIVO)
import talib
import pandas as pd
import numpy as np

def detect_candlestick_patterns(df):
    """
    Detecta patrones de velas japonesas comunes en el DataFrame OHLCV
    y añade las señales como nuevas columnas.
    Retorna los valores de TA-Lib (100, -100, 0)
    """
    
    # Aseguramos que las columnas de entrada sean las correctas para TA-Lib
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    close_price = df['close']
    
    # --- PATRONES INDIVIDUALES (REVERSIÓN) ---
    # Martillo (alcista)
    df['CDLHAMMER'] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
    # Martillo Invertido (alcista)
    df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open_price, high_price, low_price, close_price)
    # Doji (Indecisión/Posible Reversión)
    df['CDLDOJI'] = talib.CDLDOJI(open_price, high_price, low_price, close_price)
    # Estrella Fugaz (bajista)
    df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open_price, high_price, low_price, close_price)
    
    # --- PATRONES DE MÚLTIPLES VELAS ---
    # Envolvente (alcista/bajista fuerte)
    df['CDLENGULFING'] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
    # Estrella de la Mañana (Morning Star - fuerte alcista)
    df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
    
    # Reemplazamos los valores NaN iniciales (lookback) por 0, ya que NaN no es útil para el modelo.
    # Los valores de las columnas serán: 100 (fuerza alcista), -100 (fuerza bajista), o 0 (no detectado).
    df.fillna(0, inplace=True)
    
    return df