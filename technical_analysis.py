# technical_analysis.py (CÓDIGO COMPLETO MODIFICADO CON PATRONES)
import pandas as pd
import ta
import pattern_detector # <--- NUEVA IMPORTACIÓN

def analyze_data(df):
    """
    Calcula todos los indicadores técnicos y patrones de velas.
    """
    
    if df is None or df.empty:
        print("⚠️ No hay datos para analizar.")
        return None

    print("📊 Calculando indicadores técnicos con la librería 'ta'...")

    # Asegurarse de tener una copia limpia
    df_analyzed = df.copy()
    
    # 1. Indicadores de Tendencia
    df_analyzed['SMA_50'] = ta.trend.sma_indicator(df_analyzed['close'], window=50)
    df_analyzed['EMA_20'] = ta.trend.ema_indicator(df_analyzed['close'], window=20)
    # 2. ADX (Average Directional Index) - Fuerza de Tendencia
    df_analyzed['ADX'] = ta.trend.adx(df_analyzed['high'], df_analyzed['low'], df_analyzed['close'], window=14)
    
    # 3. Indicadores de Momento
    df_analyzed['RSI'] = ta.momentum.rsi(df_analyzed['close'], window=14)
    df_analyzed['CCI'] = ta.trend.cci(df_analyzed['high'], df_analyzed['low'], df_analyzed['close'], window=20)
    
    # 4. MACD (Moving Average Convergence Divergence) - Necesitamos la línea MACD y la señal
    macd_indicator = ta.trend.MACD(df_analyzed['close'], window_fast=12, window_slow=26, window_sign=9)
    df_analyzed['MACD'] = macd_indicator.macd()
    df_analyzed['MACD_Signal'] = macd_indicator.macd_signal() # ¡Añadida la Señal!
    
    # 5. ATR (Average True Range) - Volatilidad
    df_analyzed['ATR'] = ta.volatility.average_true_range(df_analyzed['high'], df_analyzed['low'], df_analyzed['close'], window=14)

    # >>> 6. Detección de Patrones de Velas Japonesas (NUEVA FASE)
    df_analyzed = pattern_detector.detect_candlestick_patterns(df_analyzed)
    
    # Eliminar las filas con NaN (las primeras filas no tienen cálculo completo)
    return df_analyzed.dropna()

# NOTA: La función generate_signal(df) ha sido eliminada.
# El sistema ahora depende completamente de la predicción del modelo ML.