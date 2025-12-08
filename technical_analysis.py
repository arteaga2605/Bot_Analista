# technical_analysis.py (CÓDIGO COMPLETO CORREGIDO)
import pandas as pd
import ta 
import talib # Asegúrate de tener 'talib' instalado: pip install ta-lib

def analyze_data(df):
    """
    Recibe un DataFrame con precios y añade indicadores técnicos usando la librería TA.
    """
    if df is None or df.empty:
        print("⚠️ No hay datos para analizar.")
        return None

    print("📊 Calculando indicadores técnicos con la librería 'ta'...")

    # 1. RSI (Índice de Fuerza Relativa)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # 2. SMA (Media Móvil Simple) - 50 periodos
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()

    # 3. EMA (Media Móvil Exponencial) - 20 periodos
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    
    # 4. ADX (Average Directional Index) - Fuerza de Tendencia (14 periodos)
    adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    
    # 5. CCI (Commodity Channel Index) - Desviación del promedio
    df['CCI'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()

    # 6. MACD (Moving Average Convergence Divergence)
    macd_indicator = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd_indicator.macd() 
    # La línea de señal es crítica para calcular MACD_SIGNAL_DIFF
    df['MACD_Signal'] = macd_indicator.macd_signal() 
    
    # 7. ATR (Average True Range) - Volatilidad
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # --- INDICADORES FALTANTES QUE DEBEN ESTAR EN EL MODELO (20 FEATURES) ---
    
    # 8. SMA (Media Móvil Simple) - 100 periodos
    df['SMA_100'] = ta.trend.SMAIndicator(close=df['close'], window=100).sma_indicator()
    
    # 9. ADX (Average Directional Index) - 50 periodos 
    adx_50_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=50)
    df['ADX_50'] = adx_50_indicator.adx()
    
    # 10. Volumen Relativo (Volumen actual / SMA de Volumen 20)
    # FIX CRÍTICO: Reemplazamos la clase de TA obsoleta con la función de Pandas para el SMA de Volumen.
    df['VOL_SMA_20'] = df['volume'].rolling(window=20).mean()
    df['REL_VOLUME'] = df['volume'] / df['VOL_SMA_20']
    df = df.drop(columns=['VOL_SMA_20'], errors='ignore')

    # 11. Patrones de Velas
    print("🕯️ Detectando patrones de velas...")
    try:
        # Se divide por 100 para normalizar el output de TALIB (que usa 100, -100, 0)
        df['CDLHAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']) / 100
        df['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close']) / 100
        df['CDLDOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']) / 100
        df['CDLENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) / 100
        df['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close']) / 100
        df['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close']) / 100
        
    except Exception as e:
        print(f"⚠️ Error al calcular patrones de velas (TALIB no disponible o datos insuficientes): {e}. Usando valores por defecto.")
        # Llenar con ceros si talib falla
        for col in ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDOJI', 'CDLENGULFING', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR']:
             df[col] = 0.0

    # Eliminar columnas auxiliares que NO van al modelo (BBL, BBU)
    df = df.drop(columns=['BBL', 'BBU'], errors='ignore')

    # Eliminar las filas con NaN (las primeras filas no tienen cálculo completo)
    df = df.dropna()

    return df

def generate_signal(df):
    """
    Genera una señal simple basada en el último dato para el reporte de texto.
    """
    if df.empty:
        return "SIN DATOS VÁLIDOS", 0, 0
        
    last_row = df.iloc[-1]
    rsi = last_row['RSI']
    close_price = last_row['close']
    ema_20 = last_row['EMA_20']

    signal = "NEUTRAL"
    confidence = "0%"

    if rsi < 30 and close_price > ema_20:
        signal = "POSIBLE COMPRA (Rebote)"
        confidence = "60%"
    elif rsi > 70 and close_price < ema_20:
        signal = "POSIBLE VENTA (Corrección)"
        confidence = "60%"
    
    return signal, rsi, close_price