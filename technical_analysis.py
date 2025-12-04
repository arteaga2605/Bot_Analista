# technical_analysis.py (Versión con ADX y CCI)
import pandas as pd
import ta 

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
    
    # --- NUEVOS INDICADORES ---
    
    # 4. ADX (Average Directional Index) - Fuerza de Tendencia
    # ADXIndicator añade las columnas ADX, +DI y -DI
    adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()

    # 5. CCI (Commodity Channel Index) - Desviación del promedio
    df['CCI'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()

    # 6. Bandas de Bollinger 
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BBL'] = bb.bollinger_lband()
    df['BBU'] = bb.bollinger_hband()
    
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