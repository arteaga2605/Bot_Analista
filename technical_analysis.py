# technical_analysis.py (Versión con librería 'ta')
import pandas as pd
import ta # Importación de la nueva librería

def analyze_data(df):
    """
    Recibe un DataFrame con precios y añade indicadores técnicos usando la librería TA.
    """
    if df is None or df.empty:
        print("⚠️ No hay datos para analizar.")
        return None

    print("📊 Calculando indicadores técnicos con la librería 'ta'...")

    # 1. RSI (Índice de Fuerza Relativa)
    # df['RSI'] = df.ta.rsi(length=14) <--- Código anterior
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # 2. SMA (Media Móvil Simple) - 50 periodos
    # df['SMA_50'] = df.ta.sma(length=50) <--- Código anterior
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()

    # 3. EMA (Media Móvil Exponencial) - 20 periodos
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()

    # 4. Bandas de Bollinger (Añade 3 columnas: alta, baja y media)
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BBL'] = bb.bollinger_lband() # Banda Baja
    df['BBU'] = bb.bollinger_hband() # Banda Alta
    
    # IMPORTANTE: La librería 'ta' puede dejar valores NaN al inicio del DataFrame 
    # (porque necesita datos previos para calcular los indicadores).
    # Eliminamos esas filas para trabajar solo con datos limpios
    df = df.dropna()

    return df

def generate_signal(df):
    """
    Genera una señal simple basada en el último dato (la vela más reciente cerrada).
    """
    if df.empty:
        return "SIN DATOS VÁLIDOS", 0, 0
        
    last_row = df.iloc[-1] # Última fila
    rsi = last_row['RSI']
    close_price = last_row['close']
    ema_20 = last_row['EMA_20']

    signal = "NEUTRAL"
    confidence = "0%"

    # Lógica simple de ejemplo
    if rsi < 30 and close_price > ema_20:
        signal = "POSIBLE COMPRA (Rebote)"
        confidence = "60%"
    elif rsi > 70 and close_price < ema_20:
        signal = "POSIBLE VENTA (Corrección)"
        confidence = "60%"
    
    return signal, rsi, close_price