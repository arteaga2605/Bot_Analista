# data_fetcher.py (CÓDIGO COMPLETO)
import pandas as pd
from binance.client import Client
import os # Importar para usar variables de entorno

# --- CONFIGURACIÓN DE ACCESO (MANTENER EN SECRETO) ---
# Se recomienda usar variables de entorno para las claves, pero se dejan aquí para la demostración.
# Si quieres operar en modo live real, descomenta estas líneas y reemplaza con tus claves
# API_KEY = os.getenv('BINANCE_API_KEY', 'TU_API_KEY_AQUI') 
# API_SECRET = os.getenv('BINANCE_API_SECRET', 'TU_SECRET_AQUI')
API_KEY = "TU_API_KEY_AQUI" 
API_SECRET = "TU_SECRET_AQUI"

# --- CONFIGURACIÓN DE TRADING ---
# Definir aquí el par y el intervalo de tiempo.
config = {
    'SYMBOL': 'LINKUSDT', # Par de trading
    # Intervalo de 4 horas (usa Client.KLINE_INTERVAL_... para otros intervalos)
    'TIMEFRAME': Client.KLINE_INTERVAL_4HOUR 
}

# Inicializa el cliente de Binance
try:
    # Intentamos inicializar con claves
    client = Client(API_KEY, API_SECRET)
except Exception as e:
    print(f"⚠️ Aviso: Error al inicializar cliente con claves. Continuando en modo solo lectura. {e}")
    # Si las claves no funcionan, inicializa sin ellas para modo solo lectura (lectura de precios)
    client = Client() 

def get_historical_data(limit=1000):
    """
    Obtiene data histórica de velas (OHLCV) de Binance y la devuelve como un DataFrame.
    
    Args:
        limit (int): Número de velas a solicitar.
        
    Returns:
        pd.DataFrame: DataFrame de precios con índice de tiempo.
    """
    try:
        # Pide las velas históricas usando el símbolo y timeframe de la configuración
        klines = client.get_historical_klines(
            symbol=config['SYMBOL'],
            interval=config['TIMEFRAME'],
            limit=limit
        )

        # Crea el DataFrame de Pandas
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Limpieza y Conversión
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # Convertir precios y volumen a float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Dejar solo las columnas esenciales para el análisis
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df
    
    except Exception as e:
        print(f"❌ Error al obtener datos de Binance: {e}")
        return None

# Puedes agregar más funciones aquí (ej. get_current_price, place_order, etc.)