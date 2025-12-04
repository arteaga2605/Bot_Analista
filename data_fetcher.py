# data_fetcher.py (COMPLETO Y CORREGIDO: Ticker LINKUSDT)
import pandas as pd
from binance.client import Client
import os

# --- CONFIGURACIÓN DE LA API DE BINANCE ---
config = {
    'SYMBOL': 'LINKUSDT', 
    'TIMEFRAME': '1h',
    'LIMIT': 1000  
}
# --- CREDENCIALES (Asegúrate de que estas variables de entorno estén configuradas) ---
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

def fetch_data():
    """
    Obtiene datos históricos (velas) de Binance.
    """
    try:
        # Inicialización del cliente (usa la API pública si las credenciales no están definidas)
        client = Client(API_KEY, API_SECRET)
        
        # Obtener los datos históricos
        klines = client.get_historical_klines(
            symbol=config['SYMBOL'],
            interval=config['TIMEFRAME'],
            limit=config['LIMIT']
        )
        
        # Estructurar los datos en un DataFrame de Pandas
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Limpieza y conversión
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        return df
        
    except Exception as e:
        print(f"❌ Error al obtener datos de Binance: {e}")
        return None