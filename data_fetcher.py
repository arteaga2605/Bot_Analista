# data_fetcher.py
import ccxt
import pandas as pd
import config

def fetch_market_data(limit=config.LIMIT): # <-- AHORA ACEPTA EL ARGUMENTO 'limit'
    """
    Conecta con el exchange y descarga datos OHLCV (Open, High, Low, Close, Volume).
    Utiliza el límite pasado como argumento o el límite por defecto de config.
    """
    print(f"🔄 Conectando a {config.EXCHANGE_ID} para obtener datos de {config.SYMBOL}...")
    
    try:
        # Inicializar el exchange
        exchange_class = getattr(ccxt, config.EXCHANGE_ID)
        exchange = exchange_class({
            'enableRateLimit': True, # Respetar límites de velocidad para no ser baneado
        })

        # Descargar velas (OHLCV)
        ohlcv = exchange.fetch_ohlcv(config.SYMBOL, config.TIMEFRAME, limit=limit) # <-- USAMOS EL NUEVO ARGUMENTO 'limit'

        # Convertir a DataFrame de Pandas (formato tabla)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convertir timestamp a fecha legible
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"✅ Datos obtenidos exitosamente: {len(df)} velas.")
        return df

    except ccxt.NetworkError as e:
        print(f"❌ Error de Red: {e}")
        return None
    except ccxt.ExchangeError as e:
        print(f"❌ Error del Exchange (posible símbolo inválido): {e}")
        return None
    except Exception as e:
        print(f"❌ Error desconocido: {e}")
        return None