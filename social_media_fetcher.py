# social_media_fetcher.py
import requests
import random 

# URL de una API simulada de noticias de criptomonedas (comportamiento real)
NEWS_API_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

def fetch_recent_headlines(symbol='BTC'):
    """
    Simula la obtención de titulares de noticias recientes usando requests.
    """
    print("📡 Obteniendo titulares de noticias recientes (vía API)...")
    
    try:
        # Hacemos la solicitud HTTP
        response = requests.get(NEWS_API_URL, timeout=10)
        response.raise_for_status() # Lanza error para códigos 4xx/5xx

        data = response.json()
        headlines = []
        
        # --- CORRECCIÓN DEL ERROR ---
        # 1. Verificamos que 'Data' existe
        # 2. Verificamos que 'Data' es una lista (o lo tratamos como tal)
        if 'Data' in data and isinstance(data['Data'], list):
            # Recorremos la lista de datos y la rebanamos
            for item in data['Data'][:10]:
                if isinstance(item, dict) and 'title' in item:
                    headlines.append(item['title'])
            
            print(f"✅ {len(headlines)} titulares obtenidos de fuente real.")
            return headlines
        
        print("⚠️ Advertencia: 'Data' no está presente o no es una lista en la respuesta de la API.")
        return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al conectar con la API de noticias: {e}")
        # Retornamos una lista vacía para no romper el programa
        return ["Error de red, sentimiento desconocido."]
    except TypeError as e:
        print(f"❌ TypeError durante el parseo de la API: {e}")
        # Si el error persiste, al menos manejamos la excepción
        return []