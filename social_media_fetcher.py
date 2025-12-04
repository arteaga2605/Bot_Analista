# social_media_fetcher.py
import requests
import random # Lo usamos para simular un desfase, pero la estructura es real

# URL de una API simulada de noticias de criptomonedas (comportamiento real)
# Nota: En un proyecto real, necesitarías una API key para CryptoPanic o NewsAPI.
# Usaremos una URL de prueba para simular la conexión HTTP real.
NEWS_API_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

def fetch_recent_headlines(symbol='BTC'):
    """
    Simula la obtención de titulares de noticias recientes usando requests.
    En la vida real, se filtraría por el símbolo (ej. Bitcoin).
    """
    print("📡 Obteniendo titulares de noticias recientes (vía API)...")
    
    try:
        # Hacemos la solicitud HTTP
        response = requests.get(NEWS_API_URL, timeout=10)
        response.raise_for_status() # Lanza error para códigos 4xx/5xx

        data = response.json()
        headlines = []
        
        # Parseamos el JSON para extraer los titulares de las últimas 10 noticias
        if 'Data' in data:
            for item in data['Data'][:10]:
                if 'title' in item:
                    headlines.append(item['title'])
            
            print(f"✅ {len(headlines)} titulares obtenidos de fuente real.")
            return headlines
        
        return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Error al conectar con la API de noticias: {e}")
        # Retornamos una lista vacía para no romper el programa
        return ["Error de red, sentimiento desconocido."]