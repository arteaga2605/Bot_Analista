# sentiment_analyzer.py (Versión Final y Estable)
from textblob import TextBlob
# Ya no importamos NaiveBayesAnalyzer porque requiere el paquete roto

def get_sentiment(text):
    """
    Analiza un texto dado y devuelve su polaridad y subjetividad usando el analizador base.
    """
    if not text:
        return 0.0, 0.0, "NEUTRAL"

    # Inicializar TextBlob con el texto
    # TextBlob detecta automáticamente el idioma y usa un analizador estándar para la polaridad.
    blob = TextBlob(text)

    # Obtener Polaridad y Subjetividad
    polaridad = blob.sentiment.polarity
    subjetividad = blob.sentiment.subjectivity

    # Clasificar el sentimiento con un umbral simple
    if polaridad > 0.2:
        clasificacion = "POSITIVO"
    elif polaridad < -0.2:
        clasificacion = "NEGATIVO"
    else:
        clasificacion = "NEUTRAL"
        
    return polaridad, subjetividad, clasificacion

def analyze_crypto_narrative(news_list):
    """
    Toma una lista de titulares/textos y calcula el sentimiento promedio.
    """
    total_polaridad = 0
    num_noticias = len(news_list)
    
    if num_noticias == 0:
        return 0.0, 0.0, "NEUTRAL"

    print(f"📰 Analizando sentimiento de {num_noticias} textos...")
    
    for text in news_list:
        polaridad, _, _ = get_sentiment(text)
        total_polaridad += polaridad
        
    # Calcular el promedio
    polaridad_promedio = total_polaridad / num_noticias

    # Clasificar el sentimiento general del mercado
    if polaridad_promedio > 0.1:
        sentimiento_general = "SENTIMIENTO BULLISH (OPTIMISTA)"
    elif polaridad_promedio < -0.1:
        sentimiento_general = "SENTIMIENTO BEARISH (PESIMISTA)"
    else:
        sentimiento_general = "SENTIMIENTO NEUTRAL/MIXTO"

    # La subjetividad promedio ya no se calcula, se devuelve 0.0
    return polaridad_promedio, 0.0, sentimiento_general