# social_media_fetcher.py (FINAL CORREGIDO)
import requests
from textblob import TextBlob
import os

# --- CONFIGURACIÓN DEL SERVICIO EXTERNO (EJEMPLO) ---
NEWS_API_URL = "https://example.com/api/v1/crypto/news" 

def fetch_recent_headlines(symbol, count=10):
    """
    Simula la obtención de titulares de noticias para un símbolo específico.
    """
    simulated_headlines = [
        f"Chainlink ({symbol}) price rallies 10% on whale accumulation.",
        "Fear and Greed Index shows high greed in crypto markets.",
        "Major institutional investment hits the DeFi sector.",
        "Chainlink network usage hits an all-time high.",
        f"{symbol} faces minor correction after profit taking.",
        "Global regulatory uncertainty weighs on altcoins.",
        "New partnership announced for LINK ecosystem.",
        "Market analysts predict strong Q4 for decentralized oracles.",
        "Trading volumes drop significantly across major exchanges.",
        "Chainlink staking proposal generates community excitement."
    ]
    
    return simulated_headlines[:count]

def analyze_sentiment(texts):
    """
    Analiza el sentimiento de una lista de textos.
    Retorna la polaridad promedio normalizada de (0 a 1).
    """
    polarities = []
    
    for text in texts:
        analysis = TextBlob(text)
        # >>> CORRECCIÓN: Usar .polarity en lugar de .polaridad
        polarities.append(analysis.sentiment.polarity) 
        
    if not polarities:
        return [0.5]
        
    avg_polaridad = sum(polarities) / len(polarities)
    normalized_polaridad = (avg_polaridad + 1) / 2
    
    return [normalized_polaridad]