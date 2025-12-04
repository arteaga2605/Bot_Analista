# main.py (Integración Fase 1 y Fase 2)
import data_fetcher
import technical_analysis
import sentiment_analyzer # NUEVO
import time

# Datos simulados de noticias (En la Fase 3, esto se obtendrá de APIs reales)
SIMULATED_NEWS = [
    "CEO de Coinbase: La regulación de las criptomonedas es inminente y positiva.",
    "Ballenas de Bitcoin mueven 500 millones USD a exchanges, generando incertidumbre.",
    "El RSI de Ethereum muestra sobreventa, posible rebote en camino.",
    "Nuevo fondo de inversión de BlackRock aprueba exposición a activos digitales.",
    "El miedo y la codicia han alcanzado niveles de euforia, ¡cuidado con la corrección!",
]

def run_bot():
    print("--- 🤖 INICIANDO CRYPTO ANALYST BOT (FASE 2) ---")
    
    # 1. Obtener Datos Técnicos
    df = data_fetcher.fetch_market_data()
    
    if df is not None:
        # 2. Análisis Técnico
        df_analyzed = technical_analysis.analyze_data(df)
        
        # 3. Análisis de Sentimiento (Fase 2)
        polaridad, subjetividad, sentimiento_gral = sentiment_analyzer.analyze_crypto_narrative(SIMULATED_NEWS)
        
        # 4. Interpretación Básica (Veredicto Técnico)
        signal, rsi, price = technical_analysis.generate_signal(df_analyzed)
        
        # 5. Reporte Final
        print("\n" + "="*50)
        print(f"📢 REPORTE COMPLETO DE ANÁLISIS")
        print("="*50)
        
        # Reporte Técnico
        print("📈 ANÁLISIS TÉCNICO:")
        print(f"  > Símbolo: {data_fetcher.config.SYMBOL}")
        print(f"  > Precio Actual: ${price:,.2f}")
        print(f"  > RSI (14): {round(rsi, 2)} ({signal})")
        
        # Reporte Sentimiento
        print("\n📰 ANÁLISIS DE SENTIMIENTO:")
        print(f"  > Polaridad Promedio: {round(polaridad * 100, 2)}%")
        print(f"  > Veredicto Social: {sentimiento_gral}")
        
        # Conclusión (Lógica simple para Fase 2)
        if sentimiento_gral == "SENTIMIENTO BEARISH (PESIMISTA)" and signal == "POSIBLE VENTA (Corrección)":
             final_pred = "ALTA PROBABILIDAD DE CAÍDA"
        elif sentimiento_gral == "SENTIMIENTO BULLISH (OPTIMISTA)" and signal == "POSIBLE COMPRA (Rebote)":
             final_pred = "ALTA PROBABILIDAD DE SUBIDA"
        else:
             final_pred = "MERCADO MIXTO, PRECAUCIÓN"

        print("\n🧠 CONCLUSIÓN (PREDICCIÓN SIMPLE):")
        print(f"  >> Veredicto Final: {final_pred}")
        print("="*50 + "\n")
        
    else:
        print("❌ No se pudo completar el análisis.")

if __name__ == "__main__":
    run_bot()