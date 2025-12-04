# main.py (Integración Fase 1, 2 y 3)
import data_fetcher
import technical_analysis
import sentiment_analyzer
import prediction_model # NUEVO
import time
import pandas as pd

# Datos simulados de noticias (Misma simulación)
SIMULATED_NEWS = [
    "CEO de Coinbase: La regulación de las criptomonedas es inminente y positiva.",
    "Ballenas de Bitcoin mueven 500 millones USD a exchanges, generando incertidumbre.",
    "El RSI de Ethereum muestra sobreventa, posible rebote en camino.",
    "Nuevo fondo de inversión de BlackRock aprueba exposición a activos digitales.",
    "El miedo y la codicia han alcanzado niveles de euforia, ¡cuidado con la corrección!",
]

def run_bot():
    print("--- 🤖 INICIANDO CRYPTO ANALYST BOT (FASE 3: PREDICCIÓN) ---")
    
    # 1. Obtener Datos Técnicos
    df = data_fetcher.fetch_market_data()
    
    if df is not None:
        # 2. Análisis Técnico
        df_analyzed = technical_analysis.analyze_data(df)
        
        # 3. Análisis de Sentimiento
        sentiment_data = sentiment_analyzer.analyze_crypto_narrative(SIMULATED_NEWS)
        
        # 4. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO (SIMULADO)
        # Aquí es donde le enseñamos al modelo a aprender de los datos pasados (simulados)
        data_for_ml = prediction_model.prepare_data_for_training(df_analyzed, sentiment_data)
        accuracy = prediction_model.train_or_update_model(data_for_ml)

        # 5. PREDICCIÓN EN TIEMPO REAL
        current_features = prediction_model.get_current_features(df_analyzed, sentiment_data)
        prob_up, prediction_text = prediction_model.predict_next_move(current_features)
        
        # 6. Reporte Final
        
        price = df_analyzed.iloc[-1]['close'] if not df_analyzed.empty else 0
        rsi = df_analyzed.iloc[-1]['RSI'] if not df_analyzed.empty else 0
        
        print("\n" + "="*50)
        print("📢 REPORTE PROFESIONAL DE PREDICCIÓN (IA)")
        print("="*50)
        
        # Resumen Técnico
        print(f"📈 Símbolo Analizado: {data_fetcher.config.SYMBOL}")
        print(f"💰 Precio Actual: ${price:,.2f}")
        print(f"RSI (14) Actual: {round(rsi, 2)}")

        # Resumen Sentimiento
        print(f"\n📰 SENTIMIENTO (Polaridad): {round(sentiment_data[0] * 100, 2)}% (Impacto en IA)")
        
        # Predicción Final de la IA
        print("\n🧠 PREDICCIÓN DEL MODELO ML:")
        print(f"  > Probabilidad de Subida: {round(prob_up * 100, 2)}%")
        print(f"  > Probabilidad de Bajada: {round((1 - prob_up) * 100, 2)}%")
        print(f"  > **VEREDICTO IA: {prediction_text}**")
        print(f"\n[Precisión simulada: {round(accuracy * 100, 2)}%]")
        print("="*50 + "\n")
        
    else:
        print("❌ No se pudo completar el análisis.")

if __name__ == "__main__":
    run_bot()