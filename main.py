# main.py (Versión con Datos Reales en Fases 1 y 2)
import data_fetcher
import technical_analysis
import sentiment_analyzer
import prediction_model
import social_media_fetcher # NUEVO IMPORT
import time
import pandas as pd

def run_bot():
    print("--- 🤖 INICIANDO CRYPTO ANALYST BOT (VERSIÓN DATOS REALES) ---")
    
    # 1. Obtener Datos Técnicos (REAL)
    df = data_fetcher.fetch_market_data()
    
    if df is not None:
        
        # 2. Obtener Datos Sociales (REAL)
        real_headlines = social_media_fetcher.fetch_recent_headlines(symbol=data_fetcher.config.SYMBOL)
        
        # 3. Análisis Técnico
        df_analyzed = technical_analysis.analyze_data(df)
        
        # 4. Análisis de Sentimiento (USANDO DATOS REALES)
        sentiment_data = sentiment_analyzer.analyze_crypto_narrative(real_headlines)
        
        # 5. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO (Aprende de datos históricos REALES)
        data_for_ml = prediction_model.prepare_data_for_training(df_analyzed, sentiment_data)
        accuracy = prediction_model.train_or_update_model(data_for_ml)

        # 6. PREDICCIÓN EN TIEMPO REAL
        current_features = prediction_model.get_current_features(df_analyzed, sentiment_data)
        prob_up, prediction_text = prediction_model.predict_next_move(current_features)
        
        # 7. Reporte Final
        price = df_analyzed.iloc[-1]['close'] if not df_analyzed.empty else 0
        rsi = df_analyzed.iloc[-1]['RSI'] if not df_analyzed.empty else 0
        
        print("\n" + "="*50)
        print("📢 REPORTE PROFESIONAL DE PREDICCIÓN (IA) - 100% REAL")
        print("="*50)
        
        # Resumen Técnico
        print(f"📈 Símbolo Analizado: {data_fetcher.config.SYMBOL}")
        print(f"💰 Precio Actual: ${price:,.2f}")
        print(f"RSI (14) Actual: {round(rsi, 2)}")

        # Resumen Sentimiento
        print(f"\n📰 SENTIMIENTO (Polaridad Promedio de Noticias): {round(sentiment_data[0] * 100, 2)}%")
        
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