# main.py (Versión con Backtesting y Corrección de Límite)
import data_fetcher
import technical_analysis
import sentiment_analyzer
import prediction_model
import social_media_fetcher
import backtester 
import time
import pandas as pd

def run_bot(mode='live'):
    
    # --- 1. PREPARACIÓN DE DATOS ---
    if mode == 'live':
        print("--- 🤖 INICIANDO CRYPTO ANALYST BOT (FASE LIVE) ---")
        limit_data = data_fetcher.config.LIMIT 
    else: 
        print("--- 🔬 INICIANDO BACKTEST (OBTENIENDO 2000 VELAS) ---")
        limit_data = 2000
    
    # 1.1 Obtener Sentimiento Real
    real_headlines = social_media_fetcher.fetch_recent_headlines(symbol=data_fetcher.config.SYMBOL)
    sentiment_data = sentiment_analyzer.analyze_crypto_narrative(real_headlines)
    
    # 1.2 Obtener Datos Técnicos
    df = data_fetcher.fetch_market_data(limit=limit_data) 
    
    if df is not None:
        
        # 2. Análisis Técnico
        df_analyzed = technical_analysis.analyze_data(df)
        
        # 3. CREACIÓN DEL DATAFRAME DE ML (ESTE CONTIENE LA COLUMNA POLARIDAD)
        data_for_ml = prediction_model.prepare_data_for_training(df_analyzed, sentiment_data)
        
        # 4. ENTRENAMIENTO y Persistencia
        accuracy = prediction_model.train_or_update_model(data_for_ml)

        # 5. EJECUCIÓN DEL BACKTEST (PASAMOS data_for_ml, NO df_analyzed)
        if mode == 'backtest':
            backtester.run_backtest(data_for_ml) # <-- CORRECCIÓN
            return

        # 6. PREDICCIÓN EN VIVO
        current_features = prediction_model.get_current_features(df_analyzed, sentiment_data)
        prob_up, prediction_text = prediction_model.predict_next_move(current_features)
        
        # 7. Reporte Final
        price = df_analyzed.iloc[-1]['close'] if not df_analyzed.empty else 0
        rsi = df_analyzed.iloc[-1]['RSI'] if not df_analyzed.empty else 0
        
        print("\n" + "="*50)
        print("📢 REPORTE PROFESIONAL DE PREDICCIÓN (IA)")
        print("="*50)
        
        print(f"📈 Símbolo Analizado: {data_fetcher.config.SYMBOL}")
        print(f"💰 Precio Actual: ${price:,.2f}")
        print(f"RSI (14) Actual: {round(rsi, 2)}")

        print(f"\n📰 SENTIMIENTO (Polaridad Promedio de Noticias): {round(sentiment_data[0] * 100, 2)}%")
        
        print("\n🧠 PREDICCIÓN DEL MODELO ML:")
        print(f"  > Probabilidad de Subida: {round(prob_up * 100, 2)}%")
        print(f"  > Probabilidad de Bajada: {round((1 - prob_up) * 100, 2)}%")
        print(f"  > **VEREDICTO IA: {prediction_text}**")
        print(f"\n[Precisión simulada: {round(accuracy * 100, 2)}%]")
        print("="*50 + "\n")
        
    else:
        print("❌ No se pudo completar el análisis.")

if __name__ == "__main__":
    run_bot(mode='backtest')