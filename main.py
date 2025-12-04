# main.py (FINAL CORREGIDO)
import data_fetcher
import technical_analysis
import social_media_fetcher
import prediction_model
import backtester
import time 

def run_bot(mode='live'):
    """
    Función principal para correr el bot en modo backtest o live.
    """
    
    # 1. OBTENER DATOS DE NOTICIAS
    sentiment_data = [0.5] 
    print("📡 Obteniendo titulares de noticias recientes (vía API)...")
    try:
        # >>> CORRECCIÓN 1: Usar corchetes para acceder al diccionario
        real_headlines = social_media_fetcher.fetch_recent_headlines(symbol=data_fetcher.config['SYMBOL'])
        if real_headlines:
            print(f"✅ {len(real_headlines)} titulares obtenidos de fuente real.")
            print(f"📰 Analizando sentimiento de {len(real_headlines)} textos...")
            sentiment_data = social_media_fetcher.analyze_sentiment(real_headlines)
        else:
            print("❌ No se pudieron obtener titulares, usando sentimiento neutral (0.5).")
    except Exception as e:
        # Si ocurre un error aquí, puede ser la sintaxis o un problema en el fetcher.
        # Imprimimos el error, pero seguimos usando el sentimiento neutral.
        # Si el error es el AttributeError, lo vemos inmediatamente al inicio.
        print(f"❌ Error crítico al obtener o analizar noticias: {e}. Usando sentimiento neutral (0.5).")
        
    # 2. OBTENER DATOS DE PRECIO
    # >>> CORRECCIÓN 2: Usar corchetes para acceder al diccionario
    print(f"🔄 Conectando a binance para obtener datos de {data_fetcher.config['SYMBOL']}...")
    data_df = data_fetcher.fetch_data()
    
    if data_df is None or data_df.empty:
        print("❌ Terminando la ejecución: No se pudieron obtener datos de precio.")
        return

    print(f"✅ Datos obtenidos exitosamente: {len(data_df)} velas.")

    # 3. ANÁLISIS TÉCNICO
    print("📊 Calculando indicadores técnicos con la librería 'ta'...")
    data_analyzed = technical_analysis.analyze_data(data_df)
    
    if data_analyzed is None or data_analyzed.empty:
        print("❌ Terminando la ejecución: El análisis técnico no produjo datos válidos.")
        return

    # 4. PREPARACIÓN DE DATOS PARA ML
    data_for_ml = prediction_model.prepare_data_for_training(data_analyzed, sentiment_data)

    if mode == 'backtest':
        print(f"--- 🔬 INICIANDO BACKTEST (OBTENIENDO {data_fetcher.config.get('LIMIT', 'N/A')} VELAS) ---")
        
        # 5a. ENTRENAMIENTO/ACTUALIZACIÓN del MODELO
        print("\n--- 🧠 ENTRENANDO EL MODELO ---")
        prediction_model.train_or_update_model(data_for_ml)
        
        # 6a. EJECUTAR BACKTESTING
        backtester.run_backtest(data_for_ml) 
        
    elif mode == 'live':
        print("\n--- 🚀 MODO DE OPERACIÓN EN VIVO INICIADO ---")
        
        # 5b. PREDECIR EL SIGUIENTE MOVIMIENTO
        latest_features = prediction_model.get_current_features(data_analyzed, sentiment_data)
        prob_up, prediction_text = prediction_model.predict_next_move(latest_features)
        
        print(f"\n--- 🤖 PREDICCIÓN DEL BOT ---")
        print(f"Probabilidad de Subida (Target=1): {prob_up*100:.2f}%")
        print(f"Decisión: {prediction_text}")
        
        if prob_up > backtester.BUY_THRESHOLD:
            print("🚀 SEÑAL DE COMPRA FUERTE GENERADA (Acción: ¡COMPRAR!)")
        elif prob_up < backtester.SELL_THRESHOLD:
            print("📉 SEÑAL DE VENTA FUERTE GENERADA (Acción: ¡VENDER!)")
        else:
            print("💤 SEÑAL NEUTRAL (Acción: ¡ESPERAR!)")

# 7. INICIO DEL BOT
if __name__ == "__main__":
    run_bot(mode='backtest')