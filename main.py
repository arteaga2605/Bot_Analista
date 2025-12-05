# main.py (CÓDIGO COMPLETO MODIFICADO)
import data_fetcher
import technical_analysis
import social_media_fetcher
import prediction_model
import backtester
import time 

# --- CONFIGURACIÓN DE AUTOMATIZACIÓN ---
# Tiempo de espera en segundos entre cada chequeo de mercado (1 hora = 3600 segundos)
CHECK_INTERVAL_SECONDS = 3600 

def run_bot(mode='live'):
    """
    Función principal para correr el bot en modo backtest o live.
    """
    
    # 1. OBTENER DATOS DE NOTICIAS
    sentiment_data = [0.5] 
    print("📡 Obteniendo titulares de noticias recientes (vía API)...")
    try:
        real_headlines = social_media_fetcher.fetch_recent_headlines(symbol=data_fetcher.config['SYMBOL'])
        if real_headlines:
            print(f"✅ {len(real_headlines)} titulares obtenidos de fuente real.")
            print(f"📰 Analizando sentimiento de {len(real_headlines)} textos...")
            sentiment_data = social_media_fetcher.analyze_sentiment(real_headlines)
        else:
            print("❌ No se pudieron obtener titulares, usando sentimiento neutral (0.5).")
    except Exception as e:
        print(f"❌ Error crítico al obtener o analizar noticias: {e}. Usando sentimiento neutral (0.5).")
        
    # 2. OBTENER DATOS DE PRECIO
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
        
        # 5b. PREDECIR EL SIGUIENTE MOVIMIENTO
        latest_features = prediction_model.get_current_features(data_analyzed, sentiment_data)
        
        # Para el modo live, re-entrenamos con la nueva data antes de predecir
        print("\n--- 🧠 ACTUALIZANDO EL MODELO CON LA ÚLTIMA DATA ---")
        prediction_model.train_or_update_model(data_for_ml)

        prob_up, prediction_text = prediction_model.predict_next_move(latest_features)
        
        # 6b. CALCULAR OBJETIVOS DE RIESGO
        last_row_analyzed = data_analyzed.iloc[-1]
        targets = backtester.calculate_sl_tp_targets(last_row_analyzed)
        
        print(f"\n--- 🤖 PREDICCIÓN DEL BOT ({data_fetcher.config['SYMBOL']}) ---")
        print(f"Precio Actual: ${targets['current_price']:.4f} (Volatilidad ATR: {targets['ATR']:.4f})")
        print(f"Probabilidad de Subida (Target=1): {prob_up*100:.2f}%")
        
        # 7b. GENERAR SEÑAL DE TRADING (Criterio de Compra/Venta)
        if prob_up > backtester.BUY_THRESHOLD:
            print(f"✅ SEÑAL DE COMPRA FUERTE (Confianza > {backtester.BUY_THRESHOLD*100}%)")
            print(f"💰 ACCIÓN RECOMENDADA: ¡COMPRAR!")
            print(f"   Objetivo de Ganancia (TP): ${targets['TP_Buy']:.4f}")
            print(f"   Límite de Pérdida (SL): ${targets['SL_Buy']:.4f}")
        elif prob_up < backtester.SELL_THRESHOLD:
            print(f"🔻 SEÑAL DE VENTA FUERTE (Confianza < {backtester.SELL_THRESHOLD*100}%)")
            print(f"❌ ACCIÓN RECOMENDADA: ¡VENDER/CERRAR POSICIÓN!")
            print(f"   Objetivo de Ganancia (TP): ${targets['TP_Sell']:.4f}")
            print(f"   Límite de Pérdida (SL): ${targets['SL_Sell']:.4f}")
        else:
            print(f"💤 SEÑAL NEUTRAL ({prediction_text})")
            print("⚖️ ACCIÓN RECOMENDADA: ¡ESPERAR!")

# 8. INICIO DEL BOT
if __name__ == "__main__":
    # >>> CAMBIO CLAVE: Iniciamos el bucle de operación en modo 'live'
    print("\n--- 🚀 MODO DE OPERACIÓN EN VIVO INICIADO ---")
    while True:
        try:
            run_bot(mode='live')
            
            # Pausa para esperar la próxima vela (1 hora)
            print(f"\n--- ⏸️ Esperando {CHECK_INTERVAL_SECONDS/60} minutos para la próxima vela ({time.ctime()}) ---")
            time.sleep(CHECK_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            print("\n👋 Ejecución detenida por el usuario.")
            break
        except Exception as e:
            print(f"\n❌ Ocurrió un error en el ciclo principal: {e}")
            print(f"Reintentando en 60 segundos...")
            time.sleep(60)