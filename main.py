# main.py (CÓDIGO COMPLETO CORREGIDO)
import data_fetcher
import technical_analysis
import social_media_fetcher
import prediction_model
import backtester
import time 
import sys 
import pandas as pd 

CHECK_INTERVAL_SECONDS = 14400 

def run_bot(mode='live'):
    sentiment_data = [0.5] 
    print("📡 Obteniendo titulares de noticias recientes (vía API)...")
    try:
        symbol = data_fetcher.config.get('SYMBOL', 'LINKUSDT')
        real_headlines = social_media_fetcher.fetch_recent_headlines(symbol=symbol)
        if real_headlines:
            print(f"✅ {len(real_headlines)} titulares obtenidos de fuente real.")
            print(f"📰 Analizando sentimiento de {len(real_headlines)} textos...")
            sentiment_data = social_media_fetcher.analyze_sentiment(real_headlines)
        else:
            print("❌ No se pudieron obtener titulares, usando sentimiento neutral (0.5).")
    except Exception as e:
        print(f"❌ Error crítico al obtener o analizar noticias: {e}. Usando sentimiento neutral (0.5).")
        
    print(f"🔄 Conectando a binance para obtener datos de {data_fetcher.config['SYMBOL']}...")
    if mode == 'backtest':
        df_raw = data_fetcher.get_historical_data(limit=3000)
        print(f"✅ Datos obtenidos exitosamente: {df_raw.shape[0]} velas.")
    else:
        df_raw = data_fetcher.get_historical_data(limit=1000)
        print(f"✅ Datos obtenidos exitosamente: {df_raw.shape[0]} velas.")

    if df_raw is None or df_raw.empty:
        print("❌ No se pudieron obtener datos históricos. Deteniendo el bot.")
        return

    data_with_ta = technical_analysis.analyze_data(df_raw)
    print("--- 🧠 PREPARANDO DATA PARA MACHINE LEARNING ---")
    data_for_ml = prediction_model.prepare_data_for_training(data_with_ta, sentiment_data) 
    
    if mode == 'live':
        print("\n--- 🧠 ACTUALIZANDO EL MODELO CON LA ÚLTIMA DATA ---")
        prediction_model.train_or_update_model(data_for_ml)
    elif mode == 'backtest':
        print("\n--- 🧠 ENTRENANDO EL MODELO ---")
        prediction_model.train_or_update_model(data_for_ml)
    
    if mode == 'live':
        last_row_for_prediction_df = prediction_model.get_current_features(data_with_ta, sentiment_data)
        
        if last_row_for_prediction_df.empty:
            print("❌ No hay datos suficientes para generar features de predicción en vivo. Saltando predicción.")
            return
            
        prob_up, prediction_text = prediction_model.predict_next_move(last_row_for_prediction_df)
        last_row = last_row_for_prediction_df.iloc[-1]  # ✅ Serie de 1 fila
        print(f"[DEBUG] last_row type: {type(last_row)}, close: {last_row['close']}, ATR: {last_row['ATR']}")
        
        targets = backtester.calculate_sl_tp_targets(last_row)

        print("\n--- 📣 GENERANDO SEÑAL DE TRADING EN VIVO ---")
        print(f"✅ Probabilidad de SUBIDA: {prob_up*100:.2f}% ({prediction_text})")
        print(f"📉 Volatilidad (ATR): {targets['ATR']:.4f}")

        if prob_up > backtester.BUY_THRESHOLD:
            print(f"📈 SEÑAL DE COMPRA FUERTE (Confianza > {backtester.BUY_THRESHOLD*100}%)")
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

    elif mode == 'backtest':
        backtester.run_backtest(data_for_ml)
        
if __name__ == "__main__":
    prediction_model.load_model()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'live'
    
    if mode == 'backtest':
        run_bot(mode='backtest')
        sys.exit()

    print("\n--- 🚀 MODO DE OPERACIÓN EN VIVO INICIADO ---")
    while True:
        try:
            run_bot(mode='live')
            print(f"\n--- ⏸️ Esperando {CHECK_INTERVAL_SECONDS/60} minutos para la próxima ejecución... ---")
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n⏹️ Detención manual por el usuario (Ctrl + C).")
            print("🧹 Cerrando bot de forma segura...")
            sys.exit(0)   # Salida limpia
        except Exception as e:
            print(f"\n❌ Ocurrió un error en el ciclo principal: {e}")
            print("Reintentando en 60 segundos...")
            time.sleep(60)