# main.py
import data_fetcher
import technical_analysis
import time

def run_bot():
    print("--- 🤖 INICIANDO CRYPTO ANALYST BOT (FASE 1) ---")
    
    # 1. Obtener Datos
    df = data_fetcher.fetch_market_data()
    
    if df is not None:
        # 2. Analizar Datos
        df_analyzed = technical_analysis.analyze_data(df)
        
        # 3. Mostrar Resultados (Últimos 3 registros)
        print("\n🔎 Últimos 3 registros analizados:")
        print(df_analyzed[['timestamp', 'close', 'RSI', 'SMA_50']].tail(3))
        
        # 4. Interpretación Básica
        signal, rsi, price = technical_analysis.generate_signal(df_analyzed)
        
        print("\n" + "="*40)
        print(f"📢 REPORTE FINAL:")
        print(f"💰 Precio Actual: {price}")
        print(f"📈 RSI Actual: {round(rsi, 2)}")
        print(f"🤖 Veredicto del Bot: {signal}")
        print("="*40 + "\n")
        
    else:
        print("❌ No se pudo completar el análisis.")

if __name__ == "__main__":
    # Ejecutar una vez
    run_bot()
    
    # Opcional: Ejecutar en bucle cada hora
    # while True:
    #     run_bot()
    #     time.sleep(3600)