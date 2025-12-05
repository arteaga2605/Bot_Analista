# backtester.py (CÓDIGO COMPLETO FINAL)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prediction_model 
from data_fetcher import config 

# --- CONFIGURACIÓN DE BACKTESTING ---
INITIAL_CAPITAL = 10.0  
COMMISSION_FEE = 0.001   
# --- PARÁMETROS DE RIESGO (LIVE TRADING - BASADO EN ATR) ---
SL_MULTIPLIER = 1.5   # Stop Loss a 1.5 veces el ATR
TP_MULTIPLIER = 3.0   # Take Profit a 3.0 veces el ATR (Relación 1:2 riesgo/recompensa)
# --- PARÁMETROS DE CONFIANZA ---
BUY_THRESHOLD = 0.70    
SELL_THRESHOLD = 0.30   
# ------------------------------------

def calculate_sl_tp_targets(last_data_row):
    """
    Calcula los niveles de Stop Loss y Take Profit basados en el ATR.
    Recibe la última fila del DataFrame analizado.
    """
    last_close = last_data_row['close']
    last_atr = last_data_row['ATR']
    
    if last_atr <= 0:
        # Esto ocurre si ATR no pudo ser calculado
        return {
            'current_price': last_close,
            'ATR': 0.0,
            'SL_Buy': last_close,
            'TP_Buy': last_close,
            'SL_Sell': last_close,
            'TP_Sell': last_close
        }

    # Calcula la distancia de riesgo/recompensa en términos de precio
    sl_distance = last_atr * SL_MULTIPLIER
    tp_distance = last_atr * TP_MULTIPLIER
    
    # Valores de SL y TP para una COMPRA (Buy Signal)
    sl_buy = last_close - sl_distance
    tp_buy = last_close + tp_distance
    
    # Valores de SL y TP para una VENTA/CORTO (Sell/Short Signal)
    sl_sell = last_close + sl_distance
    tp_sell = last_close - tp_distance
    
    # Se redondea a 4 decimales para mayor limpieza
    return {
        'current_price': last_close,
        'ATR': round(last_atr, 4),
        'SL_Buy': round(sl_buy, 4),
        'TP_Buy': round(tp_buy, 4),
        'SL_Sell': round(sl_sell, 4),
        'TP_Sell': round(tp_sell, 4)
    }


def run_backtest(data_for_ml):
    """
    Simula operaciones de compra y venta basándose en las predicciones del modelo,
    utilizando SL/TP dinámicos basados en el ATR.
    """
    model = prediction_model.model 
    feature_cols_for_prediction = prediction_model.final_feature_cols 
    
    if model is None: 
        print("❌ ERROR: El modelo de predicción no está entrenado o no se cargó.")
        return

    # Usamos los multiplicadores ATR configurados globalmente
    risk_reward_ratio = TP_MULTIPLIER / SL_MULTIPLIER
    print(f"\n--- 📈 INICIANDO BACKTESTING con GESTIÓN DE RIESGO DINÁMICA (SL={SL_MULTIPLIER}x ATR, TP={TP_MULTIPLIER}x ATR, R:R={risk_reward_ratio:.1f}:1) ---")
    
    df = data_for_ml.copy()
    
    if df.shape[0] < 2: 
        print(f"⚠️ Datos insuficientes para Backtesting: {df.shape[0]} filas. No se puede simular el trading.")
        print(f"--- ✅ RESULTADOS DEL BACKTEST ---")
        print(f"💰 Capital Inicial: ${INITIAL_CAPITAL:,.2f}")
        print(f"💵 Capital Final:   ${INITIAL_CAPITAL:,.2f}")
        print(f"📈 Ganancia Neta (%): 0.00%")
        print(f"📉 Max Drawdown: 0.00%")
        print("-" * 30)
        print("Compra y Mantén (Buy & Hold) Ganancia: No calculada por falta de datos.")
        return

    # 2. Predicción en todas las velas
    X = df[feature_cols_for_prediction] 
    
    try:
        X_scaled = prediction_model.scaler.transform(X)
        probabilities = prediction_model.model.predict_proba(X_scaled) 
    except Exception as e:
        print(f"❌ Error al escalar o predecir en Backtest: {e}")
        return

    df['Prob_Up'] = probabilities[:, 1]
    
    # 3. Generación de Señales de Compra/Venta
    df['Signal'] = np.where(df['Prob_Up'] > BUY_THRESHOLD, 1, 0)
    df['Signal'] = np.where(df['Prob_Up'] < SELL_THRESHOLD, -1, df['Signal'])

    # 4. SIMULACIÓN DE TRADING
    
    capital = INITIAL_CAPITAL
    position = 0          
    df['Capital'] = INITIAL_CAPITAL
    shares_bought = 0     
    entry_price = 0       
    sl_target = 0
    tp_target = 0
    
    # Añadir columna para registrar eventos de trading (opcional, pero útil)
    df['Trade_Type'] = np.nan 

    for i in range(1, len(df)):
        # La señal y el ATR se basan en el conocimiento de la vela anterior (i-1)
        prev_row = df.iloc[i-1]
        signal = prev_row['Signal']
        prev_atr = prev_row['ATR'] 
        
        close_price = df['close'].iloc[i] 
        high_price = df['high'].iloc[i] 
        low_price = df['low'].iloc[i] 
        exit_operation = None 

        # *** LÓGICA DE GESTIÓN DE RIESGO Y CIERRE ***
        if position == 1: # Posición de COMPRA Abierta
            # Verificamos si se alcanzó el SL o TP dentro de la vela actual (i)
            # Nota: Usamos high/low de la vela actual para simular la ejecución de órdenes
            
            # 1. ¿SL alcanzado?
            if low_price <= sl_target: 
                exit_operation = 'SL'
                exit_price = sl_target
            # 2. ¿TP alcanzado? (Si el TP es alcanzado antes que el SL en la misma vela, priorizamos el TP)
            elif high_price >= tp_target: 
                exit_operation = 'TP'
                exit_price = tp_target
            
            # 3. ¿Cierre por señal inversa? (Si no hay SL/TP, vemos la señal de la vela anterior)
            elif signal == -1 or signal == 0: 
                exit_operation = 'AI'
                exit_price = close_price

        # --- Lógica de COMPRA ---
        if signal == 1 and position == 0:
            entry_price = close_price
            
            # Calculamos SL y TP dinámicamente con el ATR de la vela anterior
            sl_distance = prev_atr * SL_MULTIPLIER
            tp_distance = prev_atr * TP_MULTIPLIER
            sl_target = entry_price - sl_distance
            tp_target = entry_price + tp_distance

            shares_bought = (capital * (1 - COMMISSION_FEE)) / entry_price
            position = 1
            df.loc[df.index[i], 'Trade_Type'] = 'BUY_OPEN'

        # --- Lógica de VENTA/CIERRE ---
        if exit_operation:
            # Si se activó SL/TP, ya tenemos exit_price
            if exit_operation in ['SL', 'TP']:
                 # Usamos el precio objetivo (SL o TP) como precio de salida
                 pass
            elif exit_operation == 'AI':
                 # Si es por señal de la IA, usamos el precio de cierre de la vela actual
                 exit_price = close_price

            gross_profit = shares_bought * exit_price
            net_capital = gross_profit * (1 - COMMISSION_FEE)
            
            capital = net_capital
            position = 0
            shares_bought = 0
            entry_price = 0 
            sl_target = 0
            tp_target = 0
            df.loc[df.index[i], 'Trade_Type'] = 'SELL_CLOSE_' + exit_operation


        # --- Lógica de MANTENER Posición ---
        elif position == 1:
            current_value = shares_bought * close_price
            capital = current_value
        
        df.loc[df.index[i], 'Capital'] = capital

    # 5. CÁLCULO DE MÉTRICAS FINALES
    final_capital = df['Capital'].iloc[-1]
    net_profit = final_capital - INITIAL_CAPITAL
    
    # Cálculo del Máximo Retroceso (Max Drawdown)
    df['Peak'] = df['Capital'].cummax()
    df['Drawdown'] = (df['Peak'] - df['Capital']) / df['Peak']
    max_drawdown = df['Drawdown'].max()
    
    # Retorno sin riesgo (Buy & Hold)
    buy_hold_value = INITIAL_CAPITAL * (df['close'].iloc[-1] / df['close'].iloc[0])
    buy_hold_profit = buy_hold_value - INITIAL_CAPITAL
    
    # 6. IMPRIMIR RESULTADOS
    print("\n--- ✅ RESULTADOS DEL BACKTEST ---")
    print(f"💰 Capital Inicial: ${INITIAL_CAPITAL:,.2f}")
    print(f"💵 Capital Final:   ${final_capital:,.2f}")
    print(f"📈 Ganancia Neta (%): {net_profit / INITIAL_CAPITAL * 100:.2f}%")
    print(f"📉 Max Drawdown: {max_drawdown * 100:.2f}%")
    print("-" * 30)
    print(f"Compra y Mantén (Buy & Hold) Ganancia: {buy_hold_profit / INITIAL_CAPITAL * 100:.2f}%")
    
    # 7. VISUALIZACIÓN
    plot_backtest_results(df)


def plot_backtest_results(df):
    """
    Genera un gráfico de la curva de capital y el precio.
    """
    import matplotlib.pyplot as plt 
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    # >>> CORRECCIÓN: Usar corchetes para TIMEFRAME
    ax1.set_xlabel(f'Fecha/Hora ({config["TIMEFRAME"]})')
    ax1.set_ylabel('Capital ($)', color=color)
    ax1.plot(df.index, df['Capital'], color=color, label='Curva de Capital (Estrategia)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Precio de Cierre', color=color) 
    ax2.plot(df.index, df['close'], color=color, alpha=0.3, label='Precio de Cierre')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    # >>> CORRECCIÓN: Usar corchetes para SYMBOL
    plt.title(f'Backtesting de Rendimiento ({config["SYMBOL"]})')
    plt.grid(True)
    plt.show()