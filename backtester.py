# backtester.py (ACTUALIZADO con Umbrales de Confianza Ajustados)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prediction_model 
from data_fetcher import config 

# --- CONFIGURACIÓN DE BACKTESTING ---
INITIAL_CAPITAL = 1000.0  
COMMISSION_FEE = 0.001   
# --- PARÁMETROS DE RIESGO ---
STOP_LOSS_PCT = 0.05    
TAKE_PROFIT_PCT = 0.10  
# --- PARÁMETROS DE CONFIANZA ---
BUY_THRESHOLD = 0.70    # Aumentado de 0.60 a 0.70
SELL_THRESHOLD = 0.30   # Disminuido de 0.40 a 0.30
# ------------------------------------

def run_backtest(data_for_ml):
    """
    Simula operaciones de compra y venta basándose en las predicciones del modelo.
    """
    model = prediction_model.model 
    feature_cols_for_prediction = prediction_model.final_feature_cols 
    
    if model is None: 
        print("❌ ERROR: El modelo de predicción no está entrenado o no se cargó.")
        return

    print(f"\n--- 📈 INICIANDO BACKTESTING (Umbrales: COMPRA > {BUY_THRESHOLD*100}%, VENTA < {SELL_THRESHOLD*100}%) ---")
    
    df = data_for_ml.copy()
    
    if df.shape[0] < 100:
        print(f"⚠️ Datos insuficientes para Backtesting: {df.shape[0]} filas.")
        return

    # 2. Predicción en todas las velas
    X = df[feature_cols_for_prediction] 
    probabilities = model.predict_proba(X) 
    df['Prob_Up'] = probabilities[:, 1]
    
    # 3. Generación de Señales de Compra/Venta
    # Señal de Compra (1) si Prob_Up > 0.70 (BUY_THRESHOLD)
    df['Signal'] = np.where(df['Prob_Up'] > BUY_THRESHOLD, 1, 0)
    
    # Señal de Venta (-1) si Prob_Up < 0.30 (SELL_THRESHOLD)
    df['Signal'] = np.where(df['Prob_Up'] < SELL_THRESHOLD, -1, df['Signal'])

    # 4. SIMULACIÓN DE TRADING
    
    capital = INITIAL_CAPITAL
    position = 0          
    df['Capital'] = INITIAL_CAPITAL
    shares_bought = 0     
    entry_price = 0       
    
    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i-1]
        close_price = df['close'].iloc[i] 
        exit_operation = None 

        # *** LÓGICA DE GESTIÓN DE RIESGO ***
        if position == 1:
            profit_pct = (close_price / entry_price) - 1.0

            if profit_pct <= -STOP_LOSS_PCT: 
                exit_operation = 'SL'
            elif profit_pct >= TAKE_PROFIT_PCT: 
                exit_operation = 'TP'
            elif signal == -1 or signal == 0: 
                exit_operation = 'AI'

        # --- Lógica de COMPRA ---
        if signal == 1 and position == 0:
            entry_price = close_price
            shares_bought = (capital * (1 - COMMISSION_FEE)) / entry_price
            position = 1

        # --- Lógica de VENTA ---
        if exit_operation:
            exit_price = close_price
            
            gross_profit = shares_bought * exit_price
            net_capital = gross_profit * (1 - COMMISSION_FEE)
            
            capital = net_capital
            position = 0
            shares_bought = 0
            entry_price = 0 

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
    ax1.set_xlabel(f'Fecha/Hora ({config.TIMEFRAME})')
    ax1.set_ylabel('Capital ($)', color=color)
    ax1.plot(df.index, df['Capital'], color=color, label='Curva de Capital (Estrategia)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Precio de Cierre', color=color) 
    ax2.plot(df.index, df['close'], color=color, alpha=0.3, label='Precio de Cierre')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.title(f'Backtesting de Rendimiento ({config.SYMBOL})')
    plt.grid(True)
    plt.show()