# backtester.py (CÓDIGO COMPLETO FINAL Y CORREGIDO)
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

def calculate_sl_tp_targets(last_data):
    """
    Calcula los niveles de Stop Loss y Take Profit basados en el ATR.
    Recibe una Serie de pandas (last_row) o dict con 'close' y 'ATR'.
    """
    try:
        if isinstance(last_data, pd.DataFrame):
            last_close = float(last_data['close'].values[0])
            atr_raw = last_data['ATR'].values[0]
            last_atr = float(atr_raw) if pd.notna(atr_raw) else 0.0
        elif isinstance(last_data, pd.Series):
            last_close = float(last_data['close'])
            atr_raw = last_data['ATR']
            if isinstance(atr_raw, pd.Series):
                atr_raw = atr_raw.iloc[0]
            last_atr = float(atr_raw) if pd.notna(atr_raw) else 0.0
        else:
            last_close = float(last_data['close'])
            atr_raw = last_data['ATR']
            if isinstance(atr_raw, pd.Series):
                atr_raw = atr_raw.iloc[0]
            last_atr = float(atr_raw) if pd.notna(atr_raw) else 0.0
    except Exception as e:
        print(f"❌ Error al extraer precio/ATR: {e}. Usando 0.0.")
        return {
            'current_price': 0.0,
            'ATR': 0.0,
            'SL_Buy': 0.0,
            'TP_Buy': 0.0,
            'SL_Sell': 0.0,
            'TP_Sell': 0.0
        }

    if last_atr <= 0:
        return {
            'current_price': last_close,
            'ATR': round(last_atr, 4),
            'SL_Buy': last_close,
            'TP_Buy': last_close,
            'SL_Sell': last_close,
            'TP_Sell': last_close
        }

    sl_distance = last_atr * SL_MULTIPLIER
    tp_distance = last_atr * TP_MULTIPLIER

    sl_buy = last_close - sl_distance
    tp_buy = last_close + tp_distance
    sl_sell = last_close + sl_distance
    tp_sell = last_close - tp_distance

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

    risk_reward_ratio = TP_MULTIPLIER / SL_MULTIPLIER
    print(f"\n--- 📈 INICIANDO BACKTESTING con GESTIÓN DE RIESGO DINÁMICA (SL={SL_MULTIPLIER}x ATR, TP={TP_MULTIPLIER}x ATR, R:R={risk_reward_ratio:.1f}:1) ---")
    
    df = data_for_ml.copy()
    
    if df.shape[0] < 2: 
        print(f"⚠️ Datos insuficientes para Backtesting: {df.shape[0]} filas. No se puede simular el trading.")
        print(f"--- ✅ RESULTADOS DEL BACKTEST ---")
        print(f"💰 Capital Inicial: ${INITIAL_CAPITAL:,.2f}")
        print(f"💵 Capital Final: ${INITIAL_CAPITAL:,.2f}")
        print(f"📈 Ganancia Neta (%): 0.00%")
        print(f"📉 Max Drawdown: 0.00%")
        print("-" * 30)
        print("Compra y Mantén (Buy & Hold) Ganancia: No calculada por falta de datos.")
        return 
        
    X = df[feature_cols_for_prediction]
    try:
        if prediction_model.scaler is None:
            print("❌ ERROR: Scaler no encontrado. No se puede predecir en Backtest.")
            return
        X_scaled = prediction_model.scaler.transform(X)
        probabilities = prediction_model.model.predict_proba(X_scaled)
    except Exception as e:
        print(f"❌ Error al escalar o predecir en Backtest: {e}")
        return 

    df['Prob_Up'] = probabilities[:, 1]
    df['Prob_Down'] = probabilities[:, 0]
    df['Signal'] = np.where(df['Prob_Up'] > BUY_THRESHOLD, 1, 
                            np.where(df['Prob_Up'] < SELL_THRESHOLD, -1, 0))

    capital = INITIAL_CAPITAL
    position = 0
    shares_bought = 0
    entry_price = 0
    sl_target = 0
    tp_target = 0
    df['Capital'] = INITIAL_CAPITAL
    df['Trade_Type'] = 'HOLD'
    trade_results = []

    for i in range(1, df.shape[0]):
        current_candle_index = df.index[i]
        close_price = df.loc[current_candle_index, 'close']
        high_price = df.loc[current_candle_index, 'high']
        low_price = df.loc[current_candle_index, 'low']
        signal = df.loc[current_candle_index, 'Signal']

        prev_atr = df.loc[df.index[i-1], 'ATR']
        if isinstance(prev_atr, pd.Series):
            prev_atr = prev_atr.iloc[0]

        df.loc[current_candle_index, 'Capital'] = capital
        df.loc[current_candle_index, 'Trade_Type'] = 'HOLD'
        exit_operation = None
        exit_price = 0

        if position != 0:
            if position == 1:
                if low_price <= sl_target:
                    exit_operation = 'SL'
                    exit_price = sl_target
                elif high_price >= tp_target:
                    exit_operation = 'TP'
                    exit_price = tp_target
                elif signal == -1:
                    exit_operation = 'AI'
                    exit_price = close_price
            elif position == -1:
                if high_price >= sl_target:
                    exit_operation = 'SL'
                    exit_price = sl_target
                elif low_price <= tp_target:
                    exit_operation = 'TP'
                    exit_price = tp_target
                elif signal == 1:
                    exit_operation = 'AI'
                    exit_price = close_price

        if exit_operation:
            if position == 1:
                gross_value = shares_bought * exit_price
                net_capital = gross_value * (1 - COMMISSION_FEE)
                capital = net_capital
            elif position == -1:
                price_diff = entry_price - exit_price
                gross_profit = price_diff * shares_bought
                net_capital = capital + (gross_profit * (1 - COMMISSION_FEE))
                capital = net_capital

            trade_results.append({
                'net_profit': capital - df.loc[df.index[i-1], 'Capital'],
                'gross_change': gross_value - (shares_bought * entry_price) if position == 1 else gross_profit,
                'exit_type': exit_operation
            })

            df.loc[current_candle_index, 'Trade_Type'] = f'CLOSE_{exit_operation}'
            df.loc[current_candle_index, 'Capital'] = capital

            position = 0
            shares_bought = 0
            entry_price = 0
            sl_target = 0
            tp_target = 0

        if position == 0:
            if signal == 1:
                entry_price = close_price
                sl_distance = prev_atr * SL_MULTIPLIER
                tp_distance = prev_atr * TP_MULTIPLIER
                sl_target = entry_price - sl_distance
                tp_target = entry_price + tp_distance
                shares_bought = (capital * (1 - COMMISSION_FEE)) / entry_price
                position = 1
                df.loc[current_candle_index, 'Trade_Type'] = 'BUY_OPEN'
            elif signal == -1:
                entry_price = close_price
                sl_distance = prev_atr * SL_MULTIPLIER
                tp_distance = prev_atr * TP_MULTIPLIER
                sl_target = entry_price + sl_distance
                tp_target = entry_price - tp_distance
                shares_bought = (capital * (1 - COMMISSION_FEE)) / entry_price
                position = -1
                df.loc[current_candle_index, 'Trade_Type'] = 'SELL_OPEN'

        if position != 0:
            if position == 1:
                df.loc[current_candle_index, 'Capital'] = shares_bought * close_price
            elif position == -1:
                current_profit_loss = (entry_price - close_price) * shares_bought
                df.loc[current_candle_index, 'Capital'] = capital + current_profit_loss

    if position != 0:
        last_index = df.index[-1]
        final_close = df.loc[last_index, 'close']
        if isinstance(final_close, pd.Series):
            final_close = final_close.iloc[0]

        if position == 1:
            gross_value = shares_bought * final_close
            net_capital = gross_value * (1 - COMMISSION_FEE)
            capital = net_capital
        elif position == -1:
            price_diff = entry_price - final_close
            gross_profit = price_diff * shares_bought
            net_capital = capital + (gross_profit * (1 - COMMISSION_FEE))
            capital = net_capital

        trade_results.append({
            'net_profit': capital - df.loc[df.index[-2], 'Capital'] if len(df.index) > 1 else capital - INITIAL_CAPITAL,
            'gross_change': gross_value - (shares_bought * entry_price) if position == 1 else gross_profit,
            'exit_type': 'FINAL_CLOSE'
        })
        df.loc[last_index, 'Capital'] = capital
        df.loc[last_index, 'Trade_Type'] = 'FINAL_CLOSE'

    final_capital = df['Capital'].iloc[-1]
    net_profit_pct = (final_capital / INITIAL_CAPITAL - 1) * 100
    df['Peak'] = df['Capital'].cummax()
    df['Drawdown'] = (df['Peak'] - df['Capital']) / df['Peak']
    max_drawdown = df['Drawdown'].max() * 100

    total_trades = len(trade_results)
    if total_trades > 0:
        df_results = pd.DataFrame(trade_results)
        winning_trades = len(df_results[df_results['net_profit'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100
        total_gross_profit = df_results[df_results['gross_change'] > 0]['gross_change'].sum()
        total_gross_loss = df_results[df_results['gross_change'] < 0]['gross_change'].sum()
        profit_factor = total_gross_profit / abs(total_gross_loss) if total_gross_loss != 0 else np.inf
        avg_profit_per_trade = df_results['net_profit'].mean()
    else:
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_profit_per_trade = 0.0

    buy_hold_shares = INITIAL_CAPITAL / df['close'].iloc[0]
    buy_hold_profit = (buy_hold_shares * df['close'].iloc[-1]) - INITIAL_CAPITAL

    print("\n--- ✅ RESULTADOS DEL BACKTEST ---")
    print(f"💰 Capital Inicial: ${INITIAL_CAPITAL:,.2f}")
    print(f"💵 Capital Final: ${final_capital:,.2f}")
    print(f"📈 Ganancia Neta (%): {net_profit_pct:.2f}%")
    print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
    print("-" * 30)
    print(f"📊 Total de Trades: {total_trades}")
    print(f"✅ Trades Ganadores: {winning_trades} ({win_rate:.2f}%)")
    print(f"❌ Trades Perdedores: {losing_trades} ({(losing_trades/total_trades*100):.2f}%)" if total_trades > 0 else "❌ Trades Perdedores: 0 (0.00%)")
    print(f"⭐ Factor de Beneficio: {profit_factor:.2f} (Debe ser > 1.0)")
    print(f"💵 Ganancia Promedio por Trade: ${avg_profit_per_trade:.4f}")
    print("-" * 30)
    print(f"Compra y Mantén (Buy & Hold) Ganancia: {buy_hold_profit / INITIAL_CAPITAL * 100:.2f}%\n")

    plot_backtest_results(df)


def plot_backtest_results(df):
    """
    Genera un gráfico de la curva de capital y el precio.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    timeframe = config.get('TIMEFRAME', 'N/A')
    ax1.set_xlabel(f'Fecha/Hora ({timeframe})')
    ax1.set_ylabel('Capital ($)', color=color)
    ax1.plot(df.index, df['Capital'], color=color, label='Curva de Capital (Estrategia)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Precio de Cierre', color=color) 
    ax2.plot(df.index, df['close'], color=color, alpha=0.3, label='Precio de Cierre')
    ax2.tick_params(axis='y', labelcolor=color)

    buy_entries = df[df['Trade_Type'] == 'BUY_OPEN']
    sell_entries = df[df['Trade_Type'] == 'SELL_OPEN']
    
    ax2.plot(buy_entries.index, buy_entries['close'], '^', markersize=5, color='green', label='Entrada Larga')
    ax2.plot(sell_entries.index, sell_entries['close'], 'v', markersize=5, color='red', label='Entrada Corta')
    
    close_signals = df[df['Trade_Type'].str.startswith('CLOSE') | (df['Trade_Type'] == 'FINAL_CLOSE')]
    ax2.plot(close_signals.index, close_signals['close'], 'o', markersize=4, color='black', fillstyle='none', label='Cierre')
    
    fig.tight_layout()
    plt.title(f'Backtest de Estrategia de Trading para {config["SYMBOL"]}')
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()