# config.py

# Configuración del Exchange
EXCHANGE_ID = 'binance'  # Puedes cambiar a 'kraken', 'coinbase', etc.

# Configuración de Monedas
SYMBOL = 'LINK/USDT'      # Par a analizar
TIMEFRAME = '1h'         # Temporalidad: 1m, 5m, 1h, 4h, 1d
LIMIT = 1000              # Cuántas velas analizar hacia atrás

# (Opcional) Claves API - Déjalas vacías para Fase 1 (Solo lectura pública)
API_KEY = ''
API_SECRET = ''