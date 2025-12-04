# 🤖 Crypto Analyst Bot (LINK/USDT)

Este bot de trading utiliza un modelo de Machine Learning (XGBoost) para predecir la dirección del precio de una criptomoneda (actualmente LINK/USDT) en el marco de tiempo de 1 hora. La predicción se basa en 11 features, que combinan análisis técnico (librería 'ta') y análisis de sentimiento de noticias (librería 'TextBlob').

## 🧠 Arquitectura y Funcionamiento del Sistema

El sistema opera en un ciclo de ejecución continuo, diseñado para actualizar su conocimiento y generar una señal de trading cada hora.

### 1. Componentes Principales

| Archivo | Función | Descripción |
| :--- | :--- | :--- |
| `main.py` | **Controlador Principal** | Inicia el bot, gestiona el bucle de ejecución, llama a todos los módulos y presenta la señal final. |
| `data_fetcher.py` | **Obtención de Datos** | Conexión a la API de Binance para obtener 1000 velas de precios históricos (OHLCV). Contiene la configuración de `SYMBOL` (`LINKUSDT`). |
| `technical_analysis.py` | **Análisis Técnico** | Calcula indicadores (RSI, MACD, etc.) y genera las 10 features técnicas usadas en el modelo. |
| `social_media_fetcher.py` | **Análisis de Sentimiento** | Simula la obtención de titulares de noticias y calcula la Polaridad de Sentimiento (una de las 11 features). |
| `prediction_model.py` | **ML Core (XGBoost)** | Contiene la lógica para la carga, entrenamiento, guardado (`.joblib`) y predicción del modelo XGBoost. También maneja la normalización de datos (`MinMaxScaler`). |
| `backtester.py` | **Simulación y Estrategia** | Define la lógica de trading (Umbrales de Confianza 70%/30%, Stop Loss, Take Profit, Capital Inicial) y ejecuta el backtesting histórico. |
| `crypto_model_xgb.joblib` | **Modelo Persistente** | El modelo XGBoost entrenado. |
| `crypto_scaler.joblib` | **Scaler Persistente** | El objeto `MinMaxScaler` necesario para normalizar los datos antes de la predicción. |

### 2. Ciclo de Operación en Modo Live (`main.py`)

1.  **Activación y Carga:** El bot se inicia en modo `live` (bucle infinito). Carga el modelo y el scaler guardados.
2.  **Obtención de Noticias:** Se obtienen 10 titulares recientes y se calcula el **Sentimiento Promedio Normalizado** (Polaridad_Sentimiento).
3.  **Obtención de Precio:** Se obtienen los **1000 datos históricos** de la criptomoneda configurada (LINK/USDT).
4.  **Feature Engineering:** Se calculan las 10 features técnicas sobre los 1000 datos.
5.  **Re-entrenamiento (Actualización):** El modelo XGBoost se actualiza con los 1000 datos para asegurar que incorpora la dinámica de mercado más reciente.
6.  **Predicción:** Se calcula la última fila de **11 features** y se le pide al modelo la **Probabilidad de Subida (Target=1)**.
7.  **Generación de Señal:**
    * Si Prob. Subida > 70%: **COMPRAR**
    * Si Prob. Subida < 30%: **VENDER/CERRAR**
    * En otro caso: **ESPERAR**
8.  **Pausa:** El bot espera 1 hora (o el intervalo configurado) y repite el ciclo.

---

## 💻 Indicaciones para Clonar y Ejecutar en Otra Computadora

Sigue estos pasos detallados para configurar y ejecutar el bot en una nueva máquina, asumiendo que ya tienes Git y Python instalados.

### Paso 1: Clonar el Repositorio

Abre la terminal (Git Bash, PowerShell o CMD) y navega al directorio donde deseas guardar el proyecto.

```bash
# Reemplaza <URL_DEL_REPOSITORIO> con la URL de tu repositorio de GitHub
git clone <URL_DEL_REPOSITORIO>
cd CryptoAnalystBot

Paso 2: Configuración del Entorno Virtual (Esencial)
Es crucial instalar las dependencias dentro de un entorno virtual para aislar el proyecto.

Crear y activar entorno virtual de python con la version 3.11

& "C:\Users\tuUsuario\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv311

.\.venv311\Scripts\Activate.ps1

deactivate (Asi solo para desactivar)

# 1. Crear el entorno virtual (llamado .venv311 o similar)
python -m venv .venv311 

# 2. Activar el entorno virtual (Windows)
.\.venv311\Scripts\activate 

# O activar el entorno virtual (Linux/macOS)
# source .venv311/bin/activate 

# Verás que el nombre del entorno aparece entre paréntesis: (.venv311)

Paso 3: Instalación de Dependencias
Instala todas las librerías necesarias. Este paso resuelve el error ModuleNotFoundError.

(.venv311) PS C:\ruta\a\CryptoAnalystBot> pip install python-binance pandas numpy scikit-learn xgboost ta textblob matplotlib requests

Paso 4: Configuración de Variables de Entorno (Opcional, pero Recomendado)
Aunque el bot usa datos públicos de Binance, las API Keys son necesarias si deseas interactuar con tu cuenta de trading (para ejecutar órdenes reales).

Si el archivo data_fetcher.py usa variables de entorno, debes configurarlas en tu sistema o modificarlas directamente en el archivo.

Para Windows (PowerShell/CMD):

$env:BINANCE_API_KEY="TU_CLAVE_API_DE_BINANCE"
$env:BINANCE_API_SECRET="TU_SECRETO_API_DE_BINANCE"

(Nota: Estas variables solo son válidas para la sesión de terminal actual.)

Paso 5: Primera Ejecución (Backtest)
El bot debe ejecutarse en modo backtest primero para re-entrenar y guardar el modelo con los datos iniciales de la nueva máquina y confirmar que todo funciona.

(.venv311) PS C:\ruta\a\CryptoAnalystBot> python main.py

Si la ejecución es exitosa, se mostrarán los resultados del backtesting y la gráfica.

Paso 6: Operación en Vivo (Live Mode)
Una vez que confirmes que el backtesting funciona, puedes iniciar el bot en modo de predicción continua, generando señales cada hora.

Nota: Si modificaste el archivo main.py para usar el bucle while True, simplemente vuelve a ejecutar el comando.

(.venv311) PS C:\ruta\a\CryptoAnalystBot> python main.py

# Para detener el bot, presiona Ctrl + C.

⚠️ Consejos de Mantenimiento
Cambiar Activo: Para cambiar de LINK/USDT a otro par (ej. ETH/USDT), edita la variable SYMBOL en data_fetcher.py (ej. 'ETHUSDT'). Deberás ejecutar el backtest de nuevo para re-entrenar el modelo con los nuevos datos.

Ajustar Umbrales: Los umbrales de confianza (70%/30%) y los parámetros de riesgo (Stop Loss/Take Profit) se ajustan en backtester.py.
