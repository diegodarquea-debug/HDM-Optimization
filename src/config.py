"""
Central configuration for the HDM optimization pipeline.

This file is intentionally verbose so business and technical users can
understand what each parameter controls and why default values were chosen.
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
# Project root folder (one level above /src)
PROJECT_ROOT = Path(__file__).parent.parent
# Input data folder (raw CSV files)
DATA_DIR = PROJECT_ROOT / "data"
# Output artifacts folder (all generated CSVs)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Source code folder
SRC_DIR = PROJECT_ROOT / "src"

# Auto-create required folders if missing
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Default input dataset path
RAW_DATA_PATH = DATA_DIR / "raw_data.csv"

# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------
# Number of random Monte Carlo scenarios (Step 4).
# QUÉ HACE: Define cuántas configuraciones aleatorias se exploran antes de Bayesian.
# SI AUMENTAS (ej: 200 → 1000):
#   ✓ Más cobertura del espacio de búsqueda
#   ✓ Mejor "x0" (semillas) para Bayesian optimization
#   ✗ Más tiempo de ejecución (lineal: 5x más configs = 5x más tiempo)
# SI DISMINUYES (ej: 200 → 50):
#   ✓ Más rápido
#   ✗ Menor exploración, puede perder configuraciones prometedoras
# RECOMENDACIÓN: 200-500 para iteración rápida, 1000-2000 para producción.
N_SIMULATIONS = 2000

# Stress analysis window (used for diagnostic slices in simulator reports)
# Rolling window size in minutes.
STRESS_WINDOW_ROLLING_SIZE = 60
# Half-window size around peak stress index (minutes before/after).
STRESS_WINDOW_HALF_SIZE = 30

# Fixed seed for reproducibility.
# QUÉ HACE: Controla el generador de números aleatorios (NumPy).
# SI CAMBIAS (ej: 42 → 999):
#   - Obtendrás diferentes configuraciones Monte Carlo pero estadísticamente equivalentes
#   - Útil para validar que resultados son robustos (no dependen de una seed específica)
# RECOMENDACIÓN: Mantén fijo durante desarrollo, cambia para validación de robustez.
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# BAYESIAN OPTIMIZATION
# -----------------------------------------------------------------------------
# Number of objective-function evaluations in Step 5.
# QUÉ HACE: Define cuántas iteraciones inteligentes (Bayesian) se ejecutan después de MC.
# SI AUMENTAS (ej: 40 → 120):
#   ✓ Más refinamiento de la mejor configuración encontrada
#   ✓ Mayor probabilidad de encontrar el óptimo global
#   ✗ Más tiempo de ejecución (cada eval simula ~50-200 escenarios)
# SI DISMINUYES (ej: 40 → 20):
#   ✓ Más rápido
#   ✗ Puede quedarse en óptimos locales, no explotar bien las semillas MC
# RECOMENDACIÓN: 40-60 para iteración rápida, 80-120 para búsqueda profunda.
N_OPTIMIZATION_CALLS = 80

# Bayesian internal settings
# QUÉ HACE:
# - n_initial_points: Cuántas evaluaciones ALEATORIAS hace Bayes antes de usar el modelo GP.
#   SI AUMENTAS (5 → 10): Más diversidad inicial pero menos aprovechamiento de x0 seeds.
#   SI DISMINUYES (5 → 2): Confía más en las semillas MC pero menos robusto si MC falla.
# - mc_seed_top_k: Cuántas configs Monte Carlo se pasan como x0 (puntos iniciales) a Bayes.
#   SI AUMENTAS (15 → 30): Bayes explora más candidatos prometedores desde el inicio.
#   SI DISMINUYES (15 → 5): Bayes se enfoca en pocos candidatos, más riesgo de sesgo.
# RECOMENDACIÓN: n_initial_points=5 es estándar, mc_seed_top_k=10-20 balancea diversidad/eficiencia.
BAYESIAN_SETTINGS = {
    "n_initial_points": 5,
    "mc_seed_top_k": 20,
}

# -----------------------------------------------------------------------------
# SEARCH SPACE (what optimizer is allowed to try)
# -----------------------------------------------------------------------------
# ⚠️ IMPORTANTE: Lógica estricta AND.
# HDM se activa SOLO cuando u1 AND u2 AND u3 son TRUE simultáneamente.
# Esto evita falsos positivos y asegura que solo se active en estrés real multidimensional.
#
# GUÍA RÁPIDA DE UMBRALES (cómo leerlos):
# - u1 (órdenes): condición se cumple si ordenes_pendientes >= u1
# - u2 (riders): condición se cumple si riders_cerca >= u2  (según lógica actual del simulador)
# - u3 (AWT):     condición se cumple si max_awt_espera_min >= u3
# - delta_ept:    minutos que se suman al EPT mientras HDM está activo
# - duracion_hdm: minutos que HDM permanece activo por cada activación
THRESHOLDS = {
    # u1 = órdenes pendientes (umbral de activación).
    # QUÉ HACE: Define cuántas órdenes pendientes deben haber para que u1=TRUE.
    # SI SUBES el mínimo (2 → 5):
    #   → HDM se activa MENOS frecuentemente (más conservador)
    #   → Menor reducción de AWT pero menor riesgo en EPT
    # SI BAJAS el mínimo (2 → 1):
    #   → HDM se activa MÁS frecuentemente (más agresivo)
    #   → Mayor reducción de AWT pero más presión en EPT
    # SI SUBES el máximo (12 → 20):
    #   → Optimizer puede elegir configs muy conservadoras (activación rara)
    # RECOMENDACIÓN ACTUAL: 3-10 cubre desde la mediana (P50≈3) hasta estrés alto (P90≈10).
    # Esto evita que el optimizador se refugie en zonas demasiado raras como 12+ órdenes.
    "u1": (3, 10),

    # u2 = riders cercanos (umbral de activación).
    # QUÉ HACE: Define cuántos riders cercanos MÍNIMO deben haber para que u2=TRUE.
    # NOTA IMPORTANTE: En la lógica ACTUAL del simulador, u2=TRUE cuando riders_cerca >= u2.
    # (Si negocio decide modelar "escasez de riders", esta desigualdad debería invertirse en código y docs.)
    # SI SUBES el umbral (5 → 8):
    #   → u2=TRUE menos frecuentemente
    #   → HDM más conservador
    # SI BAJAS el umbral (5 → 2):
    #   → u2=TRUE más frecuentemente
    #   → HDM más agresivo
    # RECOMENDACIÓN ACTUAL: 1-3 cubre el rango donde realmente vive la data
    # (P50≈1, P90≈2, P95≈3) bajo la lógica actual riders_cerca >= u2.
    # Valores mayores hacen la activación excesivamente rara y empujan la optimización a esquinas.
    "u2": (1, 3),

    # u3 = espera máxima (umbral de activación, minutos).
    # QUÉ HACE: Define cuántos minutos de espera máxima deben haber para que u3=TRUE.
    # SI SUBES el mínimo (2 → 8):
    #   → HDM solo se activa con esperas muy largas (conservador)
    #   → Reduce falsos positivos por picos transitorios cortos
    # SI BAJAS el mínimo (2 → 1):
    #   → HDM reacciona incluso a esperas cortas (agresivo)
    #   → Más activaciones pero puede sobre-reaccionar
    # RECOMENDACIÓN ACTUAL: 4-10 enfoca activación en estrés real de espera
    # (arranca sobre la media≈4.4 y llega a una zona todavía bien soportada por datos).
    "u3": (4, 10),

    # delta_ept = minutos EXTRA de EPT mientras HDM está activo.
    # QUÉ HACE: Define cuántos minutos se SUMAN al EPT cuando HDM se activa.
    # SI SUBES valores (ej: agregar 12, 15):
    #   → Optimizer puede elegir configs más agresivas (más impacto en EPT)
    #   → Potencialmente más reducción de AWT pero mayor costo operativo
    # SI BAJAS valores (ej: solo [2, 4]):
    #   → Optimizer limitado a incrementos pequeños de EPT
    #   → Más conservador, menor impacto operativo
    # VALORES DISCRETOS: Definidos por restricciones técnicas/operativas del negocio.
    # RECOMENDACIÓN ACTUAL: [2, 4, 6, 8, 10] permite explorar desde impactos muy leves hasta moderados, evitando extremos poco realistas. Ajusta según tolerancia de negocio a incrementos de EPT.
    "delta_ept": [2, 4, 6, 8, 10],

    # duracion_hdm = duración del HDM por activación (minutos).
    # QUÉ HACE: Una vez activado, cuánto tiempo permanece el HDM antes de desactivarse.
    # SI SUBES el mínimo (10 → 20):
    #   → Activaciones más largas → menos toggling → más estabilidad
    #   → Pero si situación mejora rápido, HDM sigue activo innecesariamente
    # SI BAJAS el mínimo (10 → 5):
    #   → Activaciones más cortas → más toggling → menos estable
    #   → Pero más reactivo a cambios rápidos de demanda
    # SI SUBES el máximo (30 → 60):
    #   → Optimizer puede elegir configs de muy larga duración
    # RECOMENDACIÓN ACTUAL: 10-20 cubre ventanas tácticas útiles.
    # Duraciones mayores tienden a encarecer EPT y agrandan el espacio de búsqueda
    # sin evidencia reciente de aportar mejores soluciones.
    "duracion_hdm": (10, 20),
}

# -----------------------------------------------------------------------------
# HDM IMPACT ADJUSTMENT (post-prediction calibration)
# -----------------------------------------------------------------------------
# Ajuste proporcional aplicado al AWT cuando HDM está activo:
# awt_ajustado = awt_predicho * (1 - rate * delta_ept)
#
# Este ajuste introduce sensibilidad de AWT frente a distintos valores de delta_ept,
# evitando que delta_ept bajos y altos se vean "iguales" para el optimizador.
HDM_EFFECT_SETTINGS = {
    # Reducción proporcional de AWT por cada minuto de delta_ept.
    "awt_delta_ept_reduction_per_min": 0.03,  # 3.0% (increased from 2.5% to amplify delta_ept gradient)
    # Tope de reducción total de AWT por este ajuste (30% = 0.30).
    "awt_delta_ept_max_reduction": 0.30,
}

# -----------------------------------------------------------------------------
# OPTIMIZATION CONSTRAINTS & OBJECTIVE
# -----------------------------------------------------------------------------
# Hard safety cap: aumento MÁXIMO de EPT promedio permitido (minutos).
# QUÉ HACE: Si una config aumenta EPT más de este valor, recibe penalización cuadrática fuerte.
# SI AUMENTAS (10 → 15):
#   → Optimizer puede explorar configs más agresivas con HDM
#   → Potencialmente más reducción de AWT pero mayor riesgo operativo
# SI DISMINUYES (10 → 5):
#   → Optimizer limitado a configs muy conservadoras
#   → Menor reducción de AWT pero más seguro en EPT
# RECOMENDACIÓN: 10 minutos es balance razonable; ajusta según tolerancia del negocio.
MAX_EPT_INCREASE = 15

# Activation delay before HDM impact is applied (minutes)
# QUÉ HACE: Tiempo que tarda HDM en mostrar impacto DESPUÉS de activarse.
# SI AUMENTAS (2 → 5):
#   → HDM se activa pero tarda más en reducir AWT
#   → Simulación más conservadora (menos impacto aparente)
# SI DISMINUYES (2 → 0):
#   → HDM tiene impacto instantáneo
#   → Simulación más optimista (puede sobrestimar beneficio)
# VALOR ACTUAL: Basado en observación empírica de latencia operativa real.
# RECOMENDACIÓN: Mantén 2 minutos a menos que tengas datos que confirmen otro valor.
ACTIVATION_DELAY_MINUTES = 2

# Objective weights used by optimizer:
# score = (awt_weight * awt_improvement) - (ept_penalty * ept_increase)
# Después, optimizer MINIMIZA (-score + penalties).
OBJECTIVE_WEIGHTS = {
    # Peso relativo de reducir AWT.
    # QUÉ HACE: Define cuánto "vale" reducir 1 minuto de AWT en el objetivo.
    # SI AUMENTAS (2.0 → 5.0):
    #   → Optimizer prioriza MUCHO reducir AWT (más agresivo)
    #   → Acepta mayores incrementos de EPT a cambio de reducir AWT
    # SI DISMINUYES (2.0 → 1.0):
    #   → Optimizer balancea más AWT vs EPT (menos agresivo)
    #   → Más conservador con activaciones de HDM
    # RECOMENDACIÓN: 2.0 es 2x más importante que EPT; ajusta según prioridad de negocio.
    "awt": 2.0,

    # Costo por minuto de incremento de EPT.
    # QUÉ HACE: Define cuánto "cuesta" aumentar 1 minuto de EPT en el objetivo.
    # SI AUMENTAS (0.2 → 0.5):
    #   → Optimizer MUY conservador con EPT
    #   → Reduce frecuencia/duración de HDM para evitar penalización
    #   → Probablemente obtengas u1, u2, u3 altos (pocos activaciones)
    # SI DISMINUYES (0.2 → 0.1):
    #   → Optimizer más dispuesto a aumentar EPT
    #   → Más activaciones de HDM, más reducción de AWT
    # RECOMENDACIÓN: 0.2 es intermedio; 0.4-0.6 si EPT es crítico; 0.1 si toleras más EPT.
    "ept_penalty": 0.2,
}

# Smooth penalty strengths used in optimizer objective.
# Fuerzas de penalización cuadrática en la función objetivo.
# QUÉ HACEN: Agregan penalizaciones suaves cuando ciertas condiciones empeoran.
OPTIMIZER_PENALTIES = {
    # Penalización si AWT empeora (en vez de mejorar).
    # SI AUMENTAS (50 → 100):
    #   → Optimizer evita MÁS agresivamente configs que empeoren AWT
    # SI DISMINUYES (50 → 20):
    #   → Optimizer tolera más explorar configs que empeoren AWT
    # RECOMENDACIÓN: 50 es fuerte pero no prohibitivo; ajusta si ves AWT empeorando.
    "awt_worse_quad": 50,
    
    # Penalización si AMBOS AWT y EPT empeoran simultáneamente.
    # SI AUMENTAS (30 → 60):
    #   → Optimizer descarta configs que sean malas en ambas métricas
    # SI DISMINUYES (30 → 10):
    #   → Optimizer explora más ampliamente (incluso configs sub-óptimas)
    # RECOMENDACIÓN: 30 evita configs claramente malas sin ser demasiado restrictivo.
    "combined_worse_quad": 30,
    
    # Penalización si EPT excede MAX_EPT_INCREASE (cap de seguridad).
    # SI AUMENTAS (10 → 50):
    #   → Optimizer NUNCA excederá el límite (muy conservador)
    # SI DISMINUYES (10 → 5):
    #   → Optimizer puede acercarse más al límite
    # RECOMENDACIÓN: 10 es suave; aumenta a 20-30 si necesitas hard constraint.
    "ept_excess_quad": 10,
}

# Strategy extraction rules
STRATEGY_SETTINGS = {
    # Reducción mínima de AWT (%) para considerar estrategia como "Conservadora".
    # QUÉ HACE: Filtra estrategias candidatas en get_top_3_strategies().
    # SI AUMENTAS (0.05 → 0.10):
    #   → Solo estrategias con 10%+ reducción califican como "conservadora"
    #   → Puede que no encuentres candidatas si datos tienen poco estrés
    # SI DISMINUYES (0.05 → 0.02):
    #   → Estrategias con apenas 2% reducción califican
    #   → Más fácil encontrar candidatas pero menos impacto garantizado
    # RECOMENDACIÓN: 0.05 (5%) es umbral mínimo pragmático de mejora perceptible.
    "conservative_min_awt_reduction_pct": 0.05,
}

# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------
# Fraction of rows used for training (remaining goes to test/validation).
# 0.6 means 60% train / 40% test, a common balance between learning capacity
# and reliable out-of-sample validation.
TRAIN_TEST_SPLIT = 0.6

# Model family for both predictors (AWT/EPT).
# linear_regression = interpretable + fast baseline.
# Alternatives available in code: decision_tree, random_forest.
MODEL_TYPE = "linear_regression"

# -----------------------------------------------------------------------------
# BIGQUERY (optional / future integration)
# -----------------------------------------------------------------------------
# If env vars are not set, value remains None and CSV flow is used.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", None)
BQ_DATASET = os.getenv("BQ_DATASET", None)
BQ_TABLE = os.getenv("BQ_TABLE", None)

# Verbose logs in console for debugging/troubleshooting.
DEBUG = True
