# HDM Optimization Pipeline — Documentación Técnica Completa

Sistema de optimización automática para activación de Hora de Mayor Demanda (HDM) en operaciones de delivery.

> **Versión actual:** Pipeline en producción para franquicias Chile (entidad `PY_CL`).  
> **Fuente de datos:** Google BigQuery — autenticación via Application Default Credentials.  
> **Última validación:** 120,960 filas cargadas correctamente para KFC AAA, ventana 14 días.

---

## Índice

1. [¿Qué problema resuelve?](#-qué-problema-resuelve-este-proyecto)
2. [Cómo funciona el sistema (flujo completo)](#-cómo-funciona-el-sistema-flujo-completo)
3. [Lógica de activación HDM](#-lógica-de-activación-hdm)
4. [Espacio de búsqueda — umbrales actuales](#-espacio-de-búsqueda--umbrales-actuales)
5. [Función objetivo y pesos](#-función-objetivo-y-pesos)
6. [Penalizaciones del optimizador](#-penalizaciones-del-optimizador)
7. [Modelo predictivo (AWT / EPT)](#-modelo-predictivo-awt--ept)
8. [Estrategias resultantes](#-estrategias-resultantes)
9. [Configuración central (config.py)](#-configuración-central-completa-configpy)
10. [Ejecutar en JupyterHub](#-ejecutar-en-jupyterhub)
11. [Salidas del pipeline](#-salidas-del-pipeline)
12. [Estructura del proyecto](#-estructura-del-proyecto)
13. [Interpretación de resultados](#-interpretación-de-resultados)
14. [Ajustando la agresividad](#-ajustando-la-agresividad)
15. [Limitaciones y mejoras futuras](#-limitaciones-y-mejoras-futuras)

---

## 🎯 ¿Qué problema resuelve este proyecto?

En operaciones de delivery, el **tiempo de espera (AWT)** de los clientes es crítico para la experiencia. Durante periodos de alta demanda:

- Los riders disponibles son escasos
- Las órdenes pendientes se acumulan
- El AWT aumenta drásticamente y empeora la experiencia del cliente

**HDM (Hora de Mayor Demanda)** es un mecanismo que incrementa el **tiempo de preparación estimado (EPT)** para dar más margen al restaurante y reducir el AWT percibido por el cliente.

**El problema**: No existía un proceso objetivo para decidir _cuándo_ activar HDM ni con _qué parámetros_, lo que generaba activaciones manuales inconsistentes o inexistentes.

**La solución**: Este pipeline optimiza automáticamente los umbrales de activación (órdenes pendientes, riders disponibles, tiempo de espera) y los parámetros de efecto (duración, incremento de EPT) para encontrar el balance óptimo entre reducir AWT y no inflar EPT innecesariamente.

---

## 📊 ¿Cómo funciona el sistema? (Flujo completo)

El sistema utiliza un **enfoque híbrido de 2 etapas** ejecutadas en secuencia:

### Etapa 1 — Exploración masiva (Monte Carlo)

- Genera **2,000 configuraciones aleatorias** del espacio de búsqueda usando una semilla fija (`RANDOM_SEED=42`) para reproducibilidad.
- Cada configuración define: umbrales de activación (`u1`, `u2`, `u3`), incremento de EPT (`delta_ept`), y duración de HDM (`duracion_hdm`).
- Simula **minuto a minuto** cómo cada configuración habría funcionado sobre los datos históricos reales.
- Calcula para cada config: AWT promedio resultante, incremento de EPT, tasa de activación, score objetivo.
- Es la fase más lenta (~30–90 min en producción con 2,000 sims). Se guarda en `monte_carlo_franchise_exploration.csv`.

**¿Por qué Monte Carlo primero?**  
El espacio de búsqueda es combinatorio (enteros + categórico `delta_ept`). Monte Carlo cubre amplitud de forma barata antes de que el Bayesian optimizer haga búsqueda guiada. Sin esta fase, el GP empezaría sin información y desperdiciaría iteraciones.

### Etapa 2 — Refinamiento inteligente (Bayesian Optimization)

- Toma las **top 20 configuraciones** del Monte Carlo (filtradas por `ept_increase ≤ MAX_EPT_INCREASE=15 min`) como semillas de warm-start (`x0`).
- Ejecuta **80 iteraciones** de búsqueda guiada por **Gaussian Process** con función de adquisición `EI` (Expected Improvement).
- El GP **aprende la forma de la función objetivo** a medida que evalúa: las primeras iteraciones van más lentas (el modelo está aprendiendo) y luego acelera explotando las regiones prometedoras.
- Guarda todo el historial en `optimization_history.csv`.

**¿Por qué Bayesian y no grid search?**  
Grid search sobre solo 5 parámetros a una resolución razonable requeriría millones de evaluaciones. Bayesian converge a buenas soluciones con 80–120 evaluaciones porque usa cada resultado anterior para decidir el próximo punto a evaluar.

---

## 🧠 Lógica de activación HDM

### Condición de activación (AND estricto)

HDM se activa **SOLO** cuando se cumplen las 3 condiciones **simultáneamente**:

```
activar_hdm = (ordenes_pendientes >= u1)
          AND (riders_cerca       >= u2)
          AND (max_awt_espera_min >= u3)
```

**¿Por qué AND y no OR?**  
Con OR, cualquier pico momentáneo en una sola métrica dispararía HDM. Esto generaría activaciones en situaciones donde el sistema no está realmente bajo estrés (e.g., una ola de órdenes con muchos riders disponibles no requiere HDM). El AND garantiza que las tres dimensiones de estrés operativo ocurran al mismo tiempo antes de intervenir.

### Delay de activación (2 minutos)

Cuando las condiciones AND se cumplen en el minuto **T**:

| Momento | Estado | Comportamiento |
|---------|--------|----------------|
| T a T+2 | En cola de activación | EPT no cambia aún; AWT sigue baseline |
| T+2 en adelante | HDM activo | EPT sube `delta_ept` minutos; predictor usa `hdm_activo=1` |
| Mientras activo y vuelve a triggearse | Extensión | Duración se extiende desde el nuevo trigger |
| Fin de `duracion_hdm` sin nuevo trigger | Desactiva | Vuelve a baseline |

**Justificación del delay**: Existe latencia operativa real entre la decisión de activar HDM y el momento en que el efecto es visible para el cliente. 2 minutos es la estimación conservadora basada en observación empírica.

### Calibración del efecto HDM sobre AWT

El modelo predictivo de AWT está entrenado con datos históricos que incluyen periodos sin HDM. Para evitar que el optimizador no perciba diferencia entre `delta_ept=4` y `delta_ept=10`, se aplica un ajuste proporcional cuando HDM está activo:

$$
AWT_{\text{ajustado}} = AWT_{\text{predicho}} \times \max\left(0.70,\ 1 - 0.03 \times \delta_{ept}\right)
$$

| Parámetro | Valor | Fuente |
|-----------|-------|--------|
| Reducción por minuto de `delta_ept` | **3.0%** (`awt_delta_ept_reduction_per_min`) | `src/config.py` → `HDM_EFFECT_SETTINGS` |
| Tope máximo de reducción | **30%** (`awt_delta_ept_max_reduction`) | `src/config.py` → `HDM_EFFECT_SETTINGS` |

**Ejemplo práctico**: Si `AWT_predicho = 5.0 min` y `delta_ept = 8`:
- factor = max(0.70, 1 - 0.03 × 8) = max(0.70, 0.76) = 0.76
- `AWT_ajustado = 5.0 × 0.76 = 3.8 min`

Esto hace que el optimizador siempre prefiera mayor `delta_ept` cuando la reducción de AWT lo justifica, evitando que se quede pegado en mínimos.

---

## 🔍 Espacio de búsqueda — Umbrales actuales

Todos los rangos viven en `src/config.py → THRESHOLDS`. **Estos son los valores reales en producción:**

| Parámetro | Rango actual | Tipo | Por qué este rango |
|-----------|-------------|------|-------------------|
| **u1** — órdenes pendientes | `(3, 10)` | Entero continuo | Por debajo de 3 órdenes el sistema no está bajo presión real; más de 10 es situación de crisis donde HDM ya no es suficiente |
| **u2** — riders cerca | `(1, 3)` | Entero continuo | Rango estrecho basado en análisis P95: el dato real raramente supera 3 riders activos en el mismo instante; umbrales altos nunca se cumplen |
| **u3** — espera máxima (min) | `(5, 10)` | Entero continuo | Por debajo de 5 min el AWT está dentro de rango aceptable; por encima de 10 min el HDM solo puede mitigar parcialmente |
| **delta_ept** | `[4, 6, 8, 10]` | Categórico | Solo valores discretos operacionalmente sensatos; valores intermedios no aportan información diferencial al sistema |
| **duracion_hdm** | `(10, 20)` | Entero continuo | Mínimo de 10 min para que el efecto sea medible; más de 20 min reduce la señal de reactivación real |

> **Nota importante sobre `u2`**: El rango `(1, 3)` puede parecer estrecho, pero los datos reales muestran que `riders_cerca` raramente supera 3 en momentos de alta demanda (cuando hay pocos riders, justamente). Un umbral `u2=4` o `u2=5` prácticamente nunca se cumpliría y el HDM no activaría.

---

## 🎯 Función objetivo y pesos

El optimizador **minimiza** la pérdida `total_loss`, que es la negación del score ponderado más penalizaciones:

```
weighted_gain = (2.5 × awt_improvement) − (0.10 × ept_increase) − rate_penalty

total_loss = −weighted_gain + penalty_terms
```

### Pesos base (OBJECTIVE_WEIGHTS)

| Peso | Valor actual | Significado |
|------|-------------|-------------|
| `awt` | **2.5** | Cada minuto de AWT reducido vale 2.5 puntos |
| `ept_penalty` | **0.10** | Cada minuto de EPT incrementado cuesta 0.10 puntos |

**Ratio de prioridad**: 2.5 / 0.10 = **25×** — Reducir 1 min de AWT equivale a tolerar 25 min de aumento de EPT. Esto es intencional: el EPT es visible para el restaurante pero no directamente para el cliente; el AWT sí impacta directamente la experiencia.

### Penalización por sobreactivación (rate_penalty)

HDM no debe activarse más del **25% del tiempo**. Si lo hace, pierde significado operacional (pasa de "hora pico" a "modo permanente"):

$$
\text{rate\_penalty} = \begin{cases} 20 \times (r_{\text{HDM}} - 0.25)^2 & \text{si } r_{\text{HDM}} > 0.25 \\ 0 & \text{si } r_{\text{HDM}} \leq 0.25 \end{cases}
$$

Esta penalización es **cuadrática pero no prohibitiva**: un 30% de activación suma ~5 puntos de penalización, un 50% suma ~125. Así el optimizador puede explorar configuraciones levemente por encima de 25% pero pagará un costo creciente.

---

## 🛡️ Penalizaciones del optimizador

Además de la penalización por sobreactivación, existen tres penalizaciones de seguridad (todas cuadráticas suaves):

| Penalización | Valor | Se activa cuando | Fórmula |
|-------------|-------|-----------------|---------|
| `awt_worse_quad` | **50** | AWT empeora respecto a baseline | `50 × (awt_improvement_neg)²` |
| `combined_worse_quad` | **30** | AWT + EPT combinados empeoran | `30 × (combined_improvement_neg)²` |
| `ept_excess_quad` | **10** | EPT aumenta más de 15 min | `10 × (ept_increase − 15)²` |

**¿Por qué cuadráticas y no hard constraints?**  
Los hard constraints rompen la superficie de la función objetivo (discontinuidades) y confunden al Gaussian Process. Las penalizaciones cuadráticas suaves crean una superficie continua y diferenciable: el GP puede "ver" que se está acercando a una zona peligrosa y evitarla gradualmente.

**Ejemplo**: Si una config resulta en `ept_increase = 18 min`:
- `ept_excess_quad = 10 × (18 − 15)² = 10 × 9 = 90 puntos de penalización`
- Esos 90 puntos se restan del score, casi siempre haciendo esa config inferior a alternativas válidas.

---

## 🤖 Modelo predictivo (AWT / EPT)

### AWTPredictor

| Atributo | Valor |
|---------|-------|
| Algoritmo | **Random Forest** (`MODEL_TYPE = "random_forest"`) |
| Estimators | 100 árboles, `max_depth=5` |
| Split entrenamiento/test | **60% / 40%** (`TRAIN_TEST_SPLIT = 0.6`) |
| Features de entrada | `ordenes_pendientes`, `riders_cerca`, `hdm_activo`, `ept_promedio_min_smoothed` (si disponible) |
| Variable objetivo | `max_awt_espera_min` (o `awt_promedio` si disponible) |
| R² observado | **~0.50–0.69** dependiendo de la franquicia |

**¿Por qué Random Forest y no regresión lineal?**  
La relación entre órdenes/riders y AWT no es lineal: con pocos riders y muchas órdenes, el AWT escala de forma no proporcional. Random Forest captura estas interacciones sin requerir ingeniería de features manual.

**¿Por qué `max_depth=5`?**  
Evita overfitting. Con `max_depth` sin límite, el modelo memorizaría los datos de entrenamiento y no generalizaría bien a configuraciones nuevas durante la simulación.

### EPTPredictor

| Atributo | Valor |
|---------|-------|
| Algoritmo | **Random Forest** (mismo tipo que AWT) |
| Variable objetivo | `ept_promedio_min_smoothed` (EPT suavizado, libre de ruido) |
| Uso principal | Calcular `ept_increase` relativo al baseline para la función objetivo |

**¿Por qué usar EPT suavizado (`_smoothed`) y no el raw?**  
El EPT configurado tiene saltos discretos que introducen ruido artificial. La versión suavizada representa mejor el EPT "efectivo" que experimentan los riders y clientes.

---

## 🏆 Estrategias resultantes

Al final de la optimización, el sistema extrae **3 estrategias** del historial completo de la Bayesian Optimization:

### Agresiva
- Selecciona la configuración con **mayor `awt_improvement` absoluto** en todo el historial.
- Objetivo: máxima reducción de AWT sin importar cuánto sube EPT (mientras sea ≤ `MAX_EPT_INCREASE`).
- Útil cuando AWT alto es el problema dominante y el equipo operacional acepta EPT más alto.

### Equilibrada ⭐ (estrategia principal)
- Selecciona la configuración con **mayor `combined_improvement`** = reducción total de (AWT + EPT combinados).
- Es la estrategia que el pipeline usa por defecto para el reporte final.
- Objetivo: el mejor balance global entre AWT y EPT.

### Conservadora
- Filtra configuraciones con `awt_improvement ≥ 5%` del baseline.
- Entre esas, selecciona la de **menor `hdm_activation_rate`** (la que activa HDM con menos frecuencia).
- Útil cuando el equipo operacional quiere implementar HDM con cautela, solo en situaciones claras.

> Todas las estrategias respetan el hard cap `ept_increase ≤ 15 min` como filtro previo. Si no hay candidatos válidos, se usa el historial completo (sin filtro) como fallback.

---

## ⚙️ Configuración central completa (config.py)

Todos los parámetros del sistema están en [`src/config.py`](src/config.py). Nada está hardcodeado fuera de ese archivo. La tabla siguiente refleja los **valores reales actuales en producción**:

### Espacio de búsqueda

```python
THRESHOLDS = {
    "u1": (3, 10),           # órdenes pendientes
    "u2": (1, 3),            # riders cerca
    "u3": (5, 10),           # espera máxima (min)
    "delta_ept": [4, 6, 8, 10],   # categórico: minutos extra de EPT
    "duracion_hdm": (10, 20),     # duración de activación (min)
}
```

### Efecto HDM sobre AWT

```python
HDM_EFFECT_SETTINGS = {
    "awt_delta_ept_reduction_per_min": 0.03,   # 3% de reducción de AWT por cada min de delta_ept
    "awt_delta_ept_max_reduction": 0.30,        # tope: máximo 30% de reducción por este mecanismo
}
```

### Función objetivo

```python
OBJECTIVE_WEIGHTS = {
    "awt": 2.5,          # peso de reducción de AWT
    "ept_penalty": 0.10, # costo por aumento de EPT
}
```

### Penalizaciones del optimizador

```python
OPTIMIZER_PENALTIES = {
    "awt_worse_quad": 50,       # penaliza si AWT empeora
    "combined_worse_quad": 30,  # penaliza si AWT+EPT combined empeoran
    "ept_excess_quad": 10,      # penaliza si EPT supera 15 min
}
# Más: rate_penalty = 20 × (hdm_rate − 0.25)² si hdm_rate > 25%
```

### Restricciones y delays

```python
MAX_EPT_INCREASE = 15          # cap duro: EPT no puede subir más de 15 min en promedio
ACTIVATION_DELAY_MINUTES = 2   # delay de 2 min entre trigger y efecto visible
```

### Configuración de optimización (overridable por env vars)

```python
N_SIMULATIONS = 2000       # Monte Carlo configs (env: N_SIMULATIONS)
N_OPTIMIZATION_CALLS = 80  # Bayesian iterations (env: N_OPTIMIZATION_CALLS)

BAYESIAN_SETTINGS = {
    "n_initial_points": 5,    # evaluaciones aleatorias antes de usar el GP
    "mc_seed_top_k": 20,      # top configs Monte Carlo usadas como warm-start
}
```

### Modelo predictivo

```python
MODEL_TYPE = "random_forest"   # opciones: linear_regression, random_forest, decision_tree
TRAIN_TEST_SPLIT = 0.60        # 60% entrenamiento, 40% test
RANDOM_SEED = 42               # garantiza reproducibilidad total
```

### Perfil de extracción de estrategias

```python
STRATEGY_SETTINGS = {
    "conservative_min_awt_reduction_pct": 0.05,  # conservadora requiere ≥5% de mejora AWT
}
```

---

## 🚀 Ejecutar en JupyterHub

### Primer uso (una sola vez)

```bash
cd ~/projects/HDM-Optimization
git pull origin main   # asegurarse de tener el código más reciente
```

### Antes de cada corrida

```bash
source ./setup.sh
```

Esto exporta las 3 variables de entorno necesarias:
- `GCP_PROJECT_ID=peya-chile`
- `GOOGLE_CLOUD_PROJECT=peya-chile`
- `BQ_LOCATION=US`

### Perfiles de ejecución

| Runner | Simulaciones | Optimizaciones | Uso recomendado | Tiempo estimado |
|--------|-------------|----------------|-----------------|-----------------|
| `run.sh` | 10 | 5 | Smoke test — verificar que todo funciona | ~5 min |
| `run_realistic.sh` | 200 | 20 | Validación — resultado orientativo confiable | ~30 min |
| `run_prod.sh` | 2000 | 80 | Producción — resultado óptimo definitivo | ~3–5 h |

### Comandos

**Smoke test** (verifica BigQuery + pipeline completo):
```bash
bash ./run.sh KFC AAA 2026-02-23 2026-03-01
```

**Validación realista** (14 días de datos):
```bash
bash ./run_realistic.sh KFC AAA 2026-02-16 2026-03-01
```

**Producción completa — bloquea la terminal**:
```bash
bash ./run_prod.sh KFC AAA 2026-02-16 2026-03-01
```

**Producción completa — en background (recomendado):**
```bash
nohup bash ./run_prod.sh KFC AAA 2026-02-16 2026-03-01 > prod_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
# Seguir en vivo:
tail -f prod_run_*.log
```

### Parámetros que cambia el operador

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|---------|
| `FRANCHISE` | Nombre de la franquicia | `KFC` |
| `GRADE` | Grado de los vendors | `AAA`, `AA`, `A` |
| `START_DATE` | Inicio ventana histórica | `2026-02-16` |
| `END_DATE` | Fin ventana histórica | `2026-03-01` |

Todo lo demás está fijo: país Chile (`country_id=2`), entidad `PY_CL`, fuente BigQuery, scope global.

### Ventana histórica recomendada

| Perfil | Ventana | Justificación |
|--------|---------|---------------|
| Smoke | 7 días | Solo verifica que el pipeline funciona |
| Realistic | 14 días | Suficiente variabilidad operacional, costo razonable |
| Producción | 14–21 días | Mayor representatividad de patrones de demanda |

---

## 🏗️ Arquitectura del pipeline (paso a paso)

```
┌─────────────────────────────────────────────────────────────────┐
│  PASO 1: CARGAR DATOS (BigQuery)                                │
│  SQL: sql/franchise_input.sql con parámetros @target_franchise, │
│  @target_grade, @target_country_id, @target_entity_id,         │
│  @start_date, @end_date                                         │
│  Salida: DataFrame minuto a minuto [ordenes, riders, awt, ept]  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 2: ANALIZAR BASELINE                                      │
│  Calcula métricas SIN HDM: AWT avg/P50/P95, EPT avg/P95        │
│  Establece el punto de referencia para medir mejora             │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 3: ENTRENAR PREDICTORES                                   │
│  AWTPredictor (Random Forest): f(ordenes, riders, hdm, ept)→awt │
│  EPTPredictor (Random Forest): f(ordenes, riders, hdm) → ept   │
│  Train 60% / Test 40% / Semilla fija RANDOM_SEED=42            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 4: MONTE CARLO (2,000 configs)                           │
│  Para cada config aleatoria:                                    │
│    1. Simula activación AND minuto a minuto                     │
│    2. Aplica delay de 2 minutos antes de efecto                 │
│    3. Aplica calibración AWT × (1 − 0.03 × delta_ept)         │
│    4. Calcula AWT/EPT resultantes y score                       │
│  Salida: monte_carlo_franchise_exploration.csv                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 5: BAYESIAN OPTIMIZATION (80 iteraciones)                │
│  Warm-start: top 20 configs Monte Carlo como x0                 │
│  GP aprende función objetivo; adquisición: Expected Improvement │
│  Penaliza: AWT peor (+50), EPT>15 (+10), HDM>25% (+20×quad)   │
│  Salida: optimization_history.csv                               │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 6: EXTRAER ESTRATEGIAS                                   │
│  Agresiva: max awt_improvement del historial                    │
│  Equilibrada: max combined_improvement (AWT+EPT) ← DEFAULT     │
│  Conservadora: min hdm_rate entre configs con ≥5% mejora AWT   │
│  Salida: franchise_optimal_config.csv                           │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASO 7: EVALUAR IMPACTO POR PARTNER                           │
│  Aplica config ganadora a cada restaurante individualmente      │
│  Pondera resultados por volumen (sum ordenes_pendientes)        │
│  Salida: franchise_impact_by_partner.csv                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Estructura del proyecto

```
HDM-Optimization/
│
├── sql/
│   └── franchise_input.sql          # Query BigQuery parametrizada (producción)
│
├── src/
│   ├── config.py                    # ⚙️ TODA la configuración aquí — nada hardcodeado
│   ├── data_loader.py               # Carga desde CSV o BigQuery, preprocessamento
│   ├── analytics.py                 # Métricas de baseline (sin HDM)
│   ├── model.py                     # AWTPredictor + EPTPredictor (Random Forest)
│   ├── simulator.py                 # Motor de simulación minuto a minuto
│   └── optimizer.py                 # Bayesian Optimization + extracción de estrategias
│
├── outputs/                         # 📊 Resultados consolidados (última corrida)
│   ├── runs/                        # Corridas versionadas por timestamp
│   │   └── YYYYMMDD_HHMMSS_run/    # Carpeta por cada ejecución
│   ├── latest_run_path.txt          # Puntero a la carpeta de la última corrida
│   ├── monte_carlo_franchise_exploration.csv
│   ├── optimization_history.csv
│   ├── franchise_optimal_config.csv
│   └── franchise_impact_by_partner.csv
│
├── main.py                          # 🎯 Orquestador principal del pipeline
├── run.sh                           # Smoke test (10 sims / 5 opts)
├── run_realistic.sh                 # Validación (200 sims / 20 opts)
├── run_prod.sh                      # Producción (2000 sims / 80 opts)
├── setup.sh                         # Configura env vars de GCP para JupyterHub
└── requirements.txt                 # Dependencias Python
```

---

## 📊 Salidas del pipeline

### 1. `monte_carlo_franchise_exploration.csv`

Una fila por cada una de las 2,000 configuraciones aleatorias evaluadas:

| Columna | Descripción |
|---------|-------------|
| `u1`, `u2`, `u3` | Umbrales de activación probados |
| `delta_ept`, `duracion_hdm` | Parámetros de efecto HDM |
| `awt_mean` | AWT promedio con esa config (min) |
| `awt_improvement` | Reducción vs. baseline (min, positivo = mejora) |
| `ept_increase` | Aumento de EPT vs. baseline (min) |
| `hdm_activation_rate` | Fracción del tiempo con HDM activo (0–1) |
| `objective_score` | Score ponderado: `2.5×awt_imp − 0.10×ept_inc` |

### 2. `optimization_history.csv`

Una fila por cada iteración del Bayesian Optimizer (80 en producción):

| Columna | Descripción |
|---------|-------------|
| `u1`–`duracion_hdm` | Config evaluada en esa iteración |
| `awt_improvement` | Mejora de AWT obtenida |
| `ept_increase` | Aumento de EPT |
| `combined_improvement` | Mejora combinada AWT+EPT |
| `hdm_activation_rate` | Tasa de activación |
| `total_loss` | Pérdida total (el optimizador minimiza esto) |
| `objective_score` | Score equivalente (el optimizador maximiza esto) |

### 3. `franchise_optimal_config.csv`

La configuración ganadora por estrategia:

```csv
strategy,u1,u2,u3,delta_ept,duracion_hdm,awt_improvement,ept_increase,hdm_activation_rate
Equilibrada,5,2,7,8,15,1.23,4.56,0.18
Agresiva,4,1,6,10,18,1.87,8.12,0.24
Conservadora,7,2,8,6,12,0.62,2.11,0.09
```

### 4. `franchise_impact_by_partner.csv`

Evalúa la config `Equilibrada` en cada restaurante individualmente:

| Columna | Descripción |
|---------|-------------|
| `partner_name` | Nombre del restaurante |
| `baseline_awt` | AWT promedio sin HDM (min) |
| `optimized_awt` | AWT con config óptima aplicada (min) |
| `awt_improvement` | Reducción en minutos |
| `ept_increase` | Aumento de EPT en minutos |
| `hdm_activation_rate` | % del tiempo con HDM activo |

---

## 📈 Interpretación de resultados

### ¿Qué buscar en los CSVs?

**Monte Carlo** → Buscar configs con `objective_score` alto, `ept_increase < 10`, `hdm_activation_rate < 0.25`. Muchos puntos buenos en la zona alta = espacio de búsqueda bien calibrado.

**Optimization History** → El `objective_score` debe crecer en el tiempo (convergencia). Si no crece después de 40 iteraciones, el espacio de búsqueda puede ser demasiado estrecho.

**Optimal Config** → Revisar que `ept_increase < 15` (dentro del cap), `hdm_activation_rate < 0.25` (no sobreactiva), y que `awt_improvement` sea mayor a 0 (sí mejora).

**Partner Impact** → Si algún partner tiene `awt_improvement < 0`, esa franquicia individual puede necesitar configuración separada.

### Señales de advertencia

| Situación | Posible causa | Acción |
|-----------|--------------|--------|
| `hdm_activation_rate > 0.4` en config ganadora | Umbrales demasiado bajos | Subir `u1`, `u2` o `u3` |
| `ept_increase > 12` cons. | Pesos muy agresivos en AWT | Subir `ept_penalty` en config |
| `awt_improvement ≈ 0` | HDM no tiene efecto en estos datos | Revisar ventana de datos o calibración |
| Optimization history plana | Monte Carlo no encontró buenas seeds | Aumentar `N_SIMULATIONS` |

---

## 🎨 Ajustando la agresividad

Si quieres que el sistema sea **más agresivo** (reduce más AWT aunque suba más EPT):

```python
# src/config.py
OBJECTIVE_WEIGHTS = {
    "awt": 3.5,        # ↑ subir de 2.5
    "ept_penalty": 0.05,  # ↓ bajar de 0.10
}
```

Si quieres que sea **más conservador** (protege EPT, acepta menos reducción de AWT):

```python
OBJECTIVE_WEIGHTS = {
    "awt": 1.5,        # ↓ bajar de 2.5
    "ept_penalty": 0.25,  # ↑ subir de 0.10
}
```

Si quieres explorar rangos más anchos (e.g. `u1` más alto para franquicias de alto volumen):

```python
THRESHOLDS = {
    "u1": (5, 15),    # franquicia de alto volumen
    "u2": (2, 5),
    "u3": (6, 12),
    "delta_ept": [4, 6, 8, 10],
    "duracion_hdm": (10, 25),
}
```

> Siempre correr un smoke test después de cambiar config antes de lanzar producción.

---

## 🔬 Trazabilidad y auditoría

Este sistema está diseñado para ser **100% auditable**:

1. **Configuración centralizada**: TODO está en `src/config.py`. Cero valores mágicos en el código.
2. **Semilla fija** (`RANDOM_SEED=42`): los mismos datos producen siempre los mismos resultados, permitiendo reproducir cualquier corrida.
3. **Historial completo**: Monte Carlo guarda las 2,000 configs; Bayesian guarda las 80 iteraciones. Puedes reconstruir por qué se eligió una config.
4. **Corridas versionadas**: cada ejecución crea `outputs/runs/YYYYMMDD_HHMMSS_run/`; las anteriores no se sobreescriben.
5. **Puntero a última corrida**: `outputs/latest_run_path.txt` siempre apunta a la corrida más reciente.
6. **Función objetivo explícita**: `score = 2.5×AWT − 0.10×EPT − penalties`. Calculable a mano.

---

## ⚠️ Limitaciones

| Limitación | Impacto | Mitigación actual |
|-----------|---------|------------------|
| Modelo predictivo offline | No aprende de corridas en vivo | Usar ventana histórica reciente (≤21 días) |
| Calibración HDM paramétrica (3%/min) | Supuesto operativo, no causal puro | Valor empírico validado; revisable con más datos |
| Sin validación temporal cruzada | Puede sobrestimar mejoras | Validar corrida realistic antes de producción |
| Paralelismo limitado en JupyterHub | `n_jobs=1` — más lento | Compensado con vectorización del loop principal |

---

## 🛠️ Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `scikit-optimize`, `google-cloud-bigquery`, `tqdm`, `joblib`.

**Python**: 3.10+ (requerido por scikit-optimize en el entorno de producción)

---

**Versión**: 3.0 — Revisión 360°  
**Actualizado**: 2026-03-16  
**Mantenedor**: Equipo Operaciones / Data Science PedidosYa Chile

