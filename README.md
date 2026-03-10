# HDM Optimization Pipeline

Sistema de optimización automática para activación de Hora de Mayor Demanda (HDM) en operaciones de delivery.

---

## 🎯 ¿Qué problema resuelve este proyecto?

En operaciones de delivery, el **tiempo de espera (AWT)** de los clientes es crítico para la experiencia. Durante periodos de alta demanda:
- Los riders disponibles son escasos
- Las órdenes pendientes se acumulan
- El AWT aumenta drásticamente

**HDM (Hora de Mayor Demanda)** es un mecanismo que incrementa el **tiempo de preparación estimado (EPT)** para dar más margen al restaurante y reducir el AWT percibido por el cliente.

**El problema**: No sabíamos cuándo activar HDM ni con qué parámetros para maximizar reducción de AWT sin aumentar demasiado el EPT.

**La solución**: Este pipeline optimiza automáticamente los umbrales de activación (órdenes, riders, espera) y parámetros de HDM (duración, impacto en EPT) para encontrar el balance óptimo.

---

## 📊 ¿Cómo funciona el sistema?

El sistema utiliza un **enfoque híbrido de 2 etapas**:

### 1️⃣ **Exploración Masiva (Monte Carlo)**
- Prueba **2,000 configuraciones aleatorias** del espacio de búsqueda
- Cada configuración define: umbrales de activación (u1, u2, u3), delta de EPT, duración de HDM
- Simula minuto a minuto cómo cada configuración habría funcionado con datos históricos
- Calcula métricas: AWT promedio, EPT promedio, tasa de activación

### 2️⃣ **Refinamiento Inteligente (Bayesian Optimization)**
- Toma las **mejores 30 configuraciones** del Monte Carlo como punto de partida
- Ejecuta **80 iteraciones** de búsqueda guiada por Gaussian Process
- Aprende qué regiones del espacio de búsqueda son más prometedoras
- Converge hacia la configuración óptima global

---

## 🧠 Lógica de Activación HDM

### Condición de Activación (AND estricto)
HDM se activa **SOLO** cuando se cumplen **simultáneamente** las 3 condiciones:

```python
activar_hdm = (ordenes_pendientes >= u1) AND 
              (riders_cerca >= u2) AND 
              (max_awt_espera_min >= u3)
```

**¿Por qué AND y no OR?**
- Evita falsos positivos (activar por pico momentáneo en una sola métrica)
- Garantiza que solo se active durante estrés operativo **multidimensional real**
- Más robusto ante ruido en los datos

### Delay de Activación
Cuando las condiciones AND se cumplen en el minuto **T**:
- **T a T+2**: Periodo de **delay** (2 minutos)
  - HDM marcado como "en activación"
  - EPT aún no se incrementa
  - AWT sigue comportamiento baseline
- **T+2 en adelante**: HDM **activo**
  - EPT aumenta por `delta_ept` minutos
  - AWT predicho con `hdm_activo=1`
  - Duración: `duracion_hdm` minutos

**Justificación**: Basado en observación empírica de latencia operativa real entre decisión y efecto visible.

### Ajuste Nuevo (Mar-2026): Sensibilidad de AWT a `delta_ept`

Para evitar que `delta_ept` bajo y alto se comporten casi igual en la optimización, se añadió una calibración simple cuando HDM está activo:

$$
awt_{ajustado} = awt_{predicho} \times (1 - r \times delta\_ept)
$$

Donde:
- $r = 0.025$ (2.5% por minuto de `delta_ept`)
- Reducción máxima por este ajuste: 30%

**Antes**: el optimizador tendía a elegir `delta_ept=2` porque veía poco beneficio adicional en AWT al subir `delta_ept`.

**Después**: el optimizador sí percibe beneficio incremental en AWT cuando aumenta `delta_ept`, y deja de pegarse en mínimos.

---

## ⚙️ Parámetros de Configuración

Todos los parámetros están centralizados en [`src/config.py`](src/config.py).

### 🔍 Espacio de Búsqueda (THRESHOLDS)

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| **u1** | (2, 10) | Umbral de órdenes pendientes para activación |
| **u2** | (2, 5) | Umbral de riders cerca para activación |
| **u3** | (4, 9) | Umbral de espera máxima (min) para activación |
| **delta_ept** | [4, 6, 8, 10] | Minutos a sumar al EPT cuando HDM activo |
| **duracion_hdm** | (12, 24) | Duración de HDM por activación (minutos) |

### 🎯 Función Objetivo (OBJECTIVE_WEIGHTS)

```python
score = (awt_weight × reducción_awt) - (ept_penalty × aumento_ept) - penalizaciones
```

| Peso | Valor | Significado |
|------|-------|-------------|
| **awt** | 2.5 | Reducir 1 min de AWT vale 2.5 puntos |
| **ept_penalty** | 0.15 | Aumentar 1 min de EPT cuesta 0.15 puntos |

**Interpretación**: Priorizamos **reducir AWT** 16x más que evitar aumentar EPT (2.5 / 0.15 ≈ 16.7).

### 🛡️ Penalizaciones (OPTIMIZER_PENALTIES)

| Penalización | Valor | Cuándo Aplica |
|-------------|-------|---------------|
| **awt_worse_quad** | 50 | Si AWT empeora en vez de mejorar |
| **combined_worse_quad** | 20 | Si AMBOS AWT y EPT empeoran |
| **ept_excess_quad** | 5 | Si EPT aumenta más de 15 min (cap de seguridad) |

**Tipo**: Cuadráticas suaves → No prohíben, pero desincentivan fuertemente.

### 🔧 Configuración de Optimización

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **N_SIMULATIONS** | 2000 | Configs aleatorias en Monte Carlo |
| **N_OPTIMIZATION_CALLS** | 80 | Iteraciones de Bayesian Optimization |
| **mc_seed_top_k** | 30 | Top configs MC que alimentan Bayesian |
| **n_initial_points** | 5 | Evaluaciones aleatorias antes de usar GP |
| **MAX_EPT_INCREASE** | 15 min | Hard cap de seguridad en aumento de EPT |
| **ACTIVATION_DELAY_MINUTES** | 2 | Minutos de delay antes de efecto HDM |

---

## 🏗️ Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD DATA                                              │
│  Entrada: data/raw_data.csv (83,675 filas, 18 partners)        │
│  Salida: DataFrame con columnas [ordenes, riders, awt, ept]    │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: ANALYZE BASELINE                                       │
│  Calcula métricas sin HDM: AWT, EPT, P50, P95                  │
│  Establece referencia para comparación                          │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: TRAIN PREDICTORS                                       │
│  AWTPredictor: f(ordenes, riders, hdm, ept) → awt              │
│  Ajuste HDM: awt_ajustado = awt_predicho × (1 - r×delta_ept)   │
│  EPTPredictor: Usa EPT histórico del dataset                   │
│  Métricas: AWT R²=0.69, EPT R²=0.50                            │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: MONTE CARLO SIMULATION (2000 configs)                 │
│  Para cada config aleatoria:                                    │
│    1. Simula activación AND minuto a minuto                     │
│    2. Aplica delay de 2 minutos                                 │
│    3. Calcula AWT/EPT resultantes                               │
│    4. Evalúa función objetivo                                   │
│  Salida: monte_carlo_franchise_exploration.csv                  │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: BAYESIAN OPTIMIZATION (80 iterations)                 │
│  Inicializa con top 30 configs MC (x0 seeds)                   │
│  Gaussian Process aprende función objetivo                      │
│  Explora/Explota balance con acquisition function               │
│  Salida: optimization_history.csv                               │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: EXTRACT BEST STRATEGIES                               │
│  Filtra configs por tipo: Conservadora, Equilibrada, Agresiva  │
│  Selecciona mejor de cada categoría                             │
│  Salida: franchise_optimal_config.csv                           │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: IMPACT ANALYSIS                                       │
│  Simula config óptima en cada partner individual               │
│  Calcula reducción de AWT por restaurante                       │
│  Salida: franchise_impact_by_partner.csv                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Uso

### Modo Franchise (Recomendado)
Optimiza una configuración **única** para todos los partners, ponderada por volumen de órdenes:

```bash
python main.py --mode franchise
```

**Salidas**:
- `monte_carlo_franchise_exploration.csv` - Exploración de 2000 configs
- `optimization_history.csv` - Convergencia de Bayesian Optimization
- `franchise_optimal_config.csv` - Configuración ganadora
- `franchise_impact_by_partner.csv` - Impacto por restaurante

### Modo Partner (Legacy)
Optimiza configuración para un **solo partner**:

```bash
python main.py --mode partner --partner_id 12345
```

**Nota**: Requiere suficientes datos históricos para ese partner (mín. 1000 filas).

---

## 📦 Estructura del Proyecto

```
HDM-Optimization/
│
├── data/
│   └── raw_data.csv                    # Dataset de entrada (83,675 filas)
│
├── src/
│   ├── config.py                       # ⚙️ Configuración central (TODO aquí)
│   ├── data_loader.py                  # Carga y preprocesamiento de datos
│   ├── analytics.py                    # Análisis de baseline y métricas
│   ├── model.py                        # Predictores de AWT y EPT
│   ├── simulator.py                    # Simulación Monte Carlo + evaluación
│   └── optimizer.py                    # Bayesian Optimization + extracción de estrategias
│
├── outputs/                            # 📊 Resultados generados
│   ├── monte_carlo_franchise_exploration.csv
│   ├── optimization_history.csv
│   ├── franchise_optimal_config.csv
│   └── franchise_impact_by_partner.csv
│
├── main.py                             # 🎯 Pipeline principal
├── requirements.txt                    # Dependencias Python
└── README.md                           # Este archivo
```

---

## 🔍 Interpretación de Resultados

### 1. Monte Carlo Exploration (`monte_carlo_franchise_exploration.csv`)

| Columna | Descripción |
|---------|-------------|
| `u1, u2, u3` | Umbrales de activación probados |
| `delta_ept, duracion_hdm` | Parámetros de HDM probados |
| `awt_mean` | AWT promedio resultante (minutos) |
| `ept_increase` | Aumento de EPT vs. baseline (minutos) |
| `hdm_activation_rate` | % de tiempo que HDM estuvo activo |
| `objective_score` | Puntuación de la función objetivo (más alto = mejor) |

**Buscar**: Configs con `objective_score` alto, `awt_mean` bajo, `ept_increase` < 15.

---

### 2. Optimization History (`optimization_history.csv`)

| Columna | Descripción |
|---------|-------------|
| `iteration` | Número de iteración Bayesian (0-79) |
| `u1, u2, u3, delta_ept, duracion_hdm` | Config evaluada |
| `objective_score` | Puntuación obtenida |
| `awt_improvement_pct` | % de reducción de AWT vs. baseline |
| `ept_increase` | Aumento de EPT (minutos) |

**Gráfico Útil**: `objective_score` vs. `iteration` → Debe mostrar convergencia ascendente.

---

### 3. Optimal Config (`franchise_optimal_config.csv`)

Contiene la **mejor configuración encontrada**:

```csv
strategy,u1,u2,u3,delta_ept,duracion_hdm,awt_improvement_pct,ept_increase,objective_score
Equilibrada,5,2,5,6,20,-12.5,8.2,45.3
```

**Interpretación**:
- **u1=5**: Activar cuando hay ≥5 órdenes pendientes
- **u2=2**: Activar cuando hay ≥2 riders cerca
- **u3=5**: Activar cuando AWT ≥5 minutos
- **delta_ept=6**: Aumentar EPT en 6 minutos durante HDM
- **duracion_hdm=20**: HDM dura 20 minutos por activación
- **awt_improvement_pct=-12.5%**: Reduce AWT en 12.5%
- **ept_increase=8.2**: EPT sube 8.2 minutos en promedio
- **objective_score=45.3**: Puntuación ponderada (mientras más alto, mejor)

---

### 4. Impact by Partner (`franchise_impact_by_partner.csv`)

| Columna | Descripción |
|---------|-------------|
| `partner_name` | Nombre del restaurante |
| `baseline_awt` | AWT promedio sin HDM (minutos) |
| `optimized_awt` | AWT promedio con HDM óptimo (minutos) |
| `awt_improvement_pct` | % de reducción de AWT |
| `ept_increase` | Aumento de EPT (minutos) |
| `hdm_activation_rate` | % de tiempo con HDM activo |

**Buscar**: Partners con mayor `awt_improvement_pct` → Mayor impacto del HDM.

---

## 🎨 Ajustando la Agresividad

Si quieres que el sistema sea **más agresivo** (reduce más AWT aunque aumente más EPT):

```python
# En src/config.py
OBJECTIVE_WEIGHTS = {
    "awt": 3.0,              # Aumenta de 2.5 → 3.0
    "ept_penalty": 0.10,     # Reduce de 0.15 → 0.10
}
```

Si quieres que sea **más conservador** (protege EPT aunque reduzca menos AWT):

```python
OBJECTIVE_WEIGHTS = {
    "awt": 2.0,              # Reduce de 2.5 → 2.0
    "ept_penalty": 0.25,     # Aumenta de 0.15 → 0.25
}
```

---

## 📈 Validación del Modelo

### Métricas de Predictores (Test Set)

| Modelo | R² | MAE | Evaluación |
|--------|-----|-----|------------|
| **AWTPredictor** | 0.69 | 1.38 min | ✅ Aceptable (explica 69% varianza) |
| **EPTPredictor** | 0.50 | 5.53 min | ⚠️ Razonable (suficiente para guiar optimización) |

**Features usados**:
- `ordenes_pendientes`
- `riders_cerca`
- `hdm_activo` (0 o 1)
- `ept_promedio_min`

**Target**:
- AWTPredictor → `max_awt_espera_min`
- EPTPredictor → `ept_promedio_min` (del histórico)

---

## 🧪 Dataset Utilizado

**Archivo**: `data/raw_data.csv`  
**Filas**: 83,675  
**Período**: 2025-11-10 a 2025-11-17 (8 días)  
**Partners**: 18 restaurantes Melt Pizzas  
**Granularidad**: Minuto a minuto

### Columnas Críticas:

| Columna | Descripción | Rango |
|---------|-------------|-------|
| `ordenes_pendientes` | Órdenes en cola | 1-43 |
| `riders_cerca` | Riders disponibles | 0-13 |
| `max_awt_espera_min` | Tiempo máximo de espera | 0-66 min |
| `ept_promedio_min` | EPT promedio configurado | 11-50 min |
| `hdm_activo` | Flag histórico de HDM | 0 o 1 |

### Correlaciones Observadas:

```
                   ordenes  riders   awt    ept
ordenes_pendientes   1.00    0.58   0.40   0.69
riders_cerca         0.58    1.00   0.71   0.39
max_awt_espera_min   0.40    0.71   1.00   0.30
ept_promedio_min     0.69    0.39   0.30   1.00
```

**Interpretación**:
- **riders ↔ AWT** (r=0.71): Fuerte → Scarcity de riders causa mayor espera ✅
- **ordenes ↔ EPT** (r=0.69): Fuerte → Más carga causa mayor tiempo de prep ✅
- **AWT ↔ EPT** (r=0.30): Baja → Desacoplados, optimizables independientemente ✅

---

## 🔬 Trazabilidad y Auditoría

Este sistema está diseñado para ser **100% auditable**:

### 1. Configuración Centralizada
- **TODO** está en `src/config.py` (CERO hardcodes en el código)
- Cada parámetro tiene docstring explicando su efecto
- Fail-fast: Si falta un parámetro en config, el sistema falla inmediatamente

### 2. Semilla Fija
- `RANDOM_SEED = 21` → Resultados reproducibles
- Cambiar seed → Mismos resultados estadísticamente, configs diferentes

### 3. Historial Completo
- Monte Carlo guarda TODAS las 2000 configs probadas
- Bayesian guarda TODAS las 80 iteraciones
- Puedes reconstruir por qué se eligió una config sobre otra

### 4. Transparencia de Penalizaciones
- Función objetivo explícita: `score = 2.5×AWT - 0.15×EPT - penalties`
- Penalizaciones cuadráticas (suaves, no prohibitivas)
- Sin "cajas negras": puedes calcular el score a mano

### 5. Validación por Partner
- Aunque se optimiza a nivel franchise, se valida en cada partner
- Detecta si un partner específico empeora con la config general

---

## 🛠️ Requisitos

```bash
pip install -r requirements.txt
```

**Dependencias principales**:
- `pandas` - Manipulación de datos
- `numpy` - Operaciones numéricas
- `scikit-learn` - Modelos predictivos
- `scikit-optimize` - Bayesian Optimization
- `tqdm` - Barras de progreso

**Python**: 3.8+

---

## 📝 Limitaciones y Futuras Mejoras

### Limitaciones Actuales:
1. **Modelo Lineal**: AWTPredictor usa regresión lineal (R²=0.69)
   - Podría mejorar con Random Forest o XGBoost
   - Trade-off: Interpretabilidad vs. Precisión

2. **Calibración Paramétrica de AWT**: El ajuste por `delta_ept` usa tasa fija (2.5%/min)
   - Es un supuesto operativo razonable, pero no causal puro
   - Futuro: Aprender esta relación directamente de datos históricos por partner/franja

3. **Validación Temporal**: Dataset de 8 días
   - Idealmente, validar con 1-3 meses de datos
   - Confirmar robustez estacional (fin de semana, feriados)

4. **Sin Feedback Loop**: Sistema offline
   - No aprende de activaciones reales en producción
   - Futuro: Integración con A/B testing en vivo

### Posibles Mejoras:
- [ ] Modelos no lineales (Random Forest, Gradient Boosting)
- [ ] Optimización multi-objetivo (Pareto front para AWT vs. EPT)
- [ ] Validación cruzada temporal (train en semana 1-6, test en semana 7-8)
- [ ] Dashboard interactivo (Streamlit/Plotly) para explorar configs
- [ ] Integración con API de BigQuery para datos en tiempo real
- [ ] A/B testing framework para validar en producción

---

## 📞 Soporte y Contacto

**Documentos Adicionales**:
- [`AUDIT_360_REPORT.md`](AUDIT_360_REPORT.md) - Auditoría completa del sistema
- [`src/config.py`](src/config.py) - Todos los parámetros configurables

**Preguntas Frecuentes**:

**P: ¿Por qué 2000 configs Monte Carlo?**  
R: Balance entre cobertura del espacio (10x mejor que 200) y tiempo de ejecución (30-60 min razonable).

**P: ¿Por qué lógica AND y no OR?**  
R: OR activaría HDM demasiado frecuentemente (falsos positivos). AND garantiza estrés multidimensional real.

**P: ¿Qué pasa si un partner tiene muy pocos datos?**  
R: Modo franchise pondera por número de órdenes, así que partners pequeños tienen menos influencia (razonable).

**P: ¿Puedo cambiar los umbrales durante la ejecución?**  
R: No, el pipeline es batch. Debes re-ejecutar con nuevos valores en `config.py`.

---

**Versión**: 2.0  
**Última Actualización**: 2026-03-09  
**Mantenedor**: Equipo de Operaciones / Data Science
