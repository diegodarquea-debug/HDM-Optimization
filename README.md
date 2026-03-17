# HDM Optimization Pipeline

Documentacion tecnica y ejecutiva del sistema de optimizacion de HDM (Hora de Mayor Demanda) para franquicias.

## Resumen Ejecutivo

Este pipeline busca el mejor balance entre:

- Reducir AWT (tiempo de espera del cliente)
- Controlar el aumento de EPT (tiempo prometido)
- Evitar recomendaciones operativamente inutiles (HDM casi nunca activo) o excesivas (HDM casi siempre activo)

La salida principal es una configuracion recomendada de cinco parametros:

- u1: umbral de ordenes pendientes
- u2: umbral de riders cercanos
- u3: umbral de espera maxima
- delta_ept: minutos extra de EPT durante HDM
- duracion_hdm: duracion de HDM por activacion

---

## Novedades Clave (Marzo 2026)

Se actualizaron reglas para hacer el resultado mas realista para produccion:

1. Tope de EPT mucho mas estricto:
- MAX_EPT_INCREASE paso de 15.0 a 0.50 minutos

2. Penalizacion por activacion HDM extremadamente baja:
- Si HDM esta activo menos de 5% del tiempo, el optimizador penaliza esa configuracion
- Se agrego una penalizacion cuadratica fuerte para evitar recomendaciones degeneradas con tasa cercana a 0%

3. Calibracion de sensibilidad AWT/EPT:
- ept_penalty subio de 0.10 a 0.20
- awt_delta_ept_reduction_per_min bajo de 0.03 a 0.015
- delta_ept ahora explora [2, 4, 6, 8, 10]
- u3 ahora explora (3, 10)

4. Coherencia total entre etapas:
- Monte Carlo y Bayesian usan la misma logica de score y penalizaciones
- La estrategia Equilibrada se selecciona por objective_score

---

## Como Decide el Optimizador

El sistema minimiza una perdida total:

- weighted_gain = (awt_weight * awt_improvement) - (ept_penalty * ept_increase) - rate_penalty
- total_loss = -weighted_gain + penalty_terms

Donde:

- awt_improvement: mejora en minutos versus baseline
- ept_increase: aumento en minutos versus baseline
- rate_penalty: penaliza sobre-activacion y sub-activacion

### Penalizaciones activas

1. Empeorar AWT
- awt_worse_quad = 50

2. Empeorar tiempo combinado (AWT + EPT)
- combined_worse_quad = 30

3. Superar tope de EPT
- ept_excess_quad = 10
- Tope actual: MAX_EPT_INCREASE = 0.50 min

4. HDM casi nunca activo
- hdm_rate_min_threshold = 0.05
- hdm_rate_low_coeff = 500.0

5. HDM demasiado activo
- hdm_rate_threshold = 0.25
- hdm_rate_excess_coeff = 20.0

---

## Configuracion Actual (Fuente Unica de Verdad)

Toda la configuracion vive en src/config.py.

### Espacio de busqueda

- u1: (3, 10)
- u2: (1, 3)
- u3: (3, 10)
- delta_ept: [2, 4, 6, 8, 10]
- duracion_hdm: (10, 20)

### Efecto de HDM sobre AWT

- awt_delta_ept_reduction_per_min: 0.015
- awt_delta_ept_max_reduction: 0.30

### Funcion objetivo

- awt: 2.5
- ept_penalty: 0.20

### Restricciones operativas

- MAX_EPT_INCREASE: 0.50
- ACTIVATION_DELAY_MINUTES: 2

### Modelo

- MODEL_TYPE: "random_forest"
- Opciones soportadas: "linear_regression", "random_forest", "decision_tree"
- TRAIN_TEST_SPLIT: 0.60
- RANDOM_SEED: 42

---

## Flujo del Pipeline

1. Carga de datos
- Desde BigQuery o CSV
- Filtros por franquicia, grade, pais, entidad y fecha

2. Baseline
- Calcula AWT y EPT de referencia sin intervencion

3. Entrenamiento de modelos
- AWTPredictor y EPTPredictor

4. Exploracion Monte Carlo
- Evalua N_SIMULATIONS configuraciones aleatorias

5. Optimizacion Bayesiana
- Toma seeds del Monte Carlo y refina en N_OPTIMIZATION_CALLS iteraciones

6. Estrategias finales
- Agresiva, Equilibrada (default), Conservadora

7. Export de resultados
- CSVs en outputs/

---

## Estrategias de Salida

1. Agresiva
- Maximiza awt_improvement

2. Equilibrada (principal)
- Maximiza objective_score (alineado al objetivo real)

3. Conservadora
- Entre candidatos con mejora minima de AWT, minimiza tasa de activacion

---

## Ejecucion en JupyterHub

## 1) Actualizar codigo

```bash
git pull --ff-only origin main
```

## 2) Preparar entorno

```bash
source ./setup.sh
```

## 3) Ejecutar segun perfil

### Perfil base (usa defaults de config.py)

```bash
bash ./run.sh KFC AAA 2026-02-16 2026-03-01
```

### Perfil validacion

```bash
bash ./run_realistic.sh KFC AAA 2026-02-16 2026-03-01
```

### Perfil produccion

```bash
bash ./run_prod.sh KFC AAA 2026-02-16 2026-03-01
```

Importante:

- run.sh no fuerza valores smoke. Si no defines variables, usa config.py.
- run_realistic.sh y run_prod.sh si setean perfiles por defecto (pero se pueden overridear por env var).

---

## Variables de Entorno Relevantes

Infraestructura:

- GOOGLE_CLOUD_PROJECT
- GCP_PROJECT_ID
- BQ_LOCATION
- BQ_TIMEOUT_SECONDS

Performance:

- N_SIMULATIONS
- N_OPTIMIZATION_CALLS
- HDM_SIM_N_JOBS

Si no defines N_SIMULATIONS/N_OPTIMIZATION_CALLS en run.sh, se usan los valores de src/config.py.

---

## Salidas Principales

- outputs/monte_carlo_franchise_exploration.csv
- outputs/optimization_history.csv
- outputs/franchise_optimal_config.csv
- outputs/franchise_impact_by_partner.csv

Tambien se versionan corridas en outputs/runs/ y se guarda un puntero en outputs/latest_run_path.txt.

---

## Guia Rapida de Lectura de Resultados

1. Revisar franchise_optimal_config.csv
- Verificar que Equilibrada tenga awt_improvement > 0
- Verificar ept_increase <= 0.50
- Verificar hdm_activation_rate en rango razonable (idealmente entre 0.05 y 0.25)

2. Revisar optimization_history.csv
- Buscar convergencia del objective_score
- Confirmar que no se concentre en recomendaciones con activacion casi cero

3. Revisar impact_by_partner
- Detectar partners con mejora negativa para analisis por segmento

---

## Gobierno y Auditoria

El sistema es auditable porque:

- Configuracion centralizada en src/config.py
- Semilla fija para reproducibilidad
- Historial completo de Monte Carlo y Bayesian
- Corridas versionadas por timestamp

---

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales:

- pandas
- numpy
- scikit-learn
- scikit-optimize
- google-cloud-bigquery
- tqdm
- joblib

---

## Nota para Presentacion

Mensaje clave para negocio:

"El modelo ya no optimiza solo por bajar AWT a cualquier costo. Ahora exige control estricto de EPT y descarta configuraciones que casi no activan HDM, logrando recomendaciones mas operables y defendibles en produccion."

---

Actualizado: 2026-03-16
Owner: Operaciones + Data Science
