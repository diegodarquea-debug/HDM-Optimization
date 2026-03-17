# Documentation

## 1) Que hace este proyecto

Este proyecto optimiza la activacion de HDM a nivel franquicias.

Objetivo operativo:

- Bajar AWT (tiempo de espera real del cliente)
- Controlar EPT (tiempo prometido)
- Evitar recomendaciones poco utiles (HDM casi nunca activo) o excesivas (HDM casi siempre activo)

El pipeline devuelve una configuracion optima con 5 parametros:

- u1
- u2
- u3
- delta_ept
- duracion_hdm

---

## 2) Data de entrada: que viene y que significa

### Columnas requeridas (si faltan, el pipeline falla)

| Columna | Tipo esperado | Uso |
|---|---|---|
| momento_exacto | datetime | Orden temporal de la simulacion minuto a minuto |
| partner_id | int/string | Segmentacion por partner/franquicia |
| ordenes_pendientes | numerico | Condicion u1 y feature de modelos |
| riders_cerca | numerico | Condicion u2 y feature de modelos |
| hdm_activo | 0/1 | Feature historica para entrenamiento |
| max_awt_espera_min | numerico | Condicion u3 y baseline principal de AWT |

### Columnas opcionales (mejoran estimacion)

| Columna | Uso |
|---|---|
| ept_promedio_min | Base de EPT para simulacion |
| ept_promedio_min_smoothed | EPT suavizado para entrenamiento y baseline |
| ept_promedio | Fallback de EPT si no hay smoothed/min |
| ept_configurado_min | Ultimo fallback de EPT |
| awt_promedio | Target alternativo de AWT si esta disponible |
| partner_name | Etiqueta para reporte por partner |

---

## 3) Etapas del pipeline (de punta a punta)

## Etapa A: Ingestion de datos

Archivo principal: src/data_loader.py

1. Decide origen de datos:
- auto: BigQuery si hay configuracion completa, si no CSV
- csv: lee data/raw_data.csv (o ruta enviada)
- bigquery: ejecuta SQL (por archivo o query default)

2. Si es BigQuery:
- inyecta parametros tipados (DATE, INT64, STRING, ARRAY)
- ejecuta query con timeout y location configurables

3. Aplica filtros por modo:
- franchise: filtra por rango de fechas
- partner: filtra por partner_id y fechas

Impacto de parametros en esta etapa:

- DATA_SOURCE define origen (auto/csv/bigquery)
- BQ_TIMEOUT_SECONDS puede cortar query lenta
- BQ_DEFAULT_LOOKBACK_DAYS aplica solo en query default sin archivo SQL

---

## Etapa B: Preprocesamiento

Archivo principal: src/data_loader.py -> preprocess_data

1. Valida columnas requeridas.
2. Completa nulos:
- ordenes_pendientes, riders_cerca, max_awt_espera_min -> 0
- hdm_activo -> 0 y cast a int
3. Fuerza tipos numericos en columnas EPT/AWT opcionales.
4. Logica EPT:
- si existe ept_promedio_min, pone EPT=0 cuando ordenes=0
- recorta outliers a p99 en filas activas
- crea ept_promedio_min_smoothed con rolling window=5
- setea ept_promedio = ept_promedio_min_smoothed
5. Ordena por momento_exacto.

Impacto:

- Este paso evita ruido extremo de EPT y estabiliza entrenamiento/simulacion.

---

## Etapa C: Baseline (sin intervencion)

Archivo principal: src/analytics.py

Calcula metricas base de negocio:

- AWT promedio, p50, p95
- Ordenes y riders promedio, p50, p95
- Tasa historica de hdm_activo
- EPT promedio/p95 para columnas disponibles

Estas metricas son la referencia para:

- awt_improvement = awt_baseline - awt_simulado
- ept_increase = ept_simulado - ept_baseline

---

## Etapa D: Entrenamiento de modelos

Archivo principal: src/model.py

Se entrenan dos modelos:

1. AWTPredictor
- Features base: [ordenes_pendientes, riders_cerca, hdm_activo]
- Agrega feature EPT si existe una de estas columnas (prioridad):
  ept_promedio_min_smoothed -> ept_promedio_min -> ept_promedio
- Target:
  - awt_promedio si existe y tiene media > 0
  - si no, max_awt_espera_min

2. EPTPredictor
- Features: [ordenes_pendientes, riders_cerca, hdm_activo]
- Target EPT con prioridad:
  ept_promedio_min_smoothed -> ept_promedio_min -> ept_promedio -> ept_configurado_min
- Si no hay columnas EPT validas, usa baseline heuristico

Configuracion clave:

- MODEL_TYPE: linear_regression | random_forest | decision_tree
- TRAIN_TEST_SPLIT: proporcion train/test
- MODEL_SETTINGS: hiperparametros RF/DT

---

## Etapa E: Simulacion de una configuracion HDM

Archivo principal: src/simulator.py

Para una configuracion (u1, u2, u3, delta_ept, duracion_hdm):

1. Trigger estricto AND:
- activa si ordenes >= u1 AND riders >= u2 AND max_awt >= u3

2. Delay operativo:
- al trigger entra en cola
- aplica efecto recien luego de ACTIVATION_DELAY_MINUTES

3. Duracion:
- una vez activo dura duracion_hdm
- si vuelve a triggerear estando activo, extiende fin

4. EPT simulado:
- cuando HDM esta activo: ept_with_hdm = ept_base + delta_ept
- cuando no esta activo: ept_with_hdm = ept_base

5. AWT simulado:
- predice AWT con modelo usando hdm_active_sim y ept_with_hdm
- aplica ajuste de calibracion cuando HDM esta activo:
  hdm_factor = max(1 - awt_delta_ept_max_reduction,
                   1 - awt_delta_ept_reduction_per_min * delta_ept)
  awt_ajustado = awt_predicho * hdm_factor

6. Metricas de salida:
- awt_improvement
- ept_increase
- combined_improvement
- hdm_activation_rate
- awt_p50 / awt_p95

Impacto de umbrales aqui:

- u1/u2/u3 controlan frecuencia de activacion
- delta_ept controla intensidad EPT y factor de reduccion AWT
- duracion_hdm controla permanencia de efecto

---

## Etapa F: Monte Carlo (exploracion)

Archivos: src/simulator.py + main.py

1. Genera N_SIMULATIONS configuraciones aleatorias dentro de THRESHOLDS.
2. Evalua cada configuracion con simulador.
3. En main.py calcula objective_score para rankear seeds con la misma logica del optimizador:

- weighted_gain = awt_weight * awt_improvement - ept_penalty * ept_increase - rate_penalty
- soft_penalty incluye:
  - awt_worse_quad si awt_improvement < 0
  - combined_worse_quad si combined_improvement < 0
  - ept_excess_quad si ept_increase > MAX_EPT_INCREASE
- objective_score = weighted_gain - soft_penalty

4. Toma top_k (BAYESIAN_SETTINGS.mc_seed_top_k) para warm-start de Bayesian.

Impacto de parametros:

- N_SIMULATIONS mayor = mas cobertura del espacio
- top_k mayor = mas diversidad inicial para Bayesian

---

## Etapa G: Optimizacion Bayesiana (refinamiento)

Archivo: src/optimizer.py

1. Define espacio de busqueda con THRESHOLDS.
2. Ejecuta gp_minimize o forest_minimize por N_OPTIMIZATION_CALLS.
3. En cada evaluacion usa objective_function:

- weighted_gain = awt_weight * awt_improvement - ept_penalty * ept_increase - rate_penalty

rate_penalty tiene dos lados:

- Penaliza activacion demasiado baja:
  si hdm_rate < hdm_rate_min_threshold
- Penaliza sobreactivacion:
  si hdm_rate > hdm_rate_threshold

penalty_terms adicionales:

- awt_worse_quad si awt_improvement < 0
- combined_worse_quad si combined_improvement < 0
- ept_excess_quad si ept_increase > MAX_EPT_INCREASE

Resultado interno:

- total_loss = -weighted_gain + penalty_terms
- el optimizador minimiza total_loss

---

## Etapa H: Seleccion de estrategias y resultado final

Archivo: src/optimizer.py -> get_top_3_strategies

Sobre optimization_history:

- filtra primero por ept_increase <= MAX_EPT_INCREASE (si hay candidatos)

Estrategias:

1. Agresiva
- max awt_improvement

2. Equilibrada (principal)
- max objective_score (misma logica del objetivo)

3. Conservadora
- entre candidatos con mejora minima de AWT, elige menor hdm_activation_rate

Luego main.py guarda:

- optimization_history.csv
- franchise_optimal_config.csv
- franchise_impact_by_partner.csv

---

## 4) Como afectan pesos, umbrales y limites

| Parametro | Donde impacta | Efecto directo |
|---|---|---|
| u1 | Trigger en simulacion | Subirlo reduce activaciones; bajarlo aumenta activaciones |
| u2 | Trigger en simulacion | Subirlo exige mas riders; bajarlo activa en escenarios mas escasos |
| u3 | Trigger en simulacion | Subirlo interviene tarde; bajarlo interviene temprano |
| delta_ept | EPT simulado y factor AWT | Subirlo aumenta ept_increase y puede mejorar awt_improvement |
| duracion_hdm | Estado HDM activo | Subirlo prolonga efecto (mas impacto acumulado) |
| awt (peso) | objective_score | Subirlo prioriza bajar AWT aunque cueste mas EPT |
| ept_penalty (peso) | objective_score | Subirlo castiga mas aumento de EPT |
| MAX_EPT_INCREASE | Penalizacion de exceso + filtro estrategias | Limite de seguridad de EPT promedio permitido |
| hdm_rate_min_threshold | rate_penalty | Obliga activacion minima util |
| hdm_rate_low_coeff | rate_penalty | Intensidad del castigo por activacion baja |
| hdm_rate_threshold | rate_penalty | Techo de activacion razonable |
| hdm_rate_excess_coeff | rate_penalty | Intensidad del castigo por sobreactivacion |
| awt_delta_ept_reduction_per_min | Ajuste AWT en simulador | Pendiente de mejora de AWT por cada minuto de delta_ept |
| awt_delta_ept_max_reduction | Ajuste AWT en simulador | Tope maximo de reduccion por calibracion |
| N_SIMULATIONS | Monte Carlo | Mas exploracion inicial |
| N_OPTIMIZATION_CALLS | Bayesian | Mas refinamiento local |

---

## 5) Como leer una recomendacion final

Ejemplo de log:

Global optimum | u1=9 | u2=2 | u3=3 | delta_ept=8.0 | duracion_hdm=20 | awt_improvement=0.082 | ept_increase=0.297

Interpretacion:

- u1/u2/u3: regla de activacion
- delta_ept/duracion_hdm: intensidad y permanencia de HDM
- awt_improvement > 0: mejora de espera real
- ept_increase > 0: costo en promesa de tiempo

En este framework la decision final la manda objective_score, no solo combined_improvement.

---


## 6) Referencias de implementacion

- Configuracion: src/config.py
- Ingestion/preproceso: src/data_loader.py
- Baseline: src/analytics.py
- Modelos: src/model.py
- Simulador: src/simulator.py
- Optimizador: src/optimizer.py
- Orquestacion y outputs: main.py

---

Actualizado: 2026-03-17
