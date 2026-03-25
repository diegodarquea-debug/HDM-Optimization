# Documentation

## 0) Comandos de Traspaso (JupyterHub)

Ejecutar estos comandos al inicio para garantizar que todo funcione en la version correcta:

```bash
cd /home/jovyan/projects/HDM-Optimization
git pull --ff-only origin main
source ./setup.sh
```

Comando principal para correr la simulacion y generar los graficos que interesan:

```bash
bash ./run.sh KFC AAA 2026-02-16 2026-03-01 "all"
```

Verificacion rapida de artefactos clave:

```bash
ls outputs/global_test_timeline_*.png
ls outputs/latest_run_summary.json
```

Notas:

- `run.sh` genera automaticamente recomendacion, CSVs, PNGs de timeline y `run_summary.json`.
- Si se quieren dias especificos, usar el quinto parametro, por ejemplo: `"lun,mie,vie"` o `"1,6,7"`.

---

## 1) Resumen

El proyecto busca la mejor configuracion de HDM para una franquicia.

Que optimiza:

- Mejorar AWT
- Controlar EPT
- Evitar soluciones inutiles (HDM casi nunca activo) o excesivas (HDM casi siempre activo)

Salida final del pipeline:

- u1
- u2
- u3
- delta_ept
- duracion_hdm
- metricas de impacto (awt_improvement, ept_increase, hdm_activation_rate)

---

## 2) Data de entrada

### 1.1 Que data entra

Columnas obligatorias:

| Columna | Para que se usa |
|---|---|
| momento_exacto | Orden temporal para simular minuto a minuto |
| partner_id | Separar datos por partner/franquicia |
| ordenes_pendientes | Condicion u1 y feature de modelos |
| riders_cerca | Condicion u2 y feature de modelos |
| hdm_activo | Estado historico HDM para entrenamiento |
| max_awt_espera_min | Condicion u3 y baseline principal de AWT |

Columnas opcionales (mejoran precision):

| Columna | Para que se usa |
|---|---|
| ept_promedio_min / smoothed / ept_promedio | Baseline de EPT, feature y simulacion de EPT |
| ept_configurado_min | Fallback de EPT |
| awt_promedio | Target alternativo de AWT |
| partner_name | Etiqueta en reportes |

### 1.2 Que pasa si falta data

- Si falta una columna obligatoria: el pipeline falla con error explicito.
- Si faltan columnas EPT opcionales: usa fallback, pero con menor fidelidad.

---

## 3) Etapa A - Ingestion

Archivo principal: src/data_loader.py

### Que entra

- Fuente de datos: auto/csv/bigquery
- Parametros de fecha, franquicia, grade, etc.

### Que se hace

1. Decide origen:
- auto -> BigQuery si hay config completa; si no, CSV
- csv -> lee archivo local
- bigquery -> ejecuta SQL parametrizado

2. En BigQuery:
- tipa parametros (DATE, INT64, STRING, ARRAY)
- ejecuta query con timeout y location

3. Filtra por modo:
- franchise -> por ventana de fechas
- partner -> por partner + fechas

### Que sale

- DataFrame limpio para preprocesar

### Parametros que mandan

- DATA_SOURCE
- BQ_TIMEOUT_SECONDS
- BQ_LOCATION
- BQ_DEFAULT_LOOKBACK_DAYS (solo query default)

"Primero garantizamos que estamos leyendo exactamente el universo correcto de datos y fechas antes de modelar cualquier cosa."

---

## 4) Etapa B - Preprocesamiento

Archivo principal: src/data_loader.py -> preprocess_data

### Que entra

- DataFrame crudo de ingesta

### Que se hace

1. Validacion de columnas requeridas.
2. Limpieza de nulos:
- ordenes/riders/awt -> 0
- hdm_activo -> 0 o 1
3. Cast de columnas numericas relevantes.
4. Tratamiento EPT:
- EPT=0 cuando ordenes=0
- recorte de outliers a p99 en filas activas
- suavizado rolling (window=5)
5. Orden temporal por momento_exacto.

### Que sale

- DataFrame consistente, ordenado y menos ruidoso

### Parametros que mandan

- Logica de preproceso definida en codigo (no expuesta como flags)

"Antes de entrenar, reducimos ruido y estandarizamos la data para que el modelo no aprenda patrones falsos por outliers o nulos."

---

## 5) Etapa C - Baseline

Archivo principal: src/analytics.py

### Que entra

- DataFrame preprocesado

### Que se hace

Calcula el punto de partida sin intervencion:

- AWT promedio, p50, p95
- EPT promedio/p95
- ordenes/riders
- tasa historica de HDM

### Que sale

- baseline_metrics

### Por que es critico

Todas las mejoras se miden contra este baseline:

- awt_improvement = awt_baseline - awt_simulado
- ept_increase = ept_simulado - ept_baseline

"Si no definimos bien el baseline, no podemos afirmar que una recomendacion mejora o empeora."

---

## 6) Etapa D - Entrenamiento de modelos

Archivo principal: src/model.py

### Que entra

- DataFrame preprocesado
- Config de modelo desde src/config.py

### Que se hace

Se entrenan dos modelos:

1. AWTPredictor
- Features: ordenes_pendientes, riders_cerca, hdm_activo (+ EPT agregado por trigger)
- Target: awt_promedio o max_awt_espera_min

2. EPTPredictor
- Features: ordenes_pendientes, riders_cerca, hdm_activo

### Que sale

- awt_predictor entrenado
- ept_predictor entrenado
- metricas de entrenamiento/test

### Parametros que mandan

- MODEL_TYPE: random_forest, linear_regression, decision_tree
- TRAIN_TEST_SPLIT --> Que % de la data se usa para entrenar y que % para testear
- MODEL_SETTINGS (n_estimators, max_depth, etc.)

"Los modelos aprenden como cambian AWT y EPT segun carga, riders y estado HDM. Eso permite simular escenarios que no vimos exactamente en el historico."

---

## 7) Etapa E - Simulacion de escenarios HDM

Archivo principal: src/simulator.py

### Que entra

- Modelos entrenados
- baseline_metrics
- Una configuracion candidata: u1, u2, u3, delta_ept, duracion_hdm

### Que se hace

1. Regla de activacion (AND estricto):
- activa solo si ordenes >= u1 y riders >= u2 y awt >= u3

2. Delay operativo:
- hay cola de activacion
- efecto real aplica despues de ACTIVATION_DELAY_MINUTES

3. Duracion de estado:
- una activacion dura duracion_hdm
- nuevos triggers extienden la duracion

4. Simulacion EPT:
- HDM activo -> ept_base + delta_ept
- HDM inactivo -> ept_base

5. Simulacion AWT:
- predice con modelo
- aplica calibracion por delta_ept:
  hdm_factor = max(1 - awt_delta_ept_max_reduction,
                   1 - awt_delta_ept_reduction_per_min * delta_ept)

### Que sale

Metricas por configuracion:

- awt_improvement
- ept_increase
- combined_improvement
- hdm_activation_rate
- awt_p50, awt_p95

### Parametros que mandan

- THRESHOLDS (u1,u2,u3,delta_ept,duracion_hdm)
- ACTIVATION_DELAY_MINUTES
- HDM_EFFECT_SETTINGS

"Esta etapa responde: si aplicabamos esta politica de HDM sobre la historia real, cual habria sido el impacto en espera y EPT."

---

## 8) Etapa F - Monte Carlo (exploracion amplia)

Archivos: src/simulator.py + main.py

### Que entra

- Espacio THRESHOLDS
- N_SIMULATIONS

### Que se hace

1. Genera N_SIMULATIONS configuraciones aleatorias.
2. Simula cada una.
3. Calcula objective_score para rankear seeds de forma consistente con el optimizador:

- weighted_gain = awt*w_awt - ept*w_ept - rate_penalty
- objective_score = weighted_gain - soft_penalty

4. Toma top_k para warm-start de Bayesian.

### Que sale

- monte_carlo_franchise_exploration.csv
- Lista de seeds x0 para optimizacion

### Parametros que mandan

- N_SIMULATIONS
- BAYESIAN_SETTINGS.mc_seed_top_k
- OBJECTIVE_WEIGHTS --> Peso que se le da a ganar awt y ganar ept
- OPTIMIZER_PENALTIES --> Castigo por sobre/sub activación 

"Monte Carlo barre el mapa completo para no arrancar la optimizacion en una zona ciega."

---

## 9) Etapa G - Optimizacion Bayesiana (refinamiento)

Archivo principal: src/optimizer.py

### Que entra

- Seeds del Monte Carlo
- Espacio THRESHOLDS
- N_OPTIMIZATION_CALLS

### Que se hace

1. Evalua configuraciones con objective_function.
2. Minimiza total_loss:

- weighted_gain = awt_weight*awt_improvement - ept_penalty*ept_increase - rate_penalty --> Función Objetivo
- total_loss = -weighted_gain + penalty_terms

3. Penaliza escenarios indeseados:

- awt_improvement < 0
- combined_improvement < 0
- ept_increase > MAX_EPT_INCREASE --> Ej. EPTi: 12min, EPTf: 12.4min, entonces 0.4<0.5 todo ok!
- hdm_activation_rate demasiado baja o demasiado alta

### Que sale

- optimization_history.csv
- best_config segun objective_score/total_loss

### Parametros que mandan

- OBJECTIVE_WEIGHTS
- OPTIMIZER_PENALTIES
- MAX_EPT_INCREASE
- N_OPTIMIZATION_CALLS

"Bayesian no prueba todo: aprende en que zonas hay mejores candidatos y concentra ahi el presupuesto de evaluacion."

---

## 10) Etapa H - Seleccion de estrategia final y reportes

Archivos: src/optimizer.py + main.py

### Que entra

- optimization_history completo

### Que se hace

1. Filtra candidatos validos por EPT cap.
2. Construye 3 estrategias:

- Agresiva: max awt_improvement
- Equilibrada: max objective_score (estrategia principal)
- Conservadora: menor hdm_activation_rate con piso de mejora AWT

3. Evalua impacto por partner y guarda CSVs.

### Que sale

- franchise_optimal_config.csv
- franchise_impact_by_partner.csv
- optimization_history.csv

"No entregamos un unico numero: entregamos alternativa agresiva, equilibrada y conservadora, y elegimos por defecto la equilibrada por consistencia con el objetivo de negocio."

---

## 11) Tabla maestra: impacto de cada parametro

| Parametro | Donde pega | Si sube | Si baja |
|---|---|---|---|
| u1 | Trigger HDM | Activa menos (mas conservador) | Activa mas (mas agresivo) |
| u2 | Trigger HDM | Exige mas riders para activar | Activa con menos riders |
| u3 | Trigger HDM | Interviene mas tarde | Interviene mas temprano |
| delta_ept | EPT y ajuste AWT | Mas impacto en AWT, mas costo EPT | Menor impacto y menor costo |
| duracion_hdm | Estado activo | Efecto dura mas | Efecto dura menos |
| awt (peso) | Objective score | Prioriza bajar espera real | Pierde prioridad AWT |
| ept_penalty | Objective score | Castiga mas subir EPT | Permite mas costo EPT |
| MAX_EPT_INCREASE | Restriccion de seguridad | Endurece candidatos validos | Relaja candidatos validos |
| hdm_rate_min_threshold | Penalizacion de baja activacion | Exige mas activacion minima | Tolera activacion casi nula |
| hdm_rate_low_coeff | Fuerza del castigo bajo | Castigo mas fuerte | Castigo mas suave |
| hdm_rate_threshold | Techo de activacion | Penaliza antes sobreactivar | Permite mas activacion |
| hdm_rate_excess_coeff | Fuerza del castigo alto | Castigo mas fuerte por sobreactivar | Castigo mas suave |
| awt_delta_ept_reduction_per_min | Ajuste de simulacion AWT | Hace mas potente delta_ept | Lo hace menos potente |
| awt_delta_ept_max_reduction | Tope de reduccion AWT | Permite mayor techo de mejora | Limita techo de mejora |
| N_SIMULATIONS | Monte Carlo | Mas exploracion global | Menos cobertura del espacio |
| N_OPTIMIZATION_CALLS | Bayesian | Mas refinamiento local | Menos refinamiento |

---

## 12) Como leer un resultado final

Ejemplo:

Global optimum | u1=9 | u2=2 | u3=3 | delta_ept=8.0 | duracion_hdm=20 | awt_improvement=0.082 | ept_increase=0.297

Lectura correcta:

1. Regla operativa: u1/u2/u3 define cuando entra HDM.
2. Intensidad: delta_ept + duracion_hdm define cuanto y por cuanto tiempo intervenimos.
3. Beneficio: awt_improvement > 0 mejora espera real.
4. Costo: ept_increase > 0 sube promesa de tiempo.
5. Decision final: la manda objective_score, no solo combined_improvement.

---

## 13) Referencias de implementacion

- Configuracion: src/config.py
- Ingestion y preproceso: src/data_loader.py
- Baseline: src/analytics.py
- Modelos: src/model.py
- Simulador: src/simulator.py
- Optimizador: src/optimizer.py
- Orquestacion y outputs: main.py

---

Actualizado: 2026-03-17
