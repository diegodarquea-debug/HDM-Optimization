# HDM Optimization Pipeline

## Traspaso Rapido (Primeros comandos en JupyterHub)

Este es el flujo operativo real usado en JupyterHub para dejar todo listo.

1. Preparar entorno y sincronizar repo:

```bash
pip install scikit-optimize==0.10.2
cd /home/jovyan/projects/HDM-Optimization
source ./setup.sh
git restore outputs/franchise_impact_by_partner.csv outputs/franchise_optimal_config.csv outputs/optimization_history.csv
git pull origin main
```

2. Ejecutar test/corrida principal:

```bash
bash ./run.sh NIU AA 2026-02-03 2026-03-23 "1,2,3,4,5,6,7"
```

3. Verificar graficos y resumen generados:

```bash
ls outputs/global_test_timeline_*.png
ls outputs/latest_run_summary.json
```

4. Ver graficos dentro de un Notebook (copiar en una celda):

```python
from pathlib import Path
from IPython.display import Image, display

outputs = Path("/home/jovyan/projects/HDM-Optimization/outputs")
latest_ptr = outputs / "latest_run_path.txt"
base = Path(latest_ptr.read_text().strip()) if latest_ptr.exists() else outputs

pngs = sorted(base.glob("global_test_timeline_*.png"))

print("Base usada:", base)
if not pngs:
    print("No se encontraron PNG de timeline.")
else:
    for p in pngs:
        day = p.stem.replace("global_test_timeline_", "").capitalize()
        print(f"\n=== {day} ===")
        display(Image(filename=str(p), width=1400))
```

El comando de `run.sh` genera automaticamente los PNG de timeline por dia y el archivo `run_summary.json` para compartir resultados.

Pipeline de optimizacion de HDM para franquicias, con foco en equilibrio entre:

- Mejora de AWT (tiempo de espera)
- Control de incremento de EPT
- Recomendaciones operables (evitar HDM casi nunca activo o excesivo)

Para documentacion extendida del enfoque y negocio, ver `DOCUMENTATION.md`.

## Estado Actual (Marzo 2026)

Esta version incluye:

- Timeout BigQuery por defecto en 1200s
- Heartbeat de jobs BigQuery (progreso cada 30s con `job_id`)
- Filtro de dias configurable por CLI (`--days-of-week`)
- Generacion de timeline global por dia (Lunes a Domingo)
- Nuevo artefacto integral `run_summary.json` con todo lo usado en la simulacion

## Ejemplos de Ejecucion

Ejecucion base:

```bash
bash ./run.sh KFC AAA 2026-02-16 2026-03-01
```

Ejecucion indicando dias especificos:

```bash
bash ./run.sh KFC AAA 2026-02-16 2026-03-01 "lun,mie,vie"
bash ./run.sh KFC AAA 2026-02-16 2026-03-01 "1,6,7"
```

Convencion de dias BigQuery: `1=Sunday ... 7=Saturday`.

## Uso de `run.sh`

```bash
./run.sh <FRANCHISE> <GRADE> <START_DATE> <END_DATE> [DAYS_OF_WEEK]
```

Parametros:

- `FRANCHISE`: nombre de franquicia
- `GRADE`: `AAA`, `AA` o `A`
- `START_DATE`, `END_DATE`: formato `YYYY-MM-DD`
- `DAYS_OF_WEEK` (opcional): `all` | `1-7` | `mon,tue,...` | `lun,mar,...`

Si no envias `DAYS_OF_WEEK`, usa `all`.

## Configuracion de Modelo y Optimizacion

Fuente unica de verdad: `src/config.py`.

Valores relevantes actuales:

- `N_SIMULATIONS = 2000`
- `N_OPTIMIZATION_CALLS = 80`
- `TRAIN_TEST_SPLIT = 0.6`
- `MAX_EPT_INCREASE = 0.50`
- `THRESHOLDS`: `u1(3,10)`, `u2(1,3)`, `u3(3,10)`, `delta_ept[2,4,6,8,10]`, `duracion_hdm(10,20)`
- `OBJECTIVE_WEIGHTS`: `awt=2.5`, `ept_penalty=0.20`

## Salidas Generadas

Salida consolidada (se guarda en `outputs/` y tambien en `outputs/runs/<timestamp>_run/`):

- `franchise_optimal_config.csv`
- `franchise_impact_by_partner.csv`
- `optimization_history.csv`
- `test_timeline_global_equilibrada.csv`
- `global_test_timeline_<dia>.png` (ej: `lunes`, `martes`, ...)
- `run_summary.json`  <-- resumen integral de la corrida

Punteros rapidos:

- `outputs/latest_run_path.txt`
- `outputs/latest_run_summary.json`

## Que contiene `run_summary.json`

Archivo pensado para compartir una corrida completa sin reenviar multiples CSVs.

Incluye:

- `run_metadata`: parametros de entrada (franquicia, grade, fechas, dias, pais, entidad)
- `simulation_config`: configuracion completa usada por la corrida
- `clusters[]`:
	- `data_summary`
	- `baseline_metrics`
	- `model_quality` (R2, RMSE, MAE y split metadata)
	- `best_config`
	- `strategies` (`Agresiva`, `Equilibrada`, `Conservadora`)
	- `franchise_evaluation` agregada
	- `partner_impact` por partner

Ejemplo en notebook:

```python
import json
from pathlib import Path

summary = json.loads(Path("outputs/latest_run_summary.json").read_text())
global_cluster = next(c for c in summary["clusters"] if c["cluster"] == "Global")
global_cluster["strategies"]
```

## Variables de Entorno Utiles

Infra:

- `GOOGLE_CLOUD_PROJECT` o `GCP_PROJECT_ID`
- `BQ_LOCATION`
- `BQ_TIMEOUT_SECONDS`

Performance:

- `N_SIMULATIONS`
- `N_OPTIMIZATION_CALLS`
- `HDM_SIM_N_JOBS`

Si no defines overrides, se usan defaults de `src/config.py`.

## Troubleshooting Rapido

- Error de 0 filas en query:
	- Revisar filtros de `franchise`, `grade`, fechas y dias (`--days-of-week`)
- Timeout en BigQuery:
	- Aumentar `BQ_TIMEOUT_SECONDS` (ej. `1800`)
	- Usar `job_id` logueado para inspeccion en consola BigQuery
- No aparecen PNGs de ciertos dias:
	- Verificar que existan filas en test slice para esos dias

## Tests

Suite vigente:

```bash
python -m unittest discover -s tests -q
```

## Owner

- Operaciones + Data Science
- Actualizado: 2026-03-25
