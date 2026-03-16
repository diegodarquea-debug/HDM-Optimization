-- BigQuery input template for HDM Optimization (JupyterHub version)
-- Parameters expected by the loader:
--   @start_date (DATE, optional)
--   @end_date   (DATE, optional)
--   @partner_ids (ARRAY<INT64>, optional)
-- 
-- UPDATED to use ept_promedio_min (actual kitchen WIP) instead of ept_configurado_min

SELECT
  momento_exacto,
  partner_id,
  partner_name,
  ordenes_pendientes,
  riders_cerca,
  hdm_activo,
  max_awt_espera_min,
  ept_promedio,
  ept_promedio_min,
  awt_promedio
FROM `YOUR_PROJECT.YOUR_DATASET.YOUR_TABLE`
WHERE (@start_date IS NULL OR DATE(momento_exacto) >= @start_date)
  AND (@end_date IS NULL OR DATE(momento_exacto) <= @end_date)
  AND (ARRAY_LENGTH(@partner_ids) = 0 OR SAFE_CAST(partner_id AS INT64) IN UNNEST(@partner_ids))
ORDER BY momento_exacto;