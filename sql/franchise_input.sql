-- BigQuery backtesting query for HDM Optimization (Franquicia + Grade Integration).
-- Parameters expected by the loader:
--   @start_date (DATE, required)
--   @end_date (DATE, required)
--   @target_franchise (STRING, required)
--   @target_grade (STRING, required)
--   @target_country_id (INT64, optional, default suggested: 2 for Chile)
--   @target_entity_id (STRING, optional, default: PY_CL)
--   @target_days_of_week (ARRAY<INT64>, optional, BigQuery: 1=Sun ... 7=Sat)
--   @partner_ids (ARRAY<INT64>, optional)

WITH params AS (
  SELECT
    @start_date AS start_date,
    @end_date AS end_date,
    REGEXP_REPLACE(UPPER(@target_franchise), r'[^A-Z0-9]', '') AS target_franchise,
    @target_grade AS target_grade,
    COALESCE(@target_country_id, 2) AS target_country_id,
    COALESCE(@target_entity_id, 'PY_CL') AS target_entity_id,
    COALESCE(@target_days_of_week, [1, 2, 3, 4, 5, 6, 7]) AS target_days_of_week
),

vendor_scope AS (
  SELECT DISTINCT
    CAST(p.partner_id AS STRING) AS v_code,
    p.partner_name,
    p.franchise.franchise_name
  FROM `peya-bi-tools-pro.il_core.dim_partner` p
  INNER JOIN `peya-data-origins-pro.cl_core.growth_vendor_attributes` attr
    ON CAST(p.partner_id AS STRING) = attr.vendor_code
  WHERE p.country_id = (SELECT target_country_id FROM params)
    AND REGEXP_REPLACE(UPPER(p.franchise.franchise_name), r'[^A-Z0-9]', '') LIKE CONCAT('%', (SELECT target_franchise FROM params), '%')
    AND attr.entity_id = (SELECT target_entity_id FROM params)
    AND attr.attributes.fixed_vendor_grade = (SELECT target_grade FROM params)
    AND attr.attributes.is_latest_record = TRUE
    AND p.is_active
    AND (COALESCE(ARRAY_LENGTH(@partner_ids), 0) = 0 OR p.partner_id IN UNNEST(@partner_ids))
),

time_grid AS (
  -- Filter days dynamically from @target_days_of_week (default: all days).
  SELECT CAST(ts AS DATETIME) AS dt_snapshot
  FROM params, UNNEST(GENERATE_TIMESTAMP_ARRAY(
    CAST(start_date AS TIMESTAMP),
    TIMESTAMP(DATETIME(end_date, '23:59:59')),
    INTERVAL 1 MINUTE)) AS ts
  WHERE EXTRACT(DAYOFWEEK FROM ts) IN UNNEST((SELECT target_days_of_week FROM params))
),

hdm_data AS (
  SELECT
    LOWER(TRIM(hdm.global_order_id)) AS hdm_join_id,
    hdm.vendor.code AS vendor_code,
    COALESCE(hdm.high_demand_mode.is_hd_order, FALSE) AS is_hd_order,
    hdm.high_demand_mode.author AS hdm_author
  FROM `fulfillment-dwh-production.curated_data_shared_vendor.growth_vendor_orders` hdm
  INNER JOIN vendor_scope vs ON hdm.vendor.code = vs.v_code
  CROSS JOIN params p
  WHERE hdm.created_date BETWEEN p.start_date AND p.end_date
),

raw_transitions AS (
  SELECT
    flo.platform_order_code,
    vs.v_code AS vendor_code,
    d.rider_id,
    t.state,
    t.created_at AS transition_at,
    LEAD(t.created_at) OVER (
      PARTITION BY flo.platform_order_code
      ORDER BY t.created_at
    ) AS next_transition_at,
    flo.order_placed_at,
    SAFE_DIVIDE(flo.estimated_prep_time, 60.0) AS order_ept_min,
    COALESCE(
      flo.prepared_at,
      (
        SELECT d_inner.rider_picked_up_at
        FROM UNNEST(flo.deliveries) d_inner
        WHERE d_inner.rider_picked_up_at IS NOT NULL
        ORDER BY d_inner.created_at DESC
        LIMIT 1
      )
    ) AS order_ready_at,
    h.is_hd_order,
    h.hdm_author
  FROM `peya-bi-tools-pro.il_logistics.fact_logistic_orders` flo
  INNER JOIN vendor_scope vs ON flo.vendor.vendor_code = vs.v_code
  CROSS JOIN UNNEST(flo.deliveries) d
  CROSS JOIN UNNEST(d.transitions) t
  LEFT JOIN hdm_data h ON LOWER(TRIM(flo.platform_order_code)) = h.hdm_join_id
  CROSS JOIN params p
  WHERE flo.created_date_local BETWEEN p.start_date AND p.end_date
    AND flo.country.country_id = p.target_country_id
    AND flo.order_status = 'completed'
),

rider_waiting_periods AS (
  SELECT platform_order_code, vendor_code, rider_id, transition_at AS start_at, next_transition_at AS end_at
  FROM raw_transitions
  WHERE state = 'near_pickup'
),

order_cocina_periods AS (
  SELECT DISTINCT platform_order_code, vendor_code, order_placed_at AS arrived_at, order_ready_at AS ready_at, order_ept_min, is_hd_order, hdm_author
  FROM raw_transitions
)

SELECT
  DATE(tg.dt_snapshot) AS fecha,
  tg.dt_snapshot AS momento_exacto,
  SAFE_CAST(vs.v_code AS INT64) AS partner_id,
  vs.partner_name,
  COUNT(DISTINCT op.platform_order_code) AS ordenes_pendientes,
  COUNT(DISTINCT rp.rider_id) AS riders_cerca,
  ROUND(COALESCE(AVG(op.order_ept_min), 0), 2) AS ept_promedio_min,
  MAX(CASE WHEN op.is_hd_order THEN 1 ELSE 0 END) AS hdm_activo,
  ARRAY_AGG(op.hdm_author IGNORE NULLS ORDER BY op.arrived_at ASC LIMIT 1)[SAFE_OFFSET(0)] AS hdm_autor,
  COALESCE(MAX(SAFE_DIVIDE(DATETIME_DIFF(tg.dt_snapshot, DATETIME(rp.start_at, 'America/Santiago'), SECOND), 60.0)), 0) AS max_awt_espera_min
FROM time_grid tg
CROSS JOIN vendor_scope vs
LEFT JOIN order_cocina_periods op ON vs.v_code = op.vendor_code
    AND DATETIME(op.arrived_at, 'America/Santiago') <= tg.dt_snapshot
    AND (DATETIME(op.ready_at, 'America/Santiago') > tg.dt_snapshot OR op.ready_at IS NULL)
    AND DATETIME(op.arrived_at, 'America/Santiago') >= DATETIME_SUB(tg.dt_snapshot, INTERVAL 2 HOUR)
LEFT JOIN rider_waiting_periods rp ON vs.v_code = rp.vendor_code
    AND DATETIME(rp.start_at, 'America/Santiago') <= tg.dt_snapshot
    AND (DATETIME(rp.end_at, 'America/Santiago') > tg.dt_snapshot OR rp.end_at IS NULL)
    AND DATETIME(rp.start_at, 'America/Santiago') >= DATETIME_SUB(tg.dt_snapshot, INTERVAL 2 HOUR)
GROUP BY 1, 2, 3, 4
ORDER BY 2 ASC, 3 ASC;