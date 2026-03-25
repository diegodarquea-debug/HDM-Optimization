-- ============================================================
-- DIAGNOSTIC: What franchise names and grades exist in BQ?
-- Run this directly in BigQuery console to find the exact
-- franchise name and grade combination to use in the pipeline.
-- ============================================================

-- Step 1: Find all franchise names matching "mcdonald" (case-insensitive)
--         Ignores all other filters (grade, active, entity_id)
SELECT DISTINCT
  p.franchise.franchise_name                        AS franchise_name,
  UPPER(p.franchise.franchise_name)                 AS franchise_name_upper,
  p.is_active,
  COUNT(*) OVER (PARTITION BY p.franchise.franchise_name) AS partner_count
FROM `peya-bi-tools-pro.il_core.dim_partner` p
WHERE p.country_id = 2
  AND UPPER(p.franchise.franchise_name) LIKE '%MCDONALD%'
ORDER BY franchise_name;

-- ============================================================
-- Step 2: Once you confirm the franchise name above, check grades.
-- Replace 'PUT_EXACT_FRANCHISE_NAME_HERE' with the value from Step 1.
-- ============================================================

/*
SELECT DISTINCT
  p.franchise.franchise_name                        AS franchise_name,
  attr.attributes.fixed_vendor_grade                AS grade,
  attr.entity_id,
  p.is_active,
  COUNT(*) AS n_partners
FROM `peya-bi-tools-pro.il_core.dim_partner` p
INNER JOIN `peya-data-origins-pro.cl_core.growth_vendor_attributes` attr
  ON CAST(p.partner_id AS STRING) = attr.vendor_code
WHERE p.country_id = 2
  AND UPPER(p.franchise.franchise_name) LIKE '%MCDONALD%'
  AND attr.entity_id = 'PY_CL'
  AND attr.attributes.is_latest_record = TRUE
GROUP BY 1, 2, 3, 4
ORDER BY franchise_name, grade;
*/

-- ============================================================
-- Step 3: Isolation check for the exact vendor_scope logic.
-- This tells you which filter is removing all rows.
-- Replace grade/entity_id if needed.
-- ============================================================

WITH base AS (
  SELECT
    p.partner_id,
    p.partner_name,
    p.is_active,
    p.franchise.franchise_name AS franchise_name,
    attr.entity_id,
    attr.attributes.fixed_vendor_grade AS grade,
    attr.attributes.is_latest_record AS is_latest_record
  FROM `peya-bi-tools-pro.il_core.dim_partner` p
  INNER JOIN `peya-data-origins-pro.cl_core.growth_vendor_attributes` attr
    ON CAST(p.partner_id AS STRING) = attr.vendor_code
  WHERE p.country_id = 2
    AND REGEXP_REPLACE(UPPER(p.franchise.franchise_name), r'[^A-Z0-9]', '') LIKE '%MCDONALD%'
)

SELECT 'A_name_only' AS stage, COUNT(*) AS rows FROM base
UNION ALL
SELECT 'B_active', COUNT(*) FROM base WHERE is_active
UNION ALL
SELECT 'C_latest_record', COUNT(*) FROM base WHERE is_active AND is_latest_record
UNION ALL
SELECT 'D_entity_PY_CL', COUNT(*) FROM base WHERE is_active AND is_latest_record AND entity_id = 'PY_CL'
UNION ALL
SELECT 'E_grade_AAA', COUNT(*) FROM base WHERE is_active AND is_latest_record AND entity_id = 'PY_CL' AND grade = 'AAA';
