-- Garmin Data Transformation Views
-- These views parse raw JSON and handle data quality issues

-- ============================================================
-- 1. STEPS - Parse and flag invalid days (all zeros = no data)
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_steps_daily` AS
WITH parsed AS (
  SELECT
    REGEXP_EXTRACT(filename, r'(\d{4}-\d{2}-\d{2})') AS date_str,
    JSON_EXTRACT_ARRAY(raw_json) AS intervals
  FROM `garmin_data.garmin_raw_data`
  WHERE filename LIKE 'steps_%'
),
flattened AS (
  SELECT
    date_str,
    CAST(JSON_EXTRACT_SCALAR(interval_item, '$.steps') AS INT64) AS steps,
    JSON_EXTRACT_SCALAR(interval_item, '$.primaryActivityLevel') AS activity_level
  FROM parsed
  CROSS JOIN UNNEST(intervals) AS interval_item
),
aggregated AS (
  SELECT
    PARSE_DATE('%Y-%m-%d', date_str) AS date,
    SUM(steps) AS total_steps,
    MAX(activity_level) AS max_activity_level,
    COUNT(*) AS interval_count
  FROM flattened
  GROUP BY date_str
)
SELECT
  date,
  -- Mark as NULL if total_steps is 0 (no data collected that day)
  CASE WHEN total_steps = 0 THEN NULL ELSE total_steps END AS steps,
  total_steps AS steps_raw,
  max_activity_level,
  -- Data quality flag
  CASE
    WHEN total_steps = 0 THEN 'NO_DATA'
    WHEN total_steps < 1000 THEN 'LOW'
    ELSE 'OK'
  END AS data_quality
FROM aggregated;


-- ============================================================
-- 2. BODY BATTERY - Parse daily metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_body_battery_daily` AS
SELECT
  PARSE_DATE('%Y-%m-%d', REGEXP_EXTRACT(filename, r'(\d{4}-\d{2}-\d{2})')) AS date,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$[0].charged') AS INT64) AS charged,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$[0].drained') AS INT64) AS drained,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$[0].charged') AS INT64) -
    CAST(JSON_EXTRACT_SCALAR(raw_json, '$[0].drained') AS INT64) AS net_battery
FROM `garmin_data.garmin_raw_data`
WHERE filename LIKE 'body_battery_%';


-- ============================================================
-- 3. HEART RATE - Parse daily metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_heart_rate_daily` AS
SELECT
  PARSE_DATE('%Y-%m-%d', REGEXP_EXTRACT(filename, r'(\d{4}-\d{2}-\d{2})')) AS date,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.restingHeartRate') AS INT64) AS resting_hr,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.maxHeartRate') AS INT64) AS max_hr,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.minHeartRate') AS INT64) AS min_hr,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.lastSevenDaysAvgRestingHeartRate') AS INT64) AS avg_resting_hr_7d
FROM `garmin_data.garmin_raw_data`
WHERE filename LIKE 'heart_rate_%';


-- ============================================================
-- 4. STRESS - Parse daily metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_stress_daily` AS
SELECT
  PARSE_DATE('%Y-%m-%d', REGEXP_EXTRACT(filename, r'(\d{4}-\d{2}-\d{2})')) AS date,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.avgStressLevel') AS INT64) AS avg_stress,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.maxStressLevel') AS INT64) AS max_stress,
  -- Stress category based on Garmin scale
  CASE
    WHEN CAST(JSON_EXTRACT_SCALAR(raw_json, '$.avgStressLevel') AS INT64) <= 25 THEN 'RESTING'
    WHEN CAST(JSON_EXTRACT_SCALAR(raw_json, '$.avgStressLevel') AS INT64) <= 50 THEN 'LOW'
    WHEN CAST(JSON_EXTRACT_SCALAR(raw_json, '$.avgStressLevel') AS INT64) <= 75 THEN 'MEDIUM'
    ELSE 'HIGH'
  END AS stress_category
FROM `garmin_data.garmin_raw_data`
WHERE filename LIKE 'stress_%';


-- ============================================================
-- 5. SLEEP - Parse daily metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_sleep_daily` AS
SELECT
  PARSE_DATE('%Y-%m-%d', REGEXP_EXTRACT(filename, r'(\d{4}-\d{2}-\d{2})')) AS date,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.dailySleepDTO.sleepTimeSeconds') AS INT64) / 3600.0 AS sleep_hours,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.dailySleepDTO.deepSleepSeconds') AS INT64) / 3600.0 AS deep_sleep_hours,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.dailySleepDTO.lightSleepSeconds') AS INT64) / 3600.0 AS light_sleep_hours,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.dailySleepDTO.remSleepSeconds') AS INT64) / 3600.0 AS rem_sleep_hours,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.dailySleepDTO.awakeSleepSeconds') AS INT64) / 3600.0 AS awake_hours,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.bodyBatteryChange') AS INT64) AS sleep_body_battery_change,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.restingHeartRate') AS INT64) AS sleep_resting_hr,
  CAST(JSON_EXTRACT_SCALAR(raw_json, '$.avgOvernightHrv') AS FLOAT64) AS avg_overnight_hrv
FROM `garmin_data.garmin_raw_data`
WHERE filename LIKE 'sleep_%';


-- ============================================================
-- 6. DAILY METRICS - Merged view with all metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_daily_metrics` AS
SELECT
  COALESCE(s.date, bb.date, hr.date, st.date, sl.date) AS date,

  -- Steps (NULL if no data)
  s.steps,
  s.data_quality AS steps_quality,

  -- Body Battery
  bb.charged,
  bb.drained,
  bb.net_battery,

  -- Heart Rate
  hr.resting_hr,
  hr.max_hr,
  hr.min_hr,
  hr.avg_resting_hr_7d,

  -- Stress
  st.avg_stress,
  st.max_stress,
  st.stress_category,

  -- Sleep
  sl.sleep_hours,
  sl.deep_sleep_hours,
  sl.rem_sleep_hours,
  sl.sleep_body_battery_change,
  sl.avg_overnight_hrv,

  -- Data completeness flag
  CASE
    WHEN s.steps IS NULL THEN FALSE
    ELSE TRUE
  END AS has_steps_data

FROM `garmin_data.v_steps_daily` s
FULL OUTER JOIN `garmin_data.v_body_battery_daily` bb ON s.date = bb.date
FULL OUTER JOIN `garmin_data.v_heart_rate_daily` hr ON COALESCE(s.date, bb.date) = hr.date
FULL OUTER JOIN `garmin_data.v_stress_daily` st ON COALESCE(s.date, bb.date, hr.date) = st.date
FULL OUTER JOIN `garmin_data.v_sleep_daily` sl ON COALESCE(s.date, bb.date, hr.date, st.date) = sl.date
ORDER BY date;


-- ============================================================
-- 7. DATA QUALITY SUMMARY
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_data_quality_summary` AS
SELECT
  FORMAT_DATE('%Y-%m', date) AS month,
  COUNT(*) AS total_days,
  COUNTIF(has_steps_data) AS days_with_steps,
  COUNTIF(NOT has_steps_data) AS days_without_steps,
  ROUND(COUNTIF(has_steps_data) / COUNT(*) * 100, 1) AS steps_coverage_pct,
  ROUND(AVG(CASE WHEN has_steps_data THEN steps END), 0) AS avg_steps_when_available,
  ROUND(AVG(avg_stress), 1) AS avg_stress
FROM `garmin_data.v_daily_metrics`
GROUP BY month
ORDER BY month;
