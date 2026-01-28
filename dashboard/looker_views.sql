-- Dashboard Views for Looker Studio
-- These views are optimized for visualization and reporting

-- ============================================================
-- 1. DAILY DASHBOARD - Main dashboard view with all metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_dashboard_daily` AS
SELECT
  date,
  FORMAT_DATE('%Y-%m', date) AS month,
  FORMAT_DATE('%A', date) AS day_of_week,
  EXTRACT(DAYOFWEEK FROM date) AS day_num,

  -- Stress metrics
  avg_stress,
  max_stress,
  stress_category,
  CASE
    WHEN avg_stress <= 25 THEN 1
    WHEN avg_stress <= 50 THEN 2
    WHEN avg_stress <= 75 THEN 3
    ELSE 4
  END AS stress_level_num,

  -- Body Battery
  charged,
  drained,
  net_battery,
  CASE
    WHEN net_battery > 20 THEN 'Great Recovery'
    WHEN net_battery > 0 THEN 'Good Recovery'
    WHEN net_battery > -20 THEN 'Slight Drain'
    ELSE 'Heavy Drain'
  END AS recovery_status,

  -- Heart Rate
  resting_hr,
  avg_resting_hr_7d,

  -- Sleep
  ROUND(sleep_hours, 1) AS sleep_hours,
  ROUND(deep_sleep_hours, 1) AS deep_sleep_hours,
  ROUND(rem_sleep_hours, 1) AS rem_sleep_hours,
  sleep_body_battery_change,
  avg_overnight_hrv,

  -- Steps
  steps,
  has_steps_data,
  CASE
    WHEN steps IS NULL THEN 'No Data'
    WHEN steps < 5000 THEN 'Sedentary'
    WHEN steps < 7500 THEN 'Low Active'
    WHEN steps < 10000 THEN 'Moderate'
    ELSE 'Active'
  END AS activity_level

FROM `garmin_data.v_daily_metrics`;


-- ============================================================
-- 2. WEEKLY SUMMARY - Aggregated weekly metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_dashboard_weekly` AS
SELECT
  DATE_TRUNC(date, WEEK(MONDAY)) AS week_start,
  FORMAT_DATE('%Y-W%V', date) AS week_label,

  COUNT(*) AS days_in_week,

  -- Stress
  ROUND(AVG(avg_stress), 1) AS avg_stress,
  MAX(max_stress) AS max_stress,
  COUNTIF(avg_stress > 50) AS high_stress_days,

  -- Body Battery
  ROUND(AVG(charged), 1) AS avg_charged,
  ROUND(AVG(drained), 1) AS avg_drained,
  ROUND(AVG(net_battery), 1) AS avg_net_battery,

  -- Heart Rate
  ROUND(AVG(resting_hr), 0) AS avg_resting_hr,

  -- Sleep
  ROUND(AVG(sleep_hours), 1) AS avg_sleep_hours,
  ROUND(AVG(deep_sleep_hours), 1) AS avg_deep_sleep,

  -- Steps
  ROUND(AVG(CASE WHEN steps IS NOT NULL THEN steps END), 0) AS avg_steps,
  COUNTIF(steps IS NOT NULL) AS days_with_steps

FROM `garmin_data.v_daily_metrics`
GROUP BY week_start, week_label;


-- ============================================================
-- 3. MONTHLY SUMMARY - Aggregated monthly metrics
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_dashboard_monthly` AS
SELECT
  DATE_TRUNC(date, MONTH) AS month_start,
  FORMAT_DATE('%Y-%m', date) AS month_label,
  FORMAT_DATE('%B %Y', date) AS month_name,

  COUNT(*) AS days_in_month,

  -- Stress
  ROUND(AVG(avg_stress), 1) AS avg_stress,
  ROUND(STDDEV(avg_stress), 1) AS stress_stddev,
  MAX(max_stress) AS max_stress_peak,
  COUNTIF(avg_stress > 50) AS high_stress_days,
  ROUND(COUNTIF(avg_stress > 50) / COUNT(*) * 100, 1) AS high_stress_pct,

  -- Body Battery
  ROUND(AVG(charged), 1) AS avg_charged,
  ROUND(AVG(net_battery), 1) AS avg_net_battery,

  -- Heart Rate
  ROUND(AVG(resting_hr), 0) AS avg_resting_hr,
  MIN(resting_hr) AS min_resting_hr,
  MAX(resting_hr) AS max_resting_hr,

  -- Sleep
  ROUND(AVG(sleep_hours), 1) AS avg_sleep_hours,
  ROUND(AVG(deep_sleep_hours), 1) AS avg_deep_sleep,
  COUNTIF(sleep_hours < 6) AS poor_sleep_days,

  -- Steps
  ROUND(AVG(CASE WHEN steps IS NOT NULL THEN steps END), 0) AS avg_steps,
  COUNTIF(steps IS NOT NULL) AS days_with_steps,
  ROUND(COUNTIF(steps IS NOT NULL) / COUNT(*) * 100, 1) AS steps_coverage_pct

FROM `garmin_data.v_daily_metrics`
GROUP BY month_start, month_label, month_name;


-- ============================================================
-- 4. CORRELATION METRICS - For scatter plots
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_dashboard_correlations` AS
SELECT
  date,

  -- Core metrics for correlation analysis
  avg_stress,
  resting_hr,
  sleep_hours,
  deep_sleep_hours,
  charged AS body_battery_charged,
  net_battery,
  steps,
  avg_overnight_hrv,

  -- Lagged metrics (next day)
  LEAD(avg_stress, 1) OVER (ORDER BY date) AS next_day_stress,
  LEAD(resting_hr, 1) OVER (ORDER BY date) AS next_day_resting_hr,

  -- Previous day metrics
  LAG(sleep_hours, 1) OVER (ORDER BY date) AS prev_day_sleep,
  LAG(steps, 1) OVER (ORDER BY date) AS prev_day_steps

FROM `garmin_data.v_daily_metrics`
WHERE avg_stress IS NOT NULL;


-- ============================================================
-- 5. TREND METRICS - 7-day rolling averages
-- ============================================================
CREATE OR REPLACE VIEW `garmin_data.v_dashboard_trends` AS
SELECT
  date,
  avg_stress,
  resting_hr,
  sleep_hours,
  net_battery,
  steps,

  -- 7-day rolling averages
  ROUND(AVG(avg_stress) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS stress_7d_avg,
  ROUND(AVG(resting_hr) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0) AS resting_hr_7d_avg,
  ROUND(AVG(sleep_hours) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS sleep_7d_avg,
  ROUND(AVG(net_battery) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS net_battery_7d_avg,

  -- Week-over-week change
  avg_stress - LAG(avg_stress, 7) OVER (ORDER BY date) AS stress_wow_change,
  resting_hr - LAG(resting_hr, 7) OVER (ORDER BY date) AS resting_hr_wow_change

FROM `garmin_data.v_daily_metrics`;
