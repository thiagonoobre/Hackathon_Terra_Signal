
SELECT
    customerID,
    TRIM(BOTH '"' FROM CustomerFeedback) AS CustomerFeedback_adj,
    (CASE WHEN lower(Churn) = 'yes' THEN 1 ELSE 0 END) AS TargetChurn
FROM workspace.default.{dataset}
