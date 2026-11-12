
SELECT
    customerID,
    TRIM(BOTH '"' FROM CustomerFeedback) AS CustomerFeedback_adj
FROM workspace.default.{dataset}
