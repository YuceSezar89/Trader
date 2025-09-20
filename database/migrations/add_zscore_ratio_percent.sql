-- Add zscore_ratio_percent column to signals table
-- Pine panel eşdeğeri Z-Score ratio: (close - EMA200) / stdev * 100

ALTER TABLE signals 
ADD COLUMN zscore_ratio_percent FLOAT;

-- Add comment for documentation
COMMENT ON COLUMN signals.zscore_ratio_percent IS 'Z-Score ratio percent: (close - EMA200) / stdev * 100 - Pine panel equivalent';
