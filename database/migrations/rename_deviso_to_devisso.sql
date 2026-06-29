ALTER TABLE signals RENAME COLUMN deviso_score TO devisso_score;
ALTER TABLE signals RENAME COLUMN deviso_delta TO devisso_delta;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS devisso_ratio FLOAT;
