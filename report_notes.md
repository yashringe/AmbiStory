# Short technical notes for the final report

- Problem: predict human-perceived plausibility of a candidate word sense in an ambiguous five-sentence story, outputting a 1–5 score.
- Baseline: sense-aware TF-IDF using the story, homonym, candidate definition and example sentence.
- Improvements: group-aware contrast for competing senses, tuned integer thresholds, ensemble wrapper, and optional transformer cross-encoder path.
- Important design choice: we do not use `average`, `choices`, `stdev`, `nonsensical`, or `sample_id` as input features because those are labels/metadata unavailable at test time.
- Final result: improved Spearman integer and accuracy-within-standard-deviation compared with the starter baseline.
