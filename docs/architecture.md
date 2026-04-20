# Architecture

## Pipeline

1. Load ATP main-tour match files for 2000-2024 plus ranking and player metadata.
2. Clean raw rows, remove invalid outcomes, normalize types, and persist cleaned tables.
3. Walk matches chronologically to build only pre-match features.
4. Train baseline and boosted models on chronological splits, including Elo-driven features.
5. Save the best model, model metadata, evaluation metrics, and player snapshots.
6. Load artifacts inside the Discord bot for slash-command predictions.

## Chronological Splits

- Train: seasons before 2022
- Validation: 2022-2023
- Test: 2024
