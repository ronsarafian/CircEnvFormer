# Scripts

The scripts in this directory expose the main public workflow stages as command-line entry points:

- `train.py` — initialize a CircEnvFormer encoder and rainfall forecast head.
- `cluster_events.py` — cluster saved event embeddings and report normalized WCSS.
- `evaluate_forecast.py` — compute regional daily or 3-hour forecast metrics from prepared tables.
- `evaluate_ifs.py` — build station-level IFS precipitation forecasts at the 24- or 48-hour lead.
- `attention_rollout.py` — compute normalized CLS attention rollout from saved attention tensors.
- `make_figures.py` — entry point for the manuscript figure set; the complete exploratory workflow is retained in the notebook.

Large ERA5, IMS, and IFS datasets are not included in the repository.
