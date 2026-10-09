# Analysis workflow

The manuscript analysis follows this sequence:

1. Load 3-hour ERA5 atmospheric fields and IMS rain-gauge observations.
2. Convert ERA5 fields to monthly standardized anomalies and retain missing-data masks where required.
3. Train CircEnvFormer on station-level rainfall occurrence.
4. Discard the rainfall forecast head and extract the latent atmospheric embeddings.
5. Define rainfall events and represent each event by the embedding at rainfall onset.
6. Cluster event embeddings with k-means and evaluate compactness and stability.
7. Diagnose environmental and atmospheric structure of each regime.
8. Compare against classical synoptic classification and a pretrained Aurora representation.
9. Evaluate rainfall forecast performance at 0, 24, and 48 h and compare 24/48 h with ECMWF IFS.
10. Generate attention-rollout attribution maps.

## Reproducibility note

The original research notebook is preserved in `notebooks/working_example.ipynb`. It contains exploratory analysis and references to local Weizmann filesystem paths and helper modules that are not distributed with this repository.

The modules under `src/circenvformer/` provide the cleaned reusable core implementation. Dataset-specific preprocessing, local data paths, trained checkpoints, and some plotting utilities remain experiment-specific and should be configured by the user.
