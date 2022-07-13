# Pipeline

Given a DLC model (or models), this performs inference and downstream analysis on overnight recordings for Zakharenko lab study. Recordings by Brett Teubner. A single mouse is recorded overnight, for between 4 and 10 hours. There are two genotypes (WT and HET) -- detailed in `overnight_expt_data.csv`. The videos are cropped to remove a lot of distracting non-cage space. Details of the crops are in `crops.csv`.

1. `extract_clips.ipynb`
2. `dlc_inference.ipynb`
3. `interpolate_overnight_dlc.ipynb` (optional)
4. `analyse_movement_stats.ipynb`
