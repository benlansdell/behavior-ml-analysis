# Pipeline

Given a DLC model (or models), this performs inference and downstream analysis on overnight recordings for Zakharenko lab study. Recordings by Brett Teubner. A single mouse is recorded overnight, for between 4 and 10 hours. There are two genotypes (WT and HET) -- detailed in `overnight_expt_data.csv`. The videos are cropped to remove a lot of distracting non-cage space. Details of the crops are in `crops.csv`.

Input videos are assumed to have the stucture:
```
└───base_folder
    └───animal1
    │   │   e3v813a-20220131T174814-181650.avi
    │   │   e3v813a-20220131T194207-201034.avi
    │   └   ...
    └───animal2
    │   │   e3v813a-20220131T194207-201034.avi
    │   └   ...
    ...
```

Given this, pipeline is as follows:
1. `extract_clips.ipynb` -- crops and prepares videos for inference (optionally makes a training set too)
2. `dlc_inference.ipynb` -- runs DLC inference given videos and DLC project
3. `interpolate_overnight_dlc.ipynb` -- (optional) smoothes over low-confidence tracking points
4. `analyse_movement_stats.ipynb` -- computes experiment statistics and analysis
