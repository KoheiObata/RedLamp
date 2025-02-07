# Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies

Implementation of [RedLamp]().


## Requirements

We run all the experiments in `python 3.10.14`, see `requirements.txt` for the list of `pip` dependencies.

To install packages

```
pip install -r requirements.txt
```

or

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "numpy<2"
pip install -U scikit-learn
pip install matplotlib
pip install tqdm
pip install pandas
pip install patool
```


## Datasets preparation

### UCR Anomaly Archive
Follow the instructions on [link](https://github.com/mononitogoswami/tsad-model-selection),

and place them in `./dataset/AnomalyArchive/*.txt`
(e.g., `/dataset/AnomalyArchive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt`）

Or just run the code below
```
python main.py --dataset anomaly_archive
```

UCR Link: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ and https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip

---

### AIOps
Follow the instructions on [link](https://github.com/mononitogoswami/tsad-model-selection),

Download [TSB-UAD-Public.zip](https://www.thedatum.org/datasets/TSB-UAD-Public.zip)
from [TSB-UAD](https://github.com/TheDatumOrg/TSB-UAD)

and place them in `./dataset/IOPS/*`.
(e.g., `/dataset/IOPS/KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3.test.out`）

AIOps Link: https://github.com/NetManAIOps/KPI-Anomaly-Detection

---

### Server Machine Dataset (SMD)
Follow the instructions on [link](https://github.com/NetManAIOps/OmniAnomaly),

and place them in `./dataset/ServerMachineDataset/*/machine-*.txt`

Or just run the code below
```
python main.py --dataset smd
```

---

### Soil Moisture Active Passive (SMAP) satellite and Mars Science Laboratory (MSL)
Unzip `./dataset/NASA.zip`

SMAP and MSL Link: https://arxiv.org/pdf/1802.04431

Our dataloader is implemented based on [link](https://github.com/mononitogoswami/tsad-model-selection)


## Experiments

After the preparation of datasets, run below.

```
sh scripts/experiments.sh
```


