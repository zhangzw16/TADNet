> [!IMPORTANT] 
> We are still working on the readme, stay tunned...

# [TADNet] Unravel Anomalies: An End-to-End Seasonal-Trend Decomposition Approach for Time Series Anomaly Detection

This repository This repository contains the code for the paper "[Unravel Anomalies: An End-to-End Seasonal-Trend Decomposition Approach for Time Series Anomaly Detection](https://ieeexplore.ieee.org/document/10446482)" by *Zhenwei Zhang; Ruiqi Wang; Ran Ding; Yuantao Gu*, published in the IEEE ICASSP 2024 (International Conference on Acoustics, Speech, and Signal Processing).

## Introduction

:triangular_flag_on_post: Presentation Slides for this paper can be found on [IEEE SigPort](https://sigport.org/documents/unravel-anomalies-end-end-seasonal-trend-decomposition-approach-time-series-anomaly) ([Download](https://sigport.org/sites/default/files/docs/TADNet%20Oral.pdf)). 

> Traditional Time-series Anomaly Detection (TAD) methods often struggle with the composite nature of complex time-series data and a diverse array of anomalies. We introduce TADNet, an end-to-end TAD model that leverages Seasonal-Trend Decomposition to link various types of anomalies to specific decomposition components, thereby simplifying the analysis of complex time-series and enhancing detection performance. Our training methodology, which includes pre-training on a synthetic dataset followed by fine-tuning, strikes a balance between effective decomposition and precise anomaly detection. Experimental validation on real-world datasets confirms TADNet’s state-of-the-art performance across a diverse range of anomalies.


## Datasets

For more details on the datasets used in the paper, please refer to [this repo](https://github.com/imperial-qore/TranAD/tree/main/data).

- UCR:
- SMD:
- SWaT:
- PSM:
- WADI:

## Preparation

<!-- Preprocess all datasets using the command -->

Generate the synthetic dataset using the command:
```bash
python run.py --mode synthetic 
```
## Training & Evaluation

Train the model using the command:
```bash
python run.py --mode pretrain --loss 2
python run.py --file_dir xxx.npy --mode finetune --loss 5 --number xxx --exists 1
```

Evaluate the model using the command:
```bash
python run.py --file_dir xxx.npy --mode test --number xxx --exists 1
```

## Citation

If you find this work useful, please consider citing the following paper:

```bibtex
@INPROCEEDINGS{10446482,
  author={Zhang, Zhenwei and Wang, Ruiqi and Ding, Ran and Gu, Yuantao},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Unravel Anomalies: an End-to-End Seasonal-Trend Decomposition Approach for Time Series Anomaly Detection}, 
  year={2024},
  volume={},
  number={},
  pages={5415-5419},
  keywords={Training;Analytical models;Time series analysis;Data visualization;Signal processing;Data models;Arrays;time-series anomaly detection;seasonal-trend decomposition;time-series analysis;end-to-end},
  doi={10.1109/ICASSP48485.2024.10446482}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
