# Beyond Correlation: Causal Intervention for Multi-Label Medical Image Diagnosis

Official implementation of **Beyond Correlation: Causal Intervention for Multi-Label Medical Image Diagnosis**, accepted by **IEEE Transactions on Medical Imaging**.

Please refer to our paper for the full method, experimental setup, and analysis.

Authors: Jianyang Xie, Yitian Zhao, Xiuju Chen, Yanda Meng, He Zhao, Uazman Alam, Xiaoxin Li, and Yalin Zheng.

DOI: [10.1109/TMI.2026.3698052](https://doi.org/10.1109/TMI.2026.3698052)

## Overview

This repository provides the code for reproducing the BankCausal / Causal Intervention framework for multi-label medical image diagnosis. The implementation is based on the OpenMMLab stack and follows the MMPreTrain/MMEngine config workflow.

Supported experiment configs:

- `configs/ODIR/ODIR_BankCausal.py`
- `configs/FFA/FFA_BankCausal.py`
- `configs/Endo/EndoBankCausal.py`
- `configs/ChestXPert/ChestXBankCausal.py`

## Main Files

- Causal intervention head: `models/heads/multi_label_Q2LBankcausal_head.py`
- SwAV prototype hook: `models/hooks/swav_hook.py`
- Multi-label classifier wrapper: `models/Classifier/multiLabelClassifier.py`
- Metrics: `evaluation/Rankingmulti_label.py`
- Training entry: `tools/Train.py`
- Test entry: `tools/Test.py`

## Installation

Create an environment with Python 3.9 or newer, then install PyTorch and the OpenMMLab dependencies.

```bash
pip install -U openmim
mim install "mmengine>=0.10.0" "mmcv>=2.0.0" "mmpretrain>=1.0.0rc8"
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version before running large experiments.

## Data Preparation

Datasets are not included in this repository. Please place annotation files under `data/` following the paths used in the config files.

Example layout:

```text
data/
  OIA-ODIR/
    train.pkl
    val.pkl
    onsite_test.pkl
  FFA/
    train_class6.pkl
    val_class6.pkl
    test_class6.pkl
  Chexpert/
    mmpretrain_train.pkl
    mmpretrain_val.pkl
    mmpretrain_test.pkl
```

Each pickle annotation file should contain image paths and multi-label targets expected by `datasets/FFA.py`. If your data is stored elsewhere, update the annotation paths in the corresponding config or override them with `--cfg-options`.

## Training

Single GPU:

```bash
bash scripts/train_bankcausal.sh configs/ODIR/ODIR_BankCausal.py work_dirs/ODIR_BankCausal
```

Multiple GPUs:

```bash
bash tools/dist_train.sh configs/ODIR/ODIR_BankCausal.py 4 --work-dir work_dirs/ODIR_BankCausal
```

To run another dataset, replace the config path, for example:

```bash
bash scripts/train_bankcausal.sh configs/FFA/FFA_BankCausal.py work_dirs/FFA_BankCausal
```

## Evaluation

```bash
bash scripts/test_bankcausal.sh \
  configs/ODIR/ODIR_BankCausal.py \
  work_dirs/ODIR_BankCausal/best_multi-class_mAUC_multiclass_epoch_XX.pth \
  work_dirs/ODIR_BankCausal/eval
```

To enable test-time augmentation, append `--tta`.

## Citation

If this repository is useful for your research, please refer to:

```text
Jianyang Xie, Yitian Zhao, Xiuju Chen, Yanda Meng, He Zhao, Uazman Alam,
Xiaoxin Li, and Yalin Zheng. Beyond Correlation: Causal Intervention for
Multi-Label Medical Image Diagnosis.
IEEE Transactions on Medical Imaging, 2026.
DOI: 10.1109/TMI.2026.3698052.
```

## Notes

- Checkpoints, logs, predictions, TensorBoard events, and `work_dirs*` are ignored by Git.
- Dataset paths are local and should be configured for each machine.
- Pretrained backbones are loaded from public OpenMMLab URLs defined in the configs.
- For reproduction, use the config corresponding to each dataset and keep the same image resolution, batch size, warmup epoch, and prototype ratio.
