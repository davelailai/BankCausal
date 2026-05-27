# GitHub Upload Checklist

Use this checklist before publishing the BankCausal reproduction repository.

## 1. Review The Files To Commit

```bash
git status --short
```

Expected source files include `configs/`, `datasets/`, `evaluation/`, `models/`, `tools/`, `scripts/`, `README.md`, `requirements.txt`, and `.gitignore`.

Do not commit local datasets, checkpoints, prediction pickle files, TensorBoard events, or `work_dirs*` folders.

## 2. Stop Tracking Generated Files Already In Git

`.gitignore` only affects new untracked files. If generated files were already tracked, remove them from the Git index while keeping the local copies:

```bash
git rm -r --cached -- '**/__pycache__'
git rm -r --cached data work_dirs 'work_dirs_*'
git rm --cached -- '*.pth' '*.pt' '*.pkl' '*.pickle' '*.log' 'events.out.tfevents*'
```

Some commands may report paths that are not tracked; that is fine.

## 3. Check The BankCausal Entry Points

```bash
bash -n scripts/train_bankcausal.sh
bash -n scripts/test_bankcausal.sh
```

Run a small smoke test if the dataset is available:

```bash
bash scripts/train_bankcausal.sh configs/ODIR/ODIR_BankCausal.py work_dirs/smoke_test --cfg-options train_cfg.max_epochs=1
```

## 4. Commit

```bash
git add .gitignore README.md requirements.txt scripts docs configs datasets evaluation models tools
git commit -m "Prepare BankCausal reproduction code"
```

