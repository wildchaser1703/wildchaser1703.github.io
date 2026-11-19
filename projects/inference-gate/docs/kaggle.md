# Reproduce the T4 experiment

Import [the executed notebook source](../notebooks/kaggle.ipynb) into Kaggle.
Enable Internet access, select a T4 accelerator and run all cells. The notebook
uses one device and embeds the source snapshot used for this experiment
(`45fa4c3`). It installs its dependencies and downloads the pinned public model.
An early compile-and-run check confirms that FlashInfer can link the NVIDIA driver.

The output file is `/kaggle/working/inference-gate-gpu-evidence.zip`.
Check `run-status.txt` inside the archive. A failed run also writes diagnostic
logs. The verified archive and CSVs are in [evidence/kaggle-t4](../evidence/kaggle-t4/).

To verify the published run and regenerate the two CSVs, run this command from
the Inference Gate project directory with Python 3.10 or newer.

```sh
python evidence/verify.py evidence/kaggle-t4/inference-gate-gpu-evidence.zip
```

The check recomputes the paired workload hashes, accounts for all request
outcomes, and checks GPU samples and the final reservation state in each trial.
