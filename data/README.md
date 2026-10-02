# data

Generated datasets live here and are not tracked by git. One folder per circuit:

```bash
make smoke      # data/smoke/integrated and data/smoke/reference, quick check
make dataset    # data/v1/integrated and data/v1/reference, full study
```

Each folder holds `samples.parquet`, `waveforms.npy` and `manifest.json`
(see `src/ecgfd/dataset.py`). Load it with `ecgfd.dataset.load_dataset`.
