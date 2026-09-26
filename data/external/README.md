# data/external -- raw third-party datasets (not committed)

Everything in this folder except this README is git-ignored. Re-download as needed.

| file | source | license | size | used for |
|---|---|---|---|---|
| `fmucd.csv` | Facility Management Unified Classification Database (FMUCD), 12 North American universities, 2002-2021. https://data.mendeley.com/datasets/cb8d2nsjss/1 (paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11465139/) | CC BY-NC 4.0 | 1.4 GB, 3.73 M rows | `data/field_history/build_fmucd_slice.py` derives the committed rotating-equipment slice |

Not used: the Kaggle "Maintenance Work Orders Dataset" (tinhban). It is synthetic; its
resolution notes are drawn independently of the reported issue, so it carries no signal.
