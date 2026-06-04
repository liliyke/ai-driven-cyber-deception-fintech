# data/

- **`results.json`** — metrics written by the last `python -m addf.simulate` run (baseline vs
  ADDF, plus the IOC report). Regenerated on each run; a sample is committed so the expected
  output is visible without running anything.

The FinBank test-bed and attack trials are generated in code from
[`addf.config.ADDFConfig`](../src/addf/config.py) (seeded for reproducibility), so there are no
static datasets to download.

Only synthetic data is ever used — standard PCI test PANs and synthetic accounts. No real
customer, account, or cardholder data, and no captured attacker artifacts (malware, credential
dumps), belong in this repository.
