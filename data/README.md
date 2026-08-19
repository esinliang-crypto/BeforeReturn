# Data

Raw data is downloaded from the official ASOS GraphReturns OSF project:

- https://osf.io/c793h/

ASOS GraphReturns is licensed CC BY 4.0 International on OSF. Derived model and
runtime artifacts must retain attribution, note project-specific modifications,
and must not imply endorsement by ASOS or the original authors.

Run:

```bash
python scripts/download_osf_data.py
```

Expected raw files are stored under `data/raw/` and are not committed.

Committed data is limited to tiny, anonymous demo samples under `data/samples/` after schema inspection confirms that the sample complies with the project constraints.

## Important Limitation

ASOS GraphReturns contains users with at least one return and product variants with at least one purchase. It must not be described as representative of ASOS-wide return behavior.
