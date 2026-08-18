# BeforeReturn Status

Last updated: 2026-08-18

## Current Milestone

Day 1: repository skeleton, official data access, data dictionary draft, and leakage audit draft.

## Completed

- Confirmed project location: `/Users/liangyingxin/Desktop/BeforeReturn`.
- Confirmed raw data was initially absent.
- Confirmed main model choice: CatBoost.
- Confirmed frontend stack: Next.js, TypeScript, Tailwind CSS, shadcn/ui.
- Initialized git repository.
- Created initial repository structure.
- Downloaded official OSF raw files into `data/raw/`.
- Inspected raw pickle schemas.

## Verification

- OSF API endpoint for `https://osf.io/c793h/` is reachable.
- Raw data directory is git-ignored.
- Downloaded files passed SHA-256 checks in `scripts/download_osf_data.py`.
- Observed raw data size is approximately 742 MB.

## Known Issues

- Raw event tables have no timestamp or sequence field.
- Node aggregate fields include return counts/rates and return-code distributions that may leak target labels if used directly.
- `SPEC.md` remains draft until the leakage strategy is decided.
- Local Python version is 3.13.5; project target is Python 3.11 or 3.12. A compatible Python must be used for the final reproducible environment.

## Next Step

Decide modeling leakage strategy before training.
