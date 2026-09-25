# Zenodo deposit — Likelihood-Ranked Synthetic Selection

## Files in this folder

| File | Role |
|------|------|
| `Alamzeb_2026_likelihood_rank_self_consuming_diffusion.pdf` | Manuscript PDF (upload this) |
| `arxiv_source.zip` | LaTeX source + figures (optional second file) |
| `metadata.json` | Field values for the Zenodo form / API |

## Upload steps (browser — ~5 minutes)

1. Open https://zenodo.org/deposit/new (sign in with GitHub/ORCID/email).
2. **Upload** the PDF (required). Optionally also upload `arxiv_source.zip`.
3. Fill form using `metadata.json`:
   - Upload type: **Publication** → **Preprint**
   - Title / Authors / Description / Keywords as in JSON
   - License: **Creative Commons Attribution 4.0**
   - Related identifier: GitHub URL (isSupplementTo / software)
4. Click **Publish** (assigns a DOI immediately).
5. Copy the DOI (e.g. `10.5281/zenodo.xxxxxxx`) back into `paper/SUBMIT.md`.

## API upload (optional)

```powershell
$env:ZENODO_TOKEN = "your_zenodo_personal_access_token"
python scripts/zenodo_upload.py
```

Create a token at: https://zenodo.org/account/settings/applications/tokens/new/  
Scopes: `deposit:write` and `deposit:actions`.
