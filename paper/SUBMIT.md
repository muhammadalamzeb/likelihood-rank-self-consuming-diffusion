# Submission / publish status

**Author:** Muhammad Alamzeb  
**Code:** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion  
**GitHub Release (live now):** https://github.com/muhammadalamzeb/likelihood-rank-self-consuming-diffusion/releases/tag/v1.0.0  

## Public preprint options

### A) GitHub Release — DONE
PDF + LaTeX source attached to tag `v1.0.0`.  
Cite as the release URL until Zenodo DOI exists.

### B) Zenodo DOI — do this next (~5 min)
Deposit kit: `paper/zenodo/`

1. Open https://zenodo.org/deposit/new and sign in (GitHub login works).
2. Upload `paper/zenodo/Alamzeb_2026_likelihood_rank_self_consuming_diffusion.pdf`
3. Optionally upload `paper/zenodo/arxiv_source.zip`
4. Copy fields from `paper/zenodo/metadata.json` (or follow `paper/zenodo/README.md`)
5. License: **CC BY 4.0** → **Publish**
6. Paste the DOI here / tell the agent to record it

**API path:** create a Zenodo token, then:
```powershell
$env:ZENODO_TOKEN = "YOUR_TOKEN"
python scripts/zenodo_upload.py
```

### C) arXiv — waiting on endorsement
Code `Z4GDY9` for cs.LG. After endorsement, upload `paper/arxiv_source.zip`.

## Local package
- PDF: `paper/main.pdf`
- arXiv zip: `paper/arxiv_source.zip`
- Zenodo kit: `paper/zenodo/`
