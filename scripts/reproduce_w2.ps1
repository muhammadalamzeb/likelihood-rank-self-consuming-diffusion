# One-command reproduction of principal W2-1 results (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Missing .venv — create venv and install deps first (see README)." }

& $py -m pip install matplotlib --disable-pip-version-check -q
& $py experiments\scripts\run_w2_false_stability.py --out experiments --seeds 0 1 2 --rhos 5.0 10.0 --policies mix top_k rand_k replace bottom_k --generations 5 --train-steps 600 --n-data 1500 --n-synth 1500
& $py experiments\scripts\analyze_w2_fs.py
& $py experiments\scripts\make_w2_figures.py
New-Item -ItemType Directory -Force -Path paper\figures | Out-Null
Copy-Item experiments\analysis\figures\*.png paper\figures\ -Force
Copy-Item experiments\analysis\table_*.csv paper\ -Force
Write-Host "Done. See paper/PAPER.md and experiments/analysis/"
