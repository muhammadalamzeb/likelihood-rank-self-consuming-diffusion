# One-command reproduction of principal W2-1 results (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py scripts\reproduce_w2.py
