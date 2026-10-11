<#
.SYNOPSIS
  Reset Thor to the start of the demo timeline (Windows / PowerShell).
.DESCRIPTION
  Uses the root-level `docker compose` (project name taken from the repo folder, e.g.
  `thor-main`), so it acts on the same stack the team starts by hand.

  Stops the stack, drops the TimescaleDB volume (telemetry, predictions, runs, contracts,
  approvals, registry rows) and, unless -KeepModels is given, the model + MLflow volumes too,
  then starts the stack again. On start-up the api reseeds the first 80% of the synthetic
  history, the one-shot `prep` service trains + validates a champion (MTR-042 held out),
  approves its promotion as demo-prep@thor (parity was proven at validation), exports it for the
  edge and scores the fleet; only then do the replay and the edge start, streaming the remaining 20%
  live. The Chroma index and embedding-model cache volumes are always kept.

  Expect 2-3 minutes for the prep step on first start (DEMO_PREP_TRIALS in .env sets the Optuna
  trials per family; default 6).
.EXAMPLE
  .\scripts\demo_reset.ps1               # full reset (recommended before a rehearsal)
  .\scripts\demo_reset.ps1 -KeepModels   # keep model + MLflow volumes (prep still re-registers)
#>
param([switch]$KeepModels)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$project = (docker compose config --format json | ConvertFrom-Json).name
if (-not $project) { throw "could not read the compose project name; is Docker running?" }
Write-Host "==> compose project: $project"

Write-Host "==> stopping stack"
docker compose down --remove-orphans
$vols = @("${project}_timescale-data")
if (-not $KeepModels) { $vols += @("${project}_models", "${project}_mlflow-data") }
$existing = @(docker volume ls -q)
foreach ($v in $vols) {
  if ($existing -contains $v) { Write-Host "==> removing volume $v"; docker volume rm $v | Out-Null }
}

Write-Host "==> starting stack (api seeds 80%, prep trains + promotes, then replay + edge start)"
docker compose up -d
Write-Host "==> following the prep service (train, validate incl. parity, promote, export, score)"
docker compose logs -f --no-log-prefix prep
$prepExit = docker compose ps -a --format json prep | ConvertFrom-Json | Select-Object -First 1 -ExpandProperty ExitCode
if ($prepExit -ne 0) { throw "demo prep failed with exit code $prepExit; see 'docker compose logs prep'" }

Write-Host -NoNewline "==> waiting for api"
$ready = $false
for ($i = 0; $i -lt 90 -and -not $ready; $i++) {
  try { Invoke-RestMethod http://localhost:8000/system/health -TimeoutSec 2 | Out-Null; $ready = $true }
  catch { Write-Host -NoNewline "."; Start-Sleep -Seconds 2 }
}
Write-Host (" ready" * [int]$ready)
Invoke-RestMethod http://localhost:8000/system/health | ConvertTo-Json -Compress
Write-Host "==> production model"
(Invoke-RestMethod http://localhost:8000/models) | Where-Object { $_.stage -eq "production" } | ForEach-Object {
  Write-Host ("  {0} {1} {2} IMS {3} onnx {4}" -f $_.name, $_.version, $_.family, $_.ims_total, $_.onnx_path)
}
Write-Host "==> fleet top 3 (approved-model health)"
(Invoke-RestMethod http://localhost:8000/fleet) | Select-Object -First 3 | ForEach-Object {
  Write-Host ("  {0} health {1} p {2} regime {3}" -f $_.asset.asset_id, $_.health_score, $_.failure_probability, $_.regime)
}
Write-Host "==> done. Web: http://localhost:5173  (replay streams the last 20% over ~5 min per pass)"
