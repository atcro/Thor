<#
.SYNOPSIS
  Reset Thor to the start of the demo timeline (Windows / PowerShell).
.DESCRIPTION
  Drops the TimescaleDB volume (telemetry, predictions, runs, contracts, approvals, registry
  rows) and, unless -KeepModels is given, the model + MLflow volumes too, then starts the
  stack again. The api reseeds the first 80% of the synthetic history and the replay streams
  the remaining 20% live, so MTR-042's degradation plays out on the Fleet screen during the
  demo. The Chroma index and embedding-model cache volumes are always kept.
.EXAMPLE
  .\scripts\demo_reset.ps1               # full reset (recommended before a recording)
  .\scripts\demo_reset.ps1 -KeepModels   # keep the deployed edge model + registry artifacts
#>
param([switch]$KeepModels)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$compose = "docker compose -f infra/docker-compose.yml"

Write-Host "==> stopping stack"
Invoke-Expression "$compose down --remove-orphans"
$vols = @("thor_timescale-data")
if (-not $KeepModels) { $vols += @("thor_models", "thor_mlflow-data") }
foreach ($v in $vols) {
  docker volume inspect $v *> $null
  if ($LASTEXITCODE -eq 0) { Write-Host "==> removing volume $v"; docker volume rm $v | Out-Null }
}

Write-Host "==> starting stack"
Invoke-Expression "$compose up -d"
Write-Host -NoNewline "==> waiting for api"
$ready = $false
for ($i = 0; $i -lt 90 -and -not $ready; $i++) {
  try { Invoke-RestMethod http://localhost:8000/system/health -TimeoutSec 2 | Out-Null; $ready = $true }
  catch { Write-Host -NoNewline "."; Start-Sleep -Seconds 2 }
}
Write-Host (" ready" * [int]$ready)
Invoke-RestMethod http://localhost:8000/system/health | ConvertTo-Json -Compress
Write-Host "==> fleet top 3"
(Invoke-RestMethod http://localhost:8000/fleet) | Select-Object -First 3 | ForEach-Object {
  Write-Host ("  {0} health {1} p {2}" -f $_.asset.asset_id, $_.health_score, $_.failure_probability)
}
Write-Host "==> done. Web: http://localhost:5173  (replay streams the last 20% over ~5 min per pass)"
