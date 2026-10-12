<#
.SYNOPSIS
  Reset Thor to the start of the demo (Windows / PowerShell).
.DESCRIPTION
  Uses the root-level `docker compose` (project name taken from the repo folder, e.g.
  `thor-main`), so it acts on the same stack the team starts by hand.

  Stops the stack, drops the TimescaleDB volume (telemetry, predictions, alarms, runs,
  contracts, approvals, registry rows) and, unless -KeepModels is given, the model + MLflow
  volumes too, then starts the stack again. On start-up the api seeds the first 80% of the
  recorded history and then NOTHING else happens: there is no production model, every motor
  shows "no prediction yet", and the replay container holds until a named approver promotes a
  model in ModelOps. The presenter drives it from there (docs/finals/demo-walkthrough.md):
  AutoML Studio > Train new model, ModelOps > Approve promotion, and the recorded history
  starts streaming by itself. The Chroma index and embedding-model cache volumes are kept.

  -Prep runs the manual fallback instead (docker compose run --rm prep): train, validate,
  promote as demo-prep@thor and score the fleet, for when training on stage is not an option.
.EXAMPLE
  .\scripts\demo_reset.ps1               # clean start: seeded, no model, replay held
  .\scripts\demo_reset.ps1 -KeepModels   # keep model + MLflow volumes (the stage still starts clean: the registry rows live in the DB)
  .\scripts\demo_reset.ps1 -Prep         # clean start + the fallback prep job (no training on stage)
#>
param([switch]$KeepModels, [switch]$Prep)
# "Continue", not "Stop": docker writes its progress to stderr, which Windows PowerShell 5.1
# would otherwise turn into a terminating error. Exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"
function Assert-Ok([string]$what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }
Set-Location (Join-Path $PSScriptRoot "..")

$project = (docker compose config --format json | ConvertFrom-Json).name
if (-not $project) { throw "could not read the compose project name; is Docker running?" }
Write-Host "==> compose project: $project"

Write-Host "==> stopping stack"
docker compose --profile prep down --remove-orphans
Assert-Ok "docker compose down"
# an exited fallback-prep container keeps the models volume "in use"; remove it explicitly
docker rm -f "${project}-prep-1" 2>$null | Out-Null
$vols = @("${project}_timescale-data")
if (-not $KeepModels) { $vols += @("${project}_models", "${project}_mlflow-data") }
$existing = @(docker volume ls -q)
foreach ($v in $vols) {
  if ($existing -contains $v) { Write-Host "==> removing volume $v"; docker volume rm $v | Out-Null }
}

Write-Host "==> starting stack (api seeds 80% of the recorded history; no model, no predictions, replay held)"
docker compose up -d
Assert-Ok "docker compose up"

Write-Host -NoNewline "==> waiting for the api to finish seeding"
$ready = $false
for ($i = 0; $i -lt 150 -and -not $ready; $i++) {
  try {
    $h = Invoke-RestMethod http://localhost:8000/system/health -TimeoutSec 15
    if ($h.n_telemetry_rows -gt 0) { $ready = $true } else { Write-Host -NoNewline "."; Start-Sleep -Seconds 2 }
  } catch { Write-Host -NoNewline "."; Start-Sleep -Seconds 2 }
}
if (-not $ready) { throw "api did not come up within 5 minutes; see 'docker compose logs api'" }
Write-Host " ready"
Invoke-RestMethod http://localhost:8000/system/health | ConvertTo-Json -Compress

if ($Prep) {
  Write-Host "==> fallback prep: train, validate, promote as demo-prep@thor, score the fleet (~60-90 s)"
  docker compose run --rm prep
  Assert-Ok "docker compose run prep"
}

$replay = Invoke-RestMethod http://localhost:8000/replay
$models = @(Invoke-RestMethod http://localhost:8000/models)
$prod = $models | Where-Object { $_.stage -eq "production" }
Write-Host "==> production model:" $(if ($prod) { "$($prod.name) $($prod.version) $($prod.family) IMS $($prod.ims_total)" } else { "none (expected before the demo)" })
Write-Host "==> replay: released=$($replay.released) paused=$($replay.paused) plant time $($replay.plant_time)"
# Invoke-WebRequest + ConvertFrom-Json: on Windows PowerShell 5.1 Invoke-RestMethod can hand a
# JSON array back as one object, which made this line count "1 motor".
$fleet = ConvertFrom-Json (Invoke-WebRequest -UseBasicParsing http://localhost:8000/fleet).Content
$noPred = @($fleet | Where-Object { $null -eq $_.health_score }).Count
Write-Host "==> fleet: $(@($fleet).Count) motors, $noPred without a prediction yet"
Write-Host ""
Write-Host "==> done. Open http://localhost:5173 (lands on Fleet: every motor grey, 'no prediction yet')."
Write-Host "    1. AutoML Studio > Train new model (2-3 min; talk over the split, trials, gate, parity)"
Write-Host "    2. ModelOps > Approve promotion with your name -> the fleet turns green and the replay starts"
Write-Host "    3. MTR-042 goes amber then red on its own ~20-60 s later; Run analysis, ask Bolt, approve"
Write-Host "    Recovery: Pause/Resume in the top bar; fallback training: .\scripts\demo_reset.ps1 -Prep"
Write-Host "    Full click path and timings: docs\finals\demo-walkthrough.md"
