#!/usr/bin/env bash
# Reset Thor to the start of the demo.
#
# Uses the root-level `docker compose` (project name taken from the repo folder, e.g.
# `thor-main`), so it acts on the same stack the team starts by hand.
#
# Stops the stack, drops the TimescaleDB volume (telemetry, predictions, alarms, runs, contracts,
# approvals, registry rows) and, unless --keep-models is given, the model + MLflow volumes too,
# then starts the stack again. On start-up the api seeds the first SEED_FRACTION (80%) of the
# recorded history and then NOTHING else happens: no production model, every motor shows
# "no prediction yet", and the replay container holds until a named approver promotes a model
# in ModelOps. The presenter drives it from there (docs/finals/demo-walkthrough.md): AutoML
# Studio > Train new model, ModelOps > Approve promotion, and the recorded history starts
# streaming by itself. Chroma index + embedding-model cache volumes are always kept.
#
# --prep runs the manual fallback instead (docker compose run --rm prep): train, validate,
# promote as demo-prep@thor and score the fleet, for when training on stage is not an option.
#
#   scripts/demo_reset.sh                # clean start: seeded, no model, replay held
#   scripts/demo_reset.sh --keep-models  # keep model + MLflow volumes
#   scripts/demo_reset.sh --prep         # clean start + the fallback prep job
set -euo pipefail
cd "$(dirname "$0")/.."
KEEP_MODELS=0
PREP=0
for arg in "$@"; do
  case "$arg" in
    --keep-models) KEEP_MODELS=1 ;;
    --prep) PREP=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

PROJECT="$(docker compose config --format json | sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' | head -n 1)"
[[ -n "$PROJECT" ]] || { echo "could not read the compose project name; is Docker running?" >&2; exit 1; }
echo "==> compose project: $PROJECT"

echo "==> stopping stack"
docker compose --profile prep down --remove-orphans
docker rm -f "${PROJECT}-prep-1" >/dev/null 2>&1 || true   # an exited prep container pins the models volume
VOLS=("${PROJECT}_timescale-data")
[[ $KEEP_MODELS -eq 0 ]] && VOLS+=("${PROJECT}_models" "${PROJECT}_mlflow-data")
EXISTING="$(docker volume ls -q)"
for v in "${VOLS[@]}"; do
  if grep -qx "$v" <<<"$EXISTING"; then
    echo "==> removing volume $v"; docker volume rm "$v" >/dev/null
  fi
done

echo "==> starting stack (api seeds 80% of the recorded history; no model, no predictions, replay held)"
docker compose up -d

echo -n "==> waiting for the api to finish seeding"
READY=0
for _ in $(seq 1 150); do
  if curl -sf -m 15 http://localhost:8000/system/health 2>/dev/null | grep -q '"n_telemetry_rows": *[1-9]'; then READY=1; break; fi
  echo -n "."; sleep 2
done
[[ $READY -eq 1 ]] || { echo "api did not come up within 5 minutes; see 'docker compose logs api'" >&2; exit 1; }
echo " ready"
curl -s http://localhost:8000/system/health
echo

if [[ $PREP -eq 1 ]]; then
  echo "==> fallback prep: train, validate, promote as demo-prep@thor, score the fleet (~60-90 s)"
  docker compose run --rm prep
fi

echo "==> production model"
curl -s http://localhost:8000/models | python -c "import sys,json; ms=[m for m in json.load(sys.stdin) if m['stage']=='production']; print('  none (expected before the demo)' if not ms else '\n'.join(f\"  {m['name']} {m['version']} {m['family']} IMS {m['ims_total']}\" for m in ms))" 2>/dev/null || true
echo "==> replay"
curl -s http://localhost:8000/replay
echo
echo "==> fleet"
curl -s http://localhost:8000/fleet | python -c "import sys,json; f=json.load(sys.stdin); print(f'  {len(f)} motors, {sum(1 for x in f if x[\"health_score\"] is None)} without a prediction yet')" 2>/dev/null || true
echo
echo "==> done. Open http://localhost:5173 (lands on Fleet: every motor grey, 'no prediction yet')."
echo "    1. AutoML Studio > Train new model (2-3 min; talk over the split, trials, gate, parity)"
echo "    2. ModelOps > Approve promotion with your name -> the fleet turns green and the replay starts"
echo "    3. MTR-042 goes amber then red on its own ~20-60 s later; Run analysis, ask Bolt, approve"
echo "    Recovery: Pause/Resume in the top bar; fallback training: scripts/demo_reset.sh --prep"
echo "    Full click path and timings: docs/finals/demo-walkthrough.md"
