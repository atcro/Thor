#!/usr/bin/env bash
# Reset Thor to the start of the demo timeline.
#
# Uses the root-level `docker compose` (project name taken from the repo folder, e.g.
# `thor-main`), so it acts on the same stack the team starts by hand.
#
# Stops the stack, drops the TimescaleDB volume (telemetry, predictions, runs, contracts,
# approvals, registry rows) and, unless --keep-models is given, the model + MLflow volumes too,
# then starts the stack again. On start-up the api reseeds the first SEED_FRACTION (80%) of the
# synthetic history, the one-shot `prep` service trains + validates a champion (MTR-042 held
# out), approves its promotion as demo-prep@thor (parity was proven at validation), exports it
# for the edge and scores the fleet; only then do the replay and the edge start, streaming the remaining 20%
# live. Chroma index + embedding-model cache volumes are always kept.
#
# Expect 2-3 minutes for the prep step on first start (DEMO_PREP_TRIALS in .env sets the Optuna
# trials per family; default 6).
#
#   scripts/demo_reset.sh                # full reset (recommended before a rehearsal)
#   scripts/demo_reset.sh --keep-models  # keep model + MLflow volumes (prep still re-registers)
set -euo pipefail
cd "$(dirname "$0")/.."
KEEP_MODELS=0
[[ "${1:-}" == "--keep-models" ]] && KEEP_MODELS=1

PROJECT="$(docker compose config --format json | sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' | head -n 1)"
[[ -n "$PROJECT" ]] || { echo "could not read the compose project name; is Docker running?" >&2; exit 1; }
echo "==> compose project: $PROJECT"

echo "==> stopping stack"
docker compose down --remove-orphans
VOLS=("${PROJECT}_timescale-data")
[[ $KEEP_MODELS -eq 0 ]] && VOLS+=("${PROJECT}_models" "${PROJECT}_mlflow-data")
for v in "${VOLS[@]}"; do
  if docker volume inspect "$v" >/dev/null 2>&1; then
    echo "==> removing volume $v"; docker volume rm "$v" >/dev/null
  fi
done

echo "==> starting stack (api seeds 80%, prep trains + promotes, then replay + edge start)"
docker compose up -d
echo "==> following the prep service (train, validate incl. parity, promote, export, score)"
docker compose logs -f --no-log-prefix prep
PREP_EXIT="$(docker compose ps -a --format json prep | sed -n 's/.*"ExitCode": *\([0-9]*\).*/\1/p' | head -n 1)"
if [[ "${PREP_EXIT:-1}" != "0" ]]; then
  echo "demo prep failed with exit code ${PREP_EXIT:-?}; see 'docker compose logs prep'" >&2
  exit 1
fi

echo -n "==> waiting for api"
for _ in $(seq 1 90); do
  if curl -sf http://localhost:8000/system/health >/dev/null 2>&1; then echo " ready"; break; fi
  echo -n "."; sleep 2
done
curl -s http://localhost:8000/system/health
echo
echo "==> production model"
curl -s http://localhost:8000/models | python -c "import sys,json; [print(' ', m['name'], m['version'], m['family'], 'IMS', m['ims_total'], 'onnx', m['onnx_path']) for m in json.load(sys.stdin) if m['stage']=='production']" 2>/dev/null || true
echo "==> fleet top 3 (approved-model health)"
curl -s http://localhost:8000/fleet | python -c "import sys,json; [print(' ', f['asset']['asset_id'], 'health', f['health_score'], 'p', f['failure_probability'], 'regime', f['regime']) for f in json.load(sys.stdin)[:3]]" 2>/dev/null || true
echo "==> done. Web: http://localhost:5173  (replay streams the last 20% over ~5 min per pass)"
