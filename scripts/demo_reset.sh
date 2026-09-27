#!/usr/bin/env bash
# Reset Thor to the start of the demo timeline.
#
# Drops the TimescaleDB volume (telemetry, predictions, runs, contracts, approvals, registry
# rows) and, unless --keep-models is given, the model + MLflow volumes too, then starts the
# stack again. The api reseeds the first SEED_FRACTION (80%) of the synthetic history and the
# replay streams the remaining 20% live, so MTR-042's degradation plays out on the Fleet
# screen during the demo. Chroma index + embedding-model cache volumes are always kept (they
# are slow to rebuild and contain no demo state).
#
#   scripts/demo_reset.sh                # full reset (recommended before a recording)
#   scripts/demo_reset.sh --keep-models  # keep the deployed edge model + registry artifacts
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE="docker compose -f infra/docker-compose.yml"
KEEP_MODELS=0
[[ "${1:-}" == "--keep-models" ]] && KEEP_MODELS=1

echo "==> stopping stack"
$COMPOSE down --remove-orphans
VOLS=(thor_timescale-data)
[[ $KEEP_MODELS -eq 0 ]] && VOLS+=(thor_models thor_mlflow-data)
for v in "${VOLS[@]}"; do
  if docker volume inspect "$v" >/dev/null 2>&1; then
    echo "==> removing volume $v"; docker volume rm "$v" >/dev/null
  fi
done

echo "==> starting stack"
$COMPOSE up -d
echo -n "==> waiting for api"
for _ in $(seq 1 90); do
  if curl -sf http://localhost:8000/system/health >/dev/null 2>&1; then echo " ready"; break; fi
  echo -n "."; sleep 2
done
curl -s http://localhost:8000/system/health
echo
echo "==> fleet top 3"
curl -s http://localhost:8000/fleet | python -c "import sys,json; [print(' ', f['asset']['asset_id'], 'health', f['health_score'], 'p', f['failure_probability']) for f in json.load(sys.stdin)[:3]]" 2>/dev/null || true
echo "==> done. Web: http://localhost:5173  (replay streams the last 20% over ~5 min per pass)"
