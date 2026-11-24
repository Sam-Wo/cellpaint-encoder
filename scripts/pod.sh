#!/usr/bin/env bash
# One entry point per RUNBOOK stage, run from the repo root on the pod/instance.
#   bash scripts/pod.sh setup      # venv, CUDA torch, deps, tests, weights check
#   bash scripts/pod.sh g0         # manifest + S3 / cache-rate / pixel-size checks (CPU pod is enough)
#   bash scripts/pod.sh cache      # tile cache + dev cache, then loader benchmark
#   bash scripts/pod.sh baselines  # dev-cache ImageNet/random baselines (G-B, our side; no source_6 metrics)
#   bash scripts/pod.sh pilot      # G1: 2 h of the main run + gate report       <- the GPU pilot
#   bash scripts/pod.sh main       # continue seed 0 to 100 epochs (resumes the pilot)
#   bash scripts/pod.sh seed1 | abl_natural_mix | abl_perimage_norm | abl_scratch
# Environment: CFG (default configs/full.yaml), RUN_ROOT (default runs),
#   PROCS (default: CPUs this container may use, i.e. min(nproc, cgroup CPU quota)).
set -euo pipefail
STAGE=${1:?stage}
CFG=${CFG:-configs/full.yaml}
RUN_ROOT=${RUN_ROOT:-runs}

eff_cpus() {  # vast.ai/docker: nproc can report every host core; the cgroup quota is what we get
  local n q="" p=""
  n=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
  fi
  if [[ "$q" =~ ^[0-9]+$ ]] && [[ "$p" =~ ^[0-9]+$ ]] && [ "$q" -gt 0 ] && [ "$p" -gt 0 ]; then
    local c=$(( q / p ))
    [ "$c" -ge 1 ] && [ "$c" -lt "$n" ] && n=$c
  fi
  echo "$n"
}
if [ -z "${PROCS:-}" ]; then
  PROCS=$(eff_cpus)
  echo "PROCS=$PROCS (nproc=$(nproc); set PROCS to override, e.g. to the offer's cpu_cores_effective)" >&2
fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
DINO_URL=https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
DINO_SHA=1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7
mkdir -p results "$RUN_ROOT"
[ "$STAGE" != setup ] && [ -f .venv/bin/activate ] && . .venv/bin/activate

train_loop() {  # restart after pre-emption (exit 75 = stopped by SIGTERM after checkpointing)
  local rc=75
  while [ $rc -eq 75 ]; do
    set +e; python scripts/train.py "$@"; rc=$?; set -e
    [ $rc -eq 75 ] && echo "pre-empted; resuming in 30 s" && sleep 30
  done
  return $rc
}

with_gpu_log() {  # $1 = run dir; rest = command
  local run=$1; shift
  mkdir -p "$run"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 10 >> "$run/gpu_util.csv" &
  local smi=$!
  set +e; "$@"; local rc=$?; set -e
  kill $smi 2>/dev/null || true
  return $rc
}

case "$STAGE" in
  setup)
    PY=${PYTHON:-python3}
    $PY -c 'import sys; assert sys.version_info >= (3, 11), f"Python >= 3.11 required, found {sys.version.split()[0]}: use a 3.11+ image or set PYTHON=python3.11"'
    $PY -m venv .venv && . .venv/bin/activate
    pip install --upgrade pip
    pip install torch==2.10.0 --index-url "$TORCH_INDEX"
    grep -v '^torch==' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
    pytest -q -x
    mkdir -p data/weights
    [ -f data/weights/dino_deitsmall16_pretrain.pth ] || curl -fL -o data/weights/dino_deitsmall16_pretrain.pth "$DINO_URL"
    echo "$DINO_SHA  data/weights/dino_deitsmall16_pretrain.pth" | sha256sum -c
    python scripts/fetch_jump_metadata.py   # pinned JUMP plate/well/compound tables + Target-2 targets (E3/E4, M4)
    df -h . | tail -1
    test -f data/anchor/anchor_wells_high.parquet || { echo "MISSING data/anchor/: rsync it from the laptop (RUNBOOK 2.3)"; exit 1; }
    ;;
  g0)
    python scripts/build_manifest.py --config "$CFG"
    python scripts/bench_io.py s3 --config "$CFG" --seconds 600 --workers 64 | tee results/g0_s3.json
    python scripts/bench_io.py cache --config "$CFG" --sites 200 --workers "$PROCS" | tee results/g0_cache.json
    ;;
  cache)
    python scripts/write_shards.py --config "$CFG" --procs "$PROCS"
    python scripts/bench_io.py loader --config "$CFG" --batches 100 | tee results/g0_loader.json
    ;;
  baselines)
    python scripts/dev_eval.py --config "$CFG" --encoder imagenet_dino --out results/dev_imagenet.json
    python scripts/dev_eval.py --config "$CFG" --encoder random_init --out results/dev_random.json
    ;;
  pilot)
    test -f results/dev_imagenet.json || bash scripts/pod.sh baselines
    with_gpu_log "$RUN_ROOT/full_seed0" train_loop --config configs/pilot.yaml --run-dir "$RUN_ROOT/full_seed0"
    python scripts/check_gates.py g1 --run "$RUN_ROOT/full_seed0" --imagenet-dev results/dev_imagenet.json \
      --gpu-util "$RUN_ROOT/full_seed0/gpu_util.csv" | tee results/g1_report.txt
    ;;
  main)
    with_gpu_log "$RUN_ROOT/full_seed0" train_loop --config "$CFG" --run-dir "$RUN_ROOT/full_seed0"
    ;;
  seed1)
    with_gpu_log "$RUN_ROOT/full_seed1" train_loop --config "$CFG" --run-dir "$RUN_ROOT/full_seed1" run.seed=1
    ;;
  abl_natural_mix|abl_scratch|abl_perimage_norm)
    if [ "$STAGE" = abl_perimage_norm ] && [ ! -f data/shards/train_perimage/index.parquet ]; then
      # a second full training cache (~232 GB, budget 358 GB raw); vast disks cannot grow, so check first
      need_kb=$(( ${PERIMAGE_NEED_GB:-380} * 1024 * 1024 ))
      free_kb=$(df -Pk data | awk 'NR==2 {print $4}')
      if [ "$free_kb" -lt "$need_kb" ]; then
        echo "only $((free_kb / 1048576)) GB free, need ${PERIMAGE_NEED_GB:-380} GB for the per-image cache." >&2
        echo "Run this ablation last: once seed0, seed1, abl_natural_mix and abl_scratch are finished," >&2
        echo "delete the per-plate training cache (rm -rf data/shards/train; keep data/shards/dev) and retry." >&2
        exit 1
      fi
      python scripts/write_shards.py --config "configs/ablations/$STAGE.yaml" --only train --procs "$PROCS"
    fi
    with_gpu_log "$RUN_ROOT/$STAGE" train_loop --config "configs/ablations/$STAGE.yaml" --run-dir "$RUN_ROOT/$STAGE"
    ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
