#!/usr/bin/env bash
#
# Compute TOPReward RA-BC progress for a dataset, then render a progress-overlay
# MP4/GIF for one episode.
#
# Chains two scripts:
#   1. `lerobot.rewards.topreward.compute_rabc_weights` - writes topreward_progress.parquet
#      and (always, here) pushes it to the dataset's HF Hub repo.
#   2. `examples/dataset/create_progress_videos.py`       - fetches it back and renders the overlay.
#
# Step 2 only ever fetches --progress-file from the dataset's Hub repo, never a local
# path, so step 1 always runs with --push-to-hub - otherwise step 2 has nothing to find.
#
# Usage:
#   ./run_rabc_and_render.sh --dataset-repo-id manual-cognition/MCopenarm_folding_test_20260714_081703 \
#       --episode 0 --camera-key observation.images.base \
#       --vlm-name Qwen/Qwen3-VL-4B-Instruct --num-samples 15 \
#       --task-prompt "Fold the T-shirt"

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATASET_REPO_ID="manual-cognition/MCopenarm_folding_test_20260714_081703"
# DATASET_REPO_ID="manual-cognition/mc_glove05_boxes_in_box"
EPISODE=0
CAMERA_KEY="observation.images.base"
VLM_NAME="Qwen/Qwen3-VL-8B-Instruct"
NUM_SAMPLES=15
FPS=5
DEVICE="cuda"
TASK_PROMPT="Fold the T-shirt and slide it to the upper-left corner of the table."
# TASK_PROMPT="Pick up the small cube and place it inside the large card-board box."
OUTPUT_DIR="progress_videos"
OUTPUT_NAME="Qwen3-VL-8B-Instruct_2FPS_ElaboratedPrompt"

usage() {
    cat <<EOF
Usage: $0 --dataset-repo-id REPO_ID [options]

Required:
  --dataset-repo-id REPO_ID    HuggingFace dataset repo id

Options:
  --episode N                  Episode index to render (default: $EPISODE)
  --camera-key KEY             Camera observation key (default: dataset's first camera)
  --vlm-name NAME               VLM backbone (default: $VLM_NAME)
  --num-samples N                RA-BC anchor prefix samples per episode (default: $NUM_SAMPLES)
  --fps FPS                       Target playback fps used to subsample frames before encoding
                                   (default: TOPRewardConfig.fps, currently 2.0)
  --device DEVICE                 cuda|cpu (default: $DEVICE)
  --task-prompt TEXT               Force this task instruction for every episode
  --output-dir DIR                  Where to write the MP4/GIF (default: $OUTPUT_DIR)
  --output-name NAME                  Rename the resulting MP4/GIF to NAME.mp4/NAME.gif
                                       (default: create_progress_videos.py's own
                                       {repo_id}_ep{episode}_progress naming)
  -h, --help                         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-repo-id) DATASET_REPO_ID="$2"; shift 2 ;;
        --episode) EPISODE="$2"; shift 2 ;;
        --camera-key) CAMERA_KEY="$2"; shift 2 ;;
        --vlm-name) VLM_NAME="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --task-prompt) TASK_PROMPT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-name) OUTPUT_NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASET_REPO_ID" ]]; then
    echo "Error: --dataset-repo-id is required" >&2
    usage
    exit 1
fi

echo "==> [1/2] Computing TOPReward RA-BC progress for $DATASET_REPO_ID (vlm=$VLM_NAME, num_samples=$NUM_SAMPLES)"

COMPUTE_ARGS=(
    -m lerobot.rewards.topreward.compute_rabc_weights
    --dataset-repo-id "$DATASET_REPO_ID"
    --num-samples "$NUM_SAMPLES"
    --device "$DEVICE"
    --vlm-name "$VLM_NAME"
    --push-to-hub
)
if [[ -n "$TASK_PROMPT" ]]; then
    COMPUTE_ARGS+=(--task-prompt "$TASK_PROMPT")
fi
if [[ -n "$FPS" ]]; then
    COMPUTE_ARGS+=(--fps "$FPS")
fi

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    uv run python "${COMPUTE_ARGS[@]}"

echo "==> [2/2] Rendering progress GIF for episode $EPISODE"

RENDER_ARGS=(
    examples/dataset/create_progress_videos.py
    --repo-id "$DATASET_REPO_ID"
    --episode "$EPISODE"
    --progress-file topreward_progress.parquet
    --output-dir "$OUTPUT_DIR"
    --gif
)
if [[ -n "$CAMERA_KEY" ]]; then
    RENDER_ARGS+=(--camera-key "$CAMERA_KEY")
fi
if [[ -n "$TASK_PROMPT" ]]; then
    RENDER_ARGS+=(--task-prompt "$TASK_PROMPT")
fi

uv run "${RENDER_ARGS[@]}"

if [[ -n "$OUTPUT_NAME" ]]; then
    SAFE_REPO_NAME="${DATASET_REPO_ID//\//_}"
    DEFAULT_STEM="${SAFE_REPO_NAME}_ep${EPISODE}_progress"
    for ext in mp4 gif; do
        src="$OUTPUT_DIR/${DEFAULT_STEM}.${ext}"
        dst="$OUTPUT_DIR/${OUTPUT_NAME}.${ext}"
        if [[ -f "$src" ]]; then
            mv -f "$src" "$dst"
            echo "==> Renamed $src -> $dst"
        fi
    done
fi

echo "==> Done. Output in $OUTPUT_DIR/"
