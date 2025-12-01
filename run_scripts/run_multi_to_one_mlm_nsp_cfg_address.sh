#!/bin/bash
# Run Multi-to-One training for: MLM + NSP-CFG + Address
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
source "${SCRIPT_DIR}/common_run_env.sh"

# Mode-specific flags
ENABLE_MLM=true
ENABLE_NSP_CFG=true
ENABLE_NSP_DFG=false
ENABLE_SCOPE=false
USE_ADDRESS_EMBEDDING=true
INSTRUCTION_LEVEL_SEGMENT=false
NSP_CONTENT_MAX=60

# Build task flags
TASK_FLAGS=""
if [ "${ENABLE_MLM}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_mlm"
else
  TASK_FLAGS="${TASK_FLAGS} --disable_mlm"
fi
if [ "${ENABLE_NSP_CFG}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_nsp_cfg"
fi
if [ "${ENABLE_NSP_DFG}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_nsp_dfg"
fi
if [ "${ENABLE_SCOPE}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --enable_scope"
fi
if [ "${USE_ADDRESS_EMBEDDING}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --use_address_embedding"
fi
if [ "${INSTRUCTION_LEVEL_SEGMENT}" = true ]; then
  TASK_FLAGS="${TASK_FLAGS} --instruction_level_segment"
fi

# Auto-generate model name
MODEL_NAME=""
if [ "${ENABLE_MLM}" = true ]; then MODEL_NAME="${MODEL_NAME}_mlm"; fi
if [ "${ENABLE_NSP_CFG}" = true ]; then MODEL_NAME="${MODEL_NAME}_nsp_cfg"; fi
if [ "${ENABLE_NSP_DFG}" = true ]; then MODEL_NAME="${MODEL_NAME}_nsp_dfg"; fi
if [ "${ENABLE_SCOPE}" = true ]; then MODEL_NAME="${MODEL_NAME}_scope"; fi
if [ "${USE_ADDRESS_EMBEDDING}" = true ]; then MODEL_NAME="${MODEL_NAME}_address"; fi
if [ "${INSTRUCTION_LEVEL_SEGMENT}" = true ]; then MODEL_NAME="${MODEL_NAME}_ins"; fi
MODEL_NAME="${MODEL_NAME#_}"
if [ -z "${MODEL_NAME}" ]; then MODEL_NAME="baseline"; fi

OUTPUT_DIR="/work/kliu14/haapr/output/${MODEL_NAME}_multi_to_one"
LOG_DIR="/work/kliu14/haapr/log/${MODEL_NAME}_multi_to_one"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Starting multi-to-one training: ${MODEL_NAME}"

# export CUDA_VISIBLE_DEVICES=1  <-- Removed for HPC compatibility (let scheduler decide)
python "${REPO_ROOT}/train_multi_to_one.py" \
  --cfg_train "${CFG_TRAIN}" \
  --dfg_train "${DFG_TRAIN}" \
  --cfg_val "${CFG_VAL}" \
  --dfg_val "${DFG_VAL}" \
  --cfg_test "${CFG_TEST}" \
  --dfg_test "${DFG_TEST}" \
  --scope_train "${SCOPE_TRAIN}" \
  --scope_val "${SCOPE_VAL}" \
  --vocab "${VOCAB_PATH}" \
  --hidden ${HIDDEN} \
  --layers ${LAYERS} \
  --attn_heads ${ATTN_HEADS} \
  --seq_len ${SEQ_LEN} \
  --nsp_content_max ${NSP_CONTENT_MAX} \
  --dropout ${DROPOUT} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --warmup_steps ${WARMUP_STEPS} \
  --num_workers ${NUM_WORKERS} \
  --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
  --mask_prob ${MASK_PROB} \
  --nsp_prob ${NSP_PROB} \
  --data_percentage ${TRAIN_PERCENTAGE} \
  --val_percentage ${VAL_PERCENTAGE} \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --resume \
  ${TASK_FLAGS} \
  ${CUDA} ${MULTI_GPU} \
  2>&1 | tee "${LOG_DIR}/train_multi_to_one.log"

if [ $? -eq 0 ]; then
  echo "Multi-to-one training completed: ${OUTPUT_DIR}"
else
  echo "Multi-to-one training failed for ${MODEL_NAME}" >&2
  exit 1
fi
#!/bin/bash
# wrapper to run strupos/run_multi_to_one.sh with mode mlm_nsp_cfg_address
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
repo_root="$(cd "${script_dir}/.." >/dev/null && pwd)"

echo "running multi-to-one: mlm_nsp_cfg_address"
cd "${repo_root}/strupos"
bash run_multi_to_one.sh MLM_NSP_CFG_ADDRESS
