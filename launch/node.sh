#!/bin/bash
# Start one rank of Qwen3.8-Flash-Next NVFP4 across two DGX Spark (GB10) units joined by one direct QSFP cable.
# Run it on BOTH units: rank 1 (headless) first, then rank 0 (serves the API).
#
# usage: launch/node.sh RANK THIS_UNIT_CABLE_IP RANK0_CABLE_IP
#   RANK                0 on the unit that serves the API, 1 on the other
#   THIS_UNIT_CABLE_IP  this unit's address on the direct-cable link
#   RANK0_CABLE_IP      rank 0's address on the direct-cable link (the same on both units)
#
# Environment (defaults are what we measured with):
#   CABLE_SUBNET   the direct-cable subnet, e.g. 192.168.100.0/24 (required)
#   IFACE          cable network interface        (enp1s0f0np0, the usual DGX Spark name)
#   HCA            RDMA device for that interface  (rocep1s0f0)
#   MODEL_DIR      the snapshot of RadixArk/Qwen3.8-Flash-Next-NVFP4 as seen INSIDE the container, where
#                  ~/.cache/huggingface is mounted at /hf (default: the revision we measured)
#   IMAGE          oxbyte/qwen3.8-flash-next-dual-spark:latest
#   PORT 8888  MPORT 29511  SEQS 64  UTIL 0.75  MAXLEN 131072  SPEC 4  NAME qwen38-flash-next
set -euo pipefail
RANK="$1"; HOSTIP="$2"; MASTER="$3"
: "${CABLE_SUBNET:?set CABLE_SUBNET to the direct-cable subnet, e.g. 192.168.100.0/24}"
IFACE="${IFACE:-enp1s0f0np0}"; HCA="${HCA:-rocep1s0f0}"
REV=7b719225242aacd3dbd3f9407468c2ee9a9d2594
MODEL_DIR="${MODEL_DIR:-/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$REV}"
IMAGE="${IMAGE:-oxbyte/qwen3.8-flash-next-dual-spark:latest}"
PORT="${PORT:-8888}"; MPORT="${MPORT:-29511}"; SEQS="${SEQS:-64}"; UTIL="${UTIL:-0.75}"
MAXLEN="${MAXLEN:-131072}"; SPEC="${SPEC:-4}"; NAME="${NAME:-qwen38-flash-next}"

# keep the previous container's logs before replacing it
if docker ps -aq -f "name=^${NAME}$" | grep -q .; then
  docker logs "$NAME" >"$HOME/${NAME}.prev.$(date +%s).log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi

NENV=(-e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE"
      -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$HCA" -e NCCL_IB_ADDR_FAMILY=AF_INET
      -e NCCL_IB_ADDR_RANGE="$CABLE_SUBNET" -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_QPS_PER_CONNECTION=1
      -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_MAX_NCHANNELS=8
      -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN)

ARGS=("$MODEL_DIR" --served-model-name qwen3.8-flash-next --quantization modelopt_fp4
      --tensor-parallel-size 2 --pipeline-parallel-size 1
      --nnodes 2 --node-rank "$RANK" --master-addr "$MASTER" --master-port "$MPORT"
      --distributed-executor-backend mp
      --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192
      --enable-chunked-prefill --gpu-memory-utilization "$UTIL"
      --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3
      --kv-cache-dtype auto --load-format safetensors --enforce-eager
      --enable-prefix-caching --no-enable-flashinfer-autotune --enable-expert-parallel)
[ "$SPEC" != "0" ] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC}}")
if [ "$RANK" = "0" ]; then ARGS+=(--host 0.0.0.0 --port "$PORT"); else ARGS+=(--headless); fi

docker run -d --name "$NAME" \
  --network host --ipc host --shm-size 32g --gpus all \
  --ulimit memlock=-1 --ulimit nofile=1048576 --device /dev/infiniband --cap-add IPC_LOCK \
  -v "$HOME/.cache/huggingface:/hf:ro" -v "$HOME/.cache/qwen38-vllm-cache:/cache" \
  "${NENV[@]}" \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_HOST_IP="$HOSTIP" \
  --entrypoint vllm "$IMAGE" serve "${ARGS[@]}" >/dev/null
echo "started $NAME rank=$RANK on $(hostname); follow it with: docker logs -f $NAME"
