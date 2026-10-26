# Qwen3.8-Flash-Next on two DGX Sparks

A tested launch for serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` across two NVIDIA DGX Spark (GB10) units joined by
a single direct QSFP cable: tensor parallel 2 with expert parallelism and 4-token multi-token prediction, over RDMA
(RoCE v2) on the cable.

It is the two-unit configuration Petronella Technology Group, Inc. measured on its own DGX Spark pod on
2026-10-02. Every number below comes from that run; the full results, including the four- and eight-unit
versions, are on our [Qwen3.8-Flash-Next benchmark page](https://petronellatech.com/ai/llm-benchmarks/models/qwen3-8-flash-next/).

## Measured on two DGX Sparks (2026-10-02)

| Test | Result |
|---|---|
| Decode, 1 user, prose | 35.2 tok/s |
| Decode, 1 user, code | 52.6 tok/s |
| Decode, 32 users at once | 329.6 tok/s aggregate |
| Most users served at our per-user speed floor | 32 |
| Prefill, 8K-token prompt | 3,089 tok/s |
| Prefill, 32K-token prompt | 3,069 tok/s |
| Energy | 1.16 tokens per joule |
| Long-context gate | needle retrieval passed at every tested length up to 120,001 tokens |

Conditions: both units had their GPU clocks locked to a 300 to 2300 MHz range for the whole program, so units ran at
the same speed from run to run. Speed tests are only recorded after the engine passes its correctness gate. How we
measure, and the per-user floor, are on our [methodology page](https://petronellatech.com/ai/llm-benchmarks/methodology/).

What more units buy (same image and settings, measured the same day, not covered by this script):

| Units | 1 user, prose | 32 users | At the user ceiling |
|---|---|---|---|
| 2 | 35.2 tok/s | 329.6 tok/s | 329.6 tok/s at 32 users |
| 4 | 41.4 tok/s | 482.4 tok/s | 648.7 tok/s at 64 users |
| 8 | 45.7 tok/s | 604.7 tok/s | 769.7 tok/s at 64 users |

## What you need

- Two DGX Spark (or other GB10) units with Docker and the NVIDIA container toolkit.
- One direct QSFP cable between their ConnectX-7 ports, with an IP address on each end in a small private subnet
  and RDMA working over it. Our [GB10 cluster guide](https://github.com/capetron/gb10-cluster-guide) walks through
  the cabling and network setup, and [gb10-cluster-check](https://github.com/capetron/gb10-cluster-check) verifies it.
- The checkpoint on BOTH units at the revision we measured:

```bash
huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4 --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594
docker pull oxbyte/qwen3.8-flash-next-dual-spark:latest
```

## Run

On both units, with your own cable subnet and addresses (the example uses 192.168.100.1 for rank 0 and
192.168.100.2 for rank 1):

```bash
# unit B first (rank 1, no API)
CABLE_SUBNET=192.168.100.0/24 ./launch/node.sh 1 192.168.100.2 192.168.100.1

# then unit A (rank 0, serves the API on port 8888)
CABLE_SUBNET=192.168.100.0/24 ./launch/node.sh 0 192.168.100.1 192.168.100.1

# when rank 0 logs that the server is up
./scripts/smoke-test.sh http://<unit-A-address>:8888/v1
```

`launch/node.sh` documents every setting. If your cable interface or RDMA device has a different name, set `IFACE`
and `HCA` (`ip -br link` and `ibv_devices` show them). To stop, run `docker rm -f qwen38-flash-next` on both units.

## Settings and why

| Setting | Value | Why |
|---|---|---|
| Parallelism | `--tensor-parallel-size 2 --enable-expert-parallel` | one rank per unit; expert parallel spreads the MoE experts |
| Speculative decoding | MTP, 4 tokens | the model's own multi-token prediction head |
| Concurrency | `--max-num-seqs 64`, `--gpu-memory-utilization 0.75` | the settings measured above |
| Context | `--max-model-len 131072` | the gate passed up to 120,001 tokens |
| Execution | `--enforce-eager` | the configuration we measured; CUDA graphs were not part of this run |
| Transport | NCCL over RDMA on the cable (`NCCL_NET=IB`, RoCE v2, one HCA) | tensor-parallel traffic stays on the direct link |

## Known limits

- The image tag is `latest`. We measured the build with image ID
  `sha256:eace86f3428a25554870f5a489598963ba19df6725ca92267c05979eb7079437`; check `docker image inspect` and re-test
  if yours differs.
- vLLM binds `0.0.0.0` with no API key. Keep port 8888 on a private network or behind an authenticating proxy.
- This script is two units only. The four- and eight-unit rows above used the same image and settings with more ranks.

## Credits

- Container image: [oxbyte](https://hub.docker.com/r/oxbyte/qwen3.8-flash-next-dual-spark), which packages vLLM for
  this model on dual Sparks. This repo adds the launch, transport settings and measurements, not the image.
- Checkpoint: RadixArk's NVFP4 quantization of Qwen3.8-Flash-Next.
- Engine: [vLLM](https://github.com/vllm-project/vllm).

## About

Built and measured by [Petronella Technology Group, Inc.](https://petronellatech.com/), which designs and runs
private AI on hardware its clients own. More measured configurations are on our
[LLM benchmark hub](https://petronellatech.com/ai/llm-benchmarks/).

License: MIT (see `LICENSE`).
