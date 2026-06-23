"""
Solutions -- Day 9 : Inference at Scale
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Framework choice.

    1) Startup, Llama-3-8B chatbot, 1 A100 :
       -> **vLLM**. Reasons :
         - Continuous batching out-of-the-box, very simple to launch
           (`python -m vllm.entrypoints.openai.api_server --model ...`).
         - OpenAI-compatible API, instant plug-in for the client.
         - Best perf/cost on the market in 2025 for open source models.
         - Zero complex infra, perfect for a small team.

    2) Bank, BERT classification, 4 T4s, multi-model hosting :
       -> **Triton Inference Server**. Reasons :
         - Native multi-model serving (a single server serves 10+ models).
         - Model ensembling, dynamic batching, versioning.
         - Supports PyTorch, TensorRT, ONNX, Python backend -> flexible for
           heterogeneous data science teams.
         - Excellent monitoring and Prometheus metrics.

    3) AI vendor, 100K users, maximize the margin :
       -> **TensorRT-LLM**. Reasons :
         - The deepest optimization per NVIDIA GPU (graph compilation,
           fused kernels, int8/int4 calibrated for each architecture).
         - 20-40% gain over vLLM at scale, which represents millions
           of dollars when running at 100K users.
         - Downside : steep learning curve, an expert team is
           required (model compilation, quantization, etc.).

    4) MacBook M3 Pro, personal Llama-3-70B :
       -> **llama.cpp / Ollama**. Reasons :
         - Optimized for Apple Silicon (Metal) and CPU.
         - GGUF quantization (int4 -> 35 GB, fits in the unified RAM of an M3 Pro).
         - Ollama is the simplest : `ollama run llama3:70b-q4`.

    5) HF-centric team, quick fine-tune deploy :
       -> **TGI (Text Generation Inference)**. Reasons :
         - Native integration with the HuggingFace Hub, a private model deployed
           with a single docker command.
         - Standard SSE streaming.
         - Continuous batching and quantization support.
         - Good perf/ease compromise when vLLM feels too "raw".
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- Sizing Llama-3-70B int8.

    Assumptions :
      - 1 H100 in int8 serves ~30 concurrent reqs with p99 < 3 s
      - Peak : 1000 concurrent
      - SLA : p99 < 5 s, 99.9% uptime
      - H100 : $3/h cloud

    Q1 -- Minimum number of H100s for the peak :
      1000 / 30 = 33.33 -> rounded up = **34 H100s**.
      That size will put you at the edge of the SLA. In practice, headroom is needed.

    Q2 -- Safety margin :
      Recommendation : **+25% headroom** -> ~42 H100s for the peak.
      Reasons :
        - Natural traffic variance (bursts)
        - Some requests have longer outputs (the queue fills up)
        - To hold the p99 < 5 s (not just the p50), we must stay
          far from saturation. At 90% util, the queues explode.

    Q3 -- Approximate monthly cost :
      Average case : we do not run 42 H100s all the time.
      - 30% of the time at peak (42 GPUs) = 0.30 * 720h * 42 * $3 = $27,216
      - 10% of the time at mid (20 GPUs)  = 0.10 * 720h * 20 * $3 = $4,320
      - 60% of the time at low (10 GPUs)  = 0.60 * 720h * 10 * $3 = $12,960
      Total : ~**$44K / month**.
      (The prompt says peak 30%, off-peak 10% -- interpreted as mid 10%, low 60%
      implicit. In all cases, we are in the $30-50K/month range for a
      load of 1000 concurrent 24/7.)

    Q4 -- Int4 + 2 models per H100 :
      In int4 (~35 GB per model), we can load 2 replicas of the same model
      per H100. Careful : the throughput does not necessarily double, because they
      share the compute units. In practice : 1.5-1.8x gain.
      With an effective 1.7x : 34 / 1.7 = 20 H100s -> with headroom 25 H100s.
      Savings : ~$20K / month. BUT : int4 quality loss to validate on
      a business benchmark before committing.

    Q5 -- 99.9% uptime :
      99.9% = max 8.7h of downtime per year. A routine maintenance of one
      GPU, a preempted instance, etc. consume the error budget quickly.
      Recommendation :
        - N+2 at minimum (2 replicas on top of the required minimum)
        - Deploy across 2 distinct availability zones
        - Health checks + auto eviction
        - Rolling deploys with canary, never a full swap
      -> +2-4 extra H100s compared to N, but spread across 2 AZs.
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- Tuning the dynamic batching.

    Current state :
      max_batch_size=32, max_wait=100ms
      p50=180ms, p99=950ms, throughput=180 req/s, GPU util=65%

    Observation : GPU util 65% = headroom. p99/p50 = 5.3x = the tail dominates.
    Probable causes : max_wait=100ms pushes some requests to wait
    almost 100ms before being processed.

    === Goal 1 : -30% cost per request ===
      Lever : increase the throughput to amortize the GPU's fixed cost.
      Action :
        - max_batch_size : 32 -> 48 or 64 (if the KV memory allows)
        - max_wait       : 100 -> 150 ms (we accept a bit of latency)
      Expected result : GPU util 85%+, throughput +30-40%, p50 +50ms, p99 +200ms.
      If the SLA still holds : mission accomplished.
      Metrics to watch : throughput (must go up), GPU util (must go
        up), p99 (must not explode), cost per 1K tokens (must go down).

    === Goal 2 : p99 < 500ms ===
      Lever : reduce the variance. The tail mainly comes from
      requests that wait for a full batch or the timeout.
      Action :
        - max_wait : 100 -> 20 ms (flush earlier)
        - max_batch_size : 32 -> 16 (limit the per-batch variance)
      Expected result : p99 drops to ~350-450ms, throughput drops ~20%.
      Downside : more expensive per request, GPU util drops to ~50%.
      Metrics : p99 (target), queue depth (must stay low),
        GPU util (will go down, that's normal).

    === Goal 3 : p99 < 200ms at iso-throughput ===
      Verdict : **cannot be held with the same hardware**.
      Reasons : even at batch=1, gpu_process_time is > 40ms. p99=200ms
      with batch = 32 is impossible on a single replica.
      Options :
        1. Scale horizontally : add replicas (double the fleet)
           to spread the load. Each replica has less queue -> low p99.
        2. Smaller model : more aggressive distillation / quantization
           -> gpu_process_time divided by 2-3.
        3. Speculative architecture : a small model predicts, a large
           model verifies (speculative decoding). 2x gain on latencies
           on average.
        4. Change GPU (H100 instead of A100 : ~2x faster).
      Monitoring : p99, queue depth per replica, LB consistency.

    Cross-cutting note : in ALL cases, watch the **queue depth**
    in front of the batcher. It's the leading signal that precedes a
    p99 degradation. If queue > 0 sustained -> trigger scale-up.
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
