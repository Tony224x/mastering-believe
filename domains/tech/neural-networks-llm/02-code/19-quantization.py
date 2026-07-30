"""
Day 19 — Quantization deep dive (pure NumPy, no torch required).

We implement, from scratch:
    PART 1 — Symmetric INT8 quantization. Measure error on a Gaussian matrix.
    PART 2 — Asymmetric INT4 with zero-point. Compare on skewed distributions.
    PART 3 — Per-tensor vs per-channel on a matrix with mixed column scales.
    PART 4 — Outlier impact: naive INT8 vs SmoothQuant-like channel migration.
    PART 5 — NF4 (NormalFloat 4-bit) via Gaussian quantile lookup,
             compared with linear INT4.
    PART 6 — MX (Microscaling) formats, the 2026 frontier default:
             6a. MXFP4 (FP4 E2M1 + power-of-two shared scale per 32
                 elements) benchmarked against NF4 and linear INT4 at the
                 SAME block size. MXFP4 loses on accuracy — and wins anyway.
             6b. The mixed-precision map used by Kimi K3: experts in MXFP4,
                 routers / latent projections / shared experts left alone.
             6c. The train-vs-inference divergence that quantization-aware
                 training exists to remove, measured in KL.

Run:
    python 19-quantization.py
Should finish in well under 30 s with seed=42.
"""

import numpy as np

np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report(name, x, x_hat):
    """Pretty-print MSE and max abs error between original and dequantized."""
    err = x - x_hat
    mse = float(np.mean(err ** 2))
    max_abs = float(np.max(np.abs(err)))
    # Relative MSE w.r.t. signal energy — easier to read across scales.
    rel = mse / float(np.mean(x ** 2) + 1e-12)
    print(f"  {name:<42s} MSE={mse:.6e}   max|err|={max_abs:.4f}   relMSE={rel:.4%}")


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# PART 1 — Symmetric INT8 from scratch
# ---------------------------------------------------------------------------
# Symmetric quantization: assume distribution centered on 0.
# scale = max(|x|) / 127  (INT8 signed range is [-128, 127], we keep -127..127
#                          to stay symmetric and avoid the asymmetric -128).
# q     = round(x / scale), clipped to [-127, 127]
# x_hat = q * scale

def quantize_symmetric_int8(x):
    qmax = 127
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        # Degenerate (all zeros) — pick something safe.
        scale = 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def dequantize_symmetric_int8(q, scale):
    return q.astype(np.float32) * scale


section("PART 1 — Symmetric INT8 on a centered Gaussian matrix")

x1 = np.random.randn(256, 256).astype(np.float32)  # mean=0, std=1
q1, s1 = quantize_symmetric_int8(x1)
x1_hat = dequantize_symmetric_int8(q1, s1)
print(f"  matrix shape      = {x1.shape}")
print(f"  scale (FP32)      = {s1:.6f}")
print(f"  unique INT8 codes = {len(np.unique(q1))}")
report("symmetric INT8 (per-tensor)", x1, x1_hat)


# ---------------------------------------------------------------------------
# PART 2 — Asymmetric INT4 with zero-point on a skewed distribution
# ---------------------------------------------------------------------------
# INT4 unsigned range: [0, 15] -> 16 levels.
# scale      = (max - min) / 15
# zero_point = round(-min / scale), an integer in [0, 15]
# q          = clip(round(x / scale + zero_point), 0, 15)
# x_hat      = (q - zero_point) * scale

def quantize_asymmetric_intN(x, n_bits=4):
    qmin, qmax = 0, (1 << n_bits) - 1  # e.g. 0..15 for INT4
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.int8), 1.0, 0
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = int(round(qmin - x_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int8)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point):
    return (q.astype(np.float32) - zero_point) * scale


def quantize_symmetric_intN(x, n_bits=4):
    """Symmetric N-bit, signed, centered on 0."""
    qmax = (1 << (n_bits - 1)) - 1  # e.g. 7 for INT4 signed
    scale = np.max(np.abs(x)) / qmax
    if scale == 0:
        scale = 1.0
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def dequantize_symmetric(q, scale):
    return q.astype(np.float32) * scale


section("PART 2 — Asymmetric vs symmetric INT4 on a skewed distribution")

# Skewed: like post-ReLU/SiLU activations, heavy on the positive side.
x2 = np.abs(np.random.randn(256, 256).astype(np.float32)) * 2.0 + 0.5
# Add a few near-zero entries to keep the min interesting.
x2[0, :] = 0.0

q2_sym, s2_sym = quantize_symmetric_intN(x2, n_bits=4)
x2_sym_hat = dequantize_symmetric(q2_sym, s2_sym)

q2_asym, s2_asym, zp2 = quantize_asymmetric_intN(x2, n_bits=4)
x2_asym_hat = dequantize_asymmetric(q2_asym, s2_asym, zp2)

print(f"  data range          = [{x2.min():.3f}, {x2.max():.3f}]   (skewed positive)")
print(f"  asym scale, zp      = {s2_asym:.4f}, {zp2}")
report("symmetric  INT4 on skewed data", x2, x2_sym_hat)
report("asymmetric INT4 on skewed data", x2, x2_asym_hat)
# Expected: asymmetric clearly wins because symmetric wastes ~half its range
# on negative values that don't exist in the data.


# ---------------------------------------------------------------------------
# PART 3 — Per-tensor vs per-channel
# ---------------------------------------------------------------------------
# We craft a matrix where each column has a different scale. Per-tensor will
# pick the global max, crushing small-scale columns to ~0.

section("PART 3 — Per-tensor vs per-channel INT8 on mixed-scale columns")

n_rows, n_cols = 256, 256
# Each column has a scale drawn log-uniformly over [0.01, 100].
col_scales = np.exp(np.random.uniform(np.log(0.01), np.log(100.0), size=n_cols))
x3 = (np.random.randn(n_rows, n_cols) * col_scales).astype(np.float32)
print(f"  column scale range  = [{col_scales.min():.3f}, {col_scales.max():.3f}]")

# Per-tensor (one global scale).
q3_pt, s3_pt = quantize_symmetric_int8(x3)
x3_pt_hat = dequantize_symmetric_int8(q3_pt, s3_pt)

# Per-channel (one scale per column).
def quantize_symmetric_int8_per_channel(x, axis=0):
    """axis=0 means scales are per column (one scale shared along rows)."""
    qmax = 127
    abs_max = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(abs_max == 0, 1.0, abs_max / qmax).astype(np.float32)
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return q, scale


q3_pc, s3_pc = quantize_symmetric_int8_per_channel(x3, axis=0)
x3_pc_hat = q3_pc.astype(np.float32) * s3_pc

report("per-tensor   INT8", x3, x3_pt_hat)
report("per-channel  INT8 (per column)", x3, x3_pc_hat)
# Expected: per-channel is dramatically better — orders of magnitude.


# ---------------------------------------------------------------------------
# PART 4 — Outlier impact and SmoothQuant-like migration
# ---------------------------------------------------------------------------
# Create a matrix where 0.1% of entries are 100x larger than the rest.
# Naive INT8 picks the global max -> resolution on the 99.9% normal entries
# is destroyed.
# A SmoothQuant-like fix: for each column, divide by sqrt(max), and absorb
# that scale separately. This narrows the per-column dynamic range.

section("PART 4 — Outliers: naive INT8 vs SmoothQuant-style migration")

n = 512
x4 = np.random.randn(n, n).astype(np.float32) * 0.5  # std ~0.5
# Inject 0.1% outliers at magnitude 50.
n_outliers = max(1, int(0.001 * x4.size))
idx_flat = np.random.choice(x4.size, size=n_outliers, replace=False)
flat = x4.flatten()
flat[idx_flat] = np.sign(np.random.randn(n_outliers)) * 50.0
x4 = flat.reshape(n, n)

print(f"  shape={x4.shape}   #outliers={n_outliers}   "
      f"std(non-outlier)~0.5   |outlier|=50")

# Naive symmetric per-tensor INT8.
q4_naive, s4_naive = quantize_symmetric_int8(x4)
x4_naive_hat = dequantize_symmetric_int8(q4_naive, s4_naive)

# SmoothQuant-like idea (simplified):
#   Y = X @ W  ==  (X / s) @ (s * W)
# We migrate magnitude from activations (X) to weights (W) so that, after the
# per-column rescaling, X / s has tame columns. Here we just demonstrate the
# benefit on a single matrix: split it into a "smoothed" component and a
# per-column scale, then quantize the smoothed component per-tensor.
#
# Choose s per column = max(|x4[:, j]|)^alpha, with alpha=0.5 (the typical
# SmoothQuant midpoint between activation-side and weight-side).
alpha = 0.5
col_max = np.max(np.abs(x4), axis=0, keepdims=True)
col_max = np.where(col_max == 0, 1.0, col_max)
s_smooth = np.power(col_max, alpha).astype(np.float32)  # shape (1, n)

x4_smoothed = x4 / s_smooth  # tame dynamic range per column
q4_sm, s4_sm = quantize_symmetric_int8(x4_smoothed)
x4_sm_hat = dequantize_symmetric_int8(q4_sm, s4_sm) * s_smooth  # un-do migration

report("naive          INT8 (with outliers)", x4, x4_naive_hat)
report("SmoothQuant-like INT8 (alpha=0.5)", x4, x4_sm_hat)
# Expected: SmoothQuant-like preserves precision on the 99.9% normal entries.


# ---------------------------------------------------------------------------
# PART 5 — NF4 (NormalFloat 4-bit) via Gaussian quantile lookup
# ---------------------------------------------------------------------------
# Linear INT4 spreads its 16 levels uniformly between -1 and 1.
# NF4 places its 16 levels at the quantiles of a standard Gaussian, so they
# are denser near 0 (where most weights actually live) and sparser in tails.
#
# Standard NF4 uses the asymmetric 16-level scheme published by Dettmers in
# QLoRA (one of the levels is exactly 0). We approximate it cleanly by
# computing 16 quantiles of N(0,1) and using them as the codebook.

def _erfinv_approx(y):
    """Approximate inverse error function (Winitzki 2008). Accurate to ~1e-3,
    enough for placing 16 quantile codebook levels.

    Why this approximation rather than the stdlib? `math.erfinv` does NOT
    exist in the Python standard library prior to 3.13 (and even on 3.13+ it
    is not guaranteed on every platform/build). NumPy also does not ship an
    erfinv. The cleanly-correct alternative is `scipy.special.erfinv`, but
    we keep this module dependency-free (numpy only). The Winitzki closed
    form below is more than precise enough for 16-level codebook placement.
    """
    # Avoids dependency on scipy; for an exact value, use scipy.special.erfinv.
    a = 0.147
    ln_term = np.log(1.0 - y * y)
    first = 2.0 / (np.pi * a) + ln_term / 2.0
    inside = first * first - ln_term / a
    return np.sign(y) * np.sqrt(np.sqrt(inside) - first)


def build_nf4_codebook():
    # Use the symmetric 16-level scheme: quantiles spaced uniformly in
    # probability space, then mapped back through the inverse Gaussian CDF.
    # erfinv-based Gaussian quantile: x = sqrt(2) * erfinv(2p - 1).
    probs = (np.arange(16, dtype=np.float64) + 0.5) / 16.0  # 16 midpoint probs
    levels = np.sqrt(2.0) * _erfinv_approx(2.0 * probs - 1.0)
    # Normalize so |max level| = 1 — this is what the QLoRA paper does so the
    # codebook lives in [-1, 1] just like INT4 linear.
    levels = levels / np.max(np.abs(levels))
    return levels.astype(np.float32)


NF4_CODEBOOK = build_nf4_codebook()


def quantize_nf4(x):
    """Per-block NF4 with block size 64 (same idea as QLoRA)."""
    block_size = 64
    flat = x.flatten()
    pad = (-flat.size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    # Per-block scale = max(|x|) so that the scaled block lives in [-1, 1].
    abs_max = np.max(np.abs(blocks), axis=1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scaled = blocks / abs_max  # shape (n_blocks, block_size), in [-1, 1]
    # Snap each scaled value to the nearest codebook entry.
    diffs = np.abs(scaled[..., None] - NF4_CODEBOOK[None, None, :])
    codes = np.argmin(diffs, axis=-1).astype(np.uint8)  # 0..15
    return codes, abs_max, pad, x.shape


def dequantize_nf4(codes, abs_max, pad, original_shape):
    block_size = 64
    levels = NF4_CODEBOOK[codes]  # in [-1, 1]
    blocks = levels * abs_max
    flat = blocks.flatten()
    if pad:
        flat = flat[:-pad]
    return flat.reshape(original_shape).astype(np.float32)


def quantize_int4_linear_perblock(x):
    """Per-block linear INT4 (symmetric), block size 64, for fair comparison."""
    block_size = 64
    flat = x.flatten()
    pad = (-flat.size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)
    abs_max = np.max(np.abs(blocks), axis=1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    qmax = 7  # INT4 signed: -7..7
    scale = abs_max / qmax
    q = np.clip(np.round(blocks / scale), -qmax, qmax).astype(np.int8)
    blocks_hat = q.astype(np.float32) * scale
    flat_hat = blocks_hat.flatten()
    if pad:
        flat_hat = flat_hat[:-pad]
    return flat_hat.reshape(x.shape)


section("PART 5 — NF4 vs linear INT4 on Gaussian weights (per-block, b=64)")

# Real LLM weights are approximately Gaussian — this is the regime NF4 was
# designed for. We use a fairly large matrix to average out per-block noise.
x5 = np.random.randn(1024, 1024).astype(np.float32)
print(f"  matrix shape       = {x5.shape}")
print(f"  NF4 codebook (16)  = "
      f"[{NF4_CODEBOOK[0]:.3f}, {NF4_CODEBOOK[1]:.3f}, ..., "
      f"{NF4_CODEBOOK[-2]:.3f}, {NF4_CODEBOOK[-1]:.3f}]")

# Linear INT4 per-block.
x5_int4_hat = quantize_int4_linear_perblock(x5)

# NF4 per-block.
codes, abs_max, pad, shape = quantize_nf4(x5)
x5_nf4_hat = dequantize_nf4(codes, abs_max, pad, shape)

report("linear INT4 per-block (block=64)", x5, x5_int4_hat)
report("NF4         per-block (block=64)", x5, x5_nf4_hat)
# Expected: NF4 has noticeably lower MSE on Gaussian weights. The exact
# improvement depends on block size; with block=64 we typically see ~10-25%.


# ---------------------------------------------------------------------------
# PART 6 — MX (Microscaling) formats: MXFP4 / MXFP8, the 2026 frontier default
# ---------------------------------------------------------------------------
# NF4 and per-block INT4 both store a floating-point scale per block. MX
# standardises the idea and makes it cheap enough to put in silicon:
#     block of 32 elements
#   + ONE shared scale, constrained to a power of two (E8M0: 8 exponent
#     bits, no mantissa — so the scale is exactly a bit-shift)
#   + each element in a tiny float format (FP4 E2M1, or FP8 E4M3)
# The power-of-two constraint is what makes it hardware-friendly, and it is
# also why MX is NOT the most accurate 4-bit format. Trade-off, not free lunch.

MX_BLOCK = 32

# FP4 E2M1: 2 exponent bits, 1 mantissa bit, 1 sign bit -> 8 magnitudes.
FP4_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                               dtype=np.float32)
FP4_MAX = 6.0

# FP8 E4M3 (the variant used for activations): 4 exponent bits, 3 mantissa
# bits. We only need its max value and its mantissa resolution here.
FP8_E4M3_MAX = 448.0
FP8_E4M3_MANTISSA_BITS = 3


def round_to_fp4_e2m1(x):
    """Round each element to the nearest representable FP4 E2M1 value."""
    sign = np.sign(x)
    mag = np.abs(x)
    # Nearest magnitude in the 8-entry table (ties go to the larger one,
    # which is what round-to-nearest-even does on this grid in practice).
    idx = np.argmin(np.abs(mag[..., None] - FP4_E2M1_MAGNITUDES), axis=-1)
    return sign * FP4_E2M1_MAGNITUDES[idx]


def quantize_mx(x, elem_max, round_fn, block=MX_BLOCK):
    """Quantize to an MX format: shared power-of-two scale per block.

    Returns the dequantized array. Follows the OCP MX recipe: the shared
    exponent is chosen so the block maximum lands just inside the element
    format's range.
    """
    flat = x.astype(np.float32).flatten()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, block)

    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    # E8M0: the scale is 2^e, e an integer. No mantissa, so we FLOOR the
    # exponent difference — never round up, or the block max would clip.
    shared_exp = np.floor(np.log2(amax)) - np.floor(np.log2(elem_max))
    scale = np.exp2(shared_exp).astype(np.float32)

    q = round_fn(np.clip(blocks / scale, -elem_max, elem_max))
    out = (q * scale).flatten()
    if pad:
        out = out[:-pad]
    return out.reshape(x.shape)


def round_to_fp8_e4m3(x):
    """Round to FP8 E4M3 by truncating the mantissa to 3 bits per exponent."""
    mag = np.abs(x)
    safe = np.where(mag == 0, 1.0, mag)
    exp = np.floor(np.log2(safe))
    step = np.exp2(exp - FP8_E4M3_MANTISSA_BITS)   # spacing at this exponent
    return np.sign(x) * np.where(mag == 0, 0.0, np.round(mag / step) * step)


section("PART 6a — MXFP4 vs NF4 vs linear INT4 (all per-block, b=32)")

x6 = np.random.randn(1024, 1024).astype(np.float32)
print(f"  block size for all three = {MX_BLOCK}")
print("  FP4 E2M1 magnitudes      = "
      f"{[float(v) for v in FP4_E2M1_MAGNITUDES]}")
print()

x6_mxfp4 = quantize_mx(x6, FP4_MAX, round_to_fp4_e2m1)
report("MXFP4 (E2M1 + power-of-2 shared scale)", x6, x6_mxfp4)


# Same block size for the two baselines, so the comparison is about the
# FORMAT and not about how much scale metadata we are allowed to store.
# (The PART 5 helpers hardcode block=64, hence these 32-sized variants.)
def _perblock_int4(x, block):
    flat = x.flatten()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    scale = amax / 7.0
    q = np.clip(np.round(blocks / scale), -7, 7)
    out = (q * scale).flatten()
    return (out[:-pad] if pad else out).reshape(x.shape)


def _perblock_nf4(x, block):
    flat = x.flatten()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    norm = blocks / amax
    idx = np.argmin(np.abs(norm[..., None] - NF4_CODEBOOK), axis=-1)
    out = (NF4_CODEBOOK[idx] * amax).flatten()
    return (out[:-pad] if pad else out).reshape(x.shape)


report("linear INT4 (fp32 scale)", x6, _perblock_int4(x6, MX_BLOCK))
report("NF4         (fp32 scale)", x6, _perblock_nf4(x6, MX_BLOCK))

print()
print("  -> MXFP4 is NOT the most accurate 4-bit format on Gaussian weights.")
print("     Its scale is restricted to a power of two, so a block whose max")
print("     sits just above a power of two wastes up to half the range; NF4")
print("     additionally places its 16 levels at Gaussian quantiles.")
print("     MX wins on a different axis: the scale is a bit-shift, the block")
print("     is 32 elements, and both are fixed by the OCP spec — so the")
print("     dequantize step can live inside the Tensor Core datapath instead")
print("     of costing a separate kernel. Accuracy per bit is traded for")
print("     throughput, and the gap is closed by QAT rather than by the format.")

section("PART 6b — What Kimi K3 actually quantizes (and what it does not)")

# Toy MoE layer, same shape conventions as day 16 (2 matrices per expert:
# up and down; the real SiTU-GLU FFN has 3, so absolute sizes are low).
D, LATENT, D_FF = 7168, 3584, 3072
N_ROUTED, N_SHARED = 896, 2

parts = [
    # name,                        params,                             bits
    ("routed experts (MXFP4)", N_ROUTED * 2 * LATENT * D_FF, 4.25),
    ("shared experts (BF16)", N_SHARED * 2 * D * D_FF, 16.0),
    ("latent projections (BF16)", 2 * D * LATENT, 16.0),
    ("router (BF16)", D * N_ROUTED, 16.0),
]

print(f"  MXFP4 effective bits/weight = 4 + 8/{MX_BLOCK} = "
      f"{4 + 8 / MX_BLOCK:.2f}  (element + shared scale amortised)")
print()
print(f"  {'component':<28}{'params':>16}{'bits/w':>9}{'GB':>9}{'GB if BF16':>13}")
total_k3 = total_bf16 = 0.0
for name, n, bits in parts:
    gb = n * bits / 8 / 1e9
    gb16 = n * 16 / 8 / 1e9
    total_k3 += gb
    total_bf16 += gb16
    print(f"  {name:<28}{n:>16,}{bits:>9.2f}{gb:>9.2f}{gb16:>13.2f}")
print(f"  {'TOTAL':<28}{'':>16}{'':>9}{total_k3:>9.2f}{total_bf16:>13.2f}"
      f"   ({total_bf16 / total_k3:.2f}x smaller)")
print()
print("  -> The experts are ~99% of the parameters, so quantizing THEM is the")
print("     whole game; keeping routers, latent projections and shared experts")
print("     in high precision costs almost nothing and protects the two most")
print("     error-sensitive things in the layer: the routing decision and the")
print("     path every single token goes through.")
print("     Activations follow MXFP8 (E4M3, max "
      f"{FP8_E4M3_MAX:.0f}) — which is exactly why the")
print("     activation function had to be bounded (see day 09, SiTU-GLU).")

section("PART 6c — The mismatch QAT removes")

# A policy head: hidden state -> logits over a vocabulary.
D_H, V = 512, 4096
h = np.random.randn(256, D_H).astype(np.float32)
W = (np.random.randn(D_H, V) / np.sqrt(D_H)).astype(np.float32)

W_q = quantize_mx(W, FP4_MAX, round_to_fp4_e2m1)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


print("  Same hidden states, two copies of the same weights:")
print("    'training'  = full precision")
print("    'inference' = MXFP4 (post-training quantization, no adaptation)")
print()
print("  Random weights give a nearly flat softmax, where any two policies")
print("  look identical. A trained head is PEAKED, so we sweep a logit scale")
print("  to show how the mismatch depends on how confident the model is.")
print()
print(f"  {'logit std':>11}{'mean top-1 prob':>18}{'KL(train||infer)':>20}"
      f"{'top-1 agree':>14}{'top-5 overlap':>16}")

for logit_std in [1.0, 3.0, 6.0, 10.0]:
    lo_hi = (h @ W) * logit_std
    lo_lo = (h @ W_q) * logit_std
    p_hi, p_lo = softmax(lo_hi), softmax(lo_lo)
    kl = float(np.mean(np.sum(
        p_hi * np.log((p_hi + 1e-12) / (p_lo + 1e-12)), axis=-1)))
    top1 = float(np.mean(p_hi.argmax(-1) == p_lo.argmax(-1)))
    t5h = np.argsort(-lo_hi, axis=-1)[:, :5]
    t5l = np.argsort(-lo_lo, axis=-1)[:, :5]
    overlap = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(t5h, t5l)]))
    conf = float(p_hi.max(axis=-1).mean())
    print(f"  {logit_std:>11.1f}{conf:>18.1%}{kl:>20.4f}"
          f"{top1:>14.1%}{overlap:>16.1%}")

print()
print("  Note the top-1 / top-5 columns do not move with the scale: rescaling")
print("  logits cannot reorder them. What moves is the KL - the same ranking")
print("  disagreement costs more the more confident the policy is.")
print("  Do not read those two columns as a benchmark: these weights are")
print("  random, so the top logits are near-ties and flip easily. A trained")
print("  head separates its top candidates far better. What generalises is")
print("  the SHAPE - a real, scale-dependent divergence between two things")
print("  that the training algorithm assumes are one and the same model.")
print()
print("  -> For plain text generation this gap is a quality question and a")
print("     benchmark point or two. For RL it is a CORRECTNESS question: the")
print("     rollouts are sampled from the inference engine, but the gradient")
print("     is computed against the training weights. If the two policies")
print("     differ, the on-policy assumption behind the update is false and")
print("     the algorithm is silently off-policy.")
print("     Quantization-aware training makes the training forward pass use")
print("     the SAME quantized weights, so this gap is zero by construction.")
print("     That — not the memory saving — is why Kimi K3 keeps QAT on")
print("     through the whole of post-training, SFT and RL included.")
print("     (This demo measures the gap; it does not train, so it does not")
print("      show QAT recovering accuracy. That claim is the paper's.)")


print()
print("Done. All six demos finished successfully.")
