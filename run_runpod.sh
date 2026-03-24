#!/bin/bash
# Parameter Golf — RunPod 8×H100 Benchmark Runner
# Runs variants at TRUE COMPETITION CONDITIONS:
#   • torchrun --nproc_per_node=8   (all 8 H100s)
#   • TRAIN_BATCH_TOKENS=524288     (competition batch)
#   • MAX_WALLCLOCK_SECONDS=600     (10-minute wall clock)
#   • ~7100 steps per run           (matches leaderboard exactly)
#
# Usage (from parameter-golf/ directory):
#   bash run_runpod.sh tier1        # 6 highest-priority variants (~1 hr, ~$7)
#   bash run_runpod.sh tier2        # next 6 variants (~1 hr, ~$7)
#   bash run_runpod.sh tier3        # sweep remaining variants (~2 hr, ~$14)
#   bash run_runpod.sh tier4        # WSM variants (~1 hr, ~$7)
#   bash run_runpod.sh tier5        # DiffAttn variants (~1 hr, ~$7)
#   bash run_runpod.sh tier6        # Peri-LN variants (~30 min, ~$3)
#   bash run_runpod.sh all          # everything (~5 hrs, ~$35)
#   bash run_runpod.sh V47          # single variant by ID
#
# Cost: ~$1 per variant on 8×H100 @ $6/hr
# Each run: exactly 10 minutes wall-clock, then auto-stops.
#
# DO NOT ADD:
#   TORCHDYNAMO_DISABLE=1     (not needed on Linux CUDA)
#   PYTORCH_SDP_BACKEND=...   (not needed on Linux)
#   TRAIN_SEQ_LEN=1024        (already default)

set -e

# ─── RUNPOD BASE CONFIG ────────────────────────────────────────────────────────
# Competition-accurate settings. No Windows workarounds.
BASE_VARS="DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  ITERATIONS=20000 \
  TRAIN_BATCH_TOKENS=524288 \
  MAX_WALLCLOCK_SECONDS=600 \
  VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=100 \
  MUON_WEIGHT_DECAY=0.04 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  WARMDOWN_ITERS=3500"

# Full SOTA stack base (11L, all proven techniques from 1.1233 submission)
SOTA_BASE="NUM_LAYERS=11 MLP_MULT=3 SMEARGATE=1 \
  BIGRAM_HASH_BUCKETS=2048 BIGRAM_HASH_DIM=128 \
  WARMDOWN_ITERS=3500"

# SOTA + quantization (full 1.1233 stack)
SOTA_QUANT="GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd"

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────
run_rp() {
  local run_id=$1
  local extra_vars=$2
  local log="logs/${run_id}.txt"

  # Skip if log exists and has a final val result
  if [ -f "$log" ] && grep -q "step:[0-9]*/[0-9]* val_bpb:" "$log" 2>/dev/null; then
    local cached=$(grep "val_bpb:" "$log" | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
    echo "  SKIP (cached): $run_id → val_bpb=$cached"
    return 0
  fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  RUNPOD RUN: $run_id"
  printf "║  %-65s║\n" "Vars: $extra_vars"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  mkdir -p logs
  eval "env $BASE_VARS RUN_ID=$run_id $extra_vars \
    torchrun --nproc_per_node=8 train_gpt.py" 2>&1 | tee "$log"

  local last_val=$(grep "val_bpb:" "$log" 2>/dev/null | grep "^step:" | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  echo ""
  echo "  ✓ DONE: $run_id → final val_bpb=$last_val"
  echo ""
}

TARGET="${1:-tier1}"

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Must run first (~6 variants, ~1 hr, ~$7)
# Goal: replicate SOTA baseline + test the two highest-impact novel techniques
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V47" ]]; then
  # Ground-truth SOTA replication (should match leaderboard 1.1233)
  run_rp "V47_full_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V61" ]]; then
  # LeakyReLU² on SOTA stack — easy win from PR #518 (no TTT cost)
  run_rp "V61_leaky_relu2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V73" ]]; then
  # TTT + LeakyReLU² — replicates PR #518 (1.0622 on leaderboard)
  run_rp "V73_ttt_leaky" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V82" ]]; then
  # HybridNorm + SSNorm — Layer 9 Tier S stack (both low-risk, one-line changes)
  run_rp "V82_hybrid_ss" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V83" ]]; then
  # Full SOTA (quantized) + HybridNorm + SSNorm — best submission candidate
  run_rp "V83_sota_layer9" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V70" ]]; then
  # Cosine TTT 8 epochs (fastest TTT, baseline for the technique)
  run_rp "V70_ttt_8ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=8 TTT_LR=0.0001"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — High value ablations (~6 variants, ~1 hr, ~$7)
# Goal: isolate which Layer 9 components contribute, test TTT scaling
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V80" ]]; then
  # HybridNorm alone (isolate FFN Post-Norm contribution)
  run_rp "V80_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V81" ]]; then
  # SSNorm alone (isolate outlier suppression contribution)
  run_rp "V81_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V71" ]]; then
  # TTT 10 epochs (PR #442 replication target: 1.1027)
  run_rp "V71_ttt_10ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V74" ]]; then
  # Full SOTA + TTT 10ep (stack on 1.1233 foundation)
  run_rp "V74_sota_ttt10" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V84" ]]; then
  # Full SOTA + SSNorm (quantization benefit of outlier suppression)
  run_rp "V84_sota_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V85" ]]; then
  # Full SOTA + HybridNorm (Post-Norm quality on quantized model)
  run_rp "V85_sota_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Sweep: combos, activation variants, TTT scaling (~12 variants, ~2 hr)
# Run after Tier 1+2 results tell you which direction to invest in
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V62" ]]; then
  # SwiGLU mlp_mult=2 (same params as relu² at mult=3, smooth gating)
  run_rp "V62_swiglu_m2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=swiglu MLP_MULT=2"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V63" ]]; then
  # SwiGLU mlp_mult=3 (more params, tests capacity trade-off)
  run_rp "V63_swiglu_m3" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=swiglu"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V72" ]]; then
  # TTT 20 epochs (more adaptation, slower eval)
  run_rp "V72_ttt_20ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=20 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V75" ]]; then
  # TTT higher LR (1e-3 vs default 1e-4)
  run_rp "V75_ttt_lr1e3" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V50" ]]; then
  # Value Residual on SOTA stack (ResFormer arXiv:2410.17897)
  run_rp "V50_value_residual" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V51" ]]; then
  # MoLE on SOTA stack (Mixture of Lookup Experts)
  run_rp "V51_mole" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V52" ]]; then
  # TWEO on SOTA stack (colinearity penalty, arXiv:2511.23225)
  run_rp "V52_tweo" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TWEO_LAMBDA=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V53" ]]; then
  # Tight SWA on SOTA stack (threshold-triggered weight averaging)
  run_rp "V53_tight_swa" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TIGHT_SWA=1 TIGHT_SWA_THRESHOLD=0.2 TIGHT_SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V59" ]]; then
  # Kitchen sink: SOTA + VR + MoLE + TWEO + Layer 9
  run_rp "V59_sota_all_novel" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT \
     VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64 TWEO_LAMBDA=0.0001 \
     HYBRID_NORM=1 SSNORM=1 \
     TTT_EPOCHS=10 TTT_LR=0.0001 MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4 — WSM + NuMuon variants (from research agent findings, 2026-03-24)
#   V86: WSM alone (no warmdown, stable-LR checkpoint merge)
#   V87: WSM + SOTA stack (replaces warmdown with merge)
#   V88: WSM + HybridNorm + SSNorm (all no-cost improvements stacked)
#   V89: WSM + TTT + LeakyReLU² (frontier: every technique stacked)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V86" ]]; then
  # WSM alone on SOTA stack: no warmdown, merge last 30% of checkpoints
  run_rp "V86_wsm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V87" ]]; then
  # WSM + full quantized SOTA (replaces WARMDOWN_ITERS with stable-phase merge)
  run_rp "V87_wsm_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V88" ]]; then
  # WSM + HybridNorm + SSNorm + SOTA (stack all zero-cost improvements)
  run_rp "V88_wsm_layer9" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1 WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V89" ]]; then
  # Full frontier stack: WSM + Layer9 + TTT + LeakyReLU² (kitchen sink v2)
  run_rp "V89_frontier_stack" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1 \
     WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 5 — Differential Transformer (arXiv:2410.05258, ICLR 2025 Oral)
#   Halves head_dim, two sub-heads per head (Q1/Q2, K1/K2) sharing same V.
#   Lambda scalar: attn = attn1 - λ·attn2 damps attention noise.
#   Zero parameter overhead — projection shapes identical to baseline.
#   NOTE: incompatible with VALUE_RESIDUAL (different V shape).
#   V90: DiffAttn alone (ablation — does the mechanism help at all?)
#   V91: DiffAttn + full quantized SOTA (main submission candidate)
#   V92: DiffAttn + HybridNorm + SSNorm + SOTA (maximum stacking)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V90" ]]; then
  # DiffAttn alone on SOTA stack: isolates the differential attention mechanism
  run_rp "V90_diff_attn" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V91" ]]; then
  # DiffAttn + full quantized SOTA (XSA auto-disabled by DiffAttn incompatibility)
  run_rp "V91_diff_attn_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V92" ]]; then
  # DiffAttn + Layer 9 (HybridNorm + SSNorm) + full SOTA (maximum novel stack)
  run_rp "V92_diff_layer9_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DIFF_TRANSFORMER=1 HYBRID_NORM=1 SSNORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 6 — Peri-LN (arXiv:2502.02732, Gemma/OLMo 2 families)
#   Pre-Norm + Post-Norm on BOTH attention and FFN sublayers.
#   Superset of HybridNorm (which Post-Norms FFN only). Mutually exclusive.
#   V93: Peri-LN alone on SOTA stack (ablation vs HybridNorm)
#   V94: Peri-LN + SSNorm + SOTA (replaces HybridNorm in best layer-9 stack)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier6" || "$TARGET" == "V93" ]]; then
  # Peri-LN alone on SOTA stack (compare directly to V80_hybridnorm)
  run_rp "V93_peri_ln" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     PERI_LN=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier6" || "$TARGET" == "V94" ]]; then
  # Peri-LN + SSNorm + full SOTA (replaces HybridNorm in V83 for direct comparison)
  run_rp "V94_peri_ln_ssnorm_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT PERI_LN=1 SSNORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — Power-law warmdown (arXiv:2602.06797 optimal LR scaling laws)
#   Linear decay (α=1.0) is the default. For severely undertrained models
#   (≪Chinchilla optimal), scaling theory predicts steeper decay (α>1) is better.
#   V95: α=2.0 (quadratic) on full SOTA — tests steeper warmdown shape
#   V96: α=1.5 (mid-power) on full SOTA — moderate steepening
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier7" || "$TARGET" == "V95" ]]; then
  run_rp "V95_wsd_power2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSD_POWER=2.0"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier7" || "$TARGET" == "V96" ]]; then
  run_rp "V96_wsd_power15" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSD_POWER=1.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 8 — Adaptive Group Gradient Clipping (arXiv:2601.11864)
# ═══════════════════════════════════════════════════════════════════════════════
#   V97: AGGC β=0.99 on full SOTA — tests per-group adaptive gradient clipping

if [[ "$TARGET" == "all" || "$TARGET" == "tier8" || "$TARGET" == "V97" ]]; then
  run_rp "V97_aggc" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT AGGC_BETA=0.99 AGGC_THRESHOLD=3.0"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 9 — DenseFormer: Depth-Weighted Averaging (arXiv:2402.02622, NeurIPS 2024)
# ═══════════════════════════════════════════════════════════════════════════════
#   V98: DenseFormer on full SOTA — replaces U-Net skips with learned DWA (66 scalars)

if [[ "$TARGET" == "all" || "$TARGET" == "tier9" || "$TARGET" == "V98" ]]; then
  run_rp "V98_denseformer" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DENSEFORMER=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 10 — NuMuon: Nuclear-Norm Constrained Muon (arXiv:2603.03597)
# ═══════════════════════════════════════════════════════════════════════════════
#   V99: NuMuon weight=1e-4 on full SOTA — promotes low-rank weights → better int6

if [[ "$TARGET" == "all" || "$TARGET" == "tier10" || "$TARGET" == "V99" ]]; then
  run_rp "V99_numuon" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT NUMUON_WEIGHT=1e-4"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 11 — MUDDFormer: Dynamic Dense Connections (arXiv:2502.12170, ICML 2025)
# ═══════════════════════════════════════════════════════════════════════════════
#   V100: MUDD V-stream only (highest per-ablation benefit, ~45K params)
#   V101: MUDD Q+K+V streams (full 3-stream variant, ~137K params)

if [[ "$TARGET" == "all" || "$TARGET" == "tier11" || "$TARGET" == "V100" ]]; then
  run_rp "V100_mudd_v" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MUDD_STREAMS=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier11" || "$TARGET" == "V101" ]]; then
  run_rp "V101_mudd_qkv" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MUDD_STREAMS=3"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 12 — LOTION: Smooth Quantization via Stochastic Noise (arXiv:2510.08757)
# ═══════════════════════════════════════════════════════════════════════════════
#   V102: LOTION QAT from step 0 — replaces STE with calibrated noise injection.
#         σ=sB*sqrt(Δ(1-Δ)) noise convergence guarantee. STE is biased; LOTION is not.

if [[ "$TARGET" == "all" || "$TARGET" == "tier12" || "$TARGET" == "V102" ]]; then
  run_rp "V102_lotion" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT LOTION=1 QAT_START_FRACTION=0.0"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  RUNPOD RESULTS SUMMARY — val_bpb (lower = better)             ║"
echo "║  Competition conditions: 8×H100, 600s, 524K batch/step         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
printf "%-35s %-12s %-10s\n" "RUN_ID" "final_bpb" "step"
echo "────────────────────────────────────────────────────────────────────"
for log in logs/V*.txt; do
  [ -f "$log" ] || continue
  id=$(basename "$log" .txt)
  # Match actual log output lines (start with "step:")
  last=$(grep "^step:[0-9]" "$log" 2>/dev/null | grep "val_bpb:" | tail -1)
  [ -z "$last" ] && continue
  bpb=$(echo "$last" | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  step=$(echo "$last" | grep -o "step:[0-9]*/" | sed "s/step://;s/\///")
  [ -n "$bpb" ] && printf "%-35s %-12s %-10s\n" "$id" "$bpb" "$step"
done | sort -k2 -n
echo ""
echo "Leaderboard reference: 1.2244 (baseline) | 1.1233 (SOTA) | 1.0622 (PR#518 frontier)"
