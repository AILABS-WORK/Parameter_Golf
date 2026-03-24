#!/bin/bash
# Parameter Golf — Comprehensive Benchmark Runner
# Measures val_bpb (the real competition metric) every 300 steps up to 3000 steps.
# Uses MAX_VAL_TOKENS=6_200_000 (~10% of val set) for fast ~1-min val checks.
#
# Run from: parameter-golf/ directory
# Usage:
#   bash run_benchmark.sh all             # everything (~25 min × 50 runs = 20h, run overnight)
#   bash run_benchmark.sh baseline        # just baselines (V0 variants)
#   bash run_benchmark.sh tier1           # proven competition wins
#   bash run_benchmark.sh novel           # our novel paper variants
#   bash run_benchmark.sh combos          # stacked combinations
#   bash run_benchmark.sh sota            # 1.1307→1.1233 techniques (XSA, EMA, Partial RoPE, LN Scale, GPTQ-lite)
#   bash run_benchmark.sh novel2          # beyond SOTA: Tight SWA, Overtone Init, Phase ResidMix, novel combos
#   bash run_benchmark.sh activ           # activation variants: LeakyReLU², SwiGLU (from unmerged PRs)
#   bash run_benchmark.sh ttt             # Cosine TTT variants (sub-1.1 BPB frontier, slower eval)
#   bash run_benchmark.sh layer9          # HybridNorm + SSNorm ablations (V80-V85)
#   bash run_benchmark.sh diff_attn       # Differential Transformer ablations (V90-V92)
#   bash run_benchmark.sh peri_ln         # Peri-LN ablations (V93-V94)
#   bash run_benchmark.sh V44             # single variant by ID
#
# Output:
#   logs/<RUN_ID>.txt  — step log with val_bpb every 300 steps
#   results_chart.png  — auto-generated after each group completes
#
# Extrapolation target: 7,000 steps × 524,288 tokens = 3.67B tokens (H100 8× competition)
# Each run: ~20 min training + ~5 min val overhead = ~25 min total

set -e

PYTHON="/c/Python314/python.exe"

# ─── BASE CONFIG ──────────────────────────────────────────────────────────────
# These match the competition defaults but scaled to laptop VRAM.
# TRAIN_BATCH_TOKENS=8192 keeps memory safe (vs competition's 786K).
# TRAIN_SEQ_LEN=1024 for faster steps; set SEQ2048=1 per-variant to use 2048.
BASE_VARS="TORCHDYNAMO_DISABLE=1 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  ITERATIONS=3000 \
  TRAIN_BATCH_TOKENS=8192 \
  TRAIN_SEQ_LEN=1024 \
  MAX_WALLCLOCK_SECONDS=0 \
  SKIP_ALL_VAL=0 \
  MAX_VAL_TOKENS=6200000 \
  VAL_LOSS_EVERY=300 \
  TRAIN_LOG_EVERY=10"

# Competition-scale hyperparameters (some variants activate these)
COMP_VARS="WARMDOWN_ITERS=3000 \
  MUON_WEIGHT_DECAY=0.04 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02"

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────
run_bench() {
  local run_id=$1
  local extra_vars=$2
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  BENCHMARK: $run_id"
  echo "║  Extra: $extra_vars"
  echo "╚══════════════════════════════════════════════════════════╝"
  eval "env $BASE_VARS RUN_ID=$run_id $extra_vars $PYTHON train_gpt.py"
  # Extract last val_bpb from log
  local last_val=$(grep "val_bpb:" "logs/$run_id.txt" 2>/dev/null | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  local last_step=$(grep "val_bpb:" "logs/$run_id.txt" 2>/dev/null | tail -1 | grep -o "step:[0-9]*/" | sed "s/step://;s/\///")
  echo "  RESULT: $run_id → step:$last_step val_bpb=$last_val"
  echo ""
}

TARGET="${1:-all}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Parameter Golf — 3000-Step Benchmark with val_bpb tracking     ║"
echo "║  val_bpb every 300 steps · MAX_VAL_TOKENS=6.2M (~10% val set)  ║"
echo "║  Extrapolation target: 7K H100 steps = 3.67B tokens             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "  Target group: $TARGET"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE GROUP — reference points
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "baseline" || "$TARGET" == "V0" ]]; then
  # Pure baseline — identical to competition default config
  run_bench "V0_baseline" ""
fi

if [[ "$TARGET" == "all" || "$TARGET" == "baseline" || "$TARGET" == "V0b" ]]; then
  # Baseline with competition hyperparams (longer warmdown, lower LR, WD)
  run_bench "V0b_comp_hparams" "$COMP_VARS"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Proven competition wins (from local records analysis)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V1" ]]; then
  # MLP 3x expansion — largest single contributor in best submissions
  # Why: more capacity per token with same depth; int6 compression makes it affordable
  run_bench "V1_mlp3x" "MLP_MULT=3 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V2" ]]; then
  # 10 layers — extra depth funded by compression savings
  run_bench "V2_10layers" "NUM_LAYERS=10 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V3" ]]; then
  # SmearGate — learned blend of current+prev token embedding (~512 extra params)
  run_bench "V3_smeargate" "SMEARGATE=1 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V4" ]]; then
  # BigramHash 10240 — higher-bucket version (best submission used 10240 vs 4096)
  run_bench "V4_bigram10k" "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V5" ]]; then
  # Orthogonal init — accelerates convergence, better quantization
  run_bench "V5_orthoinit" "ORTHO_INIT=1 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V6" ]]; then
  # SWA start_frac=0.4 — best submission found 0.4 better than 0.5 or 0.6
  run_bench "V6_swa04" "SWA=1 SWA_START_FRACTION=0.4 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V7" ]]; then
  # WSD cosine LR warmdown (vs default linear)
  run_bench "V7_wsd_lr" "WSD_LR=1 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V8" ]]; then
  # Full Tier-1 stack: the competition's best technique combination
  # 10L + MLP3x + SmearGate + Bigram10k + OrthoInit + SWA0.4 + CompHparams
  run_bench "V8_tier1_full" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 \
     ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 $COMP_VARS"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Novel paper variants (our implementations, not in competition records)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V10" ]]; then
  # ResFormer / Value Residual (arXiv:2410.17897)
  # Why: threads first-layer V through all blocks → 16% param savings at same quality
  run_bench "V10_value_residual" "VALUE_RESIDUAL=1 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V11" ]]; then
  # MoLE 4-expert (arXiv:2503.15798) — zero-FLOP lookup table experts
  # Why: replaces BigramHash with K learned expert tables + routing gate
  run_bench "V11_mole4" "MOLE_NUM_EXPERTS=4 MOLE_DIM=64 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V12" ]]; then
  # MoLE 8-expert — more experts, larger routing space
  run_bench "V12_mole8" "MOLE_NUM_EXPERTS=8 MOLE_DIM=64 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V13" ]]; then
  # TrigramHash — extends BigramHash to 3-token context window
  run_bench "V13_trigram" "TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V14" ]]; then
  # TWEO colinearity penalty (arXiv:2511.23225)
  # Why: eliminates activation outliers → better quantization
  run_bench "V14_tweo" "TWEO_LAMBDA=0.0001 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V15" ]]; then
  # STE QAT int6 from step 0 — train aware of quantization from the start
  run_bench "V15_ste_qat_int6" "STE_QAT=1 QAT_START_FRACTION=0.0 QUANT_BITS=6 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V16" ]]; then
  # Bigram + Trigram stacked — dual n-gram context signals
  run_bench "V16_bigram_trigram" \
    "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel" || "$TARGET" == "V17" ]]; then
  # Value Residual + MoLE — ResFormer meets lookup experts
  run_bench "V17_vr_mole4" "VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64 $COMP_VARS"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Ablation of Tier-1 stack + novel additions
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V20" ]]; then
  # Tier1 stack + Value Residual — can ResFormer improve on the competition stack?
  run_bench "V20_t1_vr" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 \
     ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 VALUE_RESIDUAL=1 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V21" ]]; then
  # Tier1 stack + MoLE replacing BigramHash
  run_bench "V21_t1_mole" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 \
     MOLE_NUM_EXPERTS=4 MOLE_DIM=64 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V22" ]]; then
  # Tier1 stack + TWEO + QAT — anti-outlier training for better quantization
  run_bench "V22_t1_tweo_qat" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 \
     ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 TWEO_LAMBDA=0.0001 STE_QAT=1 QAT_START_FRACTION=0.0 QUANT_BITS=6 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V23" ]]; then
  # Tier1 stack + Trigram — adds 3-gram context on top of bigram
  run_bench "V23_t1_trigram" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 \
     TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64 ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V24" ]]; then
  # Tier1 stack + Value Residual + MoLE + TWEO — full novel stack on top of competition best
  run_bench "V24_full_novel" \
    "NUM_LAYERS=10 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 \
     ORTHO_INIT=1 SWA=1 SWA_START_FRACTION=0.4 VALUE_RESIDUAL=1 \
     MOLE_NUM_EXPERTS=4 MOLE_DIM=64 TWEO_LAMBDA=0.0001 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "combos" || "$TARGET" == "V25" ]]; then
  # WSD + QAT + SWA (training schedule optimized) — warmdown + quant-aware + averaging
  run_bench "V25_sched_opt" \
    "WSD_LR=1 STE_QAT=1 QAT_START_FRACTION=0.0 QUANT_BITS=6 SWA=1 SWA_START_FRACTION=0.4 $COMP_VARS"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER ABLATIONS — isolate key competition choices
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V30" ]]; then
  # Muon WD sweep: 0.01 (not optimal)
  run_bench "V30_muon_wd01" "MUON_WEIGHT_DECAY=0.01 WARMDOWN_ITERS=3000 MATRIX_LR=0.02 SCALAR_LR=0.02"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V31" ]]; then
  # Muon WD sweep: 0.04 (competition optimal)
  run_bench "V31_muon_wd04" "MUON_WEIGHT_DECAY=0.04 WARMDOWN_ITERS=3000 MATRIX_LR=0.02 SCALAR_LR=0.02"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V32" ]]; then
  # LR sweep: matrix_lr=0.025 (11L submission optimal)
  run_bench "V32_lr025" "MUON_WEIGHT_DECAY=0.04 WARMDOWN_ITERS=3000 MATRIX_LR=0.025 SCALAR_LR=0.025"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V33" ]]; then
  # Warmdown=3000 vs default 1200 — longer LR decay
  run_bench "V33_warmdown3k" "WARMDOWN_ITERS=3000"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V34" ]]; then
  # SWA start_frac sweep: 0.5
  run_bench "V34_swa05" "SWA=1 SWA_START_FRACTION=0.5 $COMP_VARS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "ablation" || "$TARGET" == "V35" ]]; then
  # SWA start_frac=0.3 — very early averaging
  run_bench "V35_swa03" "SWA=1 SWA_START_FRACTION=0.3 $COMP_VARS"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SOTA GROUP — techniques from 1.1307 → 1.1233 submissions (XSA, EMA, Partial RoPE, LN Scale, GPTQ-lite)
# All use 11L to match competition architecture. MUON_MOMENTUM_WARMUP_STEPS=1500 as in top submissions.
# ═══════════════════════════════════════════════════════════════════════════════

SOTA_BASE="NUM_LAYERS=11 MLP_MULT=3 SMEARGATE=1 BIGRAM_HASH_BUCKETS=2048 BIGRAM_HASH_DIM=128 \
  ORTHO_INIT=1 MUON_WEIGHT_DECAY=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3"

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V40" ]]; then
  # XSA on last 4 layers — single biggest win (-0.0121 BPB in competition)
  run_bench "V40_xsa4" \
    "$SOTA_BASE XSA_LAST_N=4"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V41" ]]; then
  # EMA decay=0.997 every step — replaces SWA, smoother averaging
  run_bench "V41_ema" \
    "$SOTA_BASE EMA=1 EMA_DECAY=0.997"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V42" ]]; then
  # Partial RoPE: rotate only first 16 of 64 head dims — zero params, position-invariant dims
  run_bench "V42_partial_rope16" \
    "$SOTA_BASE PARTIAL_ROPE_DIMS=16"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V43" ]]; then
  # LN Scale: RMSNorm output × 1/sqrt(layer_idx+1) — damps deeper layer contributions
  run_bench "V43_ln_scale" \
    "$SOTA_BASE LN_SCALE=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V44" ]]; then
  # XSA4 + EMA — replicates the 1.1271 submission (jfprincz PR #287)
  run_bench "V44_xsa4_ema" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V45" ]]; then
  # XSA4 + EMA + Partial RoPE + LN Scale — replicates 1.1248 (PR #374)
  run_bench "V45_xsa4_ema_rope_ln" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V46" ]]; then
  # GPTQ-lite only — post-training clip percentile optimization (zero training cost)
  run_bench "V46_gptq_lite" \
    "$SOTA_BASE EMA=1 EMA_DECAY=0.997 GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V47" ]]; then
  # Full SOTA replication: XSA4 + EMA + Partial RoPE + LN Scale + GPTQ-lite + int6 zstd
  # Targets ≈1.1233 (best known score). Warmdown=3500 as in that submission.
  run_bench "V47_full_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd ZSTD_LEVEL=22 WARMDOWN_ITERS=3500"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V48" ]]; then
  # XSA ablation: last 2 layers only (test sensitivity to XSA depth)
  run_bench "V48_xsa2" \
    "$SOTA_BASE XSA_LAST_N=2 EMA=1 EMA_DECAY=0.997"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V49" ]]; then
  # Partial RoPE ablation: 8 dims (more restrictive than 16)
  run_bench "V49_partial_rope8" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=8 LN_SCALE=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V50" ]]; then
  # SOTA + Value Residual — combines new techniques with our novel VR variant
  run_bench "V50_sota_vr" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V51" ]]; then
  # SOTA + MoLE4 — new embedding technique on top of competition SOTA
  run_bench "V51_sota_mole4" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V52" ]]; then
  # SOTA + TWEO — anti-outlier training stacked on competition techniques
  run_bench "V52_sota_tweo" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TWEO_LAMBDA=0.0001 GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "sota" || "$TARGET" == "V53" ]]; then
  # EMA + Partial RoPE + LN Scale only (no XSA) — ablate XSA contribution
  run_bench "V53_ema_rope_ln" \
    "$SOTA_BASE EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# NOVEL EXTENSIONS — beyond SOTA, unreported techniques
# Uses SOTA_BASE (11L arch) + known working stack
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V54" ]]; then
  # Tight SWA (from 1.1233 submission) — activates only during warmdown when lr<0.2
  # vs regular SWA (fixed time fraction). More targeted = better basin flatness.
  run_bench "V54_tight_swa" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TIGHT_SWA=1 TIGHT_SWA_INTERVAL=50 TIGHT_SWA_THRESHOLD=0.2"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V55" ]]; then
  # Overtone Init — shapes tok_emb SVD spectrum to power law k^{-0.5}
  # Matches natural language frequency structure; better embedding geometry.
  run_bench "V55_overtone_init" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     OVERTONE_INIT=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V56" ]]; then
  # Phase-Transition ResidMix — early layers trust x0 more, late layers trust residual
  # via sigmoid initialization. Better than uniform 0 init for deep U-Net.
  run_bench "V56_phase_resid" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     PHASE_RESID_MIX=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V57" ]]; then
  # Overtone Init + Phase ResidMix + Tight SWA — full novel init/averaging stack
  run_bench "V57_novel_init_stack" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     OVERTONE_INIT=1 PHASE_RESID_MIX=1 TIGHT_SWA=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V58" ]]; then
  # Full SOTA replication + Tight SWA (stacks EMA + Tight SWA for best smoothing)
  run_bench "V58_full_sota_tswa" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd ZSTD_LEVEL=22 WARMDOWN_ITERS=3500 \
     TIGHT_SWA=1 TIGHT_SWA_INTERVAL=50 TIGHT_SWA_THRESHOLD=0.2"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V59" ]]; then
  # SOTA + Value Residual + MoLE4 + TWEO — strongest novel stack on competition SOTA
  run_bench "V59_sota_all_novel" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64 TWEO_LAMBDA=0.0001 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "novel2" || "$TARGET" == "V60" ]]; then
  # Muon momentum warmup to 0.99 (from 0.92) over 1500 steps
  # Stacked with all SOTA: XSA4+EMA+RoPE+LN+GPTQ-lite
  run_bench "V60_sota_gptq_tswa_overtone" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     OVERTONE_INIT=1 TIGHT_SWA=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION VARIANTS — LeakyReLU² and SwiGLU (from top unmerged PRs)
# PR #518: LeakyReLU(0.5)² + Cosine TTT → 1.0622; PR #462: SwiGLU + XSA + TTT → 1.0672
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "activ" || "$TARGET" == "V61" ]]; then
  # LeakyReLU(0.5)² — smooth gradient through negative activations
  # Dead neuron problem in relu² fixed: negative slope 0.5 keeps gradients flowing
  run_bench "V61_leaky_relu2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "activ" || "$TARGET" == "V62" ]]; then
  # SwiGLU (mlp_mult=2) — same param count as relu² mlp_mult=3 (fc outputs 2*hidden)
  # SwiGLU's gating is smoother than hard relu² — better gradient landscape
  run_bench "V62_swiglu_2x" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_MULT=2 MLP_ACTIVATION=swiglu"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "activ" || "$TARGET" == "V63" ]]; then
  # SwiGLU (mlp_mult=3) — larger model, tests if quality gain outweighs byte cost
  run_bench "V63_swiglu_3x" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_MULT=3 MLP_ACTIVATION=swiglu"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "activ" || "$TARGET" == "V64" ]]; then
  # LeakyReLU(0.5)² + full SOTA stack (best activation on best arch)
  run_bench "V64_leaky_relu2_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TTT GROUP — Cosine Test-Time Training (the frontier technique, sub-1.1 BPB)
# NOTE: TTT makes each eval ~ttt_epochs× slower. Use MAX_VAL_TOKENS=2000000 to stay fast.
# PR #442 (1.1027, 10ep), #390 (1.1295, 8ep), #518 (1.0622, 50ep + leaky_relu2)
# TTT_EPOCHS=10 is a good local test; use 50 for competition.
# ═══════════════════════════════════════════════════════════════════════════════

TTT_BASE_VARS="TORCHDYNAMO_DISABLE=1 DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
  ITERATIONS=3000 TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=1024 MAX_WALLCLOCK_SECONDS=0 \
  SKIP_ALL_VAL=0 MAX_VAL_TOKENS=2000000 VAL_LOSS_EVERY=300 TRAIN_LOG_EVERY=10"

run_bench_ttt() {
  local run_id=$1
  local extra_vars=$2
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║  BENCHMARK (TTT): $run_id"
  echo "║  Extra: $extra_vars"
  echo "╚══════════════════════════════════════════════════════════╝"
  eval "env $TTT_BASE_VARS RUN_ID=$run_id $extra_vars $PYTHON train_gpt.py"
  local last_val=$(grep "val_bpb:" "logs/$run_id.txt" 2>/dev/null | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  local last_step=$(grep "val_bpb:" "logs/$run_id.txt" 2>/dev/null | tail -1 | grep -o "step:[0-9]*/" | sed "s/step://;s/\///")
  echo "  RESULT: $run_id → step:$last_step val_bpb=$last_val"
  echo ""
}

if [[ "$TARGET" == "ttt" || "$TARGET" == "V70" ]]; then
  # Cosine TTT 8 epochs — baseline TTT test (PR #390 baseline)
  run_bench_ttt "V70_ttt_8ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=8 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "ttt" || "$TARGET" == "V71" ]]; then
  # Cosine TTT 10 epochs — matches PR #442 (1.1027)
  run_bench_ttt "V71_ttt_10ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "ttt" || "$TARGET" == "V72" ]]; then
  # Cosine TTT 20 epochs (more adaptation per chunk)
  run_bench_ttt "V72_ttt_20ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=20 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "ttt" || "$TARGET" == "V73" ]]; then
  # TTT + LeakyReLU² — PR #518 combo (1.0622 on H100)
  run_bench_ttt "V73_ttt_leaky" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5 TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "ttt" || "$TARGET" == "V74" ]]; then
  # Full SOTA + TTT 10ep — stack on top of 1.1233 replication
  run_bench_ttt "V74_sota_ttt10" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "ttt" || "$TARGET" == "V75" ]]; then
  # TTT higher LR sweep (1e-3 instead of 1e-4)
  run_bench_ttt "V75_ttt_lr1e3" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.001"
fi

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 9 GROUP (V80-V85) — Novel research papers (2024-2026)
#   V80: HybridNorm  (Post-Norm FFN, arXiv:2503.04598)
#   V81: SSNorm      (Single-Scale RMSNorm, arXiv:2506.19697)
#   V82: Both together (Tier S stack)
#   V83: HybridNorm + SSNorm + full SOTA
#   V84: SSNorm on full SOTA alone
#   V85: HybridNorm on full SOTA alone
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$TARGET" == "layer9" || "$TARGET" == "V80" ]]; then
  # HybridNorm alone: Post-Norm on FFN, Pre-Norm on attention (unchanged)
  run_bench "V80_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1"
fi

if [[ "$TARGET" == "layer9" || "$TARGET" == "V81" ]]; then
  # SSNorm alone: Single-Scale RMSNorm (prevents privileged axes → better int6 QAT)
  run_bench "V81_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     SSNORM=1"
fi

if [[ "$TARGET" == "layer9" || "$TARGET" == "V82" ]]; then
  # HybridNorm + SSNorm combined (Tier S stack)
  run_bench "V82_hybrid_ss" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "layer9" || "$TARGET" == "V83" ]]; then
  # Full SOTA (GPTQ-lite) + HybridNorm + SSNorm
  run_bench "V83_sota_layer9" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "layer9" || "$TARGET" == "V84" ]]; then
  # Full SOTA + SSNorm only (isolated test on quantized model)
  run_bench "V84_sota_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     SSNORM=1"
fi

if [[ "$TARGET" == "layer9" || "$TARGET" == "V85" ]]; then
  # Full SOTA + HybridNorm only (isolated test on quantized model)
  run_bench "V85_sota_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd WARMDOWN_ITERS=3500 \
     HYBRID_NORM=1"
fi

# ─────────────────────────────────────────────────────────────────────────────
# DIFF ATTN GROUP (V90-V92) — Differential Transformer (arXiv:2410.05258)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$TARGET" == "diff_attn" || "$TARGET" == "V90" ]]; then
  run_bench "V90_diff_attn" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "diff_attn" || "$TARGET" == "V91" ]]; then
  run_bench "V91_diff_attn_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd \
     DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "diff_attn" || "$TARGET" == "V92" ]]; then
  run_bench "V92_diff_layer9_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd \
     DIFF_TRANSFORMER=1 HYBRID_NORM=1 SSNORM=1"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PERI-LN GROUP (V93-V94) — Peri-LN arXiv:2502.02732 (Gemma/OLMo 2)
# ─────────────────────────────────────────────────────────────────────────────

if [[ "$TARGET" == "peri_ln" || "$TARGET" == "V93" ]]; then
  run_bench "V93_peri_ln" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     PERI_LN=1"
fi

if [[ "$TARGET" == "peri_ln" || "$TARGET" == "V94" ]]; then
  run_bench "V94_peri_ln_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd \
     PERI_LN=1 SSNORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY + CHART
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK SUMMARY — val_bpb (lower = better)                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
printf "%-35s %-12s %-12s\n" "RUN_ID" "final_val_bpb" "final_step"
echo "────────────────────────────────────────────────────────────────"
for log in logs/V*.txt; do
  [ -f "$log" ] || continue
  id=$(basename "$log" .txt)
  last_line=$(grep "val_bpb:" "$log" 2>/dev/null | tail -1)
  [ -z "$last_line" ] && continue
  bpb=$(echo "$last_line" | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  step=$(echo "$last_line" | grep -o "step:[0-9]*/" | sed "s/step://;s/\///")
  [ -n "$bpb" ] && printf "%-35s %-12s %-12s\n" "$id" "$bpb" "$step"
done | sort -k2 -n
echo ""

# Generate chart with val_bpb data
if $PYTHON -c "import matplotlib" 2>/dev/null; then
  echo "Generating val_bpb chart..."
  $PYTHON plot_results.py --val V
else
  echo "(pip install matplotlib for charts)"
fi
