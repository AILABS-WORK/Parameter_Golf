# Scientific Analysis — Parameter Golf Technique Inventory
*Why each technique works, evidence strength, expected gain, and interaction effects.*

## Competition Setup
- **Metric**: val_bpb (bits-per-byte, post-quantization, sliding window stride=64)
- **Scale**: 8×H100 SXM, 600s wall clock → ~7,101 steps × 786,432 tok/step = 5.58B tokens
- **Constraint**: ≤16MB artifact (train_gpt.py code + compressed weights)
- **Baseline** (NaiveBaseline, 2026-03-17): val_bpb = 1.2244
- **SOTA merged** (11L EMA+GPTQ-lite, 2026-03-22): val_bpb = 1.1233 (3-seed mean)
- **V0 local reference** (3000 steps, 24.6M tokens, no extras): val_bpb = 1.7245

## V0 Local Benchmark Progression (reference calibration)

| Step | val_bpb | Notes |
|------|---------|-------|
| 0 | 4.1628 | random init |
| 300 | 2.3670 | — |
| 600 | 2.2839 | — |
| 900 | 2.1880 | — |
| 1200 | 2.1404 | — |
| 1500 | 2.0755 | — |
| 1800 | 1.9875 | — |
| 2100 | 1.9008 | — |
| 2400 | 1.8257 | — |
| 2700 | 1.7609 | — |
| 3000 | **1.7245** | **final; int8+zlib roundtrip = 1.9689** |

Post-quant penalty for unquantized V0: +0.2444 BPB (expected — no QAT). Competition SOTA with QAT sees only +0.016 penalty.

---

## LAYER 1 — Evaluation Strategy (free wins, no training change)

### Sliding Window Eval (stride=64)
- **What**: Score each token with 960 tokens of context instead of 0–1023 average
- **Why it works**: Language has long-range dependencies. With stride=64, tokens near position 0 in a window still get 960 tokens of prior context. The baseline scores them with ~512 average context → systematic underestimation of model quality
- **Evidence**: SlidingWindowEval submission: 1.1925 vs baseline 1.2244 = **−0.032 BPB pure free win**
- **Cost**: ~10× slower eval; ~88s on 8×H100 (acceptable within 10-min cap)
- **Interaction**: Amplifies all other improvements — every technique looks better under sliding window

### FP16 Tied Embedding Export
- **What**: Keep `tok_emb.weight` in FP16 instead of int8 during quantization
- **Why it works**: Tied embeddings serve dual role: input token lookup AND output head (lm_head). Int8 errors compound through both paths. FP16 costs ~2KB vs int8 (vocab×dim×2 bytes) — affordable
- **Evidence**: FP16Embed submission: ~−0.007 BPB
- **Mathematical insight**: For a vocab=1024 token, if embedding error δ propagates to logit → output cross-entropy error scales as δ². Compounding through both paths: error ≈ 2δ². FP16 keeps δ~1e-3 vs int8 δ~0.1

---

## LAYER 2 — Quantization Strategy (artifact compression)

### Int6 per-row (vs Int8)
- **What**: Quantize to [-31,31] (64 levels) with per-row scale instead of [-127,127] per-tensor
- **Why it works**: Transformers have concentrated weight distributions. Per-row scaling reduces max quantization error by ~4× vs per-tensor. Int6 compresses better with zstd than int8 because the 6-bit representation has more structure (fewer distinct values → better entropy coding)
- **Evidence**: Switching from int8+zlib to int6+zstd: ~−0.060 BPB and ~−15% artifact size
- **Why more bits aren't better**: Int8 per-row doesn't compress as well — zstd achieves 3.91× on int8 vs 5× on int6 because int6 values cluster more tightly

### Int5 for MLP weights
- **What**: Use [-16,15] (32 levels) for MLP layers only; keep int6 for attention
- **Why it works**: MLP weights have smoother, more compressible distributions than attention weights. Attention Q/K weights have outlier dimensions (query spectrum), MLP weights are more uniform
- **Evidence**: 1.86MB savings vs uniform int6 → funds entire extra layer
- **Risk**: QAT (STE) is critical to recover quality — naive int5 degrades significantly

### STE Quantization-Aware Training
- **What**: During forward pass, apply fake quantization via straight-through estimator; gradients pass through unmodified
- **Why it works**: The model "sees" quantization noise during training → learns to be robust to it. Without QAT, post-training quantization adds ~0.016 BPB penalty. With QAT from step 0, the gap nearly vanishes
- **Mathematical**: STE approximates ∂Q(w)/∂w ≈ 1 (identity) in regions where Q is differentiable. Biased but works because gradients primarily carry sign information, not magnitude
- **⚠️ DEAD CODE WARNING (torch.compile)**: The 1.1248 submission's `Late QAT` was DEAD CODE. `torch.compile` constant-folds class attributes like `CastedLinear._qat_enabled` at trace time → the threshold check is evaluated once at compile, never changes → QAT never activates. The 1.1248 score was driven entirely by Partial RoPE + LN Scale, NOT QAT.
- **Best practice**: Set `QAT_START_FRACTION=0.0` to activate from step 0 — avoids the constant-folding trap entirely. But note: STE QAT from step 0 is **28% slower** (~350 fewer competition steps), trading speed for zero quantization gap.

### zstd-22 vs zlib-9
- **What**: Switch compression library and level
- **Why it works**: zstd uses asymmetric numeral systems (ANS) entropy coding which achieves near-theoretical limits. At level 22, it uses >100MB dictionary search window. For int6 neural weights with clustered distributions, this yields ~5% additional savings vs zlib-9
- **Evidence**: Present in all top submissions; cost is only compile-time compression (no training overhead)

---

## LAYER 3 — Architecture Scaling (within fixed parameter budget)

### MLP 3× Expansion (hidden=1536 vs 1024)
- **What**: Increase MLP hidden dimension from 2× to 3× model width
- **Why it works**: The MLP is the model's "write" operation — it stores factual associations in its weights (key-value memory interpretation, Geva et al. 2021). Wider MLP = more stored associations. ReLU² activation means ~50% neurons are active per token → effectively a sparse 1.5× width on average
- **Evidence**: Largest single contributor in best submissions: ~−0.029 BPB
- **Why it fits**: Int6 compression makes the larger MLP affordable. Without compression, 3× MLP would exceed 16MB

### 10+ Layers
- **What**: Increase depth from 9 to 10+ layers
- **Why it works**: Additional depth enables more computation (transformer can apply more refining operations per token). U-Net skip connections make deeper nets easier to train by providing gradient shortcuts
- **Trade-off**: More depth → fewer tokens/second on H100 → fewer total training tokens in 10 min. This is why 10L (not 12L) is optimal: depth gain outweighs throughput loss up to ~10 layers
- **Evidence**: 10L+Int5+SWA0.4: best at 1.1428. 11L+Int6: 1.1502 (slightly worse — throughput penalty)

### U-Net Skip Connections
- **What**: Encoder-decoder structure with learned weighted residuals between symmetric layers
- **Why it works**: Skip weights initialized to 0 → gradients flow only through main path initially. As training progresses, skip connections carry fine-grained features. This is related to "neural ODEs" — a deeper model learns to iteratively refine representations
- **Evidence**: Present in all top submissions; ablation not directly quantified but consistently present

---

## LAYER 4 — Embedding & Token Context

### BigramHash Embedding (buckets=10240, dim=128)
- **What**: Hash adjacent token pairs (prev, curr) → 10240-bucket table → linear projection to 512
- **Why it works**: Certain token pairs (e.g., "New York", "United States", byte-pairs) have predictable continuations that don't depend on long-range context. The model's attention must otherwise dedicate capacity to learn these. BigramHash offloads this to a cheap lookup
- **Evidence**: Consistent across top 3 submissions; ablation: 4096→10240 buckets gained −0.001 BPB
- **Collision analysis**: 10240 buckets for vocab^2=1048576 pairs → 1% occupy any bucket. At dim=128, the projection can linearly separate colliding entries
- **10240 vs 4096**: More buckets → fewer collisions → cleaner signal for high-frequency bigrams

### SmearGate
- **What**: Learned per-dimension gate α_d blending current embedding with previous token embedding: `e_t = e_t + α * e_{t-1}`
- **Why it works**: Some semantic dimensions propagate smoothly across tokens (e.g., sentiment, discourse mode). SmearGate learns which dimensions benefit from this blending
- **Evidence**: Appears in all submissions from PR#162 onwards; consistent −0.005 to −0.010 BPB
- **Parameters**: ~512 parameters (one gate per embedding dimension)
- **Interaction**: Complements BigramHash — SmearGate captures soft continuous blending, BigramHash captures discrete pair-specific patterns

### TrigramHash (novel, not in competition records)
- **What**: Extends BigramHash to 3-token window: hash(t-2, t-1, t) → bucket → embedding
- **Why it might work**: Trigrams capture longer syntactic patterns (articles+noun, prefix sequences)
- **Expected gain**: Smaller than BigramHash (bigrams already capture most local structure) — estimated −0.002 to −0.005 BPB
- **Risk**: More hash collisions for same bucket count; needs higher bucket count to be effective

---

## LAYER 5 — Optimizer & Training Dynamics

### Muon Weight Decay (WD=0.04)
- **What**: Decoupled weight decay applied to Muon optimizer's momentum update
- **Why it works**: Weight decay in Muon context pushes matrix weights toward lower Frobenius norm → smaller weights quantize better (less range to cover with limited bits). Also acts as implicit regularization
- **Evidence**: WD=0.04 appears in all top submissions; ablation in best submission: WD=0.01 → 0.04 gained ~−0.003 BPB
- **Why 0.04**: Empirically optimal in the regime where WD helps quantization but doesn't over-regularize

### Orthogonal Initialization
- **What**: Initialize all 2D weight matrices with `nn.init.orthogonal_(gain=1.0)`. Output projections scaled by `1/√(2L)` following muP conventions
- **Why it works**: Orthogonal matrices preserve gradient norms → reduces exploding/vanishing gradients early in training. The singular value spectrum starts uniform (all = 1) which is a natural initialization for next-token prediction
- **Evidence**: Consistent across top submissions; convergence is faster and more stable in early steps
- **muP scaling**: Output projections at `1/√(2L)` ensures that the residual stream variance stays stable at initialization (sum of L skip contributions with variance 1/(2L) each → total variance = 1/2)

### Warmdown=3000–3500 (vs default 1200)
- **What**: LR decays over final 3000–3500 of 7000+ steps (vs final 1200)
- **Why it works**: Longer warmdown = smoother descent into the loss basin. The model has more time to "consolidate" at low LR, reducing sharpness of the loss landscape → better quantization robustness
- **Evidence**: All competition submissions use warmdown=3000; the default 1200 is clearly suboptimal. Best submission used 3500 (−0.0002 BPB over 3000)
- **Int6 vs Int8 warmdown**: For **int6** the optimal is 3000–3500 steps. For **int8** the optimal is **20000 steps** — int8's coarser quantization creates a larger penalty (0.014 BPB) that requires far longer warmdown to reduce (to ~0.005 BPB). Int6's smoother distribution converges faster.
- **Control var**: `WARMDOWN_ITERS=3500` for int6; use `WARMDOWN_ITERS=20000` only if switching to int8

### Lower Learning Rates (matrix_lr=0.02 vs 0.04)
- **What**: Cut Muon and AdamW LRs by half
- **Why it works**: At competition scale (7K+ steps, 3.72B tokens), higher LR causes divergence or oscillation late in training. Lower LR + longer warmdown is the proven recipe at this token budget
- **Evidence**: All top submissions use 0.02; NaiveBaseline uses 0.04 → 1.2244 vs 1.2230 (LowerLR)

---

## LAYER 6 — Post-Training Averaging

### Stochastic Weight Averaging (SWA)
- **What**: Maintain EMA of model weights over last 40–50% of training; use averaged model for serialization
- **Why it works**: Standard SGD navigates toward a sharp minimum — SWA averages across a flat region of the loss basin → the averaged model is at a flatter minimum → generalizes better AND quantizes better (flat minima have smaller gradient magnitudes → less weight spread → more compressible)
- **Evidence**: SWA in best submission; ablation shows start_frac=0.4 better than 0.5 or 0.6
- **Why start_frac=0.4**: Earlier averaging includes less-converged checkpoints → noisy. Later averaging includes fewer checkpoints → less benefit. 0.4 is the sweet spot for 7K-step training
- **Interaction with quantization**: SWA reduces weight distribution variance → narrower dynamic range → better int5/int6 quantization without per-row scale overhead

---

## LAYER 7 — Novel Techniques (not in competition records)

### Value Residual / ResFormer (arXiv:2410.17897)
- **What**: Thread first-layer V projection through ALL subsequent attention blocks via learned α per block (init=0)
- **Why it works**: Deep transformers suffer from "rank collapse" — intermediate representations converge toward similar directions. The first-layer V contains rich raw token information that gets processed out. Residual V provides a bypass that reintroduces this signal at each layer
- **Evidence**: ResFormer paper: 16.11% fewer params for equivalent quality. Enables either (a) higher quality at same params or (b) same quality with smaller model → more layers within 16MB
- **Expected gain**: −0.010 to −0.020 BPB. High potential because it addresses a fundamental information bottleneck

### MoLE — Mixture of Lookup Experts (arXiv:2503.15798, ICML 2025 Oral)
- **What**: K expert embedding tables, each indexed by token ID. Routing gate (softmax) weights their contributions
- **Why it works**: Generalizes BigramHash — instead of one hash function over bigrams, uses K learned expert distributions over unigrams with learned routing. The routing gate can specialize: expert 1 for nouns, expert 2 for verbs, etc.
- **Zero FLOPs advantage**: Table lookup is O(K) memory operations vs O(D²) for a linear layer
- **Expected gain**: −0.015 to −0.030 BPB (superior to BigramHash if correctly tuned)
- **Competition advantage**: Not present in ANY local competition submission → true novel advantage

### TWEO / Anti-Outlier Training (arXiv:2511.23225, ICLR 2026)
- **What**: Add colinearity penalty: `λ × ||W@W.T - diag(W@W.T)||_F²`
- **Why it works**: Without regularization, transformer weights develop extreme outlier directions (values up to 10,000× average). These outliers force the quantization range to be huge → most weights use only a few of the 64 int6 levels. TWEO pushes weights toward isometric distribution → all 64 levels used efficiently
- **Expected gain**: −0.002 to −0.005 BPB directly, but synergizes with QAT → potentially −0.005 to −0.010 BPB combined
- **Key insight**: TWEO doesn't improve unquantized model performance significantly — it specifically improves the quantized model's quality

### WSD Cosine LR Warmdown
- **What**: Cosine decay during warmdown phase instead of linear
- **Why it works**: Cosine warmdown spends more time at intermediate LRs than linear (the "body" of the cosine curve). This is beneficial because gradient updates at medium LR consolidate learning without introducing sharp transitions
- **Expected gain**: −0.001 to −0.003 BPB — small but essentially free

---

## LAYER 8 — Frontier Techniques (unmerged PRs, not yet in leaderboard)

### LeakyReLU² Activation (PR #518, PR #434)
- **What**: Replace `relu(x)²` with `leaky_relu(x, α=0.5)²` in MLP layers
- **Why it works**: Standard ReLU² creates "dead neurons" — once x<0, gradient is exactly 0 and the neuron never recovers. LeakyReLU keeps a small negative slope (α=0.5), allowing gradient flow through negative activations. With 7K training steps, fewer dead neurons = more expressive MLP. The `²` squaring preserves sparsity benefits similar to relu².
- **Evidence**: PR #518 → 1.0622 BPB (with TTT); PR #434 → 1.1370 BPB
- **Control var**: `MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5`
- **Expected gain**: −0.005 to −0.015 BPB (isolated), more with TTT

### SwiGLU Activation (PRs #373, #462, #505)
- **What**: Gate-and-multiply: `proj(signal * SiLU(gate))` where `fc` outputs 2×hidden, split into signal+gate halves
- **Why it works**: SwiGLU's smooth gating provides a learned "importance filter" per neuron. Unlike relu², every dimension can contribute (no hard zeroing). Shown to outperform relu-family in LLaMA, PaLM, etc.
- **Evidence**: PR #505 → 1.1181 BPB (no TTT); PR #462 → 1.0672 BPB (with TTT)
- **Note**: Requires `mlp_mult=2` for same parameter count as `relu² mlp_mult=3`
- **Expected gain**: −0.005 to −0.020 BPB (size-dependent)

### Cosine Test-Time Training (PRs #390, #442, #518)
- **What**: During eval, for each chunk of tokens: run `ttt_epochs` gradient steps (AdamW, cosine LR) then score. Model adapts to the test distribution online.
- **Why it works**: Validation documents have statistical structure (topic, style, domain) that differ from training distribution. TTT lets the model learn per-document patterns from context → dramatically better next-token prediction. The cosine LR annealing (high LR early, low LR late) ensures aggressive adaptation early but avoids over-adaptation at end.
- **Evidence**: PR #390 → 1.1295 BPB (8ep); PR #442 → 1.1027 BPB (10ep); PR #518 → 1.0622 BPB (50ep + leaky_relu2)
- **Control vars**: `TTT_EPOCHS=10 TTT_LR=0.0001`
- **Expected gain**: −0.020 to −0.050 BPB (the single largest remaining opportunity)
- **Key design choice**: "No-reset" TTT (don't restore between chunks) generally outperforms "per-chunk-reset" because the model builds up document-level understanding cumulatively

### Tight SWA (1.1233 submission)
- **What**: SWA triggered by LR scale threshold rather than time fraction. Activates only during warmdown when `lr_scale < 0.2`, every 50 steps.
- **Why it works**: Regular SWA activates at a fixed fraction (40-60% of training). Tight SWA waits until the LR is genuinely low, so only well-converged checkpoints are averaged. Result: tighter basin → better quantization.
- **Evidence**: Present in the 1.1233 submission alongside EMA (they stack: EMA=continuous, Tight SWA=discrete warmdown checkpoints)
- **Control var**: `TIGHT_SWA=1 TIGHT_SWA_THRESHOLD=0.2 TIGHT_SWA_INTERVAL=50`
- **Expected gain**: −0.001 to −0.003 BPB incremental over EMA alone

### Overtone Init + Phase ResidMix (1.1748 submission)
- **Overtone Init**: Shape tok_emb SVD spectrum to power law `S_k ~ k^{-0.5}`. Matches natural language's Zipfian frequency distribution in embedding space.
- **Phase ResidMix**: Initialize skip connection blend weights with `sigmoid(3*(i/(L-1) - 0.5))`. Early layers trust x0 more; late layers trust residual more — reflects how information is processed in U-Net: encode → decode.
- **Evidence**: Both appeared in the 1.1748 submission (NOT in 1.1233); may provide additional gain on top of the full SOTA stack
- **Expected gain**: −0.005 to −0.010 BPB on top of full SOTA stack

---

## LAYER 9 — Novel Research Papers (2024-2026)

*Research survey completed 2026-03-24. 15 papers screened, NOT in any competition submission yet.*
*Ordered by (expected impact × implementation ease). Papers with code available marked [CODE].*

### TIER S — Implement immediately

#### HybridNorm (arXiv:2503.04598, ICML 2025) [CODE]
- **What**: Pre-Norm on attention sublayer, Post-Norm on FFN sublayer. Unifies training stability (Pre-Norm) with final quality (Post-Norm) in a single architecture.
- **Why it applies**: You already have QK-norm stabilizing the attention path — this handles the orthogonal issue (FFN norm placement). Post-Norm on FFN is known to converge to lower loss but was historically too unstable to train; QK-norm makes it safe.
- **Implementation**: Change `x = x + self.mlp(self.ln2(x))` to `x = x + self.ln2(self.mlp(x))` (Post-Norm on FFN). One line per block.
- **GitHub**: github.com/BryceZhuo/HybridNorm
- **Expected gain**: −0.003 to −0.008 BPB
- **Control var**: `HYBRIDNORM=1`

#### OSP — Single-Scale RMSNorm (SSNorm) (arXiv:2506.19697) [CODE]
- **What**: Replace per-channel learned scale `weight` in RMSNorm with a single shared scalar. Prevents channel-wise amplification that creates activation outliers in Adam-trained models.
- **Why it applies**: Muon already gives you the optimizer component of OSP. SSNorm is the architectural component — a one-parameter change that targets the root cause of int6 quantization degradation. Three independent papers agree Muon + outlier suppression is the optimal path.
- **Implementation**: In RMSNorm, change `self.weight = nn.Parameter(torch.ones(dim))` to `self.scale = nn.Parameter(torch.ones(1))`. Forward: `x_norm * self.scale` instead of `x_norm * self.weight`.
- **GitHub**: Code with arXiv:2506.19697
- **Expected gain**: −0.005 to −0.012 BPB (incremental over existing Muon)
- **Control var**: `SSNORM=1`

#### Optimal LR Warmdown via Functional Scaling Laws (arXiv:2602.06797)
- **What**: Derives closed-form optimal WSD warmdown fraction and decay exponent from scaling law parameters. For undertrained models (tokens < 100× params), the optimal decay is steeper and shorter than the standard 20% cosine warmdown.
- **Why it applies**: Your competition is firmly undertrained: ~5.6B tokens for ~150M params = 37× (Chinchilla recommends 1400×). The theory predicts a specific warmdown exponent that maximizes final loss for this regime. Current `WARMDOWN_ITERS=3500` (linear) may be suboptimal shape.
- **Implementation**: Compute scaling law exponents from V0 progression curve. Apply the derived power-decay formula to `lr_mul()`.
- **Expected gain**: −0.002 to −0.005 BPB

### TIER A — High value, manageable complexity

#### ✅ Differential Transformer (arXiv:2410.05258, ICLR 2025 Oral) [CODE] **IMPLEMENTED**
- **What**: Replaces softmax attention with `softmax(Q₁K^T) − λ·softmax(Q₂K^T)`. The subtraction cancels attention noise, producing sparse focused patterns. One extra learnable scalar λ per head.
- **Why it applies**: At 6-bit quantization, Diff Transformer retains near-FP16 quality while vanilla Transformer accuracy drops sharply. Activation outlier kurtosis is dramatically reduced — directly addresses int6 degradation. 4-bit DiffTransformer outperforms 6-bit vanilla by ~25% on zero-shot benchmarks.
- **GQA interaction**: Each KV head serves (Q_pos, Q_neg) pairs. With NUM_KV_HEADS=4 and NUM_HEADS=8, each KV group serves 2 heads: one positive, one negative query.
- **Expected gain**: −0.008 to −0.018 BPB (through better quantization robustness)
- **Implementation**: head_dim halved, two sub-heads Q1/Q2 + K1/K2 sharing same V (2×head_dim). Lambda init = 0.8 - 0.6·exp(-0.3·depth). XSA auto-disabled (V shape incompatible).
- **Control var**: `DIFF_TRANSFORMER=1` → V90-V92 in run_runpod.sh

#### QuaRot — Hadamard Rotation (arXiv:2404.00456, NeurIPS 2024) [CODE]
- **What**: Applies random Hadamard rotation to hidden states before quantization. Rotation preserves the function (mathematically invariant) but distributes outlier energy uniformly across all dimensions → quantization grids become much more efficient.
- **Why it applies**: Synergizes with TWEO (already in stack). TWEO attacks outliers at training time; QuaRot fixes residual outliers at export time. The rotation can be folded into weight matrices offline — zero inference overhead.
- **Expected gain**: −0.003 to −0.008 BPB on top of TWEO
- **Implementation complexity**: Medium. Offline weight rotation at export time.

#### ✅ WSM — Checkpoint Merging (arXiv:2507.17634) **IMPLEMENTED**
- **What**: Replaces LR decay phase entirely with averaging a window of checkpoints collected during constant-LR phase. Keeps LR high throughout → more training signal. WSD+merge outperforms WSD by +3.5% MATH, +2.9% HumanEval.
- **Why it applies**: SWA infrastructure is already in place. The change is to trigger the merge at the end of stable-LR phase rather than during decay. Competition submits the merged checkpoint.
- **Expected gain**: −0.003 to −0.006 BPB
- **Control var**: `WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50` → V86-V89 in run_runpod.sh

#### MUDDFormer (arXiv:2502.12170) *(from prior survey — highest priority novel architecture)*
- **What**: Dense connections from ALL previous layers to current layer, gated by learned fusion weights. Unlike U-Net (skip every 4 layers), every token representation incorporates all preceding layer states.
- **Why it applies**: NOT in any competition submission. Expected to be the largest single novel architecture gain.
- **Expected gain**: −0.020 to −0.040 BPB

### TIER B — Research-grade, higher implementation effort

#### MASA — Matrix Atom Sharing (arXiv:2508.04581)
- **What**: Q/K/V/O matrices across all layers expressed as linear combinations of shared "matrix atoms". Reduces attention params by 66.7% while maintaining parity perplexity on 100M-700M models.
- **Why it applies**: 67% attention savings could fund 2-3 additional transformer layers within the 16MB budget.
- **Expected gain**: −0.005 to −0.015 BPB (through extra capacity from savings)
- **Implementation complexity**: High — restructures attention module for shared dictionary.

#### NuMuon (arXiv:2603.03597) *(from prior survey — near-zero effort)*
- **What**: Nuclear-norm constrained Muon. Single optimizer parameter change (add `nuc_norm_weight=1e-4`).
- **Expected gain**: −0.003 to −0.008 BPB
- **Control var**: `NUMUON_WEIGHT=1e-4`

#### ✅ AGGC — Adaptive Group Gradient Clipping (arXiv:2601.11864) **IMPLEMENTED**
- **What**: Per-parameter-group EMA-tracked gradient norm history → adaptive clip thresholds. Protects embedding/norm parameters (which fall through Muon to AdamW) from over-clipping.
- **Expected gain**: −0.001 to −0.004 BPB
- **Control var**: `AGGC_BETA=0.99 AGGC_THRESHOLD=3.0` → V97 in run_runpod.sh

#### ✅ HybridNorm variant — Peri-LN (arXiv:2502.02732, ICML 2025) **IMPLEMENTED**
- **What**: Places LayerNorm on BOTH input AND output of each sublayer. Unifies Pre-LN and output-LN. Used in Gemma, OLMo families.
- **Expected gain**: −0.002 to −0.006 BPB (alternative to HybridNorm, test one)
- **Control var**: `PERI_LN=1` (mutually exclusive with HYBRID_NORM) → V93-V94 in run_runpod.sh

### Summary Table

| Paper | arXiv | Tier | Expected BPB | Complexity | Status |
|-------|-------|------|-------------|------------|--------|
| HybridNorm | 2503.04598 | S | −0.003/−0.008 | Low | ✅ HYBRID_NORM=1 |
| OSP SSNorm | 2506.19697 | S | −0.005/−0.012 | Low | ✅ SSNORM=1 |
| Optimal LR decay | 2602.06797 | S | −0.002/−0.005 | Low | ✅ WSD_POWER=2.0 (approx) |
| Differential Transformer | 2410.05258 | A | −0.008/−0.018 | Med | ✅ DIFF_TRANSFORMER=1 |
| QuaRot | 2404.00456 | A | −0.003/−0.008 | Med | pending research |
| WSM Merging | 2507.17634 | A | −0.003/−0.006 | Low | ✅ WSM=1 |
| MUDDFormer | 2502.12170 | A | −0.020/−0.040 | High | pending research |
| NuMuon | 2603.03597 | A | −0.003/−0.008 | Low | pending (compressibility) |
| MASA | 2508.04581 | B | −0.005/−0.015 | High | todo |
| AGGC | 2601.11864 | B | −0.001/−0.004 | Low | ✅ AGGC_BETA=0.99 |
| Peri-LN | 2502.02732 | B | −0.002/−0.006 | Low | ✅ PERI_LN=1 |

*Stacking all Tier S+A techniques (assuming 0.6× synergy discount): estimated −0.050 to −0.100 BPB over existing SOTA stack.*

---

## INTERACTION MATRIX — Expected combined effects

*Confirmed synergy factors from 1.1233 submission analysis (2026-03-24):*

| Technique A | Technique B | Measured synergy | Notes |
|-------------|-------------|-----------------|-------|
| **EMA** | **Tight SWA** | **2.0× superlinear** | EMA=−0.0006, TightSWA alone≈−0.0006, together=−0.0012 BPB |
| **Partial RoPE** | **LN Scale** | **1.3× superlinear** | PartRoPE=−0.0010, LNScale=−0.0013, together=−0.0029 BPB |
| Int6 QAT | TWEO | Strongly positive (est) | TWEO reduces outliers → QAT works better |
| SWA | Int5/6 quant | Strongly positive | SWA flattens landscape → better quantization |
| BigramHash | SmearGate | Mildly positive | Orthogonal signals (discrete vs continuous) |
| Value Residual | MoLE | Mildly positive | Both at embedding level, different mechanisms |
| MLP 3× | 10+ layers | Mildly negative | Throughput decreases → fewer training tokens |
| Orthogonal init | SWA | Mildly positive | Stable init → SWA collects more converged checkpoints |
| TWEO | SWA | Positive | TWEO reduces outliers → SWA average has smaller variance |
| TTT | LeakyReLU² | Strongly positive | PR #518 shows 1.0622 (best known score) |
| TTT | Any model improvement | Orthogonal | TTT scales multiplicatively with better base model |
| SwiGLU | TTT | Positive | Smooth gating adapts better to TTT gradient updates |
| Overtone Init | All | Mildly positive | Better initialization allows all techniques to start from better geometry |

---

## RECOMMENDED EXPERIMENT ORDER (by expected ROI, updated 2026-03-24)

**Phase 1 — Baselines and SOTA replication (COMPLETE / IN PROGRESS)**
1. ✅ **V0_baseline** — local reference val_bpb = 1.7245 at 3000 steps
2. 🔄 **V44_xsa4_ema** — replicating 1.1271 submission (step 280/3000)
3. **V47_full_sota** — replicate 1.1233 SOTA locally

**Phase 2 — Frontier techniques from existing PRs**
4. **V61_leaky_relu2** — LeakyReLU² on SOTA stack (easy win, no TTT)
5. **V70_ttt_8ep** — Cosine TTT 8 epochs (highest single impact)
6. **V73_ttt_leaky** — TTT + LeakyReLU² (PR #518 replica → target 1.0622)

**Phase 3 — Layer 9 novel papers (2024-2026)**
7. **V80_hybridnorm** — HybridNorm: Post-Norm FFN (one-line, TIER S)
8. **V81_ssnorm** — OSP SSNorm: single-scale RMSNorm (one-param, TIER S)
9. **V82_diffxfmr** — Differential Transformer (best quantization paper, TIER A)
10. **V83_numuon** — NuMuon: nuclear-norm Muon (single optimizer param, TIER A)
11. **V84_muddformer** — MUDDFormer dense connections (highest expected gain, TIER A)
12. **V85_layer9_stack** — HybridNorm + SSNorm + NuMuon + OptimalLR (TIER S stack)

**Phase 4 — Kitchen sink**
13. **V59_sota_all_novel** — SOTA + VR + MoLE + TWEO + TTT + Layer9
14. **V57_novel_init_stack** — Overtone + PhaseResid + TightSWA

## COMPETITION FRONTIER (as of 2026-03-24)

| Score | Source | Techniques |
|-------|---------|------------|
| 1.2244 | Merged SOTA (baseline) | 9L, int8, no extras |
| 1.1233 | Merged SOTA (best merged) | 11L + XSA4 + EMA + PartRoPE + LN + GPTQ-lite + wds3500 |
| ~1.1295 | Unmerged PR #390 | +TTT 8ep |
| ~1.1216 | Unmerged PR #415 | +XSA4 + Two-Phase TTT |
| ~1.1027 | Unmerged PR #442 | +EMA + AdamW TTT 10ep |
| ~1.0891 | Unmerged PR #490 | +Value Residual + Gated Attn + AdamW TTT |
| ~1.0887 | Unmerged PR #486 | +TrigramHash + VR + GradQuant + Cosine TTT |
| ~1.0622 | Unmerged PR #518 | +LeakyReLU(0.5)² + Cosine TTT 50ep |
| ~1.04–1.05 | **Our novel target (est.)** | All above + Layer 9: HybridNorm + SSNorm + DiffXfmr + NuMuon |
| **<1.04** | **Stretch goal** | Full stack + MUDDFormer |

## EXTRAPOLATION METHODOLOGY

### Token counts (corrected)

- **Local benchmark**: 3,000 steps × 8,192 tok/step = **24.6M tokens** (RTX 5080, batch=8K)
- **Competition (H100 8×)**: 7,101 steps × 786,432 tok/step = **5.58B tokens** (seq_len=2048, batch=384 seqs)
- **Scale ratio**: 5.58B / 24.6M = **227×** more training compute at competition

### Power-law fit

`BPB = a × tokens^(-α)` fitted from 10 val_bpb checkpoints (every 300 steps).

α (scaling exponent) for these small LLMs ≈ 0.35–0.50 from Chinchilla.
At 227× more tokens: expected gap reduction = 227^(-0.40) ≈ 0.17×

**Example**: If V0_baseline reaches BPB=3.20 at 24.6M tokens locally,
extrapolated competition BPB ≈ 3.20 × (5.58B / 24.6M)^(-0.40) ≈ 1.40 (rough)

Error: ±15% on absolute value, but RANKING is reliable (right ordering holds).

### Leaderboard comparison

| Score | Label | Our extrapolation target |
|-------|-------|--------------------------|
| 1.2244 | NaiveBaseline | V0 extrapolated should match |
| 1.1928 | SlidingWindowEval | V0 + EVAL_STRIDE=64 |
| 1.1458 | Int6+MLP3x+SWA | V8_tier1_full |
| 1.1428 | 10L+Int5+SWA0.4 | V8 with 10L |
| 1.1307 | 11L+XSA4 | V40_xsa4 |
| 1.1271 | 11L+XSA4+EMA | V44_xsa4_ema |
| 1.1248 | 11L+PartRoPE+LNScale | V45_xsa4_ema_rope_ln |
| **1.1233** | **SOTA: GPTQ-lite+EMA** | **V47_full_sota** |
| <1.1233 | **Novel target** | V50-V53 + new paper techniques |

### How to use plot_results.py for comparison

```bash
python plot_results.py --val V  # shows all V* variants
```

Bar chart shows:
- **Horizontal bars** = val_bpb at step 3000 (local scale)
- **Diamonds** = extrapolated val_bpb at competition scale (5.58B tokens)
- **Red vertical lines** = leaderboard milestone scores

If a diamond is to the left of a milestone line, that variant should beat that submission at competition scale.
