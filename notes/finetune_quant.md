# On-device fine-tuning, quantisation and pruning of small language models

Verified research note for programmersRAG (features 2, 3 and 5: opportunistic local fine-tuning, a fully local model, and a pruned context fed to a fine-tuned, custom-quantised generator).

- Verification date: 2026-09-29; third-pass re-check and additions 2026-09-30.
- Status: second-pass, adversarial check of a first-pass dossier. Every source below was opened by the checker. arXiv items were checked against the arXiv API record (title, authors, date, comment, abstract) and, where a body-level number, quotation or equation is used, against the text of the PDF. llama.cpp items were read as raw files from the `master` branch (HEAD `ba0ba54d93`, 2026-09-29) or as raw pull-request and issue bodies through the GitHub API. Model cards were fetched and cross-checked against the Hugging Face API.
- Labels used throughout: **[E]** established result, **[R]** recent result (2024-2026), **[C]** contested, **[P]** our proposal or our own derivation (not a published result).
- Evaluation context: all experiments planned for the paper run on a demo repository that is a fork (clone) of `microsoft/mimalloc`; the corpus is about 59 C source and header files, which is small. Several risks below follow directly from that size.

## Summary

The pipeline under study is: build a training set from the indexed repository and from user feedback, fine-tune a low-rank adapter when the machine is idle, merge the adapter, quantise the merged model to GGUF with a domain-calibrated importance matrix, pass a held-out evaluation gate, and swap the new model in atomically.

What the literature supports:

1. Low-rank adaptation is the right family of methods for a sub-2B local model. It forgets less than full fine-tuning [11]. In one 2026 benchmark of sub-1B models, full fine-tuning of models under 300M parameters often fell below the zero-shot baseline and was outperformed by LoRA on Qwen2.5-0.5B [15]. LoRA is nevertheless not automatically safe: it can still degrade general ability badly unless replay or an output-space penalty is added [13, 14, 12].
2. Training the generator on contexts that contain distractors is a published and effective recipe [17, 18]. This matters for us because a 25% admission threshold deliberately lets distractors through.
3. Retrieval and fine-tuning are complementary for code [22, 26], and per-project adaptation has been shown to help on at least one code task [25].
4. llama.cpp k-quants with an importance matrix are a weighted least-squares scalar quantiser whose weights are per-input-channel mean squared activations [51, 55, 58]. Calibration data enters only through those second moments.
5. Evidence that 4-bit quantisation is nearly lossless comes almost entirely from models of 7B parameters or more. The evidence that exists for small models and for code is less favourable [77, 78, 86].
6. Weight pruning is dominated by quantisation at equal compression and is more sensitive to calibration data [86, 68]. Pruning the prompt instead has measured latency benefits and, for code, 2024-2026 methods that prune at function or line granularity [88, 89, 90, 91, 92].
7. Native C/C++ training support is immature. Upstream llama.cpp offers full-parameter FP32 fine-tuning that is explicitly work in progress, and LoRA training is not implemented upstream [62, 64, 61].

### Verification status and corrections to the first-pass dossier

All 73 sources in the first-pass dossier were located and their titles and author lists matched; none had to be dropped. Eight composite entries (several model cards or files under one key) were split so that each reference has one URL, giving 81. A further 23 sources were added by the checker after opening them, for a total of 104 verified references. A third pass on 2026-09-30 re-opened every arXiv abstract page (all titles, author lists and submission dates matched), re-read the llama.cpp files from `master`, the pull-request and issue bodies through the GitHub API, the model cards and the TFX PDF, and independently re-derived every formula. It confirmed corrections 1-20 below (correction 2 was re-checked in `quantize_q4_K`, which calls `quantize_row_q4_K_ref` when no importance matrix is given) and found no further metadata errors. It added five sources [105-109] on precision scaling laws, forgetting and LoRA practice, and two formulas (F28, F29). The following errors were found and corrected in this note:

| # | First-pass statement | Correction |
|---|---|---|
| 1 | Q2_K uses 2.5625 bits per weight; k-quants have "per-block 6-bit scales" | The pull-request text says 2.5625, but the current `block_q2_K` struct holds two fp16 super-block scales, so it is 2.625 bpw, as the source comment states. Scale precision differs by type: 4 bits (Q2_K), 6 bits (Q3_K, Q4_K, Q5_K), 8 bits (Q6_K) [56]. |
| 2 | Without an importance matrix the Q4_K fit weight is sqrt(sigma^2) + abs(x) with sigma^2 computed over the 256-weight super-block | That expression sits in a branch of `quantize_row_q4_K_impl` that is not reached: when no importance matrix is supplied `quantize_q4_K` calls `quantize_row_q4_K_ref`, whose weight is sqrt(mean of x^2 over the 32-weight block) + abs(x) and which uses `make_qkx2_quants` [55]. |
| 3 | Closed-form scale and offset of the block fit | Formula correct, but the code also clamps the offset to be non-positive (if the fitted offset is positive it is set to zero and the scale becomes S_xL / S_LL), then quantises the eight scales and eight offsets to 6 bits and re-assigns the integer levels [55]. |
| 4 | "k-quants with imatrix minimise exactly the diagonal approximation of the GPTQ objective" | Overstated. The stored importance is exactly the mean squared activation, i.e. diag(X X^T)/N (verified in code), but the implemented fit weight is that importance multiplied by sqrt(sigma^2 + x^2), and no compensation of the remaining weights is performed. See formula F12 and F25. |
| 5 | "The benefit of an importance matrix grows as bit width shrinks" (10%, 17%, 25%) | Cherry-picked. The relative reduction in mean KL divergence in the same table is 10.0% (Q4_K_M), 25.9% (Q4_K_S), 16.2% (Q3_K_L), 17.2% (Q3_K_M), 12.4% (Q3_K_S), 25.4% (Q2_K): not monotone. Only the absolute reduction grows from the 4-bit to the 2-bit class [54]. |
| 6 | RAFT "Torch Hub 84.95% vs 8.60% baseline" | The 8.60 baseline is Llama2-7B with RAG. Domain-specific fine-tuning without RAG scores 84.94 on the same data set, so RAFT's gain over plain fine-tuning there is 0.01 points. The HuggingFace figure (74.00 vs 61.06) is a genuine gain [17]. |
| 7 | Recommendation "rsLoRA scaling alpha/sqrt(r) with alpha = 2r" | These are two different conventions. alpha = 2r was validated under the classical alpha/r scaling (effective scale 2) [11]; inserting it into alpha/sqrt(r) gives an effective scale 2 sqrt(r) (8, 16, 32 at r = 16, 64, 256), which neither paper tested. See F3. |
| 8 | Recommendation "consider PiSSA if the adapter will be merged and quantised" | Unsupported. PiSSA's roughly 20% lower quantisation error applies when the frozen residual is quantised and the adapter stays in full precision [5]. After merging, the full matrix is re-formed and re-quantised, so that advantage does not apply. |
| 9 | "Independent studies agree 4-bit retains performance" | Overstated; see point 5 of the summary and the contested results below. |
| 10 | Metadata | LoRA is ICLR 2022; LoRA+ is ICML 2024 (PMLR 235:17783-17806); LoftQ is ICLR 2024; "LoRA vs Full Fine-tuning" is NeurIPS 2025; Weyssow et al. is ACM TOSEM 34(7), 2025; Bonawitz et al. is SysML 2019; EWC is PNAS 114(13), 2017 (not 2016); the SmolLM2 card dates from 2024. |
| 11 | "Gemma 4 (July 2026)" | The technical report is dated 2 July 2026, but the Hugging Face repository `google/gemma-4-E2B-it` was created on 2 March 2026. The note no longer states a release month. |
| 12 | Qwen3-0.6B and Qwen3-1.7B sizes | The cards state 0.6B and 1.7B, but the stored safetensors hold 0.752B and 2.03B parameters. Memory must be budgeted from the stored count [33, 34]. |
| 13 | "a 1.5B model is already under about 1.2 GB at 6 bits" | 1,543,714,304 parameters at 6.5633 bits per weight is about 1.27 GB (1.18 GiB). This is our estimate and assumes the effective bits per weight published for Llama-3.1-8B. |
| 14 | DoRA gains "+3.7 / +1.0 / +2.9 / +4.4" | These are quoted from the introduction. In the body the full-rank DoRA gain on LLaMA2-7B is 2.1 points; 2.9 belongs to the half-rank variant [3]. Reworded as a range. |
| 15 | AWQ calibration robustness | The 0.5-0.6 versus 2.3-4.9 perplexity figures are for OPT-6.7B at 3-bit (INT3-g128), and the same passage states that matching calibration and evaluation distributions works best [48]. |
| 16 | Code-specific calibration helps at 3 and 2 bits "for code models of 1B-34B" | The calibration experiments used 7B models with AQLM. The 1.3B model was only evaluated at 2 bits [69]. |
| 17 | rsLoRA formula written with a bias term | Only the adapter form and the merged matrix could be confirmed in the paper; the formula was rewritten accordingly (F2). |
| 18 | Peak-memory reduction "26-28x" for on-device LoRA | Those factors were measured on an NVIDIA A100. On a phone the same paper reports peak memory of 7.38% of baseline at 32 tokens and 19.37% at 2048 tokens [99]. |
| 19 | Latency model: "quantisation changes v_pp and v_tg" | True, but prefill does not reliably get faster. In llama.cpp's own Llama-3.1-8B table F16 has the highest prompt-processing rate (923 tokens/s versus 822 for Q4_K_M) while generation rises from 29 to 72 tokens/s [52]. |
| 20 | Windows idle semantics | Correct as quoted, but the page documents Windows 7 and Windows 8 behaviour; it does not mention Windows 11 [97]. |

## Established results

### Parameter-efficient fine-tuning

- **[E] LoRA.** The pretrained weight is frozen and a rank-r update is learned; A is Gaussian-initialised and B is zero so training starts at the base model; the update is scaled by alpha/r; after training the update can be merged so inference latency is unchanged. The abstract reports 10,000 times fewer trainable parameters and 3 times less GPU memory than full fine-tuning of GPT-3 175B [1].
- **[E] QLoRA.** Gradients are back-propagated through a frozen 4-bit base into LoRA adapters using the NF4 data type, double quantisation (0.5 to 0.127 bits of overhead per parameter, a saving of 0.373) and paged optimisers. The authors report that 4-bit QLoRA matches 16-bit full fine-tuning and 16-bit LoRA on their benchmarks [2]. Later work disputes the generality of that equivalence (see Contested).
- **[E] Catastrophic forgetting and EWC.** A Fisher-weighted quadratic penalty anchors parameters that were important for earlier tasks [100]. Forgetting is observed in LLMs of 1B to 7B parameters under continual instruction tuning [101].

### Post-training quantisation

- **[E] GPTQ** minimises the layer-wise output reconstruction error with second-order (Optimal Brain Surgeon style) updates of the not-yet-quantised weights, using 128 random 2048-token C4 segments as calibration data, 1% diagonal dampening and a Cholesky reformulation [47, 49].
- **[E] AWQ** protects salient weight channels, identified from activation magnitudes, by per-channel scaling found with a grid search, without back-propagation [48].
- **[E] GGUF k-quants.** Weights are grouped in super-blocks of 256. Per-type layout, read from the struct definitions [56, 50]:

| Type | Blocks x weights | Reconstruction | Block scale / offset precision | Bits per weight |
|---|---|---|---|---|
| Q2_K | 16 x 16 | x = a q + b | 4 bits / 4 bits | 2.625 |
| Q3_K | 16 x 16 | x = a q | 6 bits / none | 3.4375 |
| Q4_K | 8 x 32 | x = a q + b | 6 bits / 6 bits | 4.5 |
| Q5_K | 8 x 32 | x = a q + b | 6 bits / 6 bits | 5.5 |
| Q6_K | 16 x 16 | x = a q | 8 bits / none | 6.5625 |

  Named mixes raise precision on sensitive tensors. As introduced, Q4_K_M and Q5_K_M use Q6_K for "half of" the `attention.wv` and `feed_forward.w2` tensors [50]. The current code selects those layers with `use_more_bits(i, n) = i < n/8 || i >= 7n/8 || (i - n/8) % 3 == 2` [57]. Documented effective sizes for Llama-3.1-8B are 4.8944 (Q4_K_M), 5.7036 (Q5_K_M), 6.5633 (Q6_K), 8.5008 (Q8_0) and 16.0005 (F16) bits per weight [52].
- **[E] Importance matrix.** llama.cpp accumulates the squared activations entering each tensor and a count; at quantisation time the sums are divided by the counts, so the importance of input column j is the mean squared activation [59, 58, 51]. By default the importance matrix is not applied to `output.weight` [53]. The author reports about 50,000 calibration tokens as typical and suggests that instruct-format calibration data may suit instruct-tuned models [51].
- **[E] Accuracy versus bits on large models.** LLaMA-7B perplexity: F16 5.9066, Q6_K 5.9110, Q5_K_M 5.9208, Q4_K_M 5.9601, Q3_K_M 6.1503, Q2_K 6.7764 [50]. Llama-3.1-8B-Instruct, mean of five benchmarks and perplexity: F16 69.47 / 7.32, Q5_K_M 69.36 / 7.40, Q4_K_M 69.15 / 7.56, Q3_K_M 68.07 / 7.96, Q3_K_S 65.49 / 8.96 [74]. In the second study Q5_0 scores 69.92, above F16, which indicates that differences of a few tenths of a point are within noise.

### Weight pruning

- **[E] SparseGPT** prunes large models to at least 50% unstructured sparsity in one shot [83]; **Wanda** reaches similar perplexity without weight updates (LLaMA-7B at 50% sparsity: magnitude 17.29, SparseGPT 7.22, Wanda 7.26) [84]; **LLM-Pruner** removes 20% of LLaMA-7B structurally and, after LoRA recovery, retains 94.97% of the original zero-shot classification accuracy [85].

### Opportunistic scheduling

- **[E]** Production on-device learning runs only when it cannot disturb the user: the job is invoked "when the phone is idle, charging, and connected to an unmetered network", and "the FL runtime will abort, freeing the allocated resources, if these conditions are no longer met" [95]. The Gboard deployment additionally required at least 2 GB of memory [96]. Federated averaging is the underlying algorithm [94]; only the eligibility rule is relevant to a single machine.
- **[E]** Windows Task Scheduler defines idleness by user absence plus low processor and disk use. From Windows 8 the user is by default considered absent after four minutes without keyboard or mouse input and the thresholds are 80%; idle-triggered tasks do not start on battery power by default; `StopOnIdleEnd` terminates the task when idleness ends and `RestartOnIdle` restarts it [97].

### Promotion gates and statistics

- **[E]** TFX validates that a model is "safe to serve" and compares quality "against a fixed threshold as well as against a baseline model (e.g., the current production model)"; "any new model failing any of these checks is not pushed to serving". The same section notes the tension between thresholds that are too strict and too loose [103].
- **[E]** Two models evaluated on the same questions should be compared with the paired-difference standard error; questions drawn in groups need clustered standard errors, which in the paper's examples are up to about three times the naive value; a sample-size formula gives the minimum detectable effect [104].

### Runtime support in llama.cpp

- **[E]** Adapters are loaded with `--lora`; `--lora-init-without-apply` loads them with scale zero; `POST /lora-adapters` sets the global scale and a per-request `lora` field overrides it; requests with different adapter configurations are not batched together [65]. `llama-export-lora` merges adapters into a base model and writes a merged GGUF [66].
- **[E]** Fidelity of a quantised model is measured with `llama-perplexity --kl-divergence` against stored base-model logits. The code computes the sum over the vocabulary of p_base (log p_base - log p_quant), skipping entries with log p_base <= -16, and also reports the rate at which both models agree on the top token [60, 54].

## Recent work (2024-2026)

### LoRA variants

- **[R] DoRA** separates magnitude from direction and applies LoRA to the direction; reported gains over LoRA on commonsense reasoning range from 1.0 to 4.4 points across LLaMA-7B, LLaMA-13B, LLaMA2-7B and LLaMA3-8B, with no inference overhead after merging [3].
- **[R] rsLoRA** proves that adapters are rank-stabilised if and only if the scaling factor is of order 1/sqrt(r) [4]. **LoRA+** shows that equal learning rates for A and B are inefficient at large width and sets a fixed ratio between them, reporting 1-2% improvements and up to about 2x faster fine-tuning [6]. **PiSSA** initialises the adapter from the principal singular components and freezes the residual [5].
- **[R] Quantisation-aware adapters.** LoftQ chooses the quantised backbone and the adapter initialisation jointly [9]. QA-LoRA uses group-wise operators so that, after fine-tuning, adapter and quantised weights integrate into a quantised model "without loss of accuracy" [10]. Both target a quantised base during training, which is a different pipeline from merge-then-quantise.
- **[R] Beyond low rank and uniform learning rates.** RandLoRA performs full-rank updates through learned combinations of fixed random low-rank matrices [7]. AnLR-LoRA (September 2026) gives each rank-one component its own learning rate computed from function-space velocity and Adam signal-to-noise ratio, with no extra trainable parameters [8].
- **[R] Practitioner evidence on LoRA settings (not peer reviewed).** A 2025 technical blog post from Thinking Machines Lab reports that LoRA matched full fine-tuning in a "low-regret regime" covering most post-training data-set sizes; attention-only LoRA underperformed all-layer LoRA even at a matched parameter count; LoRA was less tolerant of large batch sizes, independently of rank; and the optimal LoRA learning rate was about 10x that of full fine-tuning, consistent with Biderman et al. [108, 11]. This agrees with recommendation 2 below but should be cited as a blog-format report.
- **[R] Small models specifically.** On five sub-1B models, full fine-tuning harmed models under 300M parameters, often below zero-shot accuracy; LoRA and DoRA performed comparably; the authors recommend PEFT by default for sub-1B models [15]. The tasks were mathematical reasoning, not code.
- **[R] Hybrid architectures.** On Qwen3.5-0.8B (18 Gated DeltaNet layers and 6 softmax-attention layers) adapting only the attention pathway beat full-model adaptation with 5-10x fewer trainable parameters, while adapting the recurrent backbone alone lowered GSM8K by 14.8 points. The study used a single seed, r = 16 and alpha = 32 [16].

### Forgetting under LoRA

- **[R]** On code and mathematics with Llama-2-7B, LoRA forgets less than full fine-tuning and regularises more strongly than weight decay or attention dropout [11].
- **[R]** Replay of generic next-token-prediction data plus a penalty on the KL divergence from the adapted model to the initial model "can virtually eliminate the problem" for Qwen 2.5 Instruct models of 1.5B to 14B parameters. Reported details: replay alone gave about a 3x reduction in forgetting; a replay rate of 3 examples per training example performed best and 1 was the most economical; a KL coefficient of 0.1 removed forgetting but also nearly all plasticity, 0.01 gave the better trade-off, and 0.001 combined with replay kept the plasticity of standard fine-tuning with an over 7x average reduction in forgetting. With LoRA the frozen base is shared between the two forward passes, so no memory is added beyond LoRA itself [13].
- **[R]** When no replay corpus is available, Target-Masked KL removes the target token from both the base and adapted next-token distributions, renormalises, and penalises the divergence over the remaining vocabulary, so that the regulariser does not oppose the cross-entropy signal [14]. Preprint, in submission.
- **[R] Forgetting tracks KL divergence on the new task.** The degree of forgetting was determined by the KL divergence between the fine-tuned and base policies measured on the new task's distribution; supervised fine-tuning can move arbitrarily far in KL, whereas on-policy RL is biased towards KL-minimal solutions [109]. This gives an independent reason for the KL penalty of F18, and suggests logging the mean KL between adapted and base model on the training prompts as a cheap forgetting monitor (our inference).
- **[R]** FOREVER schedules replay by the magnitude of optimiser updates rather than by step count, evaluated on models from 0.6B to 13B parameters [102].

### Fine-tuning for retrieval-augmented generation

- **[R] RAFT.** For a fraction P of training questions the context contains the oracle document and distractors; for the rest only distractors. Targets are chain-of-thought answers that quote the oracle verbatim between `##begin_quote##` and `##end_quote##`. The best P was 40%, 60% or 100% depending on the data set; the set-up used one oracle and four distractors with Llama2-7B [17].
- **[R] Retrieval robustness.** Mixing top-1, low-ranked and random contexts with equal probability, "even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones" (Llama-2-13B) [18]. Related: RA-DIT tunes the language model and the retriever in separate steps [19]; Finetune-RAG reports a 21.2% improvement in factual accuracy from training on imperfect retrieval [20]; RoseRAG targets small models with margin-aware preference optimisation [21].
- **[R] Code.** An industrial study on more than 160,000 C++ files found that retrieval can beat fine-tuning alone, that BM25 was the most effective and efficient retriever studied, that the two approaches are "orthogonal and their combination leads to further improvement", and that retrieval scaled better with code-base size [22]. A Verilog repository-level system likewise combines domain fine-tuning with retrieval [26]. Per-project fine-tuning "can greatly improve the models' quality" on method-name prediction for Java [25].
- **[R] Synthetic instruction data.** Magicoder seeds instruction generation with real open-source snippets (75K examples) to reduce the bias of purely model-generated data [24].
- **[R] Recursion risk.** Training on model-generated data causes loss of the tails of the distribution [27]; accumulating synthetic data alongside the original real data, rather than replacing it, avoids that collapse in the settings tested [28].

### Small open models (verified from cards, reports and the Hugging Face API)

| Model | Stored parameters | Licence | Context | Notes |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct [30] | 494,032,768 (card: 0.49B, 0.36B non-embedding) | Apache-2.0 | 32,768 | 24 layers, hidden 896, tied embeddings; HumanEval 61.6, MBPP 52.4 [29] |
| Qwen2.5-Coder-1.5B-Instruct [31] | 1,543,714,304 (card: 1.54B, 1.31B non-embedding) | Apache-2.0 | 32,768 | 28 layers, hidden 1536, tied embeddings; HumanEval 70.7, MBPP 69.2 [29] |
| Qwen3-0.6B [33] | 751,632,384 (card: 0.6B, 0.44B non-embedding) | Apache-2.0 | 32,768 | 28 layers; thinking and non-thinking modes [32] |
| Qwen3-1.7B [34] | 2,031,739,904 (card: 1.7B, 1.4B non-embedding) | Apache-2.0 | 32,768 | 28 layers |
| Qwen3.5-0.8B [35] | 873,438,784 (card: 0.8B) | Apache-2.0 | 262,144 | 24 layers, 6 x (3 x Gated DeltaNet + 1 x gated attention); vision encoder |
| Qwen3.5-2B [36] | 2,274,069,824 (card: 2B) | Apache-2.0 | 262,144 | same hybrid layout |
| Gemma 3 270M [38] | 268,098,176 | Gemma | 32K | 170M embedding + 100M transformer parameters, 256k vocabulary; "not designed for complex conversational use cases" [37] |
| Gemma 3 1B [39] | 999,885,952 | Gemma | 32K input, 8,192 output | |
| Gemma 4 E2B [41] | 5,123,178,051 (card: 2.3B effective, 5.1B with embeddings) | Apache-2.0 | 128K | text, image and audio [40] |
| SmolLM2 135M / 360M / 1.7B [44] | 361,821,120 (360M variant) | Apache-2.0 | not checked | primarily English [42] |
| SmolLM3-3B [43] | 3,075,098,624 | Apache-2.0 | trained at 64k, up to 128k with extrapolation | |
| Phi-4-mini-instruct [46] | 3,836,021,760 | MIT | 128K | 200,064-token vocabulary [45] |

Qwen2.5-Coder was pretrained on more than 5.5 trillion tokens, first at file level (sequence length 8,192, next-token and fill-in-the-middle objectives) and then at repository level (32,768 tokens, about 300B tokens) with the special tokens `<|repo_name|>`, `<|file_sep|>` and `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` [29]. The llama.cpp quantisation documentation uses `google/gemma-4-E2B-it` as its worked conversion example, which confirms GGUF conversion support for that model [52].

### Quantisation

- **[R] Calibration data.** A systematic study found "substantial variations in downstream task performance" across calibration sets, with quantisation methods showing lower dispersion than pruning methods [68]. For pruning, calibration data similar to the training data works best and self-generated data improves results by up to 2.68% [70]; model-generated calibration data is competitive for both quantisation and pruning [71]. For code models of 7B parameters quantised with AQLM, a code-specific calibration set limits the loss at 3 and 2 bits, while 4 bits showed no significant loss [69].
- **[R] Importance matrix in practice.** On Llama-3-8B the mean KL divergence to FP16 falls from 0.031273 to 0.028152 for Q4_K_M, from 0.043136 to 0.031951 for Q4_K_S, from 0.101913 to 0.084358 for Q3_K_M and from 0.445132 to 0.332223 for Q2_K when an importance matrix built from Wikitext is used, and "there seems to be no consistent improvement from using more Wikitext tokens" (1 thousand to 10 million tokens) [54]. Calibration and evaluation both used Wikitext, so this is an in-distribution measurement.
- **[R] Small models and code.** A benchmark of SmolLM-135M and Qwen2.5-0.5B found relative drops generally under 0.5% at 8-bit weights and activations but sharp degradation at 4-bit weights with 8-bit activations, larger for Qwen2.5-0.5B than for SmolLM-135M [77]. These are weight-and-activation schemes, not weight-only GGUF. On automated program repair (QuixBugs, 40 bugs), GPTQ int8 changed the number of fixed bugs by -0.25 on average for Phi-3 and Qwen2.5-Coder models, but int4 by -11.25; the int4 mean includes a configuration that fixed no bugs, and the authors note the single-benchmark limitation [78].
- **[R] Scaling laws for post-training quantisation: over-trained small models suffer more.** Fitting over 465 pretraining runs (validated up to 1.7B parameters and 26B tokens, with GPTQ as the post-training quantiser), Kumar et al. model the loss increase from post-training quantisation as approximately a power law in the token-to-parameter ratio that grows exponentially as precision falls, and conclude that "overtrained language models are more sensitive to post-training quantization" [105] (F28). Independently, on Pythia checkpoints of 160M to 12B parameters trained on 1B to 206B tokens, Ouyang et al. find that smaller models trained on many tokens suffer significant quantisation-induced degradation and fit a similar law (F29) [106]. Both fits use token budgets far below those of the models considered here, so they support the direction of the risk, not a numerical prediction.
- **[R] Reasoning and code under quantisation.** On distilled reasoning models from 1.5B to 70B parameters, W8A8 and W4A16 were near-lossless while lower bit widths introduced "significant accuracy risks", with model size, model origin and task difficulty identified as determinants; the benchmarks include LiveCodeBench [107].
- **[R] Large-model evaluations.** Over 500,000 evaluations on the Llama-3.1 family found 8-bit floating point effectively lossless and 4-bit weight-only quantisation "more competitive than expected" [73]; a ten-benchmark study found 4-bit models comparable to unquantised ones and perplexity a usable proxy on most benchmarks [72].
- **[R] Fidelity metrics.** Accuracy "can exhibit unstable, non-monotonic behavior under progressive quantization"; divergence between full-vocabulary distributions is a more sensitive signal, and mixed-precision Q4_K generally gives lower divergence than uniform Q4_0 at similar size. The authors state that divergence does not by itself establish correctness or user-perceived quality [75].
- **[R] Theory.** Minimising the KL divergence between original and quantised outputs under a size constraint can be approximated by minimising squared quantisation error, and Fisher information gives a per-tensor bit allocation that saves up to 0.25 bits per parameter [76].
- **[R] Throughput.** Decode throughput can be predicted from GGUF metadata to about 12-36% error across three systems, but "fitted efficiencies are not universal" [81]. Prefill and decode reward different quantisation choices [82].
- **[R] Security and privacy.** GGUF quantisation can be exploited to build models that are benign in full precision and malicious after quantisation [79]. In one 2026 study of fine-tuned small models, GGUF Q4_K_M without a calibration corpus reproduced 5.3% of planted private records while AWQ and GPTQ reproduced none on the primary model [80].

### On-device and native training

- **[R]** llama.cpp `examples/training`: "So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP. Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory." [62]. `finetune.cpp` trains all parameters (`llama_opt_param_filter_all`) and saves a full model [63]. The public header exposes `llama_opt_init`, `llama_opt_epoch` and a parameter-filter callback, and adapter loading functions, but no LoRA-training entry point [61]. The open issue states that LoRA training "should in principle be achievable with the same tools as the full finetune, it's simply not implemented" and lists the order-of-operations comparison as open research [64].
- **[R]** A fork adds `llama_lora_training_init`, adapter saving and masked losses on Vulkan, Metal, CPU and OpenCL back-ends, and reports Qwen3-1.7B (Q8) at 5.5 minutes per epoch on an RTX 4090, 20 minutes on an Intel Arc A770, 40 minutes on an Apple M3 Pro and 7 h 40 min on a Mali G715 (data-set size per epoch not stated in the passage read; no CPU-only figure is given). Its authors describe the changes as upstream-safe and intended for later upstream integration; the functions are absent from the upstream header [61]. Vendor blog, not peer reviewed [67].
- **[R]** MobileFineTuner is a C++ framework for full and LoRA fine-tuning on phones (GPT-2, Gemma 3 270M and 1B, Qwen2.5-0.5B) with an energy-aware scheduler that slows computation when the battery falls below a threshold. LoRA on Qwen2.5-0.5B at sequence length 128 peaked at about 2,887 MB and took between 7.8 and 39.3 hours per task on the phone (four tasks) [98].
- **[R]** Four techniques (quantised base with on-the-fly dequantisation, checkpointing with disk offload, softmax approximation, logits masking) reduce LoRA peak memory for Llama-3.2 3B at 2,048 tokens from 26.20 GB to 1.02 GB on an A100 [99].

### Context pruning

- **[R]** LLMLingua reports up to 20x compression with little loss [87]. LLMLingua-2 casts compression as token classification with a bidirectional encoder and reports 1.6x-2.9x end-to-end speed-up at 2x-5x compression [88]. Provence thresholds per-token keep probabilities, rounds to whole sentences (a sentence is kept if more than half of its tokens are kept), unifies pruning with reranking and adapts the pruning ratio to the context, between 50% and 80% across data sets at fixed thresholds [89].
- **[R] Code-specific.** Hierarchical Context Pruning models the repository at function level and finds that "pruning the specific implementations of functions in all dependent files does not significantly reduce the accuracy of completions" [90]. SWE-Pruner trains a 0.6B skimmer to select lines given an explicit goal and reports 23-54% token reduction on agent tasks [91]. LaMR separates semantic evidence from dependency support using labels derived from syntax-tree analysis [92].
- **[R] Why pruning helps quality as well as latency.** Accuracy degrades when the relevant passage sits in the middle of a long context [93].

### Contested results

- **[C] Is LoRA equivalent to full fine-tuning?** On code and mathematics, LoRA at common ranks "substantially underperforms full finetuning"; full fine-tuning learns perturbations of rank 10-100x larger than typical LoRA configurations; with r = 256 LoRA matched full fine-tuning on code instruction tuning [11]. LoRA introduces "intruder dimensions" that are causally linked to forgetting and accumulate under sequential fine-tuning [12]. Naive LoRA supervised fine-tuning "can catastrophically degrade model capabilities, even when trained on very small datasets for relatively few steps" [13].
- **[C] RAG versus fine-tuning for code.** Retrieval beat fine-tuning alone on an industrial C++ code base [22], whereas PEFT was superior to in-context learning and retrieval on three Python generation data sets [23]. Only the first study evaluates the combination of the two.
- **[C] Is 4-bit safe?** Large-model studies say mostly yes [72, 73, 69]. LLM-KICK reports an 8-10% drop even at 8 bits on closed-book factoid question answering and found 8-bit matching for Vicuna-7B but 4-bit matching only for Vicuna-13B on MMLU; it also found that compressed models remain robust when the answer is present in the context [86]. Small-model evidence is unfavourable at 4 bits [77, 78], and quantisation scaling laws predict larger degradation for small, heavily over-trained models [105, 106].
- **[C] How much does the calibration distribution matter?** AWQ reports robustness to calibration shift, in one configuration, and still finds matched distributions best [48]; others find substantial variation [68]. None of the sources opened for this note reports an out-of-distribution measurement for llama.cpp importance matrices.
- **[C] Is weight pruning useful?** Pruning papers report small perplexity changes at 50-60% sparsity [83, 84]; LLM-KICK finds that "all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%)" and that "current quantization methods are more successful than pruning" [86].
- **[C] Which metric gates what?** llama.cpp documentation warns that "finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases" [54]; LLM-KICK argues that perplexity misses capability changes [86]; another study finds perplexity a usable proxy for quantised models [72].

## Formulas

Each formula is followed by its source and by the result of the checker's verification.

**F1. LoRA forward pass and merge** [1]

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k)$$

$$W_{\mathrm{merged}} = W_0 + \frac{\alpha}{r} B A$$

Checked against Sec. 4.1. The paper writes h = W_0 x + B A x and states separately that the update is scaled by alpha/r. Trainable parameters per adapted matrix: r(d + k) instead of dk.

**F2. Rank-stabilised scaling** [4]

$$\text{adapter} = \gamma_r B A, \qquad \text{all adapters rank-stabilised} \iff \gamma_r \in \Theta_r\!\left(\frac{1}{\sqrt{r}}\right), \qquad \gamma_r^{\mathrm{rsLoRA}} = \frac{\alpha}{\sqrt{r}}$$

Checked against Definition 3.1, Theorem 3.2 and Eqs. 4-5. The merged matrix is W + gamma_r B A.

**F3. [P] Effective scale under the three conventions in circulation** (our comparison; inputs from [1], [11], [4])

$$\gamma_{\mathrm{classic}} = \frac{\alpha}{r}, \qquad \gamma_{\alpha = 2r} = 2, \qquad \gamma_{\mathrm{rs}} = \frac{\alpha}{\sqrt{r}}, \qquad \gamma_{\mathrm{rs},\,\alpha = 2r} = 2\sqrt{r}$$

With a fixed alpha the classical scale vanishes as 1/r. Both alpha = 2r under classical scaling and rsLoRA with fixed alpha prevent that, but they are different choices (constant versus proportional to 1/sqrt(r)). They coincide at r = 256 when the rsLoRA alpha is 32. Substituting alpha = 2r into the rsLoRA formula gives 8, 16 and 32 at r = 16, 64 and 256 and is untested.

**F4. DoRA** [3]

$$W = m \frac{V}{\lVert V \rVert_c}, \qquad W' = m \frac{W_0 + BA}{\lVert W_0 + BA \rVert_c}$$

Checked against Eqs. 2 and 5. m is a trainable magnitude vector in R^{1 x k} with one entry per column, V in R^{d x k} is the directional matrix, the norm is taken column by column, and m, B and A are trained while W_0 stays frozen.

**F5. PiSSA** [5]

$$W = U S V^{\top}, \qquad A = U_{[:, :r]} S_{[:r, :r]}^{1/2} \in \mathbb{R}^{m \times r}, \qquad B = S_{[:r, :r]}^{1/2} V_{[:, :r]}^{\top} \in \mathbb{R}^{r \times n}$$

$$W^{\mathrm{res}} = U_{[:, r:]} S_{[r:, r:]} V_{[:, r:]}^{\top}, \qquad Y = X\left(W^{\mathrm{res}} + AB\right)$$

$$\text{error}_{\mathrm{QLoRA}} = \lVert W - \mathrm{nf4}(W) \rVert_{*}, \qquad \text{error}_{\mathrm{QPiSSA}} = \lVert W^{\mathrm{res}} - \mathrm{nf4}(W^{\mathrm{res}}) \rVert_{*}$$

Checked against Eqs. 2-3, 6 and 8. PiSSA writes the adapter as AB with A of size m x r. The second line of errors uses the nuclear norm; the reduction is about 20% relative to quantising the base model directly. It holds while the adapter is kept separate in full precision.

**F6. LoRA+** [6]

$$\eta_B = \lambda\, \eta_A, \qquad \lambda \gg 1, \qquad \frac{\eta_B}{\eta_A} = \Theta(n) \text{ for efficient feature learning at width } n$$

Checked against the discussion following Theorem 1 and Sec. 5.3. The paper fixes the ratio and tunes one learning rate. Empirically a ratio of about 2^4 improved RoBERTa, and a ratio of order 2^1 to 2^2 appeared optimal in the LLaMA experiments; the optimum is model and task dependent.

**F7. LoftQ** [9]

$$\min_{Q, A, B} \lVert W - Q - A B^{\top} \rVert_F, \qquad Q_t = q_N\!\left(W - A_{t-1} B_{t-1}^{\top}\right), \qquad (A_t, B_t) = \mathrm{SVD}_r\!\left(W - Q_t\right), \qquad A_0 = B_0 = 0$$

Checked against Eqs. 6-9 and Algorithm 1. T = 1 reproduces the QLoRA backbone with an SVD initialisation of the adapter.

**F8. QLoRA** [2]

$$X^{\mathrm{Int8}} = \mathrm{round}\!\left(\frac{127}{\mathrm{absmax}(X^{\mathrm{FP32}})} X^{\mathrm{FP32}}\right), \qquad q_i = \frac{1}{2}\left(Q_X\!\left(\frac{i}{2^k+1}\right) + Q_X\!\left(\frac{i+1}{2^k+1}\right)\right)$$

$$Y^{\mathrm{BF16}} = X^{\mathrm{BF16}}\, \mathrm{doubleDequant}\!\left(c_1^{\mathrm{FP32}}, c_2^{k\text{-bit}}, W^{\mathrm{NF4}}\right) + X^{\mathrm{BF16}} L_1^{\mathrm{BF16}} L_2^{\mathrm{BF16}}$$

$$\text{overhead: } \frac{32}{64} = 0.5 \;\longrightarrow\; \frac{8}{64} + \frac{32}{64 \cdot 256} = 0.127 \text{ bits per parameter}$$

All four expressions checked against the paper (Eqs. 1, 4 and 5 and the double-quantisation paragraph). Q_X is the quantile function of the standard normal distribution; the paper then builds an asymmetric data type so that zero is represented exactly. Arithmetic re-done: 0.125 + 0.00195 = 0.12695.

**F9. GPTQ** [47, 49]

$$\arg\min_{\widehat{W}} \lVert W X - \widehat{W} X \rVert_2^2, \qquad H_F = 2 X_F X_F^{\top}$$

$$w_q = \arg\min_{w_q} \frac{\left(\mathrm{quant}(w_q) - w_q\right)^2}{[H_F^{-1}]_{qq}}, \qquad \delta_F = -\frac{w_q - \mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}} \left(H_F^{-1}\right)_{:,q}$$

Checked against Eqs. 1 and 2. Independent check: the greedy choice and the update are invariant to a constant factor in H, so the factor 2 has no effect on the result.

**F10. AWQ** [48]

$$Q(w) = \Delta \cdot \mathrm{Round}\!\left(\frac{w}{\Delta}\right), \quad \Delta = \frac{\max(|w|)}{2^{N-1}}, \qquad s^{*} = \arg\min_{s} \left\lVert Q\!\left(W \mathrm{diag}(s)\right)\left(\mathrm{diag}(s)^{-1} X\right) - W X \right\rVert, \qquad s = s_X^{\alpha}$$

Checked against Eqs. 1, 4 and 5. s_X is the per-channel mean activation magnitude and the exponent is found by a search over a small grid. The paper also argues that scaling a salient channel by s > 1 multiplies its error by about (Delta'/Delta)(1/s), and reports that the best perplexity in its example occurs at s = 2 because larger s raises the error of the other channels.

**F11. Importance-matrix objective** [51]

$$F = \Big[\sum_j (q_j - w_j)\, a_j\Big]^2, \qquad \frac{\partial F}{\partial q_i} = 2\, a_i \sum_j a_j (q_j - w_j)$$

$$\mathbb{E}[F] = \sum_{i,j} (q_i - w_i)(q_j - w_j)\, \langle a_i a_j \rangle \;\approx\; \sum_i \langle a_i^2 \rangle\, (q_i - w_i)^2$$

The first line is from the pull request (which omits the factor 2 and contains the typographical slip a_j a_j for a_i a_j). The second line is written out by us. The approximation drops the off-diagonal second moments; the pull request justifies this by the mixed signs of activations.

**F12. k-quant block fit as implemented, Q4_K with an importance matrix** [55]

$$\omega_l = I_l \sqrt{\sigma^2 + x_l^2}, \qquad \sigma^2 = \frac{2}{256} \sum_{l=1}^{256} x_l^2$$

$$\min_{s,\; m \le 0,\; L_l \in \{0, \dots, 15\}} \; \sum_{l=1}^{32} \omega_l \left(s L_l + m - x_l\right)^2$$

$$s = \frac{S_\omega S_{xL} - S_x S_L}{S_\omega S_{LL} - S_L^2}, \qquad m = \frac{S_{LL} S_x - S_L S_{xL}}{S_\omega S_{LL} - S_L^2}, \qquad \text{if } m > 0: \; m = 0,\; s = \frac{S_{xL}}{S_{LL}}$$

with S_omega = sum of omega, S_x = sum of omega x, S_L = sum of omega L, S_LL = sum of omega L^2 and S_xL = sum of omega L x. Read from `quantize_row_q4_K_impl` and `make_qkx3_quants`, and re-derived by solving the two normal equations of weighted linear regression of x on L. Candidate level assignments come from rounding (x - min) times an inverse scale of (15 - 0.9 + 0.05 k)/(max - min) for k = 0, ..., 36; the assignment with the lowest weighted error is kept. The eight block scales and eight offsets are then quantised to 6 bits against two fp16 super-block scales, and the levels are re-assigned with the quantised values. Without an importance matrix a different routine is used:

$$\omega_l^{\text{no imatrix}} = \sqrt{\frac{1}{32} \sum_{l'=1}^{32} x_{l'}^2} + |x_l| \qquad (\texttt{quantize\_row\_q4\_K\_ref})$$

**F13. Bits per weight of the k-quant types** (derived by us from the struct definitions in [56]; the results agree with the comments in the source)

$$\mathrm{bpw}_{Q4\_K} = \frac{256 \cdot 4 + 8\,(6 + 6) + 2 \cdot 16}{256} = \frac{1152}{256} = 4.5, \qquad \mathrm{bpw}_{Q5\_K} = \frac{1280 + 96 + 32}{256} = 5.5$$

$$\mathrm{bpw}_{Q6\_K} = \frac{1536 + 16 \cdot 8 + 16}{256} = 6.5625, \qquad \mathrm{bpw}_{Q3\_K} = \frac{768 + 96 + 16}{256} = 3.4375, \qquad \mathrm{bpw}_{Q2\_K} = \frac{512 + 16 \cdot 8 + 2 \cdot 16}{256} = 2.625$$

**F14. SparseGPT** [83, 49]

$$\arg\min_{M_\ell,\, \widehat{W}_\ell} \left\lVert W_\ell X_\ell - \left(M_\ell \odot \widehat{W}_\ell\right) X_\ell \right\rVert_2^2, \qquad \varepsilon_m = \frac{w_m^2}{[H^{-1}]_{mm}}, \qquad \delta_m = -\frac{w_m}{[H^{-1}]_{mm}} H^{-1}_{:,m}, \qquad H = X X^{\top}$$

Checked against Eqs. 1 and 3. Independent check: minimising delta^T H delta subject to delta_m = -w_m gives the stated delta and the error w_m^2 / [H^{-1}]_mm with no factor one half, because H here is X X^T and the error is the squared norm itself. The original Optimal Brain Surgeon saliency carries a factor one half because it uses the Hessian of an error with a one-half convention; the two are consistent.

**F15. Wanda** [84]

$$S_{ij} = |W_{ij}| \cdot \lVert X_j \rVert_2$$

Checked against Eq. 1. Scores are compared within each output neuron; no weight update follows. The paper (its Eq. 4) shows that the SparseGPT metric reduces to the square of this score when the Hessian is approximated by its diagonal and the dampening is set to zero, which parallels F25.

**F16. LLM-Pruner group importance** [85]

$$I_{W_i} = \left|\Delta \mathcal{L}(\mathcal{D})\right| = \left|\mathcal{L}_{W_i}(\mathcal{D}) - \mathcal{L}_{W_i = 0}(\mathcal{D})\right| = \left| \frac{\partial \mathcal{L}^{\top}(\mathcal{D})}{\partial W_i} W_i - \frac{1}{2} W_i^{\top} H W_i + \mathcal{O}\!\left(\lVert W_i \rVert^3\right) \right|$$

Checked against Eq. 3. Sign re-derived: expanding the loss at W_i = 0 around the current W_i with displacement -W_i gives L(0) = L(W) - g^T W + (1/2) W^T H W, hence the expression above.

**F17. Elastic weight consolidation** [100]

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i \left(\theta_i - \theta^{*}_{A,i}\right)^2$$

Checked against Eq. 3. F is the diagonal of the Fisher information matrix of the earlier task and lambda sets how important the old task is relative to the new one.

**F18. KL-regularised fine-tuning with replay** [13]

$$\min_{\phi} \; \mathbb{E}_{(x,y) \sim d_{\mathrm{current}}} \Big[ \mathcal{L}_{\mathrm{SFT}}(x, y) + \lambda\, D_{KL}\!\big(\pi_{\theta + \phi}(\cdot \mid x) \,\Vert\, \pi_{\theta}(\cdot \mid x)\big) \Big]$$

theta is the frozen base and phi the adapter. Replay data for next-token prediction are interleaved at a ratio rho, where rho = 1 adds one replay example per fine-tuning example. For small parameter changes the KL term is approximately a quadratic form in the Fisher information, which connects it to F17. The direction of the divergence (adapted to base) is as written in the paper.

**F19. RAFT training-set construction** [17]

$$P\%: \; Q + D^{*} + D_1 + \dots + D_k \rightarrow A^{*}, \qquad (1 - P)\%: \; Q + D_1 + \dots + D_k \rightarrow A^{*}$$

Checked against the displayed scheme in the paper. The accompanying prose describes the oracle with k - 1 distractors in the first case and k distractors in the second, so that the number of documents is the same in both; the display and the prose differ by one distractor.

**F20. Token-level prompt pruning** [88]

$$h = f_\theta(x), \qquad p(x_i, \Theta) = \mathrm{softmax}(W h_i + b), \qquad \tilde{N} = \tau N$$

Checked against Eqs. 5-7. The tokens with the highest preserve probability are kept in their original order; tau is the reciprocal of the compression ratio. The training loss (Eq. 7) is the mean token-level cross-entropy.

**F21. Standard errors for evaluation** [104]

$$SE_{\mathrm{CLT}} = \sqrt{\frac{1}{n} \cdot \frac{1}{n-1} \sum_{i=1}^{n} (s_i - \bar{s})^2}, \qquad SE_{A-B,\,\mathrm{paired}} = \sqrt{\frac{1}{n} \cdot \frac{1}{n-1} \sum_{i=1}^{n} \left(s_{A-B,i} - \bar{s}_{A-B}\right)^2}$$

$$SE_{\mathrm{clustered}} = \left( SE_{\mathrm{CLT}}^2 + \frac{1}{n^2} \sum_{c} \sum_{i} \sum_{j \ne i} (s_{i,c} - \bar{s})(s_{j,c} - \bar{s}) \right)^{1/2}$$

$$n = \left(z_{\alpha/2} + z_{\beta}\right)^2 \frac{\omega^2 + \sigma_A^2 / K_A + \sigma_B^2 / K_B}{\delta^2}, \qquad \omega^2 = \mathrm{Var}(x_A) + \mathrm{Var}(x_B) - 2\,\mathrm{Cov}(x_A, x_B)$$

Checked against Eqs. 1, 4, 7 and 9. delta is the minimum detectable effect, K the number of resampled answers per question, and sigma^2 the mean within-question variance.

**F22. [P] Promotion gate** (our proposal; combines [103], [104] and [54])

$$\text{promote} \iff \left(\bar{d}_{\mathrm{dom}} - z_{1-\alpha}\, SE_{\mathrm{dom}} > 0\right) \wedge \left(\bar{d}_{\mathrm{gen}} - z_{1-\alpha}\, SE_{\mathrm{gen}} > -\delta\right) \wedge \left(\frac{1}{T} \sum_{t=1}^{T} D_{KL}\!\left(P^{\mathrm{FP}}_t \,\Vert\, P^{Q}_t\right) \le \kappa\right) \wedge \text{safe-to-serve}$$

d_i is the score of the candidate minus the score of the serving model on held-out item i. SE must be the clustered standard error of F21 with files as clusters, because held-out items from one file are correlated. The first condition is a one-sided test for improvement on the repository; the second is a non-inferiority test on a fixed general set; the third bounds the mean divergence between the merged full-precision model and its quantised version, in the direction that llama.cpp computes; the fourth is the load and resource check. alpha, delta and kappa are design parameters. If the gate is evaluated in R successive rounds at level alpha each, the probability of at least one false promotion is at most R alpha (union bound), so alpha should be divided across rounds or the held-out set refreshed.

**F23. [P] Minimum detectable effect of the gate** (our worked example using F21)

$$\delta_{\min} = \left(z_{1-\alpha} + z_{\beta}\right) \frac{\omega}{\sqrt{n}}; \qquad \alpha = 0.05,\; 1 - \beta = 0.8,\; \omega^2 = 0.25 \;\Rightarrow\; \delta_{\min} \approx \frac{1.24}{\sqrt{n}}$$

For deterministic decoding (sigma_A = sigma_B = 0) and independent items this gives 12.4 points at n = 100, 6.2 at n = 400 and 3.9 at n = 1000. With questions clustered in a dozen held-out files the effective sample size is closer to the number of files than to the number of questions, so the detectable effect on the mimalloc fork is large.

**F24. [P] Latency model and the saving from context pruning** (our model; throughput figures from [52] and [74])

$$T(N_p, N_g) \approx \frac{N_p}{v_{pp}} + \frac{N_g}{v_{tg}}, \qquad \Delta T_{\mathrm{prune}} = \frac{(1 - \rho)\, N_p}{v_{pp}}, \qquad \rho = \frac{N_p'}{N_p}$$

N_p prompt tokens are processed at prefill rate v_pp and N_g tokens generated at decode rate v_tg. The model ignores cache reuse and the growth of attention cost with context. On a dual Xeon Platinum 8488C, Llama-3.1-8B decodes at 2.83 tokens/s in F16 and 5.12 in Q4_K_M, with prefill at 79.57 and 87.70 tokens/s. Rates must be measured on the target machine [81].

**F25. [P] Relation between the importance matrix and the GPTQ Hessian** (our derivation; inputs from [47], [51], [58], [55])

$$H = 2 X X^{\top} \;\Rightarrow\; H_{ii} = 2 \sum_{n=1}^{N} x_{i,n}^2 = 2 N \langle a_i^2 \rangle, \qquad I_i = \langle a_i^2 \rangle = \frac{H_{ii}}{2N}$$

$$(w - q)^{\top} X X^{\top} (w - q) = N \sum_{i,j} (w_i - q_i)(w_j - q_j) \langle a_i a_j \rangle \;\approx\; N \sum_i I_i\, (w_i - q_i)^2$$

The first line is exact given how llama.cpp stores and normalises the statistics. The second line is the diagonal approximation of the row-wise reconstruction error. Two caveats prevent calling k-quants "GPTQ with a diagonal Hessian": the implemented weight is I_i sqrt(sigma^2 + x_i^2) rather than I_i (F12), and k-quants choose a scale, an offset and rounded levels per block without compensating the remaining weights. What does follow is that calibration data influence k-quants only through the per-input-channel second moments.

**F26. [P] Expected distractor load under a relevance threshold** (our derivation; motivates training as in [17] and [18])

$$\mathbb{E}[\#\text{irrelevant}] = \sum_{i=1}^{K} (1 - p_i) \le (1 - \tau) K, \qquad \Pr[\text{no relevant chunk}] = \prod_{i=1}^{K} (1 - p_i) \le (1 - \tau)^K$$

The first statement needs only calibrated probabilities p_i >= tau and linearity of expectation. The second additionally assumes independent relevance events. With tau = 0.25 up to three quarters of admitted chunks may be distractors, and the bound on an oracle-free context is 0.42 for K = 3, 0.24 for K = 5 and 0.10 for K = 8. The second quantity corresponds to RAFT's fraction 1 - P of training examples without the oracle.

**F27. [P] Memory budget for training** (our estimates; parameter counts and shapes from [30] and [31]; consistency check against [62])

$$M_{\text{full FP32, AdamW}} \approx 16\,\text{bytes} \times N_{\theta} + M_{\text{act}}, \qquad N_{\mathrm{LoRA}} = r \sum_{\text{adapted matrices}} (d_{\mathrm{in}} + d_{\mathrm{out}})$$

Sixteen bytes per parameter covers the weight, its gradient and two moment estimates in single precision. For a 1.24B-parameter model this is 19.8 GB before activations, consistent with the 24 GB that the llama.cpp documentation reports for LLaMA 3.2 1B. It gives 7.9 GB for Qwen2.5-Coder-0.5B and 24.7 GB for Qwen2.5-Coder-1.5B, before activations. Adapting all seven linear maps per layer gives 549,888 r trainable parameters for the 0.5B model and 1,154,048 r for the 1.5B model: 8.8M and 18.5M at r = 16, 35.2M and 73.9M at r = 64, 140.8M and 295.4M at r = 256.

**F28. Post-training-quantisation degradation scaling law** [105]

$$L(N, D, P_{\text{train}}, P_{\text{post}}) = A N_{\text{eff}}^{-\alpha} + B D^{-\beta} + E + \delta_{\text{PTQ}}(N_{\text{eff}}, D, P_{\text{train}}, P_{\text{post}}), \qquad \delta_{\text{PTQ}}(N, D, P_{\text{post}}) = C_T \left(\frac{D^{\gamma_D}}{N^{\gamma_N}}\right) e^{-P_{\text{post}}/\gamma_{\text{post}}}$$

Checked against Eqs. 1-2 of the paper; the second form is stated for models trained in BF16 or above. N is the parameter count, D the number of pretraining tokens, P_post the post-training precision in bits, and C_T, gamma_D, gamma_N, gamma_post are positive fitted constants. The paper reports gamma_D and gamma_N to be similar, so delta_PTQ is approximately a power law in D/N. Worked ratio (ours): 5.5e12 / 4.9e8 is about 1.1e4 tokens per parameter for Qwen2.5-Coder-0.5B and 5.5e12 / 1.54e9 about 3.6e3 for the 1.5B model, counting only the code-pretraining tokens reported in the Qwen2.5-Coder report. These ratios are far outside the fitted range, so the fitted constants must not be extrapolated to them.

**F29. Quantisation-induced degradation (QiD) law** [106]

$$\Delta_q \mathrm{Loss} = \mathrm{Loss}_q - \mathrm{Loss}_{16\text{-bit}}, \qquad \Delta_q \mathrm{Loss}(N, D, P) = k \cdot \frac{D^{\beta}}{N^{\alpha} P^{\gamma}}$$

Checked against Eq. 4 and the unified fit in the paper, which reports k = 0.017, beta = 0.5251, alpha = 0.2261 and gamma = 5.4967 for GPTQ on Pythia (P is the bit width, D the training tokens, N the parameters). Unlike F28 the precision dependence is a power law rather than an exponential; both agree that degradation rises with D and falls with N. Both laws are empirical fits, not derivations.

## What is novel in our proposal

Everything in this section is a proposal of this project. None of it is an established result, and each item needs its own experiment.

1. **[P] Threshold-matched distractor training.** The generator is fine-tuned on contexts produced by the system's own admission rule (relevance probability at least 0.25), with the oracle chunk present in a fraction P of examples. F26 supplies the link between the threshold and the distractor load, and suggests setting 1 - P near the measured frequency of oracle-free contexts rather than copying RAFT's values. RAFT and the retrieval-robustness work fix the distractor count or sample by rank; they do not tie the training distribution to a probabilistic admission rule.
2. **[P] A four-part promotion gate with clustered paired statistics** (F22, F23). TFX supplies the "threshold and baseline" rule and Miller the statistics; combining a one-sided domain test, a non-inferiority test on general ability, a KL fidelity bound on the quantisation step and a load test, with files as clusters and an error budget across rounds, is our construction.
3. **[P] Stage-specific metrics.** Task metrics gate the fine-tuning step and KL divergence gates the quantisation step, because perplexity is documented to move the wrong way under fine-tuning while divergence is the documented fidelity measure for quantisation.
4. **[P] Domain-calibrated importance matrix in the prompt format actually served.** Calibration text is repository code wrapped in the system's own RAG prompt template. The literature motivates this (instruct-format suggestion by the importance-matrix author, matched-distribution result in AWQ, code-specific calibration for code models) but none of the sources opened reports a measurement for k-quants on sub-2B code models at 4 to 6 bits.
5. **[P] The interpretation in F25** that calibration reaches k-quants only through per-channel second moments, with the two stated caveats.
6. **[P] Context-level pruning in place of weight pruning**, realised as the relevance threshold followed by function-body pruning that keeps signatures, and evaluated with the latency model F24.
7. **[P] Idle-time scheduling on a desktop operating system** with immediate abort and checkpointing, transplanting the federated-learning eligibility rule to a single Windows machine.

## Risks and failure modes

1. **The corpus is too small to show an effect.** About 59 files yield few independent held-out clusters. By F23 the gate may be unable to detect gains below ten points, and any reported improvement must carry a confidence interval. Training on so little text also invites memorisation rather than generalisation.
2. **Forgetting accumulates across rounds.** Intruder dimensions accumulate under sequential LoRA fine-tuning [12], and naive LoRA can degrade general ability on small data sets [13]. Continuing from the previous adapter is riskier than restarting from the base model on a cumulative data set.
3. **Self-training bias.** If the local model writes its own training questions and answers, errors can reinforce themselves [27]. Accumulating data and anchoring on real code and on user-accepted answers are the published mitigations [28, 24].
4. **4-bit quantisation may not be safe for this model size.** See the contested results. Qwen2.5-Coder-0.5B continued pretraining on over 5.5T code tokens from a Qwen2.5 initialisation, a token-to-parameter ratio above 10,000 (F28), which is the regime that precision scaling laws flag as quantisation-sensitive [105, 106]. The quantisation gate has to be measured, not assumed.
5. **Order of operations is an open question.** Whether to train on full precision, merge and quantise, or to train on the quantised model, is listed as open research by the llama.cpp maintainer [64]. Quantising after merging may perturb or round away part of a small-norm adapter update (our inference, not a measured result). QA-LoRA was designed so that adapter and quantised weights combine without a further quantisation loss [10], and LoftQ addresses the mismatch between a quantised backbone and the adapter initialisation [9].
6. **Training infrastructure.** Upstream training is FP32, full-parameter and work in progress; LoRA training must be written in-house or taken from an unmerged fork. Full fine-tuning of the 1.5B model needs about 25 GB before activations (F27), which does not fit comfortably in 32 GB alongside a desktop session. The 4 GB GPU cannot hold full fine-tuning state for either model; whether LoRA on the 0.5B model with a half-precision or quantised frozen base fits in 4 GB was not measured by any opened source.
7. **Wall-clock cost is unknown.** No opened source reports training throughput for 0.5B to 1.5B models on a laptop-class hybrid x86 processor. Published figures are 8 to 39 hours per task on a phone for a 0.5B model [98] and minutes per epoch on an RTX 4090 [67]. Idle windows may be too short to complete a useful update.
8. **Idle detection.** The documented semantics are for Windows 7 and 8. In Windows 7 a processor running only below-normal-priority threads still counts as idle; whether the same holds on Windows 11 was not verified [97].
9. **Uncalibrated relevance probabilities.** F26 and the 25% threshold mean nothing unless the retrieval scores are calibrated probabilities. That calibration is outside this note.
10. **Repeated gating inflates false promotions** unless the error budget is split across rounds (F22). Reusing the same held-out set for many rounds also leaks information into model selection.
11. **Supply chain and privacy.** Quantised files from third parties can hide behaviour that is absent in full precision [79]. Fine-tuned small models can reproduce training records, and the quantisation method affects how much [80]. A model fine-tuned on a private repository should be treated as containing that repository.
12. **Hybrid architectures.** LoRA on the recurrent layers of Qwen3.5-0.8B was destructive in the one study available [16].
13. **Context pruning can break code.** Token-level compressors trained on prose may remove syntax. The code-specific methods prune at function or line granularity for this reason [91, 90].

## Design recommendations

1. **Base model.** Use Qwen2.5-Coder-1.5B-Instruct for serving and Qwen2.5-Coder-0.5B-Instruct for CPU fine-tuning experiments. Both are Apache-2.0, standard decoder transformers with tied embeddings and repository-level pretraining. Do not adopt Qwen3.5 for fine-tuning without restricting adapters to its attention layers, and budget memory for Qwen3 from the stored parameter counts.
2. **Adapter method.** Train LoRA only, with the base frozen. Adapt all linear maps in every layer. Choose one scaling convention and report it: either classical alpha/r with alpha = 2r (effective scale 2), or rsLoRA alpha/sqrt(r) with a rank-independent alpha. Do not combine them. Use a higher learning rate for B than for A, starting from a ratio of 2 to 4 for decoder models and sweeping to 16. Sweep the base learning rate between 1e-5 and 5e-4 and keep the highest stable value.
3. **Rank.** Start at r = 64 for both models (35.2M and 73.9M trainable parameters, F27). The published code-domain recommendation of r = 256 was obtained at hidden width 4096, a ratio of about 6%; the proportional equivalent is r of about 56 at width 896 and about 96 at width 1536 (our scaling argument, not a published result). Run r = 16 as the low-cost ablation and r = 128 on the 1.5B model as the high one. At r = 256 the adapter would be 28% and 19% of the two models and is no longer parameter-efficient.
4. **Training data.** Generate one question per indexed function with the oracle chunk, distractors drawn from the system's own admission rule, and an answer that quotes code verbatim. Add fill-in-the-middle examples in the model's native token format and user-accepted answers and applied edits. Keep every real example permanently and add synthetic ones to it; never replace the data set with generated text. Split training and held-out data by file.
5. **Forgetting safeguards.** Add the KL penalty of F18 with a coefficient swept over 0.001 and 0.01, and interleave replay of generic code at one replay example per training example (the economical setting in the source; three per example performed best). If no replay corpus can be shipped, use the target-masked KL penalty. Restart each round from the base model on the cumulative data set.
6. **Scheduling.** Train only when the interface toggle is on, the user has been absent for at least four minutes, processor and disk use are low and the machine is on mains power. Run at below-normal priority, checkpoint every N steps, and abort as soon as any condition fails. Verify the idle behaviour on Windows 11 before relying on Task Scheduler semantics.
7. **Quantisation.** Merge the adapter into full-precision weights with `llama-export-lora`, compute the importance matrix with `llama-imatrix` on roughly 50,000 tokens of repository code inside the real prompt template with `--parse-special`, keep the default of not applying it to `output.weight`, and quantise with `llama-quantize`. Serve Q6_K or Q5_K_M (about 1.27 GB and 1.10 GB for the 1.5B model); treat Q4_K_M (about 0.94 GB) as an experiment that must pass the gate. Publish the calibration file with the paper.
8. **Model provenance.** Download base weights from the publisher's repository and quantise locally. Do not distribute adapters or merged models trained on private code.
9. **Pruning.** Do not prune the weights of the generator. Prune the context: admit chunks with relevance probability of at least 0.25, then keep signatures and drop the bodies of functions that are dependencies rather than targets. Report prompt-token reduction and end-to-end latency together, with throughput measured on the target machine.
10. **Evaluation gate.** Implement F22 with file-clustered paired standard errors. Compute the minimum detectable effect before training and state it in the paper. Use task metrics for the fine-tuning step and mean KL divergence plus top-token agreement for the quantisation step. Do not use Wikitext perplexity to judge fine-tuning. Divide alpha across rounds.
11. **Hot swap.** Load the candidate beside the serving model, run the gate, then swap a reference-counted handle so that in-flight requests finish on the old model. Keep the previous GGUF for one-step rollback. When serving through the llama.cpp server, adapters can be switched at run time through `/lora-adapters`.
12. **Paper hygiene.** Present the 25% threshold, the gate margins and the 16 KB work-unit size as design parameters of this system. Label F3 and F22 to F27 as the authors' derivations. State that experiments run on a fork of mimalloc and that upstream llama.cpp does not train LoRA adapters. Report the order-of-operations comparison (train on full precision then quantise, versus train on the quantised model) as an experiment, since the literature leaves it open.

## References

[1] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR 2022); arXiv:2106.09685. 2022. <https://arxiv.org/abs/2106.09685>. Key: `hu2022lora`.

[2] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Main Conference Track. 2023. <https://arxiv.org/abs/2305.14314>. Key: `dettmers2023qlora`.

[3] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation*. International Conference on Machine Learning (ICML 2024), Oral; arXiv:2402.09353. 2024. <https://arxiv.org/abs/2402.09353>. Key: `liu2024dora`.

[4] Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. arXiv preprint arXiv:2312.03732. 2023. <https://arxiv.org/abs/2312.03732>. Key: `kalajdzievski2023rslora`.

[5] Fanxu Meng, Zhaohui Wang, Muhan Zhang. *PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models*. Advances in Neural Information Processing Systems 37 (NeurIPS 2024), spotlight; arXiv:2404.02948. 2024. <https://arxiv.org/abs/2404.02948>. Key: `meng2024pissa`.

[6] Soufiane Hayou, Nikhil Ghosh, Bin Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models*. Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235:17783-17806. 2024. <https://arxiv.org/abs/2402.12354>. Key: `hayou2024loraplus`.

[7] Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad. *RandLoRA: Full-rank parameter-efficient fine-tuning of large models*. International Conference on Learning Representations (ICLR 2025); arXiv:2502.00987. 2025. <https://arxiv.org/abs/2502.00987>. Key: `albert2025randlora`.

[8] Huiyi Wang, Daijiao Liu, Lina Yao, Dong Gong. *One Rate Is Not Enough: Adaptive Anisotropic Learning Rates for LoRA Fine-Tuning*. arXiv preprint arXiv:2609.05885. 2026. <https://arxiv.org/abs/2609.05885>. Key: `wang2026anlrlora`.

[9] Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao. *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models*. International Conference on Learning Representations (ICLR 2024); arXiv:2310.08659 (submitted 2023). 2024. <https://arxiv.org/abs/2310.08659>. Key: `li2024loftq`.

[10] Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, Qi Tian. *QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models*. International Conference on Learning Representations (ICLR 2024); arXiv:2309.14717 (submitted 2023). 2024. <https://arxiv.org/abs/2309.14717>. Key: `xu2024qalora`. Added by the checker.

[11] Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham. *LoRA Learns Less and Forgets Less*. Transactions on Machine Learning Research (TMLR), August 2024, Featured Certification. 2024. <https://arxiv.org/abs/2405.09673>. Key: `biderman2024lora`.

[12] Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma. *LoRA vs Full Fine-tuning: An Illusion of Equivalence*. 39th Conference on Neural Information Processing Systems (NeurIPS 2025); arXiv:2410.21228 (submitted 2024). 2025. <https://arxiv.org/abs/2410.21228>. Key: `shuttleworth2025illusion`.

[13] Matthew Riemer, Erik Miehling, Miao Liu, Djallel Bouneffouf, Murray Campbell. *The Effectiveness of Approximate Regularized Replay for Efficient Supervised Fine-Tuning of Large Language Models*. arXiv preprint arXiv:2512.22337. 2025. <https://arxiv.org/abs/2512.22337>. Key: `riemer2025replay`.

[14] Runze Xu, Arpit Garg, Hemanth Saratchandran, Simon Lucey. *Mask the Target: A Plug-and-Play Regularizer Against LoRA Forgetting*. arXiv preprint arXiv:2605.29498 (in submission). 2026. <https://arxiv.org/abs/2605.29498>. Key: `xu2026tmkl`. Added by the checker.

[15] Rahul Nair, Chun Tao. *The Fine-Tuning Trap: Evaluating Negative Transfer and the Role of PEFT in Sub-1B Mathematical Reasoning*. arXiv preprint arXiv:2606.06920. 2026. <https://arxiv.org/abs/2606.06920>. Key: `nair2026trap`. Added by the checker.

[16] Hector Borobia, Elies Seguí-Mas, Guillermina Tormo-Carbó. *Where Should LoRA Go? Component-Type Placement in Hybrid Language Models*. arXiv preprint arXiv:2604.22127. 2026. <https://arxiv.org/abs/2604.22127>. Key: `borobia2026placement`. Added by the checker.

[17] Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez. *RAFT: Adapting Language Model to Domain Specific RAG*. arXiv preprint arXiv:2403.10131. 2024. <https://arxiv.org/abs/2403.10131>. Key: `zhang2024raft`.

[18] Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant. *Making Retrieval-Augmented Language Models Robust to Irrelevant Context*. International Conference on Learning Representations (ICLR 2024); arXiv:2310.01558 (submitted 2023). 2024. <https://arxiv.org/abs/2310.01558>. Key: `yoran2024robust`.

[19] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, Scott Yih. *RA-DIT: Retrieval-Augmented Dual Instruction Tuning*. International Conference on Learning Representations (ICLR 2024); arXiv:2310.01352 (submitted 2023). 2024. <https://arxiv.org/abs/2310.01352>. Key: `lin2024radit`.

[20] Zhan Peng Lee, Andre Lin, Calvin Tan. *Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation*. arXiv preprint arXiv:2505.10792. 2025. <https://arxiv.org/abs/2505.10792>. Key: `lee2025finetunerag`.

[21] Tianci Liu, Haoxiang Jiang, Tianze Wang, Ran Xu, Yue Yu, Linjun Zhang, Tuo Zhao, Haoyu Wang. *RoseRAG: Robust Retrieval-augmented Generation with Small-scale LLMs via Margin-aware Preference Optimization*. arXiv preprint arXiv:2502.10993. 2025. <https://arxiv.org/abs/2502.10993>. Key: `liu2025roserag`.

[22] Chaozheng Wang, Zezhou Yang, Shuzheng Gao, Cuiyun Gao, Ting Peng, Hailiang Huang, Yuetang Deng, Michael Lyu. *RAG or Fine-tuning? A Comparative Study on LCMs-based Code Completion in Industry*. Companion Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering (FSE Companion '25), Industry Track, Trondheim, Norway. 2025. <https://arxiv.org/abs/2505.15179>. Key: `wang2025ragorft`.

[23] Martin Weyssow, Xin Zhou, Kisub Kim, David Lo, Houari Sahraoui. *Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models*. ACM Transactions on Software Engineering and Methodology 34(7), 2025, doi:10.1145/3714461; arXiv:2308.10462 (submitted 2023). 2025. <https://arxiv.org/abs/2308.10462>. Key: `weyssow2025peft`.

[24] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang. *Magicoder: Empowering Code Generation with OSS-Instruct*. Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR 235. 2024. <https://arxiv.org/abs/2312.02120>. Key: `wei2024magicoder`.

[25] Egor Bogomolov, Sergey Zhuravlev, Egor Spirin, Timofey Bryksin. *Assessing Project-Level Fine-Tuning of ML4SE Models*. arXiv preprint arXiv:2206.03333. 2022. <https://arxiv.org/abs/2206.03333>. Key: `bogomolov2022project`. Added by the checker.

[26] Peiyang Wu, Nan Guo, Junliang Lv, Xiao Xiao, Xiaochun Ye. *RTLRepoCoder: Repository-Level RTL Code Completion through the Combination of Fine-Tuning and Retrieval Augmentation*. arXiv preprint arXiv:2504.08862. 2025. <https://arxiv.org/abs/2504.08862>. Key: `wu2025rtlrepocoder`. Added by the checker.

[27] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson. *The Curse of Recursion: Training on Generated Data Makes Models Forget*. arXiv preprint arXiv:2305.17493. 2023. <https://arxiv.org/abs/2305.17493>. Key: `shumailov2023curse`. Added by the checker.

[28] Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, Sanmi Koyejo. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. arXiv preprint arXiv:2404.01413. 2024. <https://arxiv.org/abs/2404.01413>. Key: `gerstgrasser2024collapse`. Added by the checker.

[29] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, Junyang Lin. *Qwen2.5-Coder Technical Report*. arXiv preprint arXiv:2409.12186. 2024. <https://arxiv.org/abs/2409.12186>. Key: `hui2024qwen25coder`.

[30] Qwen Team, Alibaba Cloud. *Qwen2.5-Coder-0.5B-Instruct (model card and config.json)*. Hugging Face model repository (created 2024-11-06; retrieved 2026-09-29). 2024. <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct>. Key: `qwen2024coder05card`.

[31] Qwen Team, Alibaba Cloud. *Qwen2.5-Coder-1.5B-Instruct (model card and config.json)*. Hugging Face model repository (created 2024-09-18; retrieved 2026-09-29). 2024. <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>. Key: `qwen2024coder15card`.

[32] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, et al. (60 authors listed on arXiv). *Qwen3 Technical Report*. arXiv preprint arXiv:2505.09388. 2025. <https://arxiv.org/abs/2505.09388>. Key: `yang2025qwen3`.

[33] Qwen Team, Alibaba Cloud. *Qwen3-0.6B (model card)*. Hugging Face model repository (created 2025-04-27; retrieved 2026-09-29). 2025. <https://huggingface.co/Qwen/Qwen3-0.6B>. Key: `qwen2025qwen3card06`.

[34] Qwen Team, Alibaba Cloud. *Qwen3-1.7B (model card)*. Hugging Face model repository (created 2025-04-27; retrieved 2026-09-29). 2025. <https://huggingface.co/Qwen/Qwen3-1.7B>. Key: `qwen2025qwen3card17`.

[35] Qwen Team, Alibaba Cloud. *Qwen3.5-0.8B (model card)*. Hugging Face model repository (created 2026-02-28; retrieved 2026-09-29). 2026. <https://huggingface.co/Qwen/Qwen3.5-0.8B>. Key: `qwen2026qwen35card08`.

[36] Qwen Team, Alibaba Cloud. *Qwen3.5-2B (model card)*. Hugging Face model repository (created 2026-02-28; retrieved 2026-09-29). 2026. <https://huggingface.co/Qwen/Qwen3.5-2B>. Key: `qwen2026qwen35card2`.

[37] Olivier Lacombe, Kathleen Kenealy, Kat Black, Ravin Kumar, Francesco Visin, Jiageng Zhang. *Introducing Gemma 3 270M: The compact model for hyper-efficient AI*. Google Developers Blog, 14 August 2025. 2025. <https://developers.googleblog.com/en/introducing-gemma-3-270m/>. Key: `lacombe2025gemma270m`.

[38] Google (Gemma Team). *Gemma 3 270M (model card)*. Hugging Face model repository, gated (created 2025-08-05; retrieved 2026-09-29). 2025. <https://huggingface.co/google/gemma-3-270m>. Key: `google2025gemma3card270m`.

[39] Google (Gemma Team). *Gemma 3 1B instruction-tuned (model card)*. Hugging Face model repository, gated (created 2025-03-10; retrieved 2026-09-29). 2025. <https://huggingface.co/google/gemma-3-1b-it>. Key: `google2025gemma3card1b`.

[40] Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Cărbune, Michelle Casbon, Mayank Chaturvedi, et al. (323 authors listed on arXiv). *Gemma 4 Technical Report*. arXiv preprint arXiv:2607.02770. 2026. <https://arxiv.org/abs/2607.02770>. Key: `gemma2026gemma4`.

[41] Google (Gemma Team). *Gemma 4 E2B instruction-tuned (model card)*. Hugging Face model repository (created 2026-03-02; retrieved 2026-09-29). 2026. <https://huggingface.co/google/gemma-4-E2B-it>. Key: `google2026gemma4e2bcard`.

[42] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf. *SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*. arXiv preprint arXiv:2502.02737. 2025. <https://arxiv.org/abs/2502.02737>. Key: `allal2025smollm2`.

[43] Hugging Face TB Research. *SmolLM3-3B (model card)*. Hugging Face model repository (created 2025-07-08; retrieved 2026-09-29). 2025. <https://huggingface.co/HuggingFaceTB/SmolLM3-3B>. Key: `hf2025smollm3card`.

[44] Hugging Face TB Research. *SmolLM2-360M-Instruct (model card)*. Hugging Face model repository (created 2024-10-31; retrieved 2026-09-29). 2024. <https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct>. Key: `hf2024smollm2card`.

[45] Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, et al. (74 authors listed on arXiv). *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs*. arXiv preprint arXiv:2503.01743. 2025. <https://arxiv.org/abs/2503.01743>. Key: `abouelenin2025phi4mini`.

[46] Microsoft. *Phi-4-mini-instruct (model card)*. Hugging Face model repository (created 2025-02-19; retrieved 2026-09-29). 2025. <https://huggingface.co/microsoft/Phi-4-mini-instruct>. Key: `microsoft2025phi4minicard`.

[47] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. International Conference on Learning Representations (ICLR 2023); arXiv:2210.17323 (submitted 2022). 2023. <https://arxiv.org/abs/2210.17323>. Key: `frantar2023gptq`.

[48] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. Proceedings of the 7th MLSys Conference (MLSys 2024), Best Paper Award; arXiv:2306.00978 (submitted 2023). 2024. <https://arxiv.org/abs/2306.00978>. Key: `lin2024awq`.

[49] Babak Hassibi, David G. Stork. *Second order derivatives for network pruning: Optimal Brain Surgeon*. Advances in Neural Information Processing Systems 5 (NIPS 1992). 1992. <https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>. Key: `hassibi1992obs`.

[50] Iwan Kawrakow (GitHub user ikawrakow). *k-quants (llama.cpp pull request #1684)*. GitHub pull request, ggml-org/llama.cpp, merged 2023-06-05. 2023. <https://github.com/ggml-org/llama.cpp/pull/1684>. Key: `kawrakow2023kquants`.

[51] Iwan Kawrakow (GitHub user ikawrakow). *Importance Matrix calculation (llama.cpp pull request #4861)*. GitHub pull request, ggml-org/llama.cpp, merged 2024-01-12. 2024. <https://github.com/ggml-org/llama.cpp/pull/4861>. Key: `kawrakow2024imatrix`.

[52] ggml-org/llama.cpp contributors. *llama.cpp tools/quantize/README.md*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md>. Key: `llamacpp2026quantize`.

[53] ggml-org/llama.cpp contributors. *llama.cpp tools/imatrix/README.md*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md>. Key: `llamacpp2026imatrix`.

[54] ggml-org/llama.cpp contributors. *llama.cpp tools/perplexity/README.md (LLaMA 3 8b scoreboard)*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md>. Key: `llamacpp2026perplexity`.

[55] ggml-org/llama.cpp contributors. *llama.cpp source file ggml/src/ggml-quants.c*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c>. Key: `llamacpp2026ggmlquants`.

[56] ggml-org/llama.cpp contributors. *llama.cpp source file ggml/src/ggml-common.h*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h>. Key: `llamacpp2026ggmlcommon`.

[57] ggml-org/llama.cpp contributors. *llama.cpp source file src/llama-quant.cpp*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp>. Key: `llamacpp2026llamaquant`. Added by the checker.

[58] ggml-org/llama.cpp contributors. *llama.cpp source file tools/quantize/quantize.cpp*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp>. Key: `llamacpp2026quantizecpp`. Added by the checker.

[59] ggml-org/llama.cpp contributors. *llama.cpp source file tools/imatrix/imatrix.cpp*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp>. Key: `llamacpp2026imatrixcpp`. Added by the checker.

[60] ggml-org/llama.cpp contributors. *llama.cpp source file tools/perplexity/perplexity.cpp*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp>. Key: `llamacpp2026perplexitycpp`. Added by the checker.

[61] ggml-org/llama.cpp contributors. *llama.cpp public header include/llama.h*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h>. Key: `llamacpp2026llamah`. Added by the checker.

[62] ggml-org/llama.cpp contributors. *llama.cpp examples/training/README.md*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29; last commit touching the directory 2026-09-02). 2026. <https://github.com/ggml-org/llama.cpp/tree/master/examples/training>. Key: `llamacpp2026training`.

[63] ggml-org/llama.cpp contributors. *llama.cpp source file examples/training/finetune.cpp*. Official repository source code (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/examples/training/finetune.cpp>. Key: `llamacpp2026finetunecpp`.

[64] Johannes Gaessler (GitHub user JohannesGaessler). *LoRA training example (llama.cpp issue #13485)*. GitHub issue, ggml-org/llama.cpp, opened 2025-05-12, still open on 2026-09-29. 2025. <https://github.com/ggml-org/llama.cpp/issues/13485>. Key: `gaessler2025loraissue`.

[65] ggml-org/llama.cpp contributors. *llama.cpp tools/server/README.md*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>. Key: `llamacpp2026server`.

[66] ggml-org/llama.cpp contributors. *llama.cpp tools/export-lora/README.md*. Official repository documentation (master at commit ba0ba54d93, retrieved 2026-09-29). 2026. <https://github.com/ggml-org/llama.cpp/blob/master/tools/export-lora/README.md>. Key: `llamacpp2026exportlora`.

[67] Subash SN, Akshay Nambiar, Patrik Lambert, Milan Gritta, Gianfranco Cordella, Amril Nurman. *An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs*. Hugging Face blog (QVAC, Tether), 1 December 2025. 2025. <https://huggingface.co/blog/qvac/fabric-llm-finetune>. Key: `sn2025qvacfabric`.

[68] Miles Williams, Nikolaos Aletras. *On the Impact of Calibration Data in Post-training Quantization and Pruning*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), Volume 1: Long Papers; arXiv:2311.09755 (submitted 2023). 2024. <https://arxiv.org/abs/2311.09755>. Key: `williams2024calibration`.

[69] Alessandro Giagnorio, Antonio Mastropaolo, Saima Afrin, Massimiliano Di Penta, Gabriele Bavota. *Evaluating the Impact of Post-Training Quantization on Large Language Models for Code Generation*. arXiv preprint arXiv:2503.07103. 2025. <https://arxiv.org/abs/2503.07103>. Key: `giagnorio2025quantcode`.

[70] Yixin Ji, Yang Xiang, Juntao Li, Qingrong Xia, Ping Li, Xinyu Duan, Zhefeng Wang, Min Zhang. *Beware of Calibration Data for Pruning Large Language Models*. International Conference on Learning Representations (ICLR 2025); arXiv:2410.17711 (submitted 2024). 2025. <https://arxiv.org/abs/2410.17711>. Key: `ji2025calibration`.

[71] Miles Williams, George Chrysostomou, Nikolaos Aletras. *Self-calibration for Language Model Quantization and Pruning*. Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2025), Volume 1: Long Papers; arXiv:2410.17170 (submitted 2024). 2025. <https://arxiv.org/abs/2410.17170>. Key: `williams2025selfcalibration`.

[72] Renren Jin, Jiangcun Du, Wuwei Huang, Wei Liu, Jian Luan, Bin Wang, Deyi Xiong. *A Comprehensive Evaluation of Quantization Strategies for Large Language Models*. Findings of the Association for Computational Linguistics: ACL 2024. 2024. <https://arxiv.org/abs/2402.16775>. Key: `jin2024quanteval`.

[73] Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh. *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization*. Annual Meeting of the Association for Computational Linguistics (ACL 2025); arXiv:2411.02355 (submitted 2024). 2025. <https://arxiv.org/abs/2411.02355>. Key: `kurtic2025bf16`.

[74] Uygar Kurt. *Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct*. arXiv preprint arXiv:2601.14277. 2026. <https://arxiv.org/abs/2601.14277>. Key: `kurt2026whichquant`.

[75] Shahzeb Qamar, Lorenz Sparrenberg, Christian Bauckhage, Baha Rababah, Carson Leung, Murat Kantarcioglu, Cuneyt Gurcan Akcora, Rafet Sifa. *Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs*. arXiv preprint arXiv:2609.07664. 2026. <https://arxiv.org/abs/2609.07664>. Key: `qamar2026divergence`.

[76] Douglas Orr, Luka Ribar, Carlo Luschi. *Optimal Formats for Weight Quantisation*. arXiv preprint arXiv:2505.12988. 2025. <https://arxiv.org/abs/2505.12988>. Key: `orr2025formats`.

[77] Jiacheng Wang, Yejun Zeng, Jinyang Guo, Yuqing Ma, Aishan Liu, Xianglong Liu. *SLMQuant:Benchmarking Small Language Model Quantization for Practical Deployment*. RichMediaGAI '25 workshop, 27-28 October 2025, Dublin, Ireland (per PDF header); arXiv:2511.13023. 2025. <https://arxiv.org/abs/2511.13023>. Key: `wang2025slmquant`. Added by the checker.

[78] Kazuki Kusama, Honglin Shu, Masanari Kondo, Yasutaka Kamei. *How Small is Enough? Empirical Evidence of Quantized Small Language Models for Automated Program Repair*. 2025 ACM/IEEE International Symposium on Empirical Software Engineering and Measurement (ESEM), pp. 393-399. 2025. <https://arxiv.org/abs/2508.16499>. Key: `kusama2025howsmall`. Added by the checker.

[79] Kazuki Egashira, Robin Staab, Mark Vero, Jingxuan He, Martin Vechev. *Mind the Gap: A Practical Attack on GGUF Quantization*. International Conference on Machine Learning (ICML 2025); arXiv:2505.23786. 2025. <https://arxiv.org/abs/2505.23786>. Key: `egashira2025mindthegap`. Added by the checker.

[80] Cristhian Kapelinski, Diego Kreutz. *Not All 4-bit Quantizers Are Equal: Deployment-Time Mitigation of PII Leakage in Fine-Tuned Small Language Models*. Accepted at SBSeg 2026 (per arXiv comment); arXiv:2609.25014. 2026. <https://arxiv.org/abs/2609.25014>. Key: `kapelinski2026pii`. Added by the checker.

[81] Xinyu Qiu, Chuhong Xu, Bo Su, Ziyao Chen, Ruiyang Xu, Shimeng Dai. *GGUF-Metadata Prediction of Single-Sequence llama.cpp Throughput Across Three Systems*. arXiv preprint arXiv:2609.14864 (submitted to ICASSP 2027 per arXiv comment). 2026. <https://arxiv.org/abs/2609.14864>. Key: `qiu2026throughput`. Added by the checker.

[82] Andrei Panferov, Maximilian Kleinegger, Sweta Priyadarshi, Tijmen Blankevoort, Dan Alistarh. *Disaggregated Quantization: Specializing LLM Prefill and Decode*. arXiv preprint arXiv:2609.26333. 2026. <https://arxiv.org/abs/2609.26333>. Key: `panferov2026disaggregated`. Added by the checker.

[83] Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. Proceedings of the 40th International Conference on Machine Learning (ICML 2023), PMLR 202:10323-10337. 2023. <https://arxiv.org/abs/2301.00774>. Key: `frantar2023sparsegpt`.

[84] Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models*. International Conference on Learning Representations (ICLR 2024); arXiv:2306.11695 (submitted 2023). 2024. <https://arxiv.org/abs/2306.11695>. Key: `sun2024wanda`.

[85] Xinyin Ma, Gongfan Fang, Xinchao Wang. *LLM-Pruner: On the Structural Pruning of Large Language Models*. Advances in Neural Information Processing Systems 36 (NeurIPS 2023). 2023. <https://arxiv.org/abs/2305.11627>. Key: `ma2023llmpruner`.

[86] Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang. *Compressing LLMs: The Truth is Rarely Pure and Never Simple*. International Conference on Learning Representations (ICLR 2024); arXiv:2310.01382 (submitted 2023). 2024. <https://arxiv.org/abs/2310.01382>. Key: `jaiswal2024compressing`.

[87] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023). 2023. <https://arxiv.org/abs/2310.05736>. Key: `jiang2023llmlingua`.

[88] Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*. Findings of the Association for Computational Linguistics: ACL 2024. 2024. <https://arxiv.org/abs/2403.12968>. Key: `pan2024llmlingua2`.

[89] Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant. *Provence: efficient and robust context pruning for retrieval-augmented generation*. International Conference on Learning Representations (ICLR 2025); arXiv:2501.16214. 2025. <https://arxiv.org/abs/2501.16214>. Key: `chirkova2025provence`.

[90] Lei Zhang, Yunshui Li, Jiaming Li, Xiaobo Xia, Jiaxi Yang, Run Luo, Minzheng Wang, Longze Chen, Junhao Liu, Min Yang. *Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs*. arXiv preprint arXiv:2406.18294. 2024. <https://arxiv.org/abs/2406.18294>. Key: `zhang2024hcp`. Added by the checker.

[91] Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, Xiaodong Gu. *SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents*. arXiv preprint arXiv:2601.16746. 2026. <https://arxiv.org/abs/2601.16746>. Key: `wang2026swepruner`. Added by the checker.

[92] Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo. *Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning*. COLM 2026 Workshop on Efficient Reasoning (per arXiv comment); arXiv:2605.15315. 2026. <https://arxiv.org/abs/2605.15315>. Key: `wang2026lamr`. Added by the checker.

[93] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts*. Transactions of the Association for Computational Linguistics 12:157-173, 2024, doi:10.1162/tacl_a_00638; arXiv:2307.03172 (submitted 2023). 2024. <https://arxiv.org/abs/2307.03172>. Key: `liu2024lost`. Added by the checker.

[94] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data*. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017), JMLR W&CP 54; arXiv:1602.05629 (submitted 2016). 2017. <https://arxiv.org/abs/1602.05629>. Key: `mcmahan2017fedavg`.

[95] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander. *Towards Federated Learning at Scale: System Design*. Proceedings of the 2nd SysML Conference, Palo Alto, CA, USA, 2019; arXiv:1902.01046. 2019. <https://arxiv.org/abs/1902.01046>. Key: `bonawitz2019scale`.

[96] Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, Daniel Ramage. *Federated Learning for Mobile Keyboard Prediction*. arXiv preprint arXiv:1811.03604. 2018. <https://arxiv.org/abs/1811.03604>. Key: `hard2018gboard`.

[97] Microsoft. *Task idle conditions (Win32 Task Scheduler)*. Microsoft Learn documentation (ms.date 2018-05-31; page updated 2025-04-15; retrieved 2026-09-29). 2018. <https://learn.microsoft.com/en-us/windows/win32/taskschd/task-idle-conditions>. Key: `microsoft2018taskidle`.

[98] Jiaxiang Geng, Lunyu Zhao, Yiyi Lu, Bing Luo. *MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications*. arXiv preprint arXiv:2512.08211. 2025. <https://arxiv.org/abs/2512.08211>. Key: `geng2025mobilefinetuner`.

[99] Hassan Dbouk, Matthias Reisser, Prathamesh Mandke, Likhita Arun Navali, Christos Louizos. *Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices*. arXiv preprint arXiv:2606.19528. 2026. <https://arxiv.org/abs/2606.19528>. Key: `dbouk2026peakmemory`.

[100] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell. *Overcoming catastrophic forgetting in neural networks*. Proceedings of the National Academy of Sciences 114(13):3521-3526, 2017, doi:10.1073/pnas.1611835114; arXiv:1612.00796 (submitted 2016). 2017. <https://arxiv.org/abs/1612.00796>. Key: `kirkpatrick2017ewc`.

[101] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning*. arXiv preprint arXiv:2308.08747. 2023. <https://arxiv.org/abs/2308.08747>. Key: `luo2023cf`.

[102] Yujie Feng, Hao Wang, Jian Li, Xu Chu, Zhaolu Kang, Yiran Liu, Yasha Wang, Philip S. Yu, Xiao-Ming Wu. *FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning*. Annual Meeting of the Association for Computational Linguistics (ACL 2026), camera-ready per arXiv comment; arXiv:2601.03938. 2026. <https://arxiv.org/abs/2601.03938>. Key: `feng2026forever`.

[103] Denis Baylor, Eric Breck, Heng-Tze Cheng, Noah Fiedel, Chuan Yu Foo, Zakaria Haque, Salem Haykal, Mustafa Ispir, Vihan Jain, Levent Koc, Chiu Yuen Koo, Lukasz Lew, Clemens Mewald, Akshay Naresh Modi, Neoklis Polyzotis, Sukriti Ramesh, Sudip Roy, Steven Euijong Whang, Martin Wicke, Jarek Wilkiewicz, Xin Zhang, Martin Zinkevich. *TFX: A TensorFlow-Based Production-Scale Machine Learning Platform*. Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17), Applied Data Science track, pp. 1387-1395, doi:10.1145/3097983.3098021. 2017. <https://research.google/pubs/tfx-a-tensorflow-based-production-scale-machine-learning-platform/>. Key: `baylor2017tfx`.

[104] Evan Miller. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*. arXiv preprint arXiv:2411.00640. 2024. <https://arxiv.org/abs/2411.00640>. Key: `miller2024errorbars`.

[105] Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan. *Scaling Laws for Precision*. arXiv preprint arXiv:2411.04330. 2024. <https://arxiv.org/abs/2411.04330>. Key: `kumar2024precision`. (Added in the third pass.)

[106] Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu. *Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens*. arXiv preprint arXiv:2411.17691 (work in progress per arXiv comment). 2024. <https://arxiv.org/abs/2411.17691>. Key: `ouyang2024undertrained`. (Added in the third pass.)

[107] Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou. *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models*. Conference on Language Modeling (COLM 2025), per arXiv comment; arXiv:2504.04823. 2025. <https://arxiv.org/abs/2504.04823>. Key: `liu2025quantreasoning`. (Added in the third pass.)

[108] John Schulman and Thinking Machines Lab. *LoRA Without Regret*. Thinking Machines Lab, Connectionism blog, 29 September 2025. 2025. <https://thinkingmachines.ai/blog/lora/>. Key: `schulman2025loraregret`. (Added in the third pass; not peer reviewed.)

[109] Idan Shenfeld, Jyothish Pari, Pulkit Agrawal. *RL's Razor: Why Online Reinforcement Learning Forgets Less*. arXiv preprint arXiv:2509.04259. 2025. <https://arxiv.org/abs/2509.04259>. Key: `shenfeld2025rlrazor`. (Added in the third pass.)
