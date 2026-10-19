# DRAFT (not yet citation-checked): On-device fine-tuning, quantisation and pruning of small language models: a defensible pipeline for programmersRAG (build training set from the indexed repository and user feedback -> LoRA fine-tune when idle -> merge -> domain-calibrated GGUF quantisation -> held-out evaluation gate -> hot-swap)

This file is the raw research dossier. The checked version is `paper/notes/finetune_quant.md` once it exists;
prefer that file when it is present.

## Recommendations

- Base model: default to Qwen2.5-Coder-1.5B-Instruct (Apache-2.0, 1.54B, 32,768 ctx, repo-level pretraining with FIM tokens) and keep Qwen2.5-Coder-0.5B-Instruct (0.49B) as the variant small enough for CPU fine-tuning experiments; both are standard decoder transformers. Treat Qwen3-0.6B/1.7B and SmolLM2 as general-purpose alternatives, Gemma 3 270M/1B as options only if the Gemma licence is acceptable, and verify llama.cpp training support before adopting Qwen3.5 (hybrid Gated DeltaNet) or Gemma 4 E2B.
- Adapter method: train LoRA only (base frozen, so rollback is dropping the adapter). Use rsLoRA scaling alpha/sqrt(r), alpha = 2r, adapters on all linear modules, and a higher learning rate for B than A (LoRA+). Start with r = 16-64 for the time budget and test larger ranks, since code-domain evidence favours high rank; consider PiSSA initialisation if the adapter will be merged and quantised.
- Training data: generate RAFT-style examples per indexed function (question, oracle chunk, k = 4 distractors, answer quoting code verbatim), with the oracle present in a fraction P of examples (tune P; RAFT found 40-100% optimal depending on dataset). Draw distractors from the system's own retrieval admission rule (relevance probability >= 0.25) so training matches inference. Add self-supervised fill-in-the-middle examples and user-accepted answers or applied edits. Split train and held-out by file, never by chunk.
- Scheduling: train only when the UI toggle is on, the user has been absent for several minutes, CPU and disk are idle and the machine is on AC power; run at below-normal priority, checkpoint every N steps, and abort immediately when the idle condition ends (mirrors federated-learning eligibility rules and Windows Task Scheduler idle semantics with StopOnIdleEnd / RestartOnIdle).
- Forgetting safeguards: interleave replay data (generic code or text) and add a KL penalty to the base model's outputs during LoRA training; keep a fixed general-ability held-out set that is never trained on and test it for non-inferiority at every promotion.
- Quantisation: merge the adapter into full-precision weights (llama-export-lora), compute an importance matrix with llama-imatrix on domain calibration text formatted like real prompts (repository code inside the RAG prompt template), then quantise with llama-quantize. Use Q4_K_M as the floor; prefer Q5_K_M or Q6_K for sub-2B models when memory allows, because small models tolerate aggressive quantisation less and a 1.5B model is already under about 1.2 GB at 6 bits. Keep the default of not applying the imatrix to output.weight, and publish the calibration file with the paper.
- Pruning: do not apply SparseGPT/Wanda/LLM-Pruner to the generator. Implement the requested pruning on the context: keep chunks with relevance probability >= 0.25, then optionally prune inside chunks at sentence or statement level, and report prompt-token reduction and end-to-end latency.
- Evaluation gate: require (a) safe-to-serve load and memory check, (b) a one-sided paired improvement on held-out domain questions versus the currently served model, (c) non-inferiority within margin delta on the general set, and (d) mean KL divergence and same-top-token rate of the quantised model versus the unquantised merged model under a bound. Do not use Wikitext perplexity to judge the fine-tuning step. Report confidence intervals, not point scores.
- Hot swap: load the candidate alongside the serving model, run the gate, then swap a reference-counted model handle atomically so in-flight requests finish on the old model; keep the previous GGUF and adapter for one-step rollback. If serving through llama.cpp's server, adapters can be enabled or disabled at run time through the /lora-adapters endpoint.
- Implementation risk: upstream llama.cpp training is FP32 full fine-tuning and explicitly work in progress, and LoRA training is not implemented upstream. Plan for either a small custom LoRA trainer on top of the llama_opt API, or an optional external trainer, and state this limitation in the paper.
- Paper hygiene: present the 25% relevance threshold, the gate margins and the 16 KB work-unit size as design parameters of this system rather than published constants; label the imatrix-Hessian identity, the gate and the latency model as the authors' derivations; and re-verify verbatim quotes and table values against the PDFs for sources that were read through HTML renderings.

## Findings

- [established] LoRA freezes the pretrained weight W0 and learns a low-rank update Delta W = BA (B in R^{d x r}, A in R^{r x k}, r << min(d,k)); A is Gaussian-initialised and B zero-initialised so training starts from the base model; the update is scaled by alpha/r; after training the update can be merged (W = W0 + BA) so there is no added inference latency. Reported reduction versus full fine-tuning of GPT-3 175B: 10,000x fewer trainable parameters and 3x less GPU memory.  (hu2021lora)
- [established] QLoRA back-propagates through a frozen 4-bit quantised base model into LoRA adapters, using the NF4 data type (quantile quantisation for normally distributed weights), double quantisation (saves 0.373 bits per parameter) and paged optimisers; the authors report that 4-bit NF4 QLoRA matches 16-bit full fine-tuning and 16-bit LoRA in their benchmarks.  (dettmers2023qlora)
- [recent] DoRA reparameterises the weight as a trainable magnitude vector times a direction that receives the LoRA update; it reports +3.7 / +1.0 / +2.9 / +4.4 points over LoRA on commonsense reasoning for LLaMA-7B / 13B / LLaMA2-7B / LLaMA3-8B with no extra inference overhead after merging.  (liu2024dora)
- [recent] Three inexpensive corrections to vanilla LoRA are supported by theory or experiments: rsLoRA proves adapters are rank-stabilised iff the scaling factor is Theta(1/sqrt(r)) and proposes gamma_r = alpha/sqrt(r); PiSSA initialises A,B from the principal singular components and freezes the residual, converging faster and giving about 20% lower quantisation error than QLoRA; LoRA+ uses different learning rates for A and B and reports 1-2% gains and up to ~2x faster fine-tuning at the same cost.  (kalajdzievski2023rslora, meng2024pissa, hayou2024loraplus)
- [recent] Newer PEFT work (2025-2026) keeps questioning the low-rank constraint and the uniform learning rate: RandLoRA (ICLR 2025) performs full-rank updates via learned combinations of fixed random low-rank matrices; AnLR-LoRA (Sept 2026) assigns each rank-one component its own learning rate from function-space velocity and Adam SNR with no extra trainable parameters; LoftQ jointly chooses the quantised backbone and the LoRA initialisation to minimise ||W - Q - AB^T||_F.  (albert2025randlora, wang2026anlrlora, li2023loftq)
- [contested] Whether LoRA is a safe default for repeated domain adaptation is contested. On code and math with Llama-2-7B, LoRA 'substantially underperforms full finetuning' in standard low-rank settings but forgets less and regularises more strongly than weight decay or dropout; full fine-tuning learns perturbations of rank 10-100x larger than typical LoRA ranks. Other work finds LoRA introduces 'intruder dimensions' that drive forgetting and accumulate in continual learning, and that naive LoRA supervised fine-tuning 'can catastrophically degrade model capabilities, even when trained on very small datasets for relatively few steps'.  (biderman2024loraforgets, shuttleworth2024illusion, riemer2025replay)
- [recent] Practical LoRA settings reported for code-domain instruction fine-tuning: target all transformer modules (not attention only), alpha = 2r, rank up to 256 for code (64 sufficed for math), learning rate swept in 1e-5 to 5e-4, and prefer instruction fine-tuning over continued pretraining.  (biderman2024loraforgets)
- [recent] RAFT fine-tunes a model for open-book domain RAG by mixing training examples: for a fraction P of questions the prompt contains the oracle document plus distractors, for the remaining (1-P) only distractors; targets are chain-of-thought answers that quote the relevant span verbatim (##begin_quote## ... ##end_quote##). The best P was dataset dependent (40%, 60%, 100%), the main setup used one oracle with four distractors, and the base model was Llama2-7B; evaluation included API-documentation (code) datasets from Gorilla (Torch Hub 84.95% vs 8.60% baseline; HuggingFace 74.00% vs 61.06% DSF).  (zhang2024raft)
- [recent] Robustness to irrelevant retrieved context is learnable from little data: mixing top-1, low-ranked and random contexts with equal probability, 'even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones'. Related RAG-specific fine-tuning: RA-DIT (separate LM and retriever tuning steps, ICLR 2024), Finetune-RAG (+21.2% factual accuracy with imperfect-retrieval training data), RoseRAG (margin-aware preference optimisation for small LMs).  (yoran2023robust, lin2023radit, lee2025finetunerag, liu2025roserag)
- [contested] For code, the relative value of RAG versus fine-tuning is contested but the combination is supported: an industrial study on 160,000+ C++ files found RAG can beat fine-tuning alone, BM25 was the most effective and efficient retriever among those studied, 'RAG and fine-tuning are orthogonal and their combination leads to further improvement', and RAG scaled better with codebase size; a separate study on Python generation datasets found PEFT superior to ICL and RAG.  (wang2025ragorft, weyssow2023peftcode)
- [recent] Instruction data for code models can be synthesised from real open-source code snippets: Magicoder trained on 75K OSS-Instruct examples seeded by code snippets and reports HumanEval+ 66.5 pass@1 for a 7B model. This supports seeding fine-tuning examples from the indexed repository itself.  (wei2023magicoder)
- [recent] Small open models suited to local code tasks (parameters / licence / context): Qwen2.5-Coder-0.5B-Instruct 0.49B (0.36B non-embedding), 24 layers, Apache-2.0, 32,768 tokens, HumanEval 61.6 / MBPP 52.4; Qwen2.5-Coder-1.5B-Instruct 1.54B (1.31B non-embedding), 28 layers, Apache-2.0, 32,768 tokens, HumanEval 70.7 / MBPP 69.2. The Qwen2.5-Coder family was trained on over 5.5 trillion tokens with file-level and repo-level pretraining (FIM and <|repo_name|>, <|file_sep|> special tokens).  (hui2024qwen25coder, qwen2024coder05card, qwen2024coder15card)
- [recent] Other small general models: Qwen3-0.6B (0.6B, 0.44B non-embedding, 28 layers, Apache-2.0, 32,768 ctx) and Qwen3-1.7B (1.7B, 1.4B non-embedding, Apache-2.0); Gemma 3 270M (270M = 170M embedding + 100M transformer, 256k vocabulary, 32K ctx, Gemma licence, 'not designed for complex conversational use cases'); Gemma 3 1B (1.0B, 32K input ctx, Gemma licence); SmolLM2 (135M / 360M / 1.7B, Apache-2.0); SmolLM3 (3B, Apache-2.0, trained at 64k ctx); Phi-4-mini (3.8B, MIT, 128K ctx, 200,064-token vocabulary).  (yang2025qwen3, qwen2025qwen3cards, google2025gemma270m, google2025gemma3cards, allal2025smollm2, hf2025smolcards, microsoft2025phi4mini)
- [recent] 2026 releases: Qwen3.5-0.8B and Qwen3.5-2B (Feb 2026, Apache-2.0, 262,144-token context, multimodal, hybrid Gated DeltaNet + gated attention architecture) and Gemma 4 (July 2026, Apache-2.0 per model card; E2B is 2.3B effective parameters, 5.1B with embeddings, 128K ctx). Their non-standard architecture (Qwen3.5) or size (Gemma 4 E2B) makes them less obviously suitable for CPU fine-tuning than Qwen2.5-Coder 0.5B/1.5B.  (qwen2026qwen35cards, gemma2026gemma4)
- [established] GGUF k-quants quantise weights in super-blocks of 256 with per-block 6-bit scales (and mins for 'type-1' w = d*q + m): Q4_K is 4.5 bits per weight, Q5_K 5.5, Q6_K 6.5625, Q3_K 3.4375, Q2_K 2.5625. The named mixes raise precision on sensitive tensors (Q4_K_M as introduced: Q6_K for half of attention.wv and feed_forward.w2 tensors, Q4_K elsewhere). For Llama-3.1-8B the documented effective sizes are Q4_K_M 4.8944 bpw, Q5_K_M 5.7036, Q6_K 6.5633, Q8_0 8.5008.  (kawrakow2023kquants, llamacpp2026ggmlsource, llamacpp2026quantize)
- [established] The llama.cpp importance matrix stores only the diagonal second moments <a_i^2> of the activations entering each tensor and uses them as weights in a weighted least-squares fit of each block's scale and offset; in the Q4_K code path the per-weight fit weight is qw_l * sqrt(sigma^2 + x_l^2) with sigma^2 = 2*sum(x^2)/256, and the scale/min are obtained in closed form by weighted linear regression over a grid of candidate scales.  (kawrakow2024imatrix, llamacpp2026ggmlsource, llamacpp2026imatrix)
- [established] Accuracy loss versus bits is small down to about 4 bits and grows quickly below: LLaMA-7B perplexity F16 5.9066, Q6_K 5.9110, Q5_K_M 5.9208, Q4_K_M 5.9601, Q3_K_M 6.1503, Q2_K 6.7764; Llama-3.1-8B (2026 study) average of five benchmarks F16 69.47, Q5_K_M 69.36, Q4_K_M 69.15, Q3_K_M 68.07, Q3_K_S 65.49 with perplexity 7.32 / 7.40 / 7.56 / 7.96 / 8.96. Independent studies agree 4-bit retains performance comparable to unquantised models, and for code models 4-bit gives ~70% memory reduction without significant loss while 3- and 2-bit degrade significantly.  (kawrakow2023kquants, kurt2026whichquant, jin2024quanteval, kurtic2024bf16, giagnorio2025quantcode)
- [recent] The benefit of an importance matrix grows as bit width shrinks. In the llama.cpp Llama-3-8B scoreboard the mean KL divergence to FP16 is 0.031273 (Q4_K_M, no imatrix) vs 0.028152 (with imatrix), 0.101913 vs 0.084358 for Q3_K_M, and 0.445132 vs about 0.332 for Q2_K, i.e. roughly 10%, 17% and 25% relative reductions (percentages computed by us from the table). The same document notes 'no consistent improvement from using more Wikitext tokens for the importance matrix'.  (llamacpp2026perplexity)
- [contested] How much the calibration distribution matters is contested. AWQ reports robustness to calibration distribution shift (perplexity +0.5-0.6 vs +2.3-4.9 for GPTQ when calibration and evaluation domains are swapped); a systematic ACL 2024 study found 'substantial variations in downstream task performance' across calibration sets, with pruning more sensitive than quantisation; for code LLMs a code-specific calibration set 'helps to limit the loss of performance' at 3 and 2 bits while 4-bit was already safe; for pruning, calibration data similar to the training data works best and model-generated (self-calibration) data is competitive. The imatrix author also suggests instruct-format calibration data for instruct-tuned models.  (lin2023awq, williams2023calibration, giagnorio2025quantcode, ji2024bewarecalibration, williams2024selfcalibration, kawrakow2024imatrix)
- [established] GPTQ and AWQ are the two reference calibration-based PTQ methods: GPTQ minimises the layer-wise output error ||WX - W_hat X||_2^2 with OBS-style second-order weight updates using H = 2XX^T (128 random 2048-token C4 segments, 3-4 bits with negligible degradation on very large models); AWQ protects salient weight channels identified by activation magnitude by per-channel scaling s = s_X^alpha chosen to minimise the layer output error, without back-propagation.  (frantar2022gptq, lin2023awq, hassibi1992obs)
- [recent] Quantisation format design has a theoretical grounding: minimising the KL divergence between original and quantised model outputs under a size constraint can be approximated by minimising squared quantisation error, and Fisher information gives an optimal per-tensor bit allocation (saving up to 0.25 bits per parameter).  (orr2025optimalformats)
- [contested] One-shot weight pruning is well developed but its usefulness is contested: SparseGPT reaches 50-60% unstructured sparsity with small perplexity increase on very large models, Wanda matches it without weight updates (LLaMA-7B at 50%: magnitude 17.29, SparseGPT 7.22, Wanda 7.26 perplexity) and LLM-Pruner removes 20% of parameters structurally and recovers 94.97% of performance with LoRA; however LLM-KICK found 'all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%)' on knowledge-intensive tasks and that 'current quantization methods are more successful than pruning'.  (frantar2023sparsegpt, sun2023wanda, ma2023llmpruner, jaiswal2023compressing)
- [recent] Pruning the prompt rather than the weights gives direct, measured latency gains: LLMLingua reports up to 20x compression with little loss; LLMLingua-2 (token classification with a bidirectional encoder) reports 1.6x-2.9x end-to-end latency speed-up at 2x-5x compression; Provence (ICLR 2025) prunes retrieved contexts by thresholding per-token keep probabilities with sentence rounding, unifies pruning with reranking, and reports negligible to no performance drop.  (jiang2023llmlingua, pan2024llmlingua2, chirkova2025provence)
- [established] Production on-device learning systems train only opportunistically: the job runs 'when the phone is idle, charging, and connected to an unmetered network', and 'the FL runtime will abort, freeing the allocated resources, if these conditions are no longer met'; the Gboard deployment additionally required at least 2 GB of memory. On Windows, Task Scheduler defines idleness by user absence (default four minutes without keyboard or mouse input) plus CPU and I/O idleness thresholds, does not start idle tasks on battery by default, and can stop the task when idle ends (StopOnIdleEnd) and restart it (RestartOnIdle).  (bonawitz2019flscale, hard2018gboard, mcmahan2016fedavg, microsoft2018taskidle)
- [recent] Native (C/C++) on-device fine-tuning of sub-2B models is feasible but immature. llama.cpp ships examples/training/finetune.cpp, which is full-parameter FP32 fine-tuning ('technically functional ... but the code is very much WIP', LLaMA 3.2 1B works with 24 GB of memory); LoRA training is 'simply not implemented' upstream per an open research issue (May 2025). A fork (qvac-fabric) adds LoRA init/train/checkpoint/merge APIs on Vulkan, Metal and CPU (not merged upstream); MobileFineTuner is a C++ framework supporting full and LoRA fine-tuning of GPT-2, Gemma 3 270M/1B and Qwen2.5-0.5B on phones (e.g. Qwen2.5-0.5B peak RSS 2887 MB) with a battery-aware scheduler; a 2026 paper reports up to 26-28x lower peak memory for LoRA on 3B models by combining a quantised base, checkpointing with disk offload, softmax approximation and logits masking.  (llamacpp2026training, llamacpp2025loraissue, qvac2025fabric, geng2025mobilefinetuner, dbouk2026peakmemory)
- [established] Catastrophic forgetting is real for models in the 1B-7B range under continual instruction tuning and has known mitigations: EWC adds a Fisher-weighted quadratic penalty anchoring parameters to the previous solution; replay of generic data plus a KL penalty to the initial model 'can virtually eliminate the problem' for LoRA fine-tuning of Qwen instruction models with modest overhead; FOREVER schedules replay using optimiser-update magnitude (tested on 0.6B-13B models).  (kirkpatrick2016ewc, luo2023cfstudy, riemer2025replay, feng2026forever)
- [established] A held-out gate before swapping a model in is standard production practice: TFX validates that a model is 'safe to serve' (loads, does not crash, bounded resources) and compares its quality 'against a fixed threshold as well as against a baseline model (e.g., the current production model)'; 'any new model failing any of these checks is not pushed to serving'. Model comparisons should use paired differences on the same questions with CLT standard errors.  (baylor2017tfx, miller2024errorbars)
- [contested] The gate metric must differ by stage. Perplexity and KL divergence to the FP16 logits are the documented fidelity measures for quantisation, and divergence is more sensitive than top-1 accuracy; but llama.cpp documentation warns that 'finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases', and LLM-KICK argues perplexity fails to capture capability changes in compressed models, whereas another study finds perplexity a usable proxy on most benchmarks. Hence: task metrics for the fine-tune gate, KL divergence for the quantisation gate.  (llamacpp2026perplexity, llamacpp2026quantize, qamar2026divergence, jaiswal2023compressing, jin2024quanteval)
- [established] Hot-swapping is supported by the llama.cpp runtime at adapter level: adapters are loaded with --lora (optionally --lora-init-without-apply), their global scale can be changed at run time with POST /lora-adapters and per request with the 'lora' field, and llama-export-lora merges adapters into a base model to produce a merged GGUF that can then be quantised.  (llamacpp2026server, llamacpp2026exportlora)
- [our_proposal] PROPOSED PIPELINE. (1) Build examples from the index: RAFT-style question/answer pairs per function with verbatim code quotes, plus self-supervised fill-in-the-middle examples, plus user-accepted answers and applied edits; split train/held-out by file. (2) When the UI toggle is on and the machine is idle and on AC power, run checkpointed LoRA training at low priority and abort on user input. (3) Regularise with replay and a KL penalty to the base model. (4) Merge the adapter into full-precision weights. (5) Compute an importance matrix on domain text formatted like real prompts and quantise with k-quants. (6) Gate on held-out task metrics (paired test vs the serving model) and on KL divergence vs the unquantised merged model, plus a safe-to-serve load test. (7) Atomically swap and retain the previous model for rollback.  (zhang2024raft, yoran2023robust, hui2024qwen25coder, bonawitz2019flscale, microsoft2018taskidle, riemer2025replay, llamacpp2026exportlora, kawrakow2024imatrix, giagnorio2025quantcode, baylor2017tfx, miller2024errorbars, llamacpp2026perplexity, llamacpp2026server)
- [our_proposal] PROPOSED link between the 25% retrieval threshold and RAFT: if every admitted chunk has calibrated relevance probability p_i >= tau, the expected number of irrelevant admitted chunks is sum_i (1 - p_i) <= (1 - tau) K, so with tau = 0.25 up to 75% of the context may be distractors. The generator should therefore be fine-tuned on contexts sampled from the same admission rule (oracle plus admitted non-oracle chunks as distractors), which is exactly the RAFT / retrieval-robustness training regime.  (zhang2024raft, yoran2023robust)
- [our_proposal] PROPOSED interpretation: the llama.cpp importance matrix is, up to a constant factor, the diagonal of the GPTQ/OBS layer-wise Hessian H = 2XX^T; k-quants with imatrix therefore minimise a diagonal approximation of the same layer output reconstruction error that GPTQ minimises with the full Hessian, and domain calibration changes the solution only through these second moments.  (kawrakow2024imatrix, frantar2022gptq, llamacpp2026ggmlsource)
- [our_proposal] PROPOSED design decision: do not weight-prune the generator. Evidence favours quantisation over pruning for equal compression, pruning is more calibration-sensitive, and small models are less resilient to aggressive compression; realise 'pruning' at the context level (relevance threshold plus sentence/function-level context pruning), where latency gains are measured and proportional to removed prompt tokens.  (jaiswal2023compressing, williams2023calibration, giagnorio2025quantcode, pan2024llmlingua2, chirkova2025provence)

## Formulas

### LoRA forward pass and merge

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d,k), \qquad W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

The paper writes h = W0 x + BAx and states that Delta W x is scaled by alpha/r; A is Gaussian-initialised and B is zero so Delta W = 0 at the start. Only A and B are trained: r(d+k) parameters per adapted matrix instead of dk. Merging gives no added inference latency.

Sources: hu2021lora

### Rank-stabilised LoRA scaling

$$
x_{out} = (W + \gamma_r B A)\, x_{in} + b, \qquad \gamma_r \in \Theta_r\!\left(\tfrac{1}{\sqrt{r}}\right), \qquad \gamma_r = \frac{\alpha}{\sqrt{r}}
$$

Theorem: all adapters are rank-stabilised (activations and gradients stay Theta(1) in r) if and only if gamma_r is Theta(1/sqrt(r)); conventional alpha/r slows learning at higher ranks.

Sources: kalajdzievski2023rslora

### DoRA weight decomposition

$$
W = m \frac{V}{\lVert V \rVert_c}, \qquad W' = \underline{m}\, \frac{V + \Delta V}{\lVert V + \Delta V \rVert_c} = \underline{m}\, \frac{W_0 + \underline{BA}}{\lVert W_0 + \underline{BA} \rVert_c}
$$

m in R^{1 x k} is a trainable magnitude vector, ||.||_c is the column-wise vector norm, and the direction receives a LoRA update BA; underlined quantities are trainable.

Sources: liu2024dora

### PiSSA initialisation

$$
W = U S V^{\top}, \quad A = U_{[:, :r]}\, S_{[:r, :r]}^{1/2}, \quad B = S_{[:r, :r]}^{1/2}\, V_{[:, :r]}^{\top}, \quad W^{res} = U_{[:, r:]}\, S_{[r:, r:]}\, V_{[:, r:]}^{\top}, \quad Y = X\,(W^{res} + A B)
$$

Adapters start from the top-r singular components and the frozen residual holds the rest (PiSSA uses the convention A in R^{m x r}, B in R^{r x n}). Because the residual has a narrower distribution it quantises with about 20% lower error than QLoRA's base.

Sources: meng2024pissa

### LoRA+ learning-rate ratio

$$
\eta_B = \lambda\, \eta_A, \qquad \lambda > 1 \text{ fixed}
$$

The abstract states that using the same learning rate for A and B prevents efficient feature learning at large width and that this is corrected by setting different learning rates for A and B with a well-chosen ratio; the symbol lambda is used here for that ratio.

Sources: hayou2024loraplus

### LoftQ objective

$$
\min_{Q, A, B} \; \lVert W - Q - A B^{\top} \rVert_F, \qquad Q_t = q_N\!\left(W - A_{t-1} B_{t-1}^{\top}\right), \quad (A_t, B_t) = \mathrm{SVD}_r\!\left(W - Q_t\right)
$$

Alternates between quantising the residual and taking the rank-r SVD of the quantisation error so that the quantised backbone plus LoRA initialisation approximates the full-precision weight.

Sources: li2023loftq

### QLoRA block quantisation, NF4 quantiles and forward pass

$$
X^{\mathrm{Int8}} = \mathrm{round}\!\left(\frac{127}{\mathrm{absmax}(X^{\mathrm{FP32}})} X^{\mathrm{FP32}}\right) = \mathrm{round}(c^{\mathrm{FP32}} \cdot X^{\mathrm{FP32}}), \qquad q_i = \frac{1}{2}\left(Q_X\!\left(\frac{i}{2^k+1}\right) + Q_X\!\left(\frac{i+1}{2^k+1}\right)\right), \qquad Y^{\mathrm{BF16}} = X^{\mathrm{BF16}}\, \mathrm{doubleDequant}(c_1^{\mathrm{FP32}}, c_2^{k\text{-bit}}, W^{\mathrm{NF4}}) + X^{\mathrm{BF16}} L_1^{\mathrm{BF16}} L_2^{\mathrm{BF16}}
$$

Absmax block quantisation; NF4 levels q_i are midpoints of quantiles of the standard normal quantile function Q_X; the forward pass dequantises the frozen base and adds the LoRA path. Double quantisation reduces the constant overhead from 32/64 = 0.5 to 8/64 + 32/(64*256) = 0.127 bits per parameter.

Sources: dettmers2023qlora

### Layer-wise quantisation error objective (GPTQ) with OBQ update

$$
\arg\min_{\widehat{W}} \lVert W X - \widehat{W} X \rVert_2^2, \qquad w_q = \arg\min_{w_q} \frac{(\mathrm{quant}(w_q) - w_q)^2}{[H_F^{-1}]_{qq}}, \qquad \delta_F = - \frac{w_q - \mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}} \cdot (H_F^{-1})_{:,q}, \qquad H_F = 2 X_F X_F^{\top}
$$

X are calibration inputs to the layer; after quantising weight q the remaining full-precision weights F are updated by delta_F to compensate. GPTQ adds dampening (about 1% of the mean diagonal) and a Cholesky reformulation for stability.

Sources: frantar2022gptq, hassibi1992obs

### AWQ activation-aware scaling

$$
Q(w) = \Delta \cdot \mathrm{Round}\!\left(\frac{w}{\Delta}\right), \; \Delta = \frac{\max(|w|)}{2^{N-1}}; \qquad s^{*} = \arg\min_{s} \big\lVert Q\big(W \cdot \mathrm{diag}(s)\big)\big(\mathrm{diag}(s)^{-1} \cdot X\big) - W X \big\rVert, \qquad s = s_X^{\alpha}, \; \alpha^{*} = \arg\min_{\alpha} \mathcal{L}(s_X^{\alpha})
$$

Scaling a salient channel by s > 1 before quantisation and dividing the activation by s reduces its relative error by about (Delta'/Delta)(1/s); s_X is the per-channel average activation magnitude and alpha is grid-searched.

Sources: lin2023awq

### llama.cpp importance-matrix objective (importance weighting)

$$
F = \Big[\sum_j (q_j - w_j)\, a_j\Big]^2, \qquad \frac{\partial F}{\partial q_i} \propto \sum_j a_i a_j (q_j - w_j), \qquad \mathbb{E}[F] = \sum_{i,j} (q_i - w_i)(q_j - w_j) \langle a_i a_j \rangle \;\approx\; \sum_i \langle a_i^2 \rangle (q_i - w_i)^2
$$

w_j are weights of one tensor row, q_j the quantised values, a_j the incoming activations. The PR argues off-diagonal <a_i a_j> are much smaller than <a_i^2> because activations have mixed signs, so only the diagonal second moments are stored and used as importances in a weighted RMSE minimisation. The last approximation step is written out by us from the PR text.

Sources: kawrakow2024imatrix

### k-quant weighted block fit as implemented (Q4_K with imatrix)

$$
\omega_l = I_l \sqrt{\sigma^2 + x_l^2}, \; \sigma^2 = \frac{2}{256}\sum_{l=1}^{256} x_l^2; \qquad \min_{s,\,m,\,L_l \in \{0..15\}} \sum_l \omega_l\,(s L_l + m - x_l)^2; \qquad s = \frac{S_\omega S_{xL} - S_x S_L}{S_\omega S_{LL} - S_L^2}, \; m = \frac{S_{LL} S_x - S_L S_{xL}}{S_\omega S_{LL} - S_L^2}
$$

Read from ggml-quants.c (quantize_row_q4_K_impl and make_qkx3_quants): I_l is the imatrix entry, x_l the weight, S_omega = sum omega, S_x = sum omega x, S_L = sum omega L, S_LL = sum omega L^2, S_xL = sum omega L x. Candidate integer assignments are generated over a grid of inverse scales and the closed-form weighted regression with the lowest weighted error is kept. Without an imatrix the weight is sqrt(sigma^2) + |x_l|.

Sources: llamacpp2026ggmlsource

### Bits per weight of Q4_K

$$
\mathrm{bpw}_{Q4\_K} = \frac{256 \cdot 4 + 8 \cdot (6 + 6) + 2 \cdot 16}{256} = \frac{1152}{256} = 4.5
$$

Derived by us from the block layout in ggml-common.h (two fp16 super-block scales, 12 bytes of 6-bit scales and mins for 8 blocks, 128 bytes of 4-bit quants per 256 weights); agrees with the 4.5 bpw stated in the k-quants PR.

Sources: llamacpp2026ggmlsource, kawrakow2023kquants

### SparseGPT layer-wise pruning objective and OBS update

$$
\arg\min_{M_\ell,\, \widehat{W}_\ell} \lVert W_\ell X_\ell - (M_\ell \odot \widehat{W}_\ell) X_\ell \rVert_2^2, \qquad \varepsilon_m = \frac{w_m^2}{[H^{-1}]_{mm}}, \qquad \delta_m = -\frac{w_m}{[H^{-1}]_{mm}} H^{-1}_{:,m}, \qquad H = X X^{\top}
$$

M is the binary sparsity mask; epsilon_m is the error incurred by removing weight m and delta_m the optimal compensation of the remaining weights.

Sources: frantar2023sparsegpt, hassibi1992obs

### Wanda pruning score

$$
S_{ij} = |W_{ij}| \cdot \lVert X_j \rVert_2
$$

Weight magnitude times the L2 norm of the corresponding input feature over calibration tokens; weights are compared within each output neuron and the lowest scores are removed, with no weight update.

Sources: sun2023wanda

### LLM-Pruner group importance

$$
I_{W_i} = |\Delta \mathcal{L}(\mathcal{D})| = \left| \frac{\partial \mathcal{L}^{\top}(\mathcal{D})}{\partial W_i} W_i - \frac{1}{2} W_i^{\top} H W_i + \mathcal{O}(\lVert W_i \rVert^3) \right|
$$

Taylor expansion of the loss change when a coupled structure W_i is removed; the first-order term is kept because the calibration data is not the training data, and the Hessian is approximated by the Fisher information.

Sources: ma2023llmpruner

### Elastic weight consolidation

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i \left(\theta_i - \theta^{*}_{A,i}\right)^2
$$

L_B is the new-task loss, F the diagonal of the Fisher information matrix of the old task, theta*_A the old solution and lambda the importance of the old task.

Sources: kirkpatrick2016ewc

### RAFT training-set construction

$$
P\%: \; Q + D^{*} + D_1 + D_2 + \dots + D_k \rightarrow A^{*}, \qquad (1-P)\%: \; Q + D_1 + D_2 + \dots + D_k \rightarrow A^{*}
$$

Q is the question, D* the oracle document, D_i distractors, A* a chain-of-thought answer quoting the oracle verbatim. The best P was 40%, 60% or 100% depending on dataset; the main experiments used one oracle and four distractors.

Sources: zhang2024raft

### Token-level prompt pruning (LLMLingua-2)

$$
h = f_\theta(x), \qquad p(x_i, \Theta) = \mathrm{softmax}(W h_i + b), \qquad \mathcal{L}(\Theta) = \frac{1}{N} \sum_{i=1}^{N} \mathrm{CrossEntropy}(y_i, p(x_i, \Theta)), \qquad \tilde{N} = \tau N
$$

A bidirectional encoder classifies each token as preserve or discard; the N-tilde tokens with highest preserve probability are kept in original order for a target ratio tau.

Sources: pan2024llmlingua2

### Paired-difference standard error for model comparison

$$
SE_{\mathrm{CLT}} = \sqrt{\frac{1}{n}\cdot\frac{1}{n-1} \sum_{i=1}^{n} (s_i - \bar{s})^2}, \qquad SE_{A-B,\mathrm{paired}} = \sqrt{\frac{1}{n}\cdot\frac{1}{n-1} \sum_{i=1}^{n} (s_{A-B,i} - \bar{s}_{A-B})^2}, \qquad CI_{95\%} = \bar{s} \pm 1.96 \cdot SE
$$

s_{A-B,i} is the per-question score difference of two models evaluated on the same questions; pairing removes question-difficulty variance.

Sources: miller2024errorbars

### Proposed promotion gate (non-inferiority plus fidelity)

$$
\text{promote} \iff \big(\bar{d}_{dom} - z_{1-\alpha}\, SE_{dom} > 0\big) \;\wedge\; \big(\bar{d}_{gen} - z_{1-\alpha}\, SE_{gen} > -\delta\big) \;\wedge\; \Big(\tfrac{1}{T}\sum_{t=1}^{T} D_{KL}\big(P^{FP}_t \,\Vert\, P^{Q}_t\big) \le \kappa\Big) \;\wedge\; \text{safe-to-serve}
$$

OUR PROPOSAL. d_i = s_i(new) - s_i(serving) on held-out items, d-bar its mean and SE the paired standard error (Miller). Condition 1 requires a one-sided significant gain on the held-out domain set; condition 2 is a non-inferiority test with margin delta on a general set to catch forgetting; condition 3 bounds the mean KL divergence between the unquantised merged model and its quantised version (llama.cpp --kl-divergence); condition 4 is TFX's load/resource check. It combines TFX's 'fixed threshold as well as baseline model' rule with paired statistics; alpha, delta and kappa are design parameters to be calibrated empirically.

Sources: baylor2017tfx, miller2024errorbars, llamacpp2026perplexity

### Proposed latency model and context-pruning saving

$$
T(N_p, N_g) \approx \frac{N_p}{v_{pp}} + \frac{N_g}{v_{tg}}, \qquad \Delta T_{prune} = \frac{(1-\rho)\, N_p}{v_{pp}}, \quad \rho = \frac{N_p'}{N_p}
$$

OUR PROPOSAL. N_p prompt tokens processed at prefill throughput v_pp and N_g generated tokens at decode throughput v_tg, the two quantities llama.cpp benchmarks report (pp512, tg128). Removing a fraction (1 - rho) of the retrieved context saves prefill time linearly; quantisation changes v_pp and v_tg (e.g. on a Xeon server Llama-3.1-8B decode 2.83 tok/s at F16 vs 5.12 tok/s at Q4_K_M). Ignores KV-cache reuse and attention cost growth with context.

Sources: kurt2026whichquant, llamacpp2026quantize, pan2024llmlingua2

### Proposed identity: imatrix as the diagonal of the GPTQ Hessian

$$
H = 2 X X^{\top} \;\Rightarrow\; H_{ii} = 2 \sum_{n=1}^{N} x_{i,n}^2 = 2N \langle a_i^2 \rangle, \qquad (w - q)^{\top} X X^{\top} (w - q) = N \sum_{i,j} (w_i - q_i)(w_j - q_j) \langle a_i a_j \rangle \approx N \sum_i \langle a_i^2 \rangle (w_i - q_i)^2
$$

OUR PROPOSAL (derivation). Expanding GPTQ's row-wise reconstruction error and dropping off-diagonal activation correlations yields exactly the weighted squared error used by llama.cpp, so imatrix entries equal diag(H)/(2N). Domain calibration therefore affects k-quants only through per-input-channel second moments.

Sources: frantar2022gptq, kawrakow2024imatrix

### Proposed expected distractor load under a relevance threshold

$$
\mathbb{E}[\#\text{irrelevant}] = \sum_{i=1}^{K} (1 - p_i) \le (1 - \tau) K, \qquad \tau = 0.25 \Rightarrow \mathbb{E}[\#\text{irrelevant}] \le 0.75 K
$$

OUR PROPOSAL (derivation). If the K admitted chunks have calibrated relevance probabilities p_i >= tau, linearity of expectation bounds the expected number of distractors. With tau = 0.25 the generator may see mostly distractors, which motivates RAFT-style fine-tuning with distractors sampled from the same admission rule.

Sources: zhang2024raft, yoran2023robust

## Open questions

- Order of operations: is quality better when LoRA is trained on full-precision weights, merged and then quantised, or trained on the already-quantised model (merged or kept as a separate adapter)? llama.cpp lists this as an open research question (issue #13485); it must be measured on the chosen model.
- Wall-clock cost: no opened source reports LoRA or full fine-tuning throughput for 0.5B-1.5B models on a laptop-class hybrid x86 CPU (i9-12950HX) or on a 4 GB RTX A1000; published numbers are for phones, an RTX 4090 or servers. Idle-window length needed per useful update is therefore unknown.
- Does domain (C source) calibration of the importance matrix give a measurable benefit at 4-6 bits for sub-2B models? Published evidence shows benefit at 3 and 2 bits for code models of 1B-34B using AQLM, and only about 10% KL reduction at Q4_K_M for an 8B model with generic calibration text.
- Is fine-tuning worthwhile at all for a corpus as small as mimalloc (about 59 files)? The industrial study found RAG scales better than fine-tuning and that the two combine; with so little data the risk is memorisation and an under-powered held-out set. The minimum detectable effect of the gate should be computed before claiming improvement.
- Quality of synthetic question/answer pairs when the generator of the training data is the same small local model that is being fine-tuned (self-training bias); whether user feedback volume is sufficient to correct it.
- How well calibrated are the retrieval relevance probabilities? The distractor bound and the 25% threshold are only meaningful if p_i is calibrated; calibration of KDE-based relevance scores is outside this dossier and needs its own evaluation.
- Non-inferiority margin delta, KL bound kappa and significance level alpha for the promotion gate are not given by the literature and must be chosen and justified empirically; TFX notes the tension between over-conservative and over-loose thresholds.
- Does LoRA forgetting accumulate over many successive opportunistic rounds (intruder dimensions), and is restarting each round from the base model plus a cumulative dataset safer than continuing from the previous adapter?
- Can upstream llama.cpp training (FP32, full-parameter, work in progress) be relied on in a C++20 Windows/MSVC build, or is a custom LoRA trainer required? The fork that adds LoRA training is not merged upstream.
- Suitability of the newest small models (Qwen3.5 0.8B/2B with Gated DeltaNet layers, Gemma 4 E2B) for llama.cpp-based training and k-quant quantisation has not been verified here.
- Verification caveat: arXiv and Hugging Face pages were read through an HTML-to-text summariser; llama.cpp documents, pull requests, source code and the TFX PDF were read as raw text. Verbatim quotes and table values from the summarised sources should be re-checked against the PDFs before publication, and the LoRA paper's conference venue could not be confirmed from an opened page so it is cited by arXiv id.

## Sources

- **hu2021lora**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
  - evidence: Sec. 4.1: 'h = W0 x + Delta W x = W0 x + BAx'; 'We use a random Gaussian initialization for A and zero for B, so Delta W = BA is zero at the beginning of training'; Delta W x scaled by alpha/r. Abstract: 'LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.' Merging W = W0 + BA adds no inference latency.
- **dettmers2023qlora**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023 (Advances in Neural Information Processing Systems 36). https://arxiv.org/abs/2305.14314
  - evidence: Abstract: 'QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters'; introduces NF4, double quantization and paged optimizers. Double quantization: 'from 32/64=0.5 bits, to 8/64+32/(64*256)=0.127 bits', 'a reduction of 0.373 bits per parameter'. '4-bit QLoRA with NF4 data type matches 16-bit full finetuning and 16-bit LoRA finetuning performance'. Venue confirmed on proceedings.neurips.cc.
- **liu2024dora**: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024 (Oral). https://arxiv.org/abs/2402.09353
  - evidence: Abstract: 'DoRA decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates'. Equations for W' = m (W0 + BA)/||W0 + BA||_c; gains over LoRA +3.7 (LLaMA-7B), +1.0 (13B), +2.9 (LLaMA2-7B), +4.4 (LLaMA3-8B).
- **kalajdzievski2023rslora**: Damjan Kalajdzievski (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv:2312.03732. https://arxiv.org/abs/2312.03732
  - evidence: Abstract: 'prove that LoRA adapters should be divided by a factor of the square root of the rank'. Body: 'all adapters are rank-stabilized if and only if gamma_r in Theta_r(1/sqrt(r))'; 'gamma_r = alpha/sqrt(r) for some hyperparameter alpha'.
- **meng2024pissa**: Fanxu Meng, Zhaohui Wang, Muhan Zhang (2024). PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. NeurIPS 2024 (spotlight). https://arxiv.org/abs/2404.02948
  - evidence: Abstract: 'initializes the adaptor matrices A and B with the principal components of the original matrix W, and put the remaining components into a residual matrix W^res ... which is frozen'; GSM8K Mistral-7B 72.86% vs LoRA 67.7%; 'Compared to QLoRA, QPiSSA exhibits smaller quantization errors'. Body: about 20% quantisation error reduction; fast SVD initialisation in seconds.
- **hayou2024loraplus**: Soufiane Hayou, Nikhil Ghosh, Bin Yu (2024). LoRA+: Efficient Low Rank Adaptation of Large Models. arXiv:2402.12354. https://arxiv.org/abs/2402.12354
  - evidence: Abstract: 'using the same learning rate for A and B does not allow efficient feature learning ... corrected simply by setting different learning rates for the LoRA adapter matrices A and B with a well-chosen ratio'; 'LoRA+ improves performance (1-2 % improvements) and finetuning speed (up to ~2X SpeedUp), at the same computational cost as LoRA'.
- **albert2025randlora**: Paul Albert, Frederic Z. Zhang, Hemanth Saratchandran, Cristian Rodriguez-Opazo, Anton van den Hengel, Ehsan Abbasnejad (2025). RandLoRA: Full-rank parameter-efficient fine-tuning of large models. ICLR 2025. https://arxiv.org/abs/2502.00987
  - evidence: Abstract: 'a parameter-efficient method that performs full-rank updates using a learned linear combinations of low-rank, non-trainable random matrices'; 'full-rank updates are beneficial across vision and language tasks'.
- **wang2026anlrlora**: Huiyi Wang, Daijiao Liu, Lina Yao, Dong Gong (2026). One Rate Is Not Enough: Adaptive Anisotropic Learning Rates for LoRA Fine-Tuning. arXiv:2609.05885. https://arxiv.org/abs/2609.05885
  - evidence: Abstract: 'assigns each rank-one component its own effective learning rate, computed online from training-time signals and mean-normalized per module'; uses 'function-space velocity and Adam SNR, as a lightweight scheme with no extra trainable parameters'.
- **li2023loftq**: Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao (2023). LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models. arXiv:2310.08659. https://arxiv.org/abs/2310.08659
  - evidence: Abstract: 'a novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning'. Objective 'min_{Q,A,B} ||W - Q - A B^T||_F' with alternating quantisation and SVD steps.
- **biderman2024loraforgets**: Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, John P. Cunningham (2024). LoRA Learns Less and Forgets Less. Transactions on Machine Learning Research (Featured Certification). https://arxiv.org/abs/2405.09673
  - evidence: Abstract: 'in the standard low-rank settings, LoRA substantially underperforms full finetuning. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain.' Body: Llama-2-7B on code and math; full fine-tuning finds perturbations of rank 10-100x greater than typical LoRA; recommendations: target all modules, alpha = 2r, r = 256 for code, LR 1e-5 to 5e-4, prefer instruction fine-tuning.
- **shuttleworth2024illusion**: Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma (2024). LoRA vs Full Fine-tuning: An Illusion of Equivalence. arXiv:2410.21228. https://arxiv.org/abs/2410.21228
  - evidence: Abstract page: LoRA and full fine-tuning yield weight matrices with different spectral structure; LoRA introduces 'intruder dimensions' which causally drive forgetting of pre-training knowledge and accumulate during continual learning.
- **riemer2025replay**: Matthew Riemer, Erik Miehling, Miao Liu, Djallel Bouneffouf, Murray Campbell (2025). The Effectiveness of Approximate Regularized Replay for Efficient Supervised Fine-Tuning of Large Language Models. arXiv:2512.22337. https://arxiv.org/abs/2512.22337
  - evidence: Abstract: 'LoRA-based supervised fine-tuning can catastrophically degrade model capabilities, even when trained on very small datasets for relatively few steps'; remedy is 'a regularized approximate replay approach which penalizes KL divergence with respect to the initial model and interleaves in data for next token prediction'; applied to Qwen instruction-tuned models.
- **zhang2024raft**: Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez (2024). RAFT: Adapting Language Model to Domain Specific RAG. arXiv:2403.10131. https://arxiv.org/abs/2403.10131
  - evidence: Abstract: 'we train the model to ignore those documents that don't help in answering the question, which we call, distractor documents. RAFT accomplishes this by citing verbatim the right sequence from the relevant document'. Body: P% of data 'Q + D* + D1 + ... + Dk -> A*', (1-P)% 'Q + D1 + ... + Dk -> A*'; optimal P 40%, 60%, 100% by dataset; one golden plus four distractors; Llama2-7B; Torch Hub 84.95%, HuggingFace 74.00%.
- **yoran2023robust**: Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant (2023). Making Retrieval-Augmented Language Models Robust to Irrelevant Context. arXiv:2310.01558. https://arxiv.org/abs/2310.01558
  - evidence: Abstract: 'even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones'. Body: 'we use either the top-1, low-ranked, or random evidence with equal probability at each step'.
- **lin2023radit**: Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, Scott Yih (2023). RA-DIT: Retrieval-Augmented Dual Instruction Tuning. ICLR 2024. https://arxiv.org/abs/2310.01352
  - evidence: Abstract: two fine-tuning steps, '(1) one updates a pre-trained LM to better use retrieved information, while (2) the other updates the retriever to return more relevant results, as preferred by the LM'; up to +8.9% in 0-shot.
- **lee2025finetunerag**: Zhan Peng Lee, Andre Lin, Calvin Tan (2025). Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation. arXiv:2505.10792. https://arxiv.org/abs/2505.10792
  - evidence: Abstract: 'RAG training dataset constructed to mimic real-world imperfections. Experimental results show that Finetune-RAG improves factual accuracy by 21.2% over the base model.'
- **liu2025roserag**: Tianci Liu, Haoxiang Jiang, Tianze Wang, Ran Xu, Yue Yu, Linjun Zhang, Tuo Zhao, Haoyu Wang (2025). RoseRAG: Robust Retrieval-augmented Generation with Small-scale LLMs via Margin-aware Preference Optimization. arXiv:2502.10993. https://arxiv.org/abs/2502.10993
  - evidence: Abstract: 'imperfect retrieval can introduce distracting noise that misleads SLMs. We propose RoseRAG, a robust RAG framework for SLMs via Margin-aware Preference Optimization.'
- **wang2025ragorft**: Chaozheng Wang, Zezhou Yang, Shuzheng Gao, Cuiyun Gao, Ting Peng, Hailiang Huang, Yuetang Deng, Michael Lyu (2025). RAG or Fine-tuning? A Comparative Study on LCMs-based Code Completion in Industry. FSE 2025 Industry Track. https://arxiv.org/abs/2505.15179
  - evidence: Abstract: 'we collect over 160,000 internal C++ files'; 'RAG ... can achieve higher accuracy than fine-tuning alone. Specifically, BM25 presents superior retrieval effectiveness and efficiency among studied RAG methods. Moreover, RAG and fine-tuning are orthogonal and their combination leads to further improvement. We also observe that RAG demonstrates better scalability than FT'.
- **weyssow2023peftcode**: Martin Weyssow, Xin Zhou, Kisub Kim, David Lo, Houari Sahraoui (2023). Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models. arXiv:2308.10462. https://arxiv.org/abs/2308.10462
  - evidence: Abstract: PEFT shows 'superiority and potential over ICL and RAG across a diverse set of LLMs and three representative Python code generation datasets: Conala, CodeAlpacaPy, and APPS' and 'significant reductions in memory usage by combining PEFT with quantization'.
- **wei2023magicoder**: Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang (2023). Magicoder: Empowering Code Generation with OSS-Instruct. ICML 2024. https://arxiv.org/abs/2312.02120
  - evidence: Abstract: 'trained on 75K synthetic instruction data using OSS-Instruct, a novel approach to enlightening LLMs with open-source code snippets to generate diverse instruction data for code'; MagicoderS-CL-7B 66.5 on HumanEval+.
- **hui2024qwen25coder**: Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, Junyang Lin (2024). Qwen2.5-Coder Technical Report. arXiv:2409.12186. https://arxiv.org/abs/2409.12186
  - evidence: Abstract: 'This series includes six models: Qwen2.5-Coder-(0.5B/1.5B/3B/7B/14B/32B)'; trained on 'over 5.5 trillion tokens'. Body: file-level pretraining with next-token prediction and FIM, repo-level pretraining at 32,768 tokens with <|repo_name|>, <|file_sep|>, <|fim_prefix|> tokens; instruct models HumanEval/MBPP 61.6/52.4 (0.5B) and 70.7/69.2 (1.5B).
- **qwen2024coder05card**: Qwen Team (Alibaba Cloud) (2024). Qwen2.5-Coder-0.5B-Instruct model card. Hugging Face model card. https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct
  - evidence: License Apache 2.0; 'Number of Parameters: 0.49B', non-embedding 0.36B; 24 layers; attention heads (GQA) 14 for Q and 2 for KV; context length 32,768 tokens; architecture 'transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings'.
- **qwen2024coder15card**: Qwen Team (Alibaba Cloud) (2024). Qwen2.5-Coder-1.5B-Instruct model card. Hugging Face model card. https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
  - evidence: License Apache 2.0; 1.54B parameters, 1.31B non-embedding; 28 layers; attention heads (GQA) 12 for Q and 2 for KV; context length 32,768 tokens.
- **yang2025qwen3**: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, et al. (2025). Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388
  - evidence: Abstract: 'models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion'; integrates thinking and non-thinking modes.
- **qwen2025qwen3cards**: Qwen Team (Alibaba Cloud) (2025). Qwen3-0.6B and Qwen3-1.7B model cards. Hugging Face model cards. https://huggingface.co/Qwen/Qwen3-0.6B
  - evidence: Qwen3-0.6B: license apache-2.0, 0.6B parameters (0.44B non-embedding), 28 layers, 16 Q / 8 KV heads, context 32,768. Qwen3-1.7B (https://huggingface.co/Qwen/Qwen3-1.7B): apache-2.0, 1.7B parameters (1.4B non-embedding), 28 layers, context 32,768.
- **qwen2026qwen35cards**: Qwen Team (Alibaba Cloud) (2026). Qwen3.5-0.8B and Qwen3.5-2B model cards. Hugging Face model cards. https://huggingface.co/Qwen/Qwen3.5-0.8B
  - evidence: License apache-2.0; 'Number of Parameters: 0.8B' (2B for https://huggingface.co/Qwen/Qwen3.5-2B); 'Context Length: 262,144 natively'; 24 layers arranged as '6 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))'; causal LM with vision encoder; non-thinking mode by default; citation dated February 2026.
- **google2025gemma270m**: Olivier Lacombe, Kathleen Kenealy, Kat Black, Ravin Kumar, Francesco Visin, Jiageng Zhang (2025). Introducing Gemma 3 270M: The compact model for hyper-efficient AI. Google Developers Blog (14 Aug 2025). https://developers.googleblog.com/en/introducing-gemma-3-270m/
  - evidence: '170 million embedding parameters due to a large vocabulary size and 100 million for our transformer blocks'; 256k-token vocabulary; 'the INT4-quantized model used just 0.75% of the battery for 25 conversations'; QAT checkpoints at INT4; the model 'is not designed for complex conversational use cases'.
- **google2025gemma3cards**: Google (Gemma Team) (2025). Gemma 3 270M and Gemma 3 1B model cards. Hugging Face model cards. https://huggingface.co/google/gemma-3-270m
  - evidence: gemma-3-270m: license Gemma, 32K input context. gemma-3-1b-it (https://huggingface.co/google/gemma-3-1b-it): license Gemma, 1.0B params, 'Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B size', output 8192 tokens.
- **gemma2026gemma4**: Gemma Team: Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, Mayank Chaturvedi, et al. (2026). Gemma 4 Technical Report. arXiv:2607.02770; model card huggingface.co/google/gemma-4-E2B-it. https://arxiv.org/abs/2607.02770
  - evidence: Abstract: 'dense and Mixture-of-Experts architectures, ranging from 2.3B to 31B parameters'. Model card: license Apache 2.0; E2B 2.3B effective (5.1B with embeddings), 128K context; small models 'specifically designed for efficient local execution on laptops and mobile devices'.
- **allal2025smollm2**: Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martin Blazquez, Guilherme Penedo, Lewis Tunstall, Andres Marafioti, Hynek Kydlicek, Agustin Piqueres Lajarin, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clementine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf (2025). SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model. arXiv:2502.02737. https://arxiv.org/abs/2502.02737
  - evidence: Abstract: 'a state-of-the-art "small" (1.7 billion parameter) language model ... we overtrain SmolLM2 on ~11 trillion tokens of data using a multi-stage training process that mixes web text with specialized math, code, and instruction-following data'; 'outperforms other recent small LMs including Qwen2.5-1.5B and Llama3.2-1B'.
- **hf2025smolcards**: Hugging Face TB Research (2025). SmolLM2-360M-Instruct and SmolLM3-3B model cards. Hugging Face model cards. https://huggingface.co/HuggingFaceTB/SmolLM3-3B
  - evidence: SmolLM3-3B: Apache 2.0, 3B parameters, 'Trained on 64k context and supports up to 128k tokens using YARN extrapolation', 'pretrained on 11.2T tokens'. SmolLM2 card (https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct): Apache 2.0; family sizes 135M, 360M, 1.7B; 'SmolLM2 models primarily understand and generate content in English.'
- **microsoft2025phi4mini**: Microsoft (Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, et al.) (2025). Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs; Phi-4-mini-instruct model card. arXiv:2503.01743; Hugging Face model card. https://huggingface.co/microsoft/Phi-4-mini-instruct
  - evidence: Card: 'The model is licensed under the MIT license'; 3.8B parameters; 128K tokens context; vocabulary 'up to 200064 tokens'; intended for 'Memory/compute constrained environments' and 'Latency bound scenarios'. Report abstract: 'Phi-4-Mini is a 3.8-billion-parameter language model'.
- **frantar2022gptq**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. https://arxiv.org/abs/2210.17323
  - evidence: Abstract: 'a new one-shot weight quantization method based on approximate second-order information ... reducing the bitwidth down to 3 or 4 bits per weight, with negligible accuracy degradation'. Body: objective argmin ||WX - W_hat X||_2^2, OBQ update with H_F = 2 X_F X_F^T, dampening 1% of mean diagonal, Cholesky reformulation; calibration '128 random 2048 token segments from the C4 dataset'.
- **lin2023awq**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024 (Best Paper Award). https://arxiv.org/abs/2306.00978
  - evidence: 'Protecting only 1% salient weights can greatly reduce quantization error'; Eq. 1 Q(w) = Delta * Round(w/Delta); Eq. 4 s* = argmin ||Q(W diag(s))(diag(s)^-1 X) - WX||; Eq. 5 s = s_X^alpha. Calibration robustness: 'AWQ only increases the perplexity by 0.5-0.6, while GPTQ has 2.3-4.9 worse perplexity'.
- **hassibi1992obs**: Babak Hassibi, David G. Stork (1992). Second order derivatives for network pruning: Optimal Brain Surgeon. Advances in Neural Information Processing Systems 5 (NIPS 1992). https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
  - evidence: Abstract: 'We investigate the use of information from all second order derivatives of the error function to perform network pruning'; 'Crucial to OBS is a recursion relation for calculating the inverse Hessian matrix H-1 from training data and structural information of the net.'
- **kawrakow2023kquants**: Iwan Kawrakow (ikawrakow) (2023). k-quants (llama.cpp pull request #1684). GitHub pull request, ggml-org/llama.cpp (merged 5 June 2023). https://github.com/ggml-org/llama.cpp/pull/1684
  - evidence: 'GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.' Q4_K_M 'uses GGML_TYPE_Q6_K for half of the attention.wv and feed_forward.w2 tensors, else GGML_TYPE_Q4_K'. 7B perplexity: F16 5.9066, Q2_K 6.7764, Q3_K_M 6.1503, Q4_K_M 5.9601, Q5_K_M 5.9208, Q6_K 5.9110. 'the 6-bit quantized perplexity is within 0.1% or better from the original fp16 model'.
- **kawrakow2024imatrix**: Iwan Kawrakow (ikawrakow) (2024). Importance Matrix calculation (llama.cpp pull request #4861). GitHub pull request, ggml-org/llama.cpp (merged 12 Jan 2024). https://github.com/ggml-org/llama.cpp/pull/4861
  - evidence: 'we want to minimize, e.g., F = [Sum_j (q_j - w_j) a_j]^2 ... we can greatly simplify things by computing just the diagonal elements <a_i^2> ... We can then use these <a_i^2> as importances (weights) in a weighted RMSE minimization when quantizing the tensor.' Also: 'for instruct tuned models, it may actually be better to use instruct tuning training datasets for computing the importance matrix'; about 50k tokens typical; ~10 minutes for a 7B model on a 32-core CPU.
- **llamacpp2026quantize**: ggml-org/llama.cpp contributors (2026). llama.cpp tools/quantize README. Official repository documentation (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
  - evidence: 'This process however, may introduce some accuracy loss which is usually measured in Perplexity (ppl) and/or Kullback-Leibler Divergence (kld). This can be minimized by using a suitable imatrix file.' Llama-3.1-8B table: Q4_K_M 4.8944 bits/weight 4.58 GiB; Q5_K_M 5.7036; Q6_K 6.5633; Q8_0 8.5008; F16 16.0005, 14.96 GiB. Options include --imatrix, --tensor-type and --prune-layers.
- **llamacpp2026imatrix**: ggml-org/llama.cpp contributors (2026). llama.cpp tools/imatrix README. Official repository documentation (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
  - evidence: 'Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.' '--process-output ... Typically, it is better not to utilize the importance matrix when quantizing output.weight, so this is set to false by default.' '--in-file one or more existing imatrix files to load and combine.' Statistics: 'Sum(Act^2): sum of all squared activations (the importance scores)'.
- **llamacpp2026perplexity**: ggml-org/llama.cpp contributors (2026). llama.cpp tools/perplexity README (LLaMA 3 8b scoreboard). Official repository documentation (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
  - evidence: 'finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases.' KLD vs FP16: q8_0 0.001355; q6_K 0.005452; q5_K_M 0.010762; q4_K_M 0.031273 (no imatrix) vs 0.028152 (imatrix WT 10m); q3_K_M 0.101913 vs 0.084358; q2_K 0.445132 vs 0.332223. 'There seems to be no consistent improvement from using more Wikitext tokens for the importance matrix.' Documents --kl-divergence statistics including 'Same top p'.
- **llamacpp2026ggmlsource**: ggml-org/llama.cpp contributors (2026). llama.cpp source: ggml/src/ggml-quants.c and ggml/src/ggml-common.h. Official repository source code (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
  - evidence: quantize_row_q4_K_impl: 'float sigma2 = 2*sum_x2/QK_K;' and 'weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);' else 'weights[l] = av_x + fabsf(x[32*j + l]);'. make_qkx3_quants computes 'this_scale = (sum_w * sum_xl - sum_x * sum_l)/D'. ggml-common.h: '#define QK_K 256', '#define K_SCALE_SIZE 12', block_q4_K comment 'Effectively 4.5 bits per weight'.
- **llamacpp2026training**: ggml-org/llama.cpp contributors (2026). llama.cpp examples/training README and finetune.cpp. Official repository documentation and source (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/tree/master/examples/training
  - evidence: 'So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP. Finetuning of Stories 260K and LLaMA 3.2 1b seems to work with 24 GB of memory. For CPU training, compile llama.cpp without any additional backends such as CUDA.' finetune.cpp uses llama_opt_init with llama_opt_param_filter_all, llama_opt_epoch, and llama_model_save_to_file; latest commit touching the directory dated 2026-09-02.
- **llamacpp2025loraissue**: Johannes Gaessler (JohannesGaessler) (2025). LoRA training example (llama.cpp issue #13485). GitHub issue, ggml-org/llama.cpp (12 May 2025). https://github.com/ggml-org/llama.cpp/issues/13485
  - evidence: 'LoRA training should in principle be achievable with the same tools as the full finetune, it's simply not implemented.' Proposes comparing: LoRA on full-precision model then merge then quantise; LoRA on quantised model and merge; LoRA on quantised model without merging.
- **llamacpp2026server**: ggml-org/llama.cpp contributors (2026). llama.cpp tools/server README. Official repository documentation (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
  - evidence: '--lora-init-without-apply: load LoRA adapters without applying them (apply later via POST /lora-adapters)'; 'POST /lora-adapters: Set list of LoRA adapters. This sets the global scale for LoRA adapters'; per-request 'lora' field; 'requests with different LoRA configurations will not be batched together'.
- **llamacpp2026exportlora**: ggml-org/llama.cpp contributors (2026). llama.cpp tools/export-lora README. Official repository documentation (master, retrieved 2026-09-29). https://github.com/ggml-org/llama.cpp/blob/master/tools/export-lora/README.md
  - evidence: 'Apply LORA adapters to base model and export the resulting model.' Usage: llama-export-lora -m base.gguf -o merged.gguf --lora adapter.gguf; multiple adapters via --lora-scaled FNAME:SCALE.
- **qvac2025fabric**: Subash SN, Akshay Nambiar, Patrik Lambert, Milan Gritta, Gianfranco Cordella, Amril Nurman (Tether AI Research) (2025). An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs. Hugging Face blog (1 Dec 2025). https://huggingface.co/blog/qvac/fabric-llm-finetune
  - evidence: Adds 'llama_lora_training_init(): Initializes and allocates memory for a LoRA adapter' and uses 'llama_opt_epoch()'; backends Vulkan, Metal, CPU; tested Qwen3 0.6B/1.7B, Gemma3, TinyLlama; Qwen3-1.7B (Q8) epoch 5.5 min on RTX 4090, 7h 40min on Mali G715; 'we did not modify any existing llama.cpp public APIs'; not merged upstream.
- **williams2023calibration**: Miles Williams, Nikolaos Aletras (2023). On the Impact of Calibration Data in Post-training Quantization and Pruning. ACL 2024. https://arxiv.org/abs/2311.09755
  - evidence: Abstract: 'we find substantial variations in downstream task performance, contrasting existing work that suggests a greater level of robustness to the calibration data.' Body: GPTQ, SpQR, SparseGPT, Wanda on LLaMA/Vicuna/OPT with C4, CNN-DM, RedPajama, RefinedWeb, Wikipedia; 'quantization methods generally exhibit lower levels of dispersion than pruning methods'; recommends releasing calibration data and evaluating across several calibration sets.
- **giagnorio2025quantcode**: Alessandro Giagnorio, Antonio Mastropaolo, Saima Afrin, Massimiliano Di Penta, Gabriele Bavota (2025). Evaluating the Impact of Post-Training Quantization on Large Language Models for Code Generation. arXiv:2503.07103. https://arxiv.org/abs/2503.07103
  - evidence: Abstract: 'the new frontier for LLM quantization is 4-bit precision, resulting in an average memory footprint reduction of 70% ... without observing any significant decrease in performance. Additionally, when the quantization becomes even more extreme (3 and 2 bits), a code-specific calibration dataset helps to limit the loss of performance.' Body: CodeLlama and DeepSeek-Coder (1B-34B), AQLM, MultiPL-E and McEval; larger models more resilient to extreme quantisation.
- **ji2024bewarecalibration**: Yixin Ji, Yang Xiang, Juntao Li, Qingrong Xia, Ping Li, Xinyu Duan, Zhefeng Wang, Min Zhang (2024). Beware of Calibration Data for Pruning Large Language Models. ICLR 2025. https://arxiv.org/abs/2410.17711
  - evidence: Abstract: 'calibration data is also crucial to post-training pruning, especially for high sparsity.' Body: calibration data similar to training data performs best; self-generated calibration data improves commonsense reasoning by 0.8-2.6% at 60% sparsity.
- **williams2024selfcalibration**: Miles Williams, George Chrysostomou, Nikolaos Aletras (2024). Self-calibration for Language Model Quantization and Pruning. NAACL 2025. https://arxiv.org/abs/2410.17170
  - evidence: Abstract: 'unrepresentative calibration examples can harm model performance'; 'leveraging the model itself to generate synthetic calibration data'; 'consistently competitive in maximizing downstream task performance, frequently outperforming even using real data'.
- **jin2024quanteval**: Renren Jin, Jiangcun Du, Wuwei Huang, Wei Liu, Jian Luan, Bin Wang, Deyi Xiong (2024). A Comprehensive Evaluation of Quantization Strategies for Large Language Models. Findings of ACL 2024. https://arxiv.org/abs/2402.16775
  - evidence: Abstract: 'LLMs with 4-bit quantization can retain performance comparable to their non-quantized counterparts, and perplexity can serve as a proxy metric for quantized LLMs on most benchmarks'; quantisation 'can also slow down the inference speed' without engineering and hardware support.
- **kurtic2024bf16**: Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh (2024). "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization. ACL 2025. https://arxiv.org/abs/2411.02355
  - evidence: Abstract: over 500,000 evaluations on the Llama-3.1 family; 'FP8 (W8A8-FP) is effectively lossless', 'well-tuned INT8 ... achieves surprisingly low (1-3%) accuracy degradation', 'INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization'.
- **kurt2026whichquant**: Uygar Kurt (2026). Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct. arXiv:2601.14277. https://arxiv.org/abs/2601.14277
  - evidence: Tables: perplexity F16 7.32, Q8_0 7.33, Q6_K 7.35, Q5_K_M 7.40, Q4_K_M 7.56, Q3_K_M 7.96, Q3_K_S 8.96; five-benchmark average F16 69.47, Q5_K_M 69.36, Q4_K_M 69.15, Q3_K_M 68.07, Q3_K_S 65.49; decode tg128 F16 2.83 vs Q4_K_M 5.12 tok/s, prefill pp512 79.57 vs 87.70 tok/s on dual Xeon Platinum 8488C. No importance matrix is mentioned.
- **qamar2026divergence**: Shahzeb Qamar, Lorenz Sparrenberg, Christian Bauckhage, Baha Rababah, Carson Leung, Murat Kantarcioglu, Cuneyt Gurcan Akcora, Rafet Sifa (2026). Accuracy is Not Enough: A Divergence-Based Approach to Evaluate Fidelity Loss in Quantized LLMs. arXiv:2609.07664. https://arxiv.org/abs/2609.07664
  - evidence: Abstract: 'accuracy can exhibit unstable, non-monotonic behavior under progressive quantization, masking substantial fidelity loss'; uses Jensen-Shannon divergence and total variation distance from BF16 down to Q2_K; 'mixed-precision Q4_K generally yields lower divergence than uniform Q4_0 at similar memory footprints'; findings 'do not directly establish correctness, calibration, safety, or user-perceived quality'.
- **orr2025optimalformats**: Douglas Orr, Luka Ribar, Carlo Luschi (2025). Optimal Formats for Weight Quantisation. arXiv:2505.12988. https://arxiv.org/abs/2505.12988
  - evidence: Abstract: 'We frame the problem as minimising the KL divergence between original and quantised model outputs under a model size constraint, which can be approximated by minimising the squared quantisation error'; 'by using the relationship between the Fisher information and KL divergence, we derive the optimal allocation of bit-widths ... saving up to 0.25 bits per parameter'.
- **frantar2023sparsegpt**: Elias Frantar, Dan Alistarh (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML 2023 (PMLR 202, pp. 10323-10337). https://arxiv.org/abs/2301.00774
  - evidence: Abstract: 'can be pruned to at least 50% sparsity in one-shot, without any retraining, at minimal loss of accuracy'; 'compatible with weight quantization approaches'. Body: objective argmin ||W X - (M o W_hat) X||_2^2, OBS error w_m^2/[H^-1]_mm, H = X X^T; calibration 128 2048-token C4 segments. Venue confirmed at proceedings.mlr.press/v202/frantar23a.html.
- **sun2023wanda**: Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter (2023). A Simple and Effective Pruning Approach for Large Language Models. ICLR 2024. https://arxiv.org/abs/2306.11695
  - evidence: Metric 'S_ij = |W_ij| * ||X_j||_2' compared per output; pruned LLM 'can be used as is'; LLaMA-7B at 50% sparsity: magnitude 17.29, SparseGPT 7.22, Wanda 7.26 perplexity; '128 sequences ... sampled from C4 training set'; metric computation '300 times faster' than SparseGPT. arXiv comments: 'ICLR 2024'.
- **ma2023llmpruner**: Xinyin Ma, Gongfan Fang, Xinchao Wang (2023). LLM-Pruner: On the Structural Pruning of Large Language Models. NeurIPS 2023. https://arxiv.org/abs/2305.11627
  - evidence: Abstract: 'adopts structural pruning that selectively removes non-critical coupled structures based on gradient information ... performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data'. Body: removing 20% of LLaMA-7B parameters retains 94.97% of performance after tuning.
- **jaiswal2023compressing**: Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang, Zhangyang Wang, Yinfei Yang (2023). Compressing LLMs: The Truth is Rarely Pure and Never Simple. ICLR 2024. https://arxiv.org/abs/2310.01382
  - evidence: Abstract: 'all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%), and fail for N:M sparsity in knowledge-intensive tasks; current quantization methods are more successful than pruning; yet, pruned LLMs even at >=50% sparsity are robust in-context retrieval and summarization systems'.
- **jiang2023llmlingua**: Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023. https://arxiv.org/abs/2310.05736
  - evidence: Abstract: 'a coarse-to-fine prompt compression method that involves a budget controller ... a token-level iterative compression algorithm'; 'allows for up to 20x compression with little performance loss'.
- **pan2024llmlingua2**: Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Ruhle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. Findings of ACL 2024. https://arxiv.org/abs/2403.12968
  - evidence: Abstract: 'We formulate prompt compression as a token classification problem'; 'our model is 3x-6x faster than existing prompt compression methods, while accelerating the end-to-end latency by 1.6x-2.9x with compression ratios of 2x-5x'. Body: p(x_i, Theta) = softmax(W h_i + b); keep top N~ = tau N tokens.
- **chirkova2025provence**: Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stephane Clinchant (2025). Provence: efficient and robust context pruning for retrieval-augmented generation. ICLR 2025. https://arxiv.org/abs/2501.16214
  - evidence: Abstract: 'dynamically detects the needed amount of pruning for a given context'; key ingredients 'formulating the context pruning task as sequence labeling, unifying context pruning capabilities with context reranking, and training on diverse data'. Body: threshold T (0.1, 0.5) on token keep probabilities with sentence rounding; pruning with negligible to no performance drop.
- **mcmahan2016fedavg**: H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas (2016). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. https://arxiv.org/abs/1602.05629
  - evidence: Abstract: 'leaves the training data distributed on the mobile devices, and learns a shared model by aggregating locally-computed updates. We term this decentralized approach Federated Learning.'
- **bonawitz2019flscale**: Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konecny, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander (2019). Towards Federated Learning at Scale: System Design. arXiv:1902.01046. https://arxiv.org/abs/1902.01046
  - evidence: Sec. 3: 'the job scheduler only invoke the job when the phone is idle, charging, and connected to an unmetered network such as WiFi'; 'The FL runtime will abort, freeing the allocated resources, if these conditions are no longer met'; 'avoid any negative impact on the user experience, data usage, or battery life'; example stores should 'automatically remove old data after a pre-designated expiration time'.
- **hard2018gboard**: Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy, Francoise Beaufays, Sean Augenstein, Hubert Eichner, Chloe Kiddon, Daniel Ramage (2018). Federated Learning for Mobile Keyboard Prediction. arXiv:1811.03604. https://arxiv.org/abs/1811.03604
  - evidence: 'the devices must have at least 2 gigabytes of memory available. Additionally, the clients are only allowed to participate if they are charging, connected to an un-metered network, and idle.' Training data 'are stored in local caches on Gboard client devices'.
- **microsoft2018taskidle**: Microsoft (2018). Task idle conditions (Win32 Task Scheduler). Microsoft Learn documentation (updated 2025). https://learn.microsoft.com/en-us/windows/win32/taskschd/task-idle-conditions
  - evidence: 'By default, the user is considered absent after four minutes of no keyboard or mouse input'; 'In Windows 8, the CPU and IO thresholds are set to 80%'; DisallowStartIfOnBatteries true by default; 'To terminate the task when the idle condition ends, set the StopOnIdleEnd property'; RestartOnIdle restarts the task when the computer becomes idle again.
- **geng2025mobilefinetuner**: Jiaxiang Geng, Lunyu Zhao, Yiyi Lu, Bing Luo (2025). MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications. arXiv:2512.08211. https://arxiv.org/abs/2512.08211
  - evidence: C++ framework with parameter sharding, activation checkpointing, gradient accumulation, memory-efficient attention and an energy-aware scheduler: 'When the battery percentage drops below a user-defined threshold, the computation frequency is dynamically adjusted'. Models GPT-2, Gemma 3 270M/1B, Qwen2.5-0.5B; supports full and LoRA fine-tuning; reported Qwen2.5-0.5B peak RSS 2887 MB.
- **dbouk2026peakmemory**: Hassan Dbouk, Matthias Reisser, Prathamesh Mandke, Likhita Arun Navali, Christos Louizos (2026). Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs on Edge Devices. arXiv:2606.19528. https://arxiv.org/abs/2606.19528
  - evidence: Abstract: four techniques (quantised base with on-the-fly dequantisation, memory-efficient checkpointing with disk offloading, softmax approximation, logits masking) give 'up to 26x and 28x reduction in peak memory' on Llama-3.2 3B and Qwen-2.5 3B. Body: Llama-3.2 3B at 2048 tokens 26.20 GB to 1.02 GB.
- **kirkpatrick2016ewc**: James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell (2016). Overcoming catastrophic forgetting in neural networks. PNAS (arXiv:1612.00796). https://arxiv.org/abs/1612.00796
  - evidence: Abstract: 'Our approach remembers old tasks by selectively slowing down learning on the weights important for those tasks.' Loss L(theta) = L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_A,i)^2 with F 'the diagonal of the Fisher information matrix'.
- **luo2023cfstudy**: Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang (2023). An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning. arXiv:2308.08747. https://arxiv.org/abs/2308.08747
  - evidence: Abstract: 'catastrophic forgetting is generally observed in LLMs ranging from 1b to 7b parameters'; 'general instruction tuning can help alleviate the forgetting phenomenon in LLMs during subsequent fine-tuning'.
- **feng2026forever**: Yujie Feng, Hao Wang, Jian Li, Xu Chu, Zhaolu Kang, Yiran Liu, Yasha Wang, Philip S. Yu, Xiao-Ming Wu (2026). FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning. ACL 2026. https://arxiv.org/abs/2601.03938
  - evidence: Abstract: 'defines model time using the magnitude of optimizer updates, allowing forgetting curve-inspired replay intervals to align with the model's internal evolution'; experiments on 'models ranging from 0.6B to 13B parameters'.
- **baylor2017tfx**: Denis Baylor, Eric Breck, Heng-Tze Cheng, Noah Fiedel, Chuan Yu Foo, Zakaria Haque, Salem Haykal, Mustafa Ispir, Vihan Jain, Levent Koc, Chiu Yuen Koo, Lukasz Lew, Clemens Mewald, Akshay Naresh Modi, Neoklis Polyzotis, Sukriti Ramesh, Sudip Roy, Steven Euijong Whang, Martin Wicke, Jarek Wilkiewicz, Xin Zhang, Martin Zinkevich (2017). TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. KDD 2017 (Applied Data Science). https://research.google/pubs/tfx-a-tensorflow-based-production-scale-machine-learning-platform/
  - evidence: Sec. 5 (read from PDF text): 'We suggest two pieces: that a model is safe to serve, and that it has the desired prediction quality.' Sec. 5.3: 'We evaluate prediction quality by comparing the model quality against a fixed threshold as well as against a baseline model (e.g., the current production model). Any new model failing any of these checks is not pushed to serving'. Also: 'most bugs cause dramatic changes to model quality metrics that can be caught by using loose thresholds'.
- **miller2024errorbars**: Evan Miller (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640. https://arxiv.org/abs/2411.00640
  - evidence: Abstract: 'we present formulas for analyzing evaluation data, measuring differences between two models, and planning an evaluation experiment'. Formulas: SE_CLT = sqrt(Var(s)/n); SE_{A-B,paired} = sqrt(Var(s_{A-B})/n); CI_95% = mean +/- 1.96 SE; recommends paired-difference tests, clustered standard errors and power analysis.
