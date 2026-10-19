# Context rot and recursive mitigation

Verified research note for programmersRAG. It covers feature 1 (removing "context rot" with recursive metrics) and supplies the pruning rule that feature 5 passes to the fine-tuned, quantised local model. The evaluation corpus is the demo repository, a clone/fork of microsoft/mimalloc (C).

- Verification date: 2026-09-30. I opened every source listed under References myself: the arXiv abstract page and full PDF text, the ACL Anthology, OpenReview, NeurIPS, AAAI, Springer or ICLR pages, or the web page itself. I checked each numeric claim against the source text. Formulas were checked against the typeset equations (arXiv HTML `alttext`) where available.
- Citation keys `[n]` refer to the numbered References at the end. Machine-readable metadata and BibTeX are in `context_rot.sources.json`.
- Labels: **Established** = peer-reviewed or classic result. **Recent** = 2024-2026, often a preprint (the venue is stated). **Proposal** = our own definition or derivation. Proposals are NOT published results and must be presented as ours in the paper.

---

## Summary

1. "Context rot" is an empirical phenomenon, not a defined metric. As input length grows, LLM accuracy degrades even on simple tasks, across 18 frontier models [1]. Degradation continues when retrieval is perfect and the filler is whitespace or masked [7]. It also appears in literal-match-free retrieval [8], agentic search [11], long-transcript monitoring [12] and a code-audit agent workflow [14]. No opened source defines a single scalar "context-rot" quantity. The closest operational measures are NoLiMa's normalized score and effective length [8] (building on RULER's effective length [9]), Xia et al.'s premature-termination rate [11] and miss-rate inflation [12].
2. The mechanisms that matter for RAG over code are: (a) length itself [7]; (b) *distractors*, meaning near-miss, topically related text, which hurt more than random text [1][4][6]; (c) position [2], which one controlled test found to be much weaker than noise [10]; and (d) redundancy, where duplicates do not help but diverse evidence does [53]. So a mitigation must reduce length, remove near-misses and redundancy, and only then re-order.
3. The published mitigations are compression and pruning [25]-[31], code-specific structure-aware pruning [36]-[41], thresholded or statistically calibrated pruning [30][32], hierarchical and recursive indexes [21][22], and fixed-point graph relevance [50][51]. LLM-driven recursion (Recursive Language Models [18]) helps far beyond the context window. However, follow-ups find it can hurt *within* the window and at greater depth [19][20]. For a 4 GB-GPU local model this favours a deterministic recursive metric over recursive LLM calls.
4. Our proposal is **Recursive Context Quality (RCQ)**. It is a density metric defined bottom-up over token → block → function → file → context and discounted for redundancy. It is combined with (i) personalised-PageRank-style relevance propagation over the call/include graph with a proven contraction bound, (ii) a threshold-pruning rule that provably guarantees $Q(C')\ge\theta$ for the proxy metric, (iii) a terminating prune-and-rescore fixed point, and (iv) a decision-theoretic reading of the owner's 25% rule. These are our derivations. They were re-checked here (algebra plus numerical tests) but have no link to accuracy yet, and that link must be shown experimentally on the mimalloc demo repository.

---

## Established results

**E1. Context rot (definition and scope) [1].** The Chroma technical report (Hong, Troynikov, Huber, 14 July 2025) evaluates 18 LLMs, including GPT-4.1, Claude 4, Gemini 2.5 and Qwen3. It reports that "models do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows." Each needle-haystack configuration is tested over 8 input lengths × 11 needle positions. The repeated-words task is scored with normalized Levenshtein distance. The report defines no scalar metric. Its conclusion is that how relevant information is presented matters, not only whether it is present. It is a vendor technical report and is not peer-reviewed.

**E2. Distractors and structure [1].** "Even a single distractor reduces performance relative to the baseline (needle only), and adding four distractors compounds this degradation further." Across all 18 models, performance was better on *shuffled* haystacks than on logically structured ones. On LongMemEval, focused prompts (about 300 tokens) beat full prompts (about 113k tokens) across the model families tested.

**E3. Position (lost in the middle) [2].** Accuracy is U-shaped in the position of the relevant document. In the worst case for GPT-3.5-Turbo with 20 or 30 documents, it falls below closed-book accuracy (56.1%). Extended-context variants were not better when the input fit in both windows. *Metadata correction:* this is TACL vol. 12 (2024), pp. 157-173, not 2023.

**E4. Irrelevant and near-miss context [3][4][5][6].**
- GSM-IC [3]: accuracy drops sharply when an irrelevant sentence is added. Fewer than 30% of base problems are solved consistently across distractor variants (macro accuracy).
- Cuconasu et al. [4]: "distracting" documents (high retriever score, no answer) hurt the LLM.
- Jin et al. [6]: with a strong retriever, answer quality follows an inverted U as the number of passages grows. Retrieved "hard negatives" are "a key contributor". The paper proposes retrieval reordering (high scores at both ends), implicit robustness fine-tuning and reasoning-augmented fine-tuning.
- Yoran et al. [5] (ICLR 2024): retrieval can reduce accuracy. Fine-tuning on a mix of relevant and irrelevant contexts makes the model robust, and "even 1,000 examples suffice". This is directly relevant to feature (2), the fine-tuning pipeline.

**E5. Length alone hurts [7][8][9].**
- Du et al. [7] (Findings of EMNLP 2025) test 5 models on math, QA and code. Even when models can perfectly retrieve all evidence, performance drops by 13.9%-85% as length grows. The drop persists when the filler is whitespace and when irrelevant tokens are masked.
- NoLiMa [8] (ICML 2025) removes literal overlap between question and needle. At 32K tokens, 11 of 13 models score half or less of their short-context base score (GPT-4o: 99.3% → 69.7%).
- RULER [9] (COLM 2024) introduced the "effective length vs claimed length" methodology that NoLiMa adapts.
- Consequence: a context-quality proxy must penalise length itself, not only irrelevance.

**E6. Recursive / hierarchical organisation [21][22][23].**
- RAPTOR [21] (ICLR 2024) recursively embeds, clusters and summarises. It uses GMM soft clustering after UMAP, and BIC selects the number of clusters. The "collapsed tree" (all layers at once, 2000-token budget, about top-20 nodes) beat layer-by-layer traversal. Summaries are about 0.28× the length of their children (72% compression), and 18.5%-57% of retrieved nodes come from non-leaf layers.
- GraphRAG [22] builds hierarchical Leiden communities and creates bottom-up summaries in which "higher levels ... recursively incorporat[e] lower-level summaries".
- Recursive dialogue summarisation [23] is a Markov update $P(M_i\mid S_i,M_{i-1})$.

**E7. Compression and pruning for natural-language RAG [25]-[31].**
- LLMLingua [25] (EMNLP 2023) works coarse to fine. Its iterative token-level stage scores each segment conditioned on the already-compressed previous segments and keeps tokens whose perplexity exceeds a segment threshold. It reports up to 20× compression with little loss.
- LongLLMLingua [26] (ACL 2024) makes compression question-aware and adds document reordering against position bias. It reports up to +21.4% on NaturalQuestions with about 4× fewer tokens (GPT-3.5-Turbo).
- LLMLingua-2 [27] argues entropy is a suboptimal compression metric and recasts compression as bidirectional token classification.
- Selective Context [28] filters lexical units by summed self-information with a percentile rule. It reports 50% context reduction, 36% less memory and 32% less inference time.
- RECOMP [29] (ICLR 2024) compressors may return an empty string (selective augmentation), with compression rate as low as 6%.
- Information-bottleneck filtering [31] (ACL 2024) reports a 2.5% compression rate.

**E8. Thresholds instead of fixed ratios [30][32][33].**
- Provence [30] (ICLR 2025) casts pruning as sequence labelling unified with reranking. Because the optimal compression ratio is specific to each question-context pair, "using a threshold as a hyperparameter is more appropriate". With T ∈ {0.1, 0.5} the pruning ratio varies automatically between 50% and 80% depending on the dataset.
- Chakraborty et al. [32] (ECIR 2026) set the threshold by split conformal prediction so that a relevant snippet is retained with probability at least $1-\alpha$. Filtering met target coverage and cut retained context 2-3×. On NeuCLIR, ARGUE F1 improved under strict filtering.
- The finite-sample quantile correction $\lceil (n+1)(1-\alpha)\rceil$ is standard [33].
- Together these are the published basis for a *calibrated* probability threshold such as the owner's 25% rule.

**E9. Code-specific pruning is hierarchical and structure-aware [36][37][41].**
- LongCodeZip [36] (ASE 2025) finds token-level compressors (LLMLingua, LLMLingua-2) perform poorly on RepoQA because "token-level compression corrupts code syntax and structure". It ranks function chunks by approximated mutual information (conditional perplexity reduction), then chooses blocks inside functions by 0/1 knapsack. It reaches up to 5.6× compression without degrading task performance. In ablation, conditional-perplexity ranking beat similarity ranking by 7.89 ES points.
- HCP [37] (AAAI 2025) keeps file-level topological dependencies but prunes function bodies of dependent files without a significant accuracy loss, going from over 50,000 to about 8,000 tokens.
- cAST [41] (Findings of EMNLP 2025) chunks by recursive AST split-then-merge and measures size in non-whitespace characters. It reports +4.3 Recall@5 on RepoEval and +2.67 Pass@1 on SWE-bench.

**E10. RAG context-quality metrics are flat ratios [42][43][44][45].**
- RAGAS context relevance [42] is CR = extracted sentences / total sentences ("aims to penalise the inclusion of redundant information").
- Current Ragas docs [43] define rank-weighted Context Precision@K and claim-level Context Recall.
- RAGChecker [44] (NeurIPS 2024 D&B) defines claim recall, context precision $|\{\text{r-chunk}\}|/k$, context utilization, and relevant and irrelevant noise sensitivity. Raising k from 5 to 20 raised claim recall 61.5 → 77.6 and faithfulness 88.1 → 92.2, but noise sensitivity also rose 34.0 → 35.4.
- Sufficient context [45] (ICLR 2025): an instance has sufficient context iff a plausible answer exists given the context.
- None of these is defined recursively across granularities or used as the pruning operator.

**E11. Diversity-aware selection [46]-[49].**
- MMR [46][47] greedily maximises $\lambda\,\mathrm{Sim}_1-(1-\lambda)\max\mathrm{Sim}_2$.
- Lin & Bilmes [48] prove the MMR-induced set function is submodular but non-monotone, so the greedy constant-factor guarantee does not apply. They propose a monotone submodular coverage-plus-diversity objective. They cite $(1-1/e)$ for greedy under a cardinality constraint (Nemhauser et al.), $(1-1/e)$ with partial enumeration under a knapsack constraint (Sviridenko) and $(1-1/\sqrt e)$ for their practical cost-scaled greedy.
- DPPs [49] trade quality against similarity via $\det(L_Y)$. Exact mode finding is NP-hard (Theorem 2.9).

**E12. Recursively defined scores as fixed points [50][51][52].**
- LexRank [50] (JAIR 2004) defines salience through neighbours' salience. By Perron-Frobenius, an irreducible aperiodic chain has a unique stationary distribution, and the power method finds it.
- HippoRAG [51] (NeurIPS 2024) runs Personalized PageRank seeded at query concepts. Its "damping factor" of 0.5 is the *restart* probability. It reports up to 20% improvement, and single-step retrieval that is 10-30× cheaper (abstract) and 6-13× faster than iterative retrieval such as IRCoT.
- Banach's contraction principle [52]: a contraction on a complete metric space has a unique fixed point, reached from any start.

**E13. Cost-sensitive decisions and calibration [34][35].** The Bayes-optimal decision threshold on a class probability is set by the misclassification cost matrix [35]. Modern neural networks are often mis-calibrated, and temperature scaling is a simple, effective post-hoc fix [34]. A probability threshold such as 25% is only meaningful on calibrated probabilities.

---

## Recent work (2024-2026)

**R1. Noise dominates position in a controlled test [10]** (arXiv:2605.00505; arXiv comment: SIGIR 2026). LLaMA-2-7B-Chat, 500 NQ questions:
- EM rises from 47.0% to 61.0% as gold passages go from 1 to 5.
- With 3 gold passages, adding 2 and 7 noise passages gives 51.4% and 41.8%.
- 1 gold among 9 noise gives 26.6% (closed book: 23.6%). Pure noise gives 8.0%.
- 1 gold alone gives 43.2%, but 1 gold plus 9 noise gives 22.2-26.6% "regardless of position". The mild U-shape is "dwarfed by the noise effect".
- This supports pruning first and ordering second.

**R2. Is unrelated context harmless? (CONTESTED) [4][10][1].** Cuconasu et al. report that *random* documents improved accuracy by up to 35% in their setting (a specific prompt layout, with the gold document near the query). Dai et al. find pure-noise contexts far below closed book, and Chroma finds irrelevant filler degrades performance with length. The effect is model-, task- and layout-dependent and must be measured for the local model.

**R3. Quantitative context-rot measures (no agreed scalar).**
- NoLiMa normalized score and effective length [8] (the RULER lineage [9]).
- Premature-termination rate (share of give-up plus uncertain-incorrect outcomes), which is positively correlated with context length when query difficulty is controlled [11].
- Monitor miss-rate inflation: Opus 4.6, GPT 5.4 and Gemini 3.1 miss dangerous actions 2×-30× more often after 800K tokens of benign activity. On MonitorBench, Opus 4.6 recall falls from 98.6% to 88% [12] (preprint).
- A "critical threshold" at 40-50% of the maximum context for Qwen2.5-7B, where F1 falls from about 0.55 to 0.3, with degradation defined as a drop of more than 30% [13]. This is an unreviewed preprint; only its abstract-level claims should be used.
- Code auditing [14] (single-author preprint, n = 10 runs per condition): pass rate 8/10 in an 11k-character clean context versus 3/10 in 299k-character contexts. The long contexts are either relevant or irrelevant, and the drop is the same for both. The difference is only trend-level (Fisher p = 0.0698), a second task showed no degradation, and the author explicitly finds no universal length threshold.

**R4. Vendor guidance [16] and survey [17].** Anthropic (29 Sep 2025) describes context rot, says that "context ... must be treated as a finite resource with diminishing marginal returns", ties this to an "attention budget" ($n^2$ pairwise relationships for $n$ tokens), and recommends "the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome". The recommended techniques are compaction, structured note-taking, sub-agents and just-in-time retrieval. Mei et al. [17] survey over 1400 papers on context engineering.

**R5. Recursive Language Models [18]** (arXiv:2512.24601, v3 May 2026). The prompt lives in a REPL variable, and the model writes code that inspects it and recursively sub-calls a model on snippets. With GPT-5 (sub-calls to GPT-5-mini), RLM depth 1 vs base: CodeQA 62.0 vs 24.0, BrowseComp+ (1K) 91.3 vs 0.0, OOLONG 56.0 vs 44.0, OOLONG-Pairs 58.0 vs 0.1. RLM-Qwen3-8B improves over Qwen3-8B by a **median** of 28.3% across the four tasks. *Correction: the dossier said "on average".*

**R6. Does recursion help, and how deep? (CONTESTED) [18][19][20].**
- In [18] Table 1, depth effects are model-dependent. With Qwen3-Coder-480B, depth 0 (no sub-calls) is best on CodeQA: 66.0 vs 56.0/54.0/44.0 at depths 1/2/3. The authors attribute this to syntax errors that propagate into sub-calls. With GPT-5, deeper recursion helped on CodeQA (depth 2: 66.0 > depth 1: 62.0) and on OOLONG-Pairs (58.0 → 65.5 → 76.0 at depths 1/2/3). *Correction: the dossier presented only the negative side.*
- Wang [19] (single-author reproduction, no significance tests): on S-NIAH with DeepSeek v3.2, accuracy is 100% base, 85% at depth 1 and 70% at depth 2, and latency grows from 3.6 s to 344.5 s.
- Alizadeh et al. [20] report up to 22% improvement over RLM at an equal time budget. They find "recursion itself is not the primary driver of performance", and within the native window recursive RLMs "often degrade performance relative to the base model".

**R7. Theory of compaction [15] and context management [11].**
- Tirmazi et al. [15] formalise selection-based compaction (choose a subset under a size budget; granularity-agnostic) and generation-based compaction. The generation game is equivalent to one-way randomized communication complexity. They exhibit a query family on which selection needs a Θ(log n) factor more budget than generation. Purely extractive pruning therefore has provable limits.
- Xia et al. [11] show that context-management methods act as test-time scaling that lowers premature termination, and that the best method is model-dependent. Sub-agent isolation suits strongly agentic backbones; compaction plus trimming suits weaker ones.

**R8. Code-specific pruning, 2026 [38][39][40].**
- LaMR [38] shows that support lines, such as imports and scope lines, have *conditional* value, so a single unconditional relevance score cannot represent code relevance (Proposition 1). It reports up to 31% more token savings on multi-turn agent tasks. **Our check found a flaw in its Proposition 2** (see Formulas F18).
- SWEzze [39] distils a one-minimal "minimal sufficient context". It reports about 6× compression, a 51.8%-71.3% total-token reduction and +5.0 to +9.2 points in issue resolution on SWE-bench Verified.
- SWE-Pruner [40] trains a 0.6B line-level "skimmer" guided by an agent-stated goal. It reports 23-54% token reduction on SWE-bench Verified with improved success, and up to 14.84× compression on LongCodeQA.

**R9. Redundancy [53]** (preprint). On FictionalQA, duplicates and LLM paraphrases of the same document did not significantly improve correctness, whereas diverse documents improved it by 17%-47%. This supports *discounting* near-duplicate functions.

**R10. Weakly evidenced: SVD-RAG [24]** (single-author preprint). It replaces LLM summaries in a RAPTOR-style tree with SVD-based extractive selection. It was tested only on corpora of 38 and 205 chunks, and its summaries were 1.7-1.9× *longer* than the originals on these small corpora. It is not a reliable basis for design.

---

## Formulas

Each formula gives its source. Items marked **[OUR PROPOSAL]** are our own. Corrections relative to the dossier are marked *Correction*.

**F1. RAGAS context relevance [42].**
$$\mathrm{CR}=\frac{\text{number of extracted sentences}}{\text{total number of sentences in }c(q)}$$

**F2. Ragas Context Precision@K and Context Recall [43]** (current docs; $v_k\in\{0,1\}$ is the relevance indicator at rank $k$).
$$\text{Context Precision@K}=\frac{\sum_{k=1}^{K}\left(\text{Precision@k}\times v_k\right)}{\text{Total number of relevant items in the top }K\text{ results}},\qquad \text{Precision@k}=\frac{\mathrm{TP@k}}{\mathrm{TP@k}+\mathrm{FP@k}}$$
$$\text{Context Recall}=\frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}$$

**F3. RAGChecker diagnostics [44]** (checked against the typeset equations). $c^{(gt)}$ are ground-truth claims, $c^{(m)}$ are response claims, $m$ is the response, $k$ is the number of retrieved chunks, and an r-chunk entails at least one ground-truth claim.
$$\text{Claim Recall}=\frac{|\{c^{(gt)}_i\mid c^{(gt)}_i\in\{\text{chunk}_j\}\}|}{|\{c^{(gt)}_i\}|},\qquad \text{Context Precision}=\frac{|\{\text{r-chunk}_j\}|}{k}$$
$$\text{Context Utilization}=\frac{|\{c^{(gt)}_i\mid c^{(gt)}_i\in\{\text{chunk}_j\}\text{ and }c^{(gt)}_i\in m\}|}{|\{c^{(gt)}_i\mid c^{(gt)}_i\in\{\text{chunk}_j\}\}|},\qquad \text{Relevant NS}=\frac{|\{c^{(m)}_i\mid c^{(m)}_i\notin gt\text{ and }c^{(m)}_i\in\{\text{r-chunk}_j\}\}|}{|\{c^{(m)}_i\}|}$$
Irrelevant NS is the same with irr-chunk in place of r-chunk.

**F4. MMR [46][47].**
$$\mathrm{MMR}=\arg\max_{D_i\in R\setminus S}\Bigl[\lambda\,\mathrm{Sim}_1(D_i,Q)-(1-\lambda)\max_{D_j\in S}\mathrm{Sim}_2(D_i,D_j)\Bigr]$$
$\lambda=1$ gives the relevance ranking and $\lambda=0$ a maximal-diversity ranking [47]. The induced set function is non-monotone submodular [48, Thm 2].

**F5. Submodularity and budgeted selection [48].**
$$F(A\cup\{v\})-F(A)\ge F(B\cup\{v\})-F(B)\quad\forall A\subseteq B\subseteq V\setminus\{v\};\qquad \max_{S\subseteq V}F(S)\ \text{s.t.}\ \sum_{i\in S}c_i\le b$$

**F6. Lin-Bilmes objective [48]** ($w_{ij}\ge0$, $0\le\alpha\le1$, partition $P_1..P_K$, $r_j\ge0$).
$$F(S)=L(S)+\lambda R(S),\quad L(S)=\sum_{i\in V}\min\{C_i(S),\,\alpha\,C_i(V)\},\ C_i(S)=\sum_{j\in S}w_{ij},\quad R(S)=\sum_{k=1}^{K}\sqrt{\sum_{j\in P_k\cap S}r_j},\quad L_{\mathrm{FL}}(S)=\sum_{i\in V}\max_{j\in S}w_{ij}$$

**F7. DPP with quality-diversity decomposition [49].**
$$\mathcal{P}_L(\boldsymbol Y=Y)=\frac{\det(L_Y)}{\det(L+I)},\qquad L_{ij}=q_i\,\phi_i^{\top}\phi_j\,q_j,\qquad S_{ij}=\phi_i^\top\phi_j=\frac{L_{ij}}{\sqrt{L_{ii}L_{jj}}}\ \ (\|\phi_i\|=1)$$
$\det(L_Y)=\bigl(\prod_{i\in Y}q_i^2\bigr)\det(S_Y)$ is the squared volume spanned by the quality-scaled vectors. MAP/mode finding is NP-hard [49, Thm 2.9].

**F8. LexRank / PageRank [50].** Here $d$ is the *jump* probability, "typically chosen in the interval [0.1, 0.2]" in the paper's convention.
$$p(u)=\frac{d}{N}+(1-d)\sum_{v\in\mathrm{adj}[u]}\frac{p(v)}{\deg(v)},\qquad \mathbf p=\bigl[d\,\mathbf U+(1-d)\,\mathbf B\bigr]^{\top}\mathbf p$$
Note that the paper's own Table 2 example uses "damping factor 0.85", which follows the opposite convention. State the convention explicitly in our paper.

**F9. LLMLingua [25]** (checked against the typeset equations).
$$\min_{\tilde{\boldsymbol x},\tau}\mathrm{KL}\bigl(P(\tilde{\boldsymbol x}_G\mid\tilde{\boldsymbol x}),P(\boldsymbol x_G\mid\boldsymbol x)\bigr),\quad \tau=\tilde L/L,\qquad p(\tilde{\boldsymbol s}_j)\approx\prod_{i}p\bigl(s_{j,i}\mid s_{j,<i},\tilde{\boldsymbol s}_{<j}\bigr),\qquad \tilde{\boldsymbol s}_j=\{s_{j,i}:\ \mathrm{PPL}(s_{j,i})>\gamma_j\}$$
*Correction:* the dossier wrote the retention rule as $p(s_{j,i})>\gamma_j$. The paper keeps tokens whose (conditional) **perplexity** exceeds $\gamma_j$, i.e. high-surprisal tokens, not high-probability ones.

**F10. Selective Context self-information [28].**
$$I(x_i)=-\log_2P(x_i\mid x_0,\dots,x_{i-1}),\qquad I(u)=\sum_{i=t}^{t+\alpha}I(x_i)\ \ \text{for } u=(x_t,\dots,x_{t+\alpha}),\qquad C'=\{U_i\mid I(U_i)\ge I_p\}$$
*Correction:* the paper (and the dossier) prints the upper summation limit as $\alpha$. It must be $t+\alpha$ for the unit $(x_t,\dots,x_{t+\alpha})$ that the paper defines. $I_p$ is the $p$-th percentile of the unit scores.

**F11. LongCodeZip [36].**
$$\mathrm{AMI}(c,q)=\mathrm{PPL}(q)-\mathrm{PPL}(q\mid c),\quad \mathrm{PPL}(q\mid c)=\exp\Bigl(-\tfrac1N\sum_{i=1}^{N}\log P(q_i\mid q_{<i},c)\Bigr),\quad R_{\mathrm{biased},i}=R_{\mathrm{base}}\bigl(1+\beta(2\,\mathrm{AMI}_{\mathrm{norm},i}-1)\bigr)$$
The rates are clamped to [0,1] and rescaled to the budget, and blocks are then chosen by a 0/1 knapsack.

**F12. Information bottleneck [31].**
$$\min\ \mathcal L_{\mathrm{IB}}=I(\tilde X;X\mid Q)-\beta\,I(\tilde X;Y\mid Q)$$
The first term is conciseness and the second is correctness.

**F13. Provence loss [30].** As typeset in the paper:
$$\mathcal L=\sum_{n=1}^{N}\Bigl\{\sum_{k=1}^{L_n}\log P(y_{n,k}\mid z_{n,k})+\lambda\bigl(s_n-z_{n,0}\bigr)^2\Bigr\}$$
*Correction (sign):* read as a loss to be minimised, this has the wrong sign on the likelihood term. The intended objective is token-level cross-entropy plus an MSE ranking regulariser, $\sum_n\{-\sum_k\log P(y_{n,k}\mid z_{n,k})+\lambda(s_n-z_{n,0})^2\}$. The paper uses $\lambda=0.05$. At inference, per-token keep probabilities are binarised with a threshold $T$ (0.1 or 0.5).

**F14. Split-conformal context filter [32][33].**
$$\hat\tau_\alpha=\mathrm{Quantile}_{1-\alpha}\bigl(\{A(q,s):(q,s)\in\mathcal D_{\mathrm{cal}},\ r(q,s)=1\}\bigr),\qquad K_q=\{s:A(q,s)\le\hat\tau_\alpha\},\qquad P\bigl(s\in K_q\mid r(q,s)=1\bigr)\ge1-\alpha$$
For a finite-sample guarantee, $\hat\tau_\alpha$ is the $\lceil(n+1)(1-\alpha)\rceil$-th smallest of the $n$ relevant calibration scores. The guarantee is marginal (averaged over calibration draws and test pairs), requires exchangeability, and controls recall only, not precision.

**F15. RAPTOR clustering [21].**
$$P(\mathbf x)=\sum_{k=1}^{K}\pi_k\,\mathcal N(\mathbf x;\boldsymbol\mu_k,\boldsymbol\Sigma_k),\qquad \mathrm{BIC}=\ln(N)\,k-2\ln\hat L$$

**F16. Recursive memory update [23].**
$$P(M_N\mid S)=\prod_{i=1}^{N}P(M_i\mid S_i,M_{i-1})$$

**F17. Context Selection Game [15].**
$$S\subseteq\mathcal X,\ \ \sum_{x_i\in S}s(x_i)\le B,\qquad \mathrm{val}(S,q)=v_q(S)\in[0,1],\qquad \mathrm{err}(S,q)=1-\mathrm{val}(S,q)$$

**F18. LaMR dependency-aware utility [38], with a counterexample to Proposition 2.**
$$U(S)=\sum_{i\in S}r_i-\lambda\sum_{i\in S}c_i-M\sum_{(i,j)\in E}\mathbf 1[i\in S,\ j\notin S]$$
The paper claims that if $M>\max_j(r_j-\lambda c_j)_+$, every optimal $S^*$ is dependency-closed. **Correction:** this is false as stated, because adding the missing node $j$ can itself be costly and can create new violations.
- *Counterexample* (checked by exhaustive enumeration): take the chain $a\to b\to c$ with net values $r-\lambda c$ of $100, 100, -1000$ and $M=101>100$. Then $U(\{a,b\})=200-101=99$, while every closed set has $U\le0$ ($\emptyset$: 0, $\{c\}$: -1000, $\{b,c\}$: -900, $\{a,b,c\}$: -800). The optimum $\{a,b\}$ is **not** dependency-closed.
- The paper's proof ignores the negative value of $j$ and the new penalties created by $j$'s own dependencies.
- The conclusion does hold if every node has non-negative net value ($r_j-\lambda c_j\ge0$), because then closing the set never loses value and removes every penalty.
- Proposition 1 (conditional marginal value $M-\lambda$ vs $-\lambda$) is unaffected.
- Design consequence: enforce dependency closure as a hard constraint (our F21 operator). Do not rely on a penalty to produce it.

**F19. NoLiMa measures [8]** (our symbolic rendering of the paper's verbal definitions; base = mean over 250/500/1K contexts).
$$\mathrm{NS}(\ell)=\frac{\mathrm{Acc}(\ell)}{\mathrm{Acc}_{\mathrm{base}}},\qquad \ell_{\mathrm{eff}}=\max\{\ell\ \text{tested}:\ \mathrm{Acc}(\ell)\ge0.85\,\mathrm{Acc}_{\mathrm{base}}\}$$
RULER [9] uses a fixed absolute threshold (Llama2-7B at 4K) instead.

**F20. [OUR PROPOSAL] Recursive Context Quality (RCQ).** For token $t$: $M(t)=r(t)\in[0,1]$ and $L(t)=1$. For an internal node $v$ with children $\mathrm{ch}(v)$ in processing order:
$$M(v)=\sum_{u\in\mathrm{ch}(v)}\bigl(1-\rho_v(u)\bigr)M(u),\qquad L(v)=\sum_{u\in\mathrm{ch}(v)}L(u),\qquad Q(v)=\frac{M(v)}{L(v)}=\sum_{u\in\mathrm{ch}(v)}\frac{L(u)}{L(v)}\bigl(1-\rho_v(u)\bigr)Q(u),\qquad \mathrm{CRI}(C)=1-Q(C)$$
Here $\rho_v(u_j)=\max_{i<j}\max\bigl(0,\cos(e_{u_i},e_{u_j})\bigr)\in[0,1]$ is the redundancy of $u_j$ with respect to earlier kept siblings (the MMR redundancy term [46]).

Properties (re-derived):
- (i) $Q(v)\in[0,1]$ by induction, because $Q(v)$ is a convex combination (weights $L(u)/L(v)$) of values in $[0,1]$ scaled by factors in $[0,1]$.
- (ii) Unrolled: $Q(C)=\frac{1}{L(C)}\sum_{t}r(t)\prod_{a\in\pi(t)}\bigl(1-\rho(a)\bigr)$, where $\pi(t)$ is the path from $t$ **up to but excluding the root, including $t$ itself**. *Correction:* the dossier wrote "ancestors strictly below the root", which omits the token's own redundancy factor.
- (iii) Special cases:
  - With $\rho\equiv0$ and binary $r$, $Q$ is the fraction of relevant leaves.
  - With sentences as unit-length leaves, $Q$ equals RAGAS CR [42].
  - With chunks as unit-length leaves and binary chunk relevance, $Q$ equals RAGChecker context precision [44].
  - With tokens as leaves, $Q$ is token-weighted rather than count-weighted.

CRI is a property of the prompt, not a measured accuracy loss.

**F21. [OUR PROPOSAL] Density criterion and threshold guarantee.**
$$Q(v\setminus u)>Q(v)\iff\bigl(1-\rho_v(u)\bigr)Q(u)<Q(v);\qquad \text{keep }u\iff\bigl(1-\rho_v(u)\bigr)Q(u)\ge\theta\ \Longrightarrow\ Q(v')\ge\theta;\qquad Q(v')\ge\theta\,\frac{L_K}{L_K+L_D}\ \text{with support set }D$$

Derivation:
- First equivalence (other redundancies held fixed, $0<L(u)<L(v)$): $Q(v\setminus u)=\frac{M(v)-(1-\rho)M(u)}{L(v)-L(u)}$. Cross-multiplying gives $(1-\rho)M(u)L(v)<M(v)L(u)$, which is the stated condition.
- Second statement: prune bottom-up. At each node, process surviving children in descending $Q$ and keep a child if its discounted quality is at least $\theta$. Redundancy is a max over *earlier kept* siblings, so its value at decision time is its final value. $Q(v')$ is then a length-weighted mean of values that are each $\ge\theta$, and induction over levels gives $Q(C')\ge\theta$.
- This needs at least one surviving child. If nothing survives, $C'=\emptyset$ and a fallback is required (see Risks).
- Third statement: re-adding support units $D$ with $M(D)\ge0$ and outside the redundancy computation gives $M\ge\theta L_K$ over length $L_K+L_D$.
- Re-checked numerically here: 1989 random 5-level trees at $\theta\in\{0.1,0.25,0.4\}$ with random embeddings, and 0 violations.
- The guarantee is about the proxy $Q$ under the chosen scorer $r$. It says nothing directly about answer accuracy.

**F22. [OUR PROPOSAL] Fixed-point relevance propagation with contraction bound** (a personalised-PageRank instance [50][51] plus Banach [52]).
$$\mathbf r^{(k+1)}=(1-\alpha)\mathbf p_q+\alpha W^{\top}\mathbf r^{(k)},\qquad \mathbf r^*=(1-\alpha)(I-\alpha W^\top)^{-1}\mathbf p_q,\qquad \|\mathbf r^{(k)}-\mathbf r^*\|_1\le\alpha^k\|\mathbf r^{(0)}-\mathbf r^*\|_1\le2\alpha^k,\qquad k\ge\frac{\ln(2/\varepsilon)}{\ln(1/\alpha)}$$
- $W$ is row-stochastic (dangling rows get a self-loop or are redirected to $\mathbf p_q$), $\mathbf p_q$ is a probability vector, and $\alpha\in[0,1)$ is the continuation probability.
- Because $W^\top$ is column-stochastic, $\|W^\top x\|_1\le\|x\|_1$, so the map is an $\ell_1$ contraction with modulus $\alpha$.
- Re-checked: $\alpha=0.5,\varepsilon=10^{-6}\Rightarrow k=21$, and $\alpha=0.85\Rightarrow k=90$.
- HippoRAG's damping 0.5 is a restart probability, so it corresponds to $\alpha=0.5$ here.
- $\mathbf r^*$ sums to 1 over all functions, so it is **not** a per-function relevance probability. Calibrate it (F24, [34]) before applying any 25% threshold.

**F23. [OUR PROPOSAL] Iterated prune-and-rescore operator.**
$$K_\theta(S)=\{u\in S:\ s(u\mid S)\ge\theta\},\qquad F_\theta(S)=\mathrm{cl}_S\bigl(K_\theta(S)\bigr)\subseteq S,\qquad S_{k+1}=F_\theta(S_k),\qquad S_{k^*+1}=S_{k^*}\ \text{for some } k^*\le|S_0|$$
- $\mathrm{cl}_S$ is the dependency closure inside $S$, imposed as a hard constraint (see F18).
- Since $F_\theta(S)\subseteq S$, the sizes $|S_k|$ are non-increasing and strictly decrease until a fixed point is reached. That happens after at most $|S_0|$ steps.
- The closest published precedent is LLMLingua's iterative segment-wise conditioning [25]. The fixed-point formulation with closure and the termination bound is ours.
- Caveat: the procedure is conservative. A unit dropped as redundant is not re-admitted if its duplicate is later dropped.

**F24. [OUR PROPOSAL] Decision-theoretic 25% rule and depth bound** (cost-sensitive threshold as in [35]).
$$\text{keep }u\iff p_uB-(1-p_u)C\ge0\iff p_u\ge\theta=\frac{C}{B+C};\qquad \theta=\tfrac14\iff B=3C;\qquad \mathbb E\bigl[\#\text{relevant in }K\bigr]=\sum_{u\in K}p_u\ge\theta|K|;\qquad \prod_{i=1}^{m}p_i\ge\theta,\ p_i\le p_{\max}<1\Rightarrow m\le\Bigl\lfloor\frac{\ln\theta}{\ln p_{\max}}\Bigr\rfloor$$
- Re-checked: $\theta=0.25$ gives 2 hops at $p_{\max}=0.5$ and 6 hops at $p_{\max}=0.8$.
- Assumptions:
  - $p_u$ are **calibrated** probabilities [34].
  - Path probability is the product of step probabilities. This is a Markov/independence modelling assumption, not a fact about code.
  - $B$ and $C$ are constant across units.
- The conformal rule (F14) provides the complementary recall guarantee.

**F25. [OUR PROPOSAL] Empirical rot curve and pruning gain** (derived from NoLiMa's normalized score [8]).
$$\mathrm{Rot}(\ell)=1-\frac{\mathrm{Acc}(\ell)}{\mathrm{Acc}(\ell_0)},\qquad \Delta_{\mathrm{prune}}=\mathrm{Acc}(C')-\mathrm{Acc}(C)$$

---

## What is novel in our proposal

Within the sources opened here, the individual ingredients are established:
- flat context-quality ratios [42][43][44];
- iterative perplexity-based compression [25][26];
- two-level function → block code compression [36];
- dependency-level hierarchical pruning [37];
- recursive tree and graph indexes [21][22];
- fixed-point graph relevance [50][51];
- thresholded and conformal pruning [30][32];
- dependency-aware code pruning [38][39][40].

We found no source that combines three things:
1. one context-quality metric defined **recursively across token, block, function, file and context**, with multiplicative redundancy discounting (RCQ, F20);
2. that metric **coupled to a fixed-point relevance propagation** over the call/include graph with an explicit iteration bound (F22);
3. the metric used **as the pruning operator**, with a proven lower bound $Q(C')\ge\theta$ (F21), a terminating dependency-closed prune-and-rescore fixed point (F23), and a decision-theoretic justification of the 25% threshold together with a conformal recall guarantee (F24 + F14).

Also new here are the counterexample to LaMR's Proposition 2 and the resulting "closure as a hard constraint" design (F18). This is a statement about the literature reviewed in this note, not a proof that no such work exists. All RCQ claims concern a proxy, and whether CRI predicts accuracy loss is an open empirical question.

---

## Risks and failure modes

1. **Proxy validity.** $Q$/CRI may not correlate with accuracy for a small quantised model. Context rot also occurs with *relevant* long context [14] and with whitespace filler [7], which a density metric cannot see. Mitigation: always penalise absolute length as well (a token budget), and report the correlation of CRI with measured loss.
2. **Calibration.** Similarity scores and PPR mass are not probabilities. A fixed 25% cut on uncalibrated scores has no guarantee. Neural scores are often mis-calibrated [34].
3. **Empty or near-empty context.** The threshold rule can prune everything. A fallback is needed, e.g. keep the top-1 function plus its closure, or answer "insufficient context" [45].
4. **Dependency closure blow-up.** Macro- and header-heavy C (mimalloc's `mimalloc-internal.h`, `prim/*`) can pull in large support sets, which weakens the bound to $\theta L_K/(L_K+L_D)$. Mitigation: include signatures only, not bodies (following HCP [37]).
5. **Penalty-based closure is unreliable.** See the F18 counterexample. Use hard closure.
6. **Redundancy on platform variants.** mimalloc has many structurally similar platform-specific functions. Cosine redundancy may drop the one variant that is actually relevant (e.g. Windows vs POSIX), so key redundancy on the build configuration as well.
7. **Conservative fixed point.** Order effects in F23 can lose units. Order by descending score and cap the number of passes.
8. **Conformal assumptions.** Exchangeability between calibration questions and user questions may fail across question types, the coverage is marginal only, and a single repository gives a small calibration set.
9. **Contested evidence.** The effect of random or unrelated context is contested [4][10]. RLM depth effects are model-dependent [18][19][20]. Several 2026 sources are preprints, some single-author ([13], [14], [19], [24]). Cite them as such.
10. **Selection vs generation gap.** Extractive pruning can need Θ(log n) more budget than generated summaries for some query families [15], so pure pruning has a ceiling.
11. **Token-level deletion breaks C.** Never delete tokens inside code. The minimum removable unit is a block or line group [36].

---

## Design recommendations

1. **Units.** Score at token level, but remove only blocks (line groups) and functions. Build the hierarchy with AST chunking (tree-sitter C; split-then-merge, size in non-whitespace characters) [41][36].
2. **RCQ in the UI.** Compute $M, L, Q$ bottom-up. For every query, display $Q(C')$, CRI, compression rate $\tau=L(C')/L(C)$ and the kept/dropped units. Label RCQ/CRI as our definitions.
3. **Calibrated 25% rule.** Treat 25% as $B=3C$ on *calibrated* probabilities. Fit temperature/isotonic calibration on labelled mimalloc questions [34]. Also report a split-conformal threshold for $\alpha=0.1$ with the $\lceil(n+1)(1-\alpha)\rceil$ correction [32][33].
4. **Graph propagation.** Run the PPR fixed point on call, include and similarity edges with $\alpha=0.5$ [51]. Stop at an $\ell_1$ change below $10^{-6}$ or after 21 iterations (the proven bound). Handle dangling rows, and calibrate $\mathbf r^*$ before thresholding.
5. **Tracing depth.** Chain caller → callee tracing while the product of calibrated step probabilities is at least 0.25. This bounds the depth by $\lfloor\ln 0.25/\ln p_{\max}\rfloor$.
6. **Hard dependency closure.** After thresholding, add the declarations, typedefs, macros and callee *signatures* required by kept functions, as signatures only [37]. Do not use a penalty to get closure (F18). Report the weakened bound.
7. **Iterate to a fixed point.** Re-score after each prune pass until the kept set is stable (at most $|S_0|$ passes). Cap at 3-5 passes and log the count.
8. **Redundancy.** Apply the MMR-style discount inside RCQ [46][53]. If a guarantee is needed under a token budget, use the monotone Lin-Bilmes objective with cost-scaled greedy [48], because plain MMR has no approximation guarantee.
9. **Order last.** Put the highest-scoring units at the start and end [2][6][26], but denoise first, because noise effects were several times larger than position effects [10].
10. **Absolute budget.** Choose the token budget from the measured rot curve $\mathrm{Rot}(\ell)$ of the actual quantised local model (F25). Do not use the 40-50% figure from [13] as a rule.
11. **No default LLM recursion.** Do not use RLM-style recursive LLM calls by default on the 4 GB GPU [19][20]. At most offer an optional depth-1 mode for inputs beyond the context window [18].
12. **Fine-tune for robustness.** Include hard-negative and irrelevant contexts (including survivors of pruning) in the opportunistic fine-tuning data [5][6].
13. **Evaluation.** On the mimalloc demo repository, report Rot(ℓ), accuracy with and without pruning, τ, context precision, claim recall and noise sensitivity [44], and the CRI-vs-loss correlation. Use top-k, MMR and LongCodeZip-style AMI ranking as baselines [36]. State that results come from one C codebase.

---

## References

1. Kelly Hong, Anton Troynikov, Jeff Huber. *Context Rot: How Increasing Input Tokens Impacts LLM Performance.* Chroma Technical Report, 14 July 2025. https://www.trychroma.com/research/context-rot
2. Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 12:157-173, 2024. doi:10.1162/tacl_a_00638. arXiv:2307.03172. https://aclanthology.org/2024.tacl-1.9/
3. Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, Denny Zhou. *Large Language Models Can Be Easily Distracted by Irrelevant Context.* ICML 2023. arXiv:2302.00093. https://arxiv.org/abs/2302.00093
4. Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, Fabrizio Silvestri. *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. doi:10.1145/3626772.3657834. https://arxiv.org/abs/2401.14887
5. Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant. *Making Retrieval-Augmented Language Models Robust to Irrelevant Context.* ICLR 2024. arXiv:2310.01558. https://openreview.net/forum?id=ZS4m74kZpH
6. Bowen Jin, Jinsung Yoon, Jiawei Han, Sercan O. Arik. *Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG.* ICLR 2025. https://arxiv.org/abs/2410.05983
7. Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A. Huerta, Hao Peng. *Context Length Alone Hurts LLM Performance Despite Perfect Retrieval.* Findings of EMNLP 2025, pp. 23281-23298. https://aclanthology.org/2025.findings-emnlp.1264/
8. Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schütze. *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* ICML 2025. https://arxiv.org/abs/2502.05167
9. Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg. *RULER: What's the Real Context Size of Your Long-Context Language Models?* COLM 2024. https://arxiv.org/abs/2404.06654
10. Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong. *LLM-Oriented Information Retrieval: A Denoising-First Perspective.* arXiv:2605.00505, 2026 (arXiv comment: SIGIR 2026). https://arxiv.org/abs/2605.00505
11. Shijie Xia, Yikun Wang, Zhen Huang, Pengfei Liu. *Diagnosing and Mitigating Context Rot in Long-horizon Search.* arXiv:2606.29718, 2026. https://arxiv.org/abs/2606.29718
12. Sam Martin, Fabien Roger. *Classifier Context Rot: Monitor Performance Degrades with Context Length.* arXiv:2605.12366, 2026. https://arxiv.org/abs/2605.12366
13. Weiwei Wang, Jiyong Min, Weijie Zou. *Intelligence Degradation in Long-Context LLMs: Critical Threshold Determination via Natural Length Distribution Analysis.* arXiv:2601.15300, 2026 (unreviewed). https://arxiv.org/abs/2601.15300
14. Yue Xue. *When and How Context Rot Appears in Coding Agents: A White-Box Study of Agent Skills in Code Auditing.* arXiv:2607.17937, 2026. https://arxiv.org/abs/2607.17937
15. Hayder Tirmazi, Sam Markelon, Allison Bishop, Michael Mitzenmacher. *Context Compaction Theory.* arXiv:2608.01326, 2026 (preliminary version). https://arxiv.org/abs/2608.01326
16. Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield (Anthropic Applied AI team). *Effective context engineering for AI agents.* Anthropic Engineering blog, 29 Sep 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
17. Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu. *A Survey of Context Engineering for Large Language Models.* arXiv:2507.13334, 2025. https://arxiv.org/abs/2507.13334
18. Alex L. Zhang, Tim Kraska, Omar Khattab. *Recursive Language Models.* arXiv:2512.24601, 2025 (v3 May 2026). https://arxiv.org/abs/2512.24601
19. Daren Wang. *Think, But Don't Overthink: Reproducing Recursive Language Models.* arXiv:2603.02615, 2026. https://arxiv.org/abs/2603.02615
20. Keivan Alizadeh, Parshin Shojaee, Minsik Cho, Mehrdad Farajtabar. *Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context.* arXiv:2603.15653, 2026. https://arxiv.org/abs/2603.15653
21. Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning. *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR 2024. https://arxiv.org/abs/2401.18059
22. Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130, 2024. https://arxiv.org/abs/2404.16130
23. Qingyue Wang, Yanhe Fu, Yanan Cao, Shuai Wang, Zhiliang Tian, Liang Ding. *Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.* arXiv:2308.15022, 2023 (arXiv comment: accepted by Neurocomputing). https://arxiv.org/abs/2308.15022
24. Zhihui Sun. *SVD-RAG: Efficient Tree-Organized Retrieval-Augmented Generation via Singular Value Decomposition.* arXiv:2607.10316, 2026. https://arxiv.org/abs/2607.10316
25. Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP 2023, pp. 13358-13376. https://aclanthology.org/2023.emnlp-main.825/
26. Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* ACL 2024. arXiv:2310.06839. https://arxiv.org/abs/2310.06839
27. Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* Findings of ACL 2024, pp. 963-981. https://aclanthology.org/2024.findings-acl.57/
28. Yucheng Li, Bo Dong, Frank Guerin, Chenghua Lin. *Compressing Context to Enhance Inference Efficiency of Large Language Models.* EMNLP 2023, pp. 6342-6353. https://aclanthology.org/2023.emnlp-main.391/
29. Fangyuan Xu, Weijia Shi, Eunsol Choi. *RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation.* ICLR 2024. arXiv:2310.04408. https://arxiv.org/abs/2310.04408
30. Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant. *Provence: efficient and robust context pruning for retrieval-augmented generation.* ICLR 2025. https://arxiv.org/abs/2501.16214
31. Kun Zhu, Xiaocheng Feng, Xiyuan Du, Yuxuan Gu, Weijiang Yu, Haotian Wang, Qianglong Chen, Zheng Chu, Jingchang Chen, Bing Qin. *An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation.* ACL 2024, pp. 1044-1069. https://aclanthology.org/2024.acl-long.59/
32. Debashish Chakraborty, Eugene Yang, Daniel Khashabi, Dawn Lawrie, Kevin Duh. *Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction.* ECIR 2026, LNCS Advances in Information Retrieval, pp. 537-546. doi:10.1007/978-3-032-21300-6_45. https://arxiv.org/abs/2511.17908
33. Anastasios N. Angelopoulos, Stephen Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511, 2021. https://arxiv.org/abs/2107.07511
34. Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. https://arxiv.org/abs/1706.04599
35. Charles Elkan. *The Foundations of Cost-Sensitive Learning.* IJCAI 2001. https://cseweb.ucsd.edu/~elkan/rescale.pdf
36. Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu. *LongCodeZip: Compress Long Context for Code Language Models.* ASE 2025, pp. 141-153. doi:10.1109/ASE63991.2025.00020. https://arxiv.org/abs/2510.00446
37. Lei Zhang, Yunshui Li, Jiaming Li, Xiaobo Xia, Jiaxi Yang, Run Luo, Minzheng Wang, Longze Chen, Junhao Liu, Qiang Qu, Min Yang. *Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs.* AAAI 2025, 39(24):25886-25894. doi:10.1609/aaai.v39i24.34782. https://ojs.aaai.org/index.php/AAAI/article/view/34782
38. Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo. *Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning.* arXiv:2605.15315, 2026 (arXiv comment: COLM 2026 Workshop on Efficient Reasoning). https://arxiv.org/abs/2605.15315
39. Haoxiang Jia, Earl T. Barr, Sergey Mechtaev. *Compressing Code Context for LLM-based Issue Resolution.* arXiv:2603.28119, 2026. https://arxiv.org/abs/2603.28119
40. Yuhang Wang, Yuling Shi, Mo Yang, Rongrui Zhang, Shilin He, Heng Lian, Yuting Chen, Siyu Ye, Kai Cai, Xiaodong Gu. *SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents.* arXiv:2601.16746, 2026. https://arxiv.org/abs/2601.16746
41. Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu. *cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree.* Findings of EMNLP 2025, pp. 8106-8116. https://aclanthology.org/2025.findings-emnlp.430/
42. Shahul Es, Jithin James, Luis Espinosa Anke, Steven Schockaert. *RAGAs: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024 System Demonstrations, pp. 150-158. https://aclanthology.org/2024.eacl-demo.16/
43. Ragas project. *Context Precision / Context Recall* (stable documentation), accessed 2026-09-30. https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ and https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
44. Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, Zheng Zhang. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation.* NeurIPS 2024 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html
45. Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, Cyrus Rashtchian. *Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.* ICLR 2025. https://arxiv.org/abs/2411.06037
46. Jaime Carbonell, Jade Goldstein. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.* SIGIR 1998, pp. 335-336. doi:10.1145/290941.291025. https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
47. Jade Goldstein, Jaime Carbonell. *Summarization: (1) Using MMR for Diversity-Based Reranking and (2) Evaluating Summaries.* TIPSTER Text Program Phase III Workshop, 1998, pp. 181-195. https://aclanthology.org/X98-1025/
48. Hui Lin, Jeff Bilmes. *A Class of Submodular Functions for Document Summarization.* ACL-HLT 2011, pp. 510-520. https://aclanthology.org/P11-1052/
49. Alex Kulesza, Ben Taskar. *Determinantal Point Processes for Machine Learning.* Foundations and Trends in Machine Learning 5(2-3):123-286, 2012. https://arxiv.org/abs/1207.6083
50. Güneş Erkan, Dragomir R. Radev. *LexRank: Graph-based Lexical Centrality as Salience in Text Summarization.* JAIR 22:457-479, 2004. https://arxiv.org/abs/1109.2128
51. Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.* NeurIPS 2024. https://arxiv.org/abs/2405.14831
52. V. I. Sobolev. *Contracting-mapping principle.* Encyclopedia of Mathematics (EMS Press/Springer). https://encyclopediaofmath.org/wiki/Contracting-mapping_principle
53. Jonathan J. Ross, Bevan Koopman, Anton van der Vegt, Guido Zuccon. *How retriever redundancy and diversity impact RAG effectiveness.* arXiv:2608.13956, 2026. https://arxiv.org/abs/2608.13956
