# DRAFT (not yet citation-checked): Context rot and recursive mitigation: evidence base and candidate mathematical definitions for a recursively defined context-quality metric (token -> block -> function -> file -> context, and/or iterated to a fixed point) used to prune retrieved code context before generation in programmersRAG (evaluation corpus: clone/fork of microsoft/mimalloc)

This file is the raw research dossier. The checked version is `paper/notes/context_rot.md` once it exists;
prefer that file when it is present.

## Recommendations

- Pruning granularity: score at token level but never delete individual tokens of C code. Use block (line-group) and function as the smallest removable units, because token-level compressors were reported to corrupt code syntax and structure (LongCodeZip) and because support lines have conditional value that a single per-token score cannot express (LaMR).
- Build the hierarchy with structure-aware chunking: parse mimalloc .c/.h files into functions, types and macros (AST-based split-then-merge as in cAST; tree-sitter style function/class/file nodes as in HCP) and measure unit size in tokens or non-whitespace characters, not lines.
- Implement RCQ bottom-up (mass M, length L, density Q per node) and expose Q(C'), CRI = 1 - Q(C') and the compression rate tau in the UI for every query, so the 'context rot removed' claim is a displayed, reproducible number. State clearly in the paper that RCQ/CRI are our definitions.
- Treat the 25% rule as a policy on CALIBRATED probabilities (equivalent to valuing a missed relevant function at three times the cost of an admitted irrelevant one). Calibrate raw scores on a labelled set of mimalloc questions, and additionally compute a split-conformal threshold for a chosen miss rate alpha (for example 0.1) so the paper can report an empirical coverage guarantee alongside the fixed 25% operating point.
- Propagate function relevance on the call/include/similarity graph with a personalised-PageRank style fixed point; start with continuation probability 0.5 (the HippoRAG setting), stop when the l1 change is below 1e-6 or after the proven bound of ceil(ln(2/eps)/ln(1/alpha)) iterations, and handle dangling nodes explicitly. Do not threshold r* directly: normalise/calibrate it first.
- After thresholding, enforce dependency closure inside the candidate set (declarations, typedefs, macros and callee signatures required by kept functions) and, following HCP, prefer keeping signatures while dropping bodies of dependency-only functions. Report the weakened bound theta*L_K/(L_K+L_D) when support units are added.
- Iterate prune-and-rescore until the kept set stops changing (guaranteed within |S_0| passes); in practice cap passes at a small number and log how many were needed.
- Penalise redundancy rather than reward repetition: use the MMR redundancy term inside RCQ, or a monotone submodular objective (saturated coverage or facility location plus cluster-diversity reward) with cost-scaled greedy selection under the token budget when an approximation guarantee is wanted. Note in the paper that plain MMR has no such guarantee because its objective is non-monotone.
- Order the final prompt after pruning: put the highest-scoring units at the beginning and the end (retrieval reordering of Jin et al.) because of the lost-in-the-middle effect, but prioritise denoising over ordering since the measured noise effect was several times larger than the positional effect.
- Keep the prompt short in absolute terms, not just dense: length alone degrades accuracy even with perfect retrieval, and one preprint reports a collapse at 40-50% of the maximum context for a 7B Qwen model. Choose the token budget empirically from the measured rot curve of the actual quantised local model.
- Do not implement LLM-based recursion (RLM-style sub-calls) as the default mitigation on the 4 GB GPU: published follow-ups report that deeper recursion and recursion inside the native window can reduce accuracy and multiply latency. If offered, make it an optional mode limited to depth 1 for prompts that exceed the context window.
- Optional offline hierarchy: build file- and module-level summaries once at index time (RAPTOR / GraphRAG style) and retrieve over all levels at once ('collapsed tree') under a token budget; if LLM summarisation is too costly locally, an extractive summariser is a possible substitute but the available evidence for it is weak.
- Use fine-tuning as a complement to pruning: include irrelevant and hard-negative contexts in the fine-tuning data so that the local model is robust to the noise that survives pruning (Yoran et al.; Jin et al.).
- Evaluation and reporting: measure Rot(l), accuracy with and without pruning, tau, context precision, claim recall and noise sensitivity on the mimalloc demo repository; compare against top-k, MMR and perplexity-ranking baselines; test whether CRI correlates with measured loss. State in the paper that all experiments are run on the demo repository, which is a fork/clone of mimalloc, and that results on one C codebase may not generalise.
- In the paper, separate three kinds of statements explicitly: published empirical findings (cite), published formulas (cite and reproduce exactly), and our definitions/derivations (RCQ, threshold bound, contraction bound, termination bound, decision-theoretic threshold), each with its assumptions (calibration, exchangeability, fixed redundancy, monotone operator).

## Findings

- [established] ESTABLISHED (definition and scope of 'context rot'). The Chroma technical report (Hong, Troynikov, Huber, July 2025) evaluated 18 LLMs (including GPT-4.1, Claude 4, Gemini 2.5, Qwen3) and reports that 'models do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows', even on simple tasks. Degradation is measured with task-specific accuracy as a function of input length (8 input lengths x 11 needle positions per configuration; normalized Levenshtein distance for the repeated-word replication task). The report does NOT define a single scalar 'context rot' metric; it concludes that what matters is not only whether relevant information is present but how it is presented, and calls for 'context engineering'.  (hong2025contextrot)
- [established] ESTABLISHED (distractors and structure). In the Chroma experiments 'even a single distractor reduces performance relative to the baseline (needle only), and adding four distractors compounds this degradation further'; the impact of distractors grows with input length; across all 18 models performance was better on shuffled haystacks than on logically structured ones; and on LongMemEval all models scored significantly higher on focused prompts (about 300 tokens) than on full prompts (about 113k tokens). Distractors (topically related but not answering) are distinguished from merely irrelevant content.  (hong2025contextrot)
- [established] ESTABLISHED (position effect). Liu et al. show a U-shaped accuracy curve over the position of the relevant document: performance is highest when relevant information is at the beginning or end of the input and degrades in the middle; for GPT-3.5-Turbo, multi-document QA accuracy with the answer in the middle of 20 or 30 documents can fall below closed-book accuracy (56.1%). Extended-context variants were not better at using their context when the input fit in both windows.  (liu2023lost)
- [established] ESTABLISHED (irrelevant and near-miss context reduces accuracy). Shi et al. (GSM-IC) find that adding one irrelevant sentence dramatically lowers arithmetic-reasoning accuracy, with fewer than 30% of base problems solved consistently across distractor variants (macro accuracy). In RAG, Cuconasu et al. find that the retriever's highest-scoring documents that do not contain the answer ('distracting' documents) negatively impact the LLM; Jin et al. find that with a strong retriever answer quality follows an inverted-U as the number of retrieved passages grows and attribute this to retrieved 'hard negatives'; Yoran et al. show retrieval can reduce accuracy and that fine-tuning on a mix of relevant and irrelevant contexts (even 1,000 examples) restores robustness, which is relevant to the planned fine-tuning pipeline.  (shi2023distracted, cuconasu2024noise, jin2025longrag, yoran2023robust)
- [contested] CONTESTED (is unrelated context harmless?). Cuconasu et al. report that adding RANDOM (unrelated) documents improved LLM accuracy by up to 35% in their setting, whereas Dai et al. report that pure-noise contexts yielded 8.0% exact match, far below the 23.6% closed-book baseline for LLaMA-2-7B-Chat on Natural Questions, and the Chroma report finds irrelevant filler degrades performance as length grows. The effect of unrelated context therefore appears to depend on model, task and prompt layout; it should be measured, not assumed, for the local model.  (cuconasu2024noise, dai2026denoising, hong2025contextrot)
- [established] ESTABLISHED (length alone hurts). Du et al. show across 5 LLMs on math, QA and coding tasks that even when models can perfectly retrieve (recite) all relevant evidence, performance drops 13.9% to 85% as input length increases, including when the filler is whitespace and when irrelevant tokens are masked. NoLiMa (ICML 2025) shows that when question and needle share minimal lexical overlap, 11 of 13 models fall to 50% or less of their short-context base score at 32K tokens (GPT-4o: 99.3% base to 69.7%). Consequently a context-quality metric must penalise length itself, not only irrelevance.  (du2025length, modarressi2025nolima)
- [recent] RECENT (noise dominates position in one controlled test). Dai et al. (SIGIR 2026 perspective paper) report for LLaMA-2-7B-Chat on 500 NQ questions: exact match rises from 47.0% to 61.0% when gold passages go from 1 to 5; with 3 gold passages, adding 2 and 7 noise passages reduces EM to 51.4% and 41.8%; one gold among nine noise passages gives 26.6% (closed-book 23.6%); a single gold passage alone gives 43.2% but with 9 noise passages 22.2-26.6% regardless of position, i.e. the noise effect (about 20 points) dwarfs the positional effect (about 4 points). This supports pruning first and re-ordering second.  (dai2026denoising)
- [recent] RECENT (quantitative measures of context rot, 2025-2026). Among the sources opened there is no agreed scalar definition. Candidates: (a) NoLiMa's normalized score (accuracy divided by short-context base score) and 'effective length' (largest tested length with score at least 85% of base); (b) the 'premature termination rate' of Xia et al. (proportion of give-up and uncertain-incorrect outcomes in long-horizon search, positively correlated with context length when query difficulty is controlled); (c) miss-rate inflation of LLM monitors (Martin and Roger: frontier monitors miss dangerous actions 2x to 30x more often after 800K tokens of benign activity; Opus 4.6 recall 98.6% to 88% on MonitorBench); (d) a 'critical threshold' at 40-50% of maximum context length for Qwen2.5-7B where F1 drops from about 0.55 to 0.3 (Wang et al., unreviewed preprint, degradation defined as a drop of more than 30%); (e) the 'effective context window' notion used by Tirmazi et al.  (modarressi2025nolima, xia2026rot, martin2026monitor, wang2026degradation, tirmazi2026compaction, hong2025contextrot)
- [recent] RECENT (vendor guidance). Anthropic's engineering guidance (29 Sep 2025) defines context rot as 'as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases', states that 'context, therefore, must be treated as a finite resource with diminishing marginal returns', attributes this to an 'attention budget' (n tokens give n^2 pairwise relationships) and recommends 'finding the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome', via compaction, structured note-taking, sub-agent isolation and just-in-time retrieval. A 2025 survey (Mei et al., over 1400 papers) frames context engineering as a discipline covering retrieval/generation, processing and management of context.  (anthropic2025context, mei2025survey, hong2025contextrot)
- [recent] RECENT (Recursive Language Models). Zhang, Kraska and Khattab define an RLM as an inference scaffold that places the prompt in a REPL variable (not in the context window) and lets the model write code that inspects, decomposes and recursively sub-calls a model on snippets. With GPT-5 (sub-calls to GPT-5-mini), RLM depth=1 vs base model: CodeQA 62.0 vs 24.0, BrowseComp+ (1K docs) 91.3 vs 0.0, OOLONG 56.0 vs 44.0, OOLONG-Pairs 58.0 vs 0.1. They observe that base-model performance degrades as a function of both input length and task complexity (roughly constant, linear, quadratic information needs), and they fine-tune RLM-Qwen3-8B, which improves over base Qwen3-8B by 28.3% on average.  (zhang2025rlm)
- [contested] CONTESTED (does recursion itself help, and how deep?). In the RLM paper's own Table 1 the no-sub-call variant (depth=0) with Qwen3-Coder-480B is best on CodeQA (66.0 vs 56.0/54.0/44.0 for depths 1/2/3) and the authors attribute worse deep-recursion results to propagated syntax errors. An independent reproduction (Wang, 2026) reports that depth-2 recursion and RLMs on simple retrieval degrade accuracy (S-NIAH with DeepSeek v3.2: 100.0% base, 85.0% depth 1, 70.0% depth 2) and inflate latency (3.6 s to 344.5 s). Alizadeh et al. (SRLM) report up to 22% improvement over RLM under the same time budget and conclude that 'recursion itself is not the primary driver of performance in RLMs', and that within the native context window recursive RLMs often perform worse than the base model. For a small local model this argues for a deterministic, non-LLM recursive metric rather than recursive LLM calls.  (zhang2025rlm, wang2026overthink, alizadeh2026srlm)
- [established] ESTABLISHED (recursive / hierarchical organisation of context). RAPTOR recursively embeds, clusters (GMM with BIC after UMAP) and summarises 100-token chunks into a tree; querying the 'collapsed tree' (all levels at once, 2000-token budget) outperformed layer-by-layer traversal; summaries averaged a 0.28 length ratio (72% compression) and 18.5% to 57% of retrieved nodes came from non-leaf layers. GraphRAG recursively detects Leiden communities and builds bottom-up community summaries in which higher levels incorporate lower-level summaries, answered by map-reduce. Recursive summarisation of dialogue memory is a Markov update M_i from (S_i, M_{i-1}). RECENT and weakly evidenced: SVD-RAG replaces LLM summaries in the tree by deterministic SVD-based extractive selection (single-author preprint, corpora of 38 and 205 chunks; the author notes summaries were longer than the originals on small corpora).  (sarthi2024raptor, edge2024graphrag, wang2023recsum, sun2026svdrag)
- [established] ESTABLISHED (context compression and pruning for natural-language RAG). LLMLingua: coarse-to-fine, perplexity-based, with an iterative token-level stage that conditions each segment on the already compressed previous segments; up to 20x compression with little performance loss. Selective Context: prunes lexical units by self-information with percentile filtering (50% context reduction, 36% less memory, 32% less inference time). LLMLingua-2: argues information entropy is a suboptimal compression metric and recasts compression as token classification with a bidirectional encoder trained on distilled labels. RECOMP: extractive and abstractive compressors that may return an empty string (selective augmentation), compression rate as low as 6%. Zhu et al.: information-bottleneck objective, 2.5% compression rate reported.  (jiang2023llmlingua, li2023selective, pan2024llmlingua2, xu2024recomp, zhu2024ib)
- [recent] ESTABLISHED/RECENT (thresholds rather than fixed ratios, and statistical guarantees). Provence (ICLR 2025) casts pruning as sequence labelling unified with reranking and argues that because the optimal compression ratio is specific to each question-context pair, 'using a threshold as a hyperparameter is more appropriate'; with T=0.1 and T=0.5 the pruning ratio varied automatically from 50% to 80% across datasets. Chakraborty et al. (ECIR 2026) apply split conformal prediction to set the threshold so that a relevant snippet is retained with probability at least 1-alpha; filtering met target coverage and reduced retained context 2-3x, with factual accuracy (ARGUE F1) improving under strict filtering on NeuCLIR. This is the published basis for a probability-threshold pruning rule such as the owner's 25% rule, and it shows the threshold should be calibrated rather than fixed a priori.  (chirkova2025provence, chakraborty2026conformal)
- [recent] RECENT (code-specific pruning is hierarchical and structure-aware). LongCodeZip (ASE 2025) reports that token-level compressors (LLMLingua, LLMLingua-2) perform poorly on code because token-level compression corrupts syntax and structure; it ranks function-level chunks by conditional perplexity (AMI) and then selects blocks inside functions by 0/1 knapsack, reaching up to 5.6x compression without degrading task performance; conditional-perplexity ranking beat similarity ranking by 7.89% ES in ablation. Hierarchical Context Pruning (AAAI 2025) finds that pruning function bodies in dependent files does not significantly reduce completion accuracy while keeping topological dependencies helps, reducing inputs from over 50,000 to about 8,000 tokens. LaMR (2026) shows a single relevance score cannot represent code relevance and separates semantic evidence from dependency support. SWEzze (2026) distils 'minimal sufficient context', holds about 6x compression and improves issue-resolution rates by 5.0%-9.2%. cAST (Findings of EMNLP 2025) chunks along AST boundaries by recursive split-then-merge and measures chunk size in non-whitespace characters.  (shi2025longcodezip, zhang2025hcp, wang2026lamr, jia2026swezze, zhang2025cast)
- [established] ESTABLISHED (RAG context-quality metrics are flat ratios). RAGAS defines context relevance CR as extracted sentences over total sentences; current Ragas documentation defines Context Precision@K (rank-weighted precision) and Context Recall (supported reference claims over all reference claims); RAGChecker defines claim recall, context precision (relevant chunks over k), context utilization and relevant/irrelevant noise sensitivity, and observes a trade-off: raising k from 5 to 20 raised claim recall 61.5 to 77.6 and faithfulness 88.1 to 92.2 but also noise sensitivity 34.0 to 35.4. Joren et al. define 'sufficient context' (an instance has sufficient context iff a plausible answer exists given the context). None of these opened sources defines the metric recursively across granularity levels or uses it as the pruning operator.  (es2023ragas, ragas2026docs, ru2024ragchecker, joren2024sufficient)
- [established] ESTABLISHED (diversity-aware selection). MMR greedily maximises lambda*Sim1(D,Q) - (1-lambda)*max Sim2(D,Dj). Lin and Bilmes prove the MMR objective is submodular but non-monotone (so the greedy constant-factor guarantee does not apply) and propose a monotone submodular objective (saturated coverage plus a diversity reward) for which budgeted greedy selection has constant-factor guarantees ((1-1/e) with partial enumeration, (1-1/sqrt(e)) for the practical cost-scaled greedy, as cited there). DPPs give a probabilistic alternative in which det(L_Y) with L_ij = q_i phi_i^T phi_j q_j balances per-item quality against similarity; exact mode finding is NP-hard.  (carbonell1998mmr, goldstein1998mmr, lin2011submodular, kulesza2012dpp)
- [recent] RECENT (what kind of redundancy helps). In a controlled study on FictionalQA, Ross et al. (2026 preprint) find that duplicate copies and LLM paraphrases of the same document do not significantly improve answer correctness, whereas diverse documents improve it by 17%-47%. This supports discounting near-duplicate functions (redundancy term) rather than rewarding repetition.  (ross2026redundancy)
- [established] ESTABLISHED (recursively defined scores computed as fixed points). PageRank/LexRank define the salience of a unit through the salience of its neighbours, p(u) = d/N + (1-d) * sum p(v)/deg(v); convergence of the power method to a unique stationary distribution follows from the Perron-Frobenius theorem for irreducible aperiodic chains. HippoRAG (NeurIPS 2024) uses Personalized PageRank seeded at query nodes for single-step multi-hop retrieval (damping factor 0.5; up to 20% better than prior methods; 10-30x cheaper and 6-13x faster than iterative retrieval such as IRCoT). The contracting-mapping principle (Banach) guarantees existence and uniqueness of a fixed point of a contraction on a complete metric space and convergence of successive approximations from any starting element.  (erkan2004lexrank, gutierrez2024hipporag, sobolev2011contracting)
- [recent] RECENT (theory and practice of context management). Tirmazi et al. formalise compaction as a Context Selection Game (choose a subset under a size budget) and a Context Generation Game (arbitrary bounded message), prove the latter equivalent to one-way communication complexity and exhibit queries where selection needs a Theta(log n) factor more budget than generation; hence purely extractive pruning has provable limits. Xia et al. find context-management methods (compaction, trimming, isolation) act as test-time scaling that lowers premature termination at extra cost, and that the best method is model-dependent: sub-agent isolation for strongly agentic backbones, compaction plus trimming for weaker ones.  (tirmazi2026compaction, xia2026rot)
- [our_proposal] OUR PROPOSAL (Recursive Context Quality, RCQ). Define effective relevant mass M and length L bottom-up over the tree token -> block -> function -> file -> context, M(v) = sum over children of (1 - rho_v(u)) M(u), L(v) = sum L(u), Q(v) = M(v)/L(v), with token scores r(t) in [0,1] and sibling redundancy rho in [0,1]. Q is in [0,1] at every level, unrolls to a product of (1 - rho) factors along each token's ancestor chain, and reduces to RAGAS context relevance and to RAGChecker context precision in flat special cases. CRI = 1 - Q is proposed as a structural context-rot index. This is a definition of ours, not a published metric, and it is a proxy computed on the prompt: its link to measured accuracy loss must be demonstrated experimentally.  (es2023ragas, ru2024ragchecker, carbonell1998mmr, li2023selective)
- [our_proposal] OUR PROPOSAL (threshold pruning with a guarantee). Removing a child u raises the parent's quality iff (1 - rho(u)) Q(u) < Q(v); pruning every child whose discounted quality is below theta, applied greedily in descending quality at each level from the bottom up, guarantees Q(C') >= theta for the final context (theta = 0.25 means at least 25% effective relevant token mass). If low-relevance support units are re-added for dependency closure the bound weakens to theta * L_K / (L_K + L_D). Derivation given in the formulas section; checked numerically in this session (300 random 4-level trees, 0 violations).  (wang2026lamr, chirkova2025provence, carbonell1998mmr)
- [our_proposal] OUR PROPOSAL (fixed-point relevance propagation). Function-level relevance is propagated over the call/similarity graph by r <- (1-alpha) p_q + alpha W^T r. The map is an l1 contraction with modulus alpha, so it has a unique fixed point r* = (1-alpha)(I - alpha W^T)^{-1} p_q and k >= ln(2/eps)/ln(1/alpha) iterations suffice for l1 error eps (21 iterations for alpha=0.5, 90 for alpha=0.85 at eps=1e-6). This is an application of personalised PageRank to a code graph with an explicit iteration bound; r* is a distribution over functions and must be calibrated before a probability threshold is applied.  (erkan2004lexrank, gutierrez2024hipporag, sobolev2011contracting)
- [our_proposal] OUR PROPOSAL (iterating prune-and-rescore to a fixed point). Because relevance and redundancy of a unit depend on what else is kept, scores are recomputed after each pruning pass; with a monotone (subset-returning) operator that also keeps dependencies of kept units, the sequence of contexts is decreasing and stabilises after at most |S_0| passes. The stable set is our operational definition of a 'rot-free' context for a given scorer and threshold. LLMLingua's iterative token-level compression is the closest published precedent (segment-wise conditioning on compressed prefixes); the fixed-point formulation with dependency closure and termination bound is ours.  (jiang2023llmlingua, wang2026lamr)
- [our_proposal] OUR PROPOSAL (meaning of the 25% rule). Under a Bayes decision rule with benefit B for including a relevant unit and cost C for including an irrelevant one, the optimal inclusion threshold on a calibrated relevance probability is theta = C/(B+C); theta = 0.25 corresponds to B = 3C. With calibrated probabilities the kept set has expected precision at least theta; chained tracing with step probabilities at most p_max stops after at most floor(ln theta / ln p_max) hops. These statements require calibration; the split-conformal threshold of Chakraborty et al. supplies the complementary recall (coverage) guarantee and is the recommended way to set the operating point.  (chakraborty2026conformal)
- [our_proposal] OUR PROPOSAL (novelty statement for the paper). Within the literature opened for this dossier, individual ingredients are established: flat context-quality ratios (RAGAS, RAGChecker), iterative perplexity-based compression (LLMLingua), two-level function-then-block code compression (LongCodeZip), hierarchical pruning by dependency level (HCP), recursive tree indexes (RAPTOR, GraphRAG), fixed-point graph relevance (PageRank/LexRank, HippoRAG), thresholded and conformal pruning (Provence, Chakraborty et al.), dependency-closed pruning (LaMR). We did not find a source that (i) defines one context-quality metric recursively across the levels token, block, function, file and context, (ii) couples it with a fixed-point relevance propagation, and (iii) uses it as the pruning operator with an explicit lower bound on the quality of the pruned prompt. That combination, the bound Q(C') >= theta, the termination bound and the decision-theoretic reading of the 25% threshold would be our contribution. This is a statement about the sources reviewed here, not a proof of absence in the wider literature.  (es2023ragas, ru2024ragchecker, jiang2023llmlingua, shi2025longcodezip, zhang2025hcp, sarthi2024raptor, gutierrez2024hipporag, chirkova2025provence, chakraborty2026conformal, wang2026lamr)
- [our_proposal] OUR PROPOSAL (evaluation protocol). Report, on questions over the mimalloc demo repository, (a) the empirical rot curve Rot(l) = 1 - Acc(l)/Acc(l_0) for the local model, (b) accuracy with pruned versus unpruned context, (c) compression rate tau = pruned length / original length, (d) RAGChecker-style context precision, claim recall and noise sensitivity, and (e) the correlation between the structural index CRI = 1 - Q and measured accuracy loss. Baselines should include plain top-k, MMR, and a LongCodeZip-style perplexity ranking.  (modarressi2025nolima, hong2025contextrot, ru2024ragchecker, jiang2023llmlingua, shi2025longcodezip, carbonell1998mmr)

## Formulas

### RAGAS context relevance (CR)

$$
\text{CR}=\frac{\text{number of extracted sentences}}{\text{total number of sentences in }c(q)}
$$

Reference-free context-quality score from the original RAGAS paper: an LLM extracts from the retrieved context c(q) the subset of sentences that are crucial to answer q; CR is the kept fraction. The paper states the metric 'aims to penalise the inclusion of redundant information'. It is a flat (single-level) density; our recursive metric reduces to it in a special case.

Sources: es2023ragas

### RAGAS Context Precision@K and Context Recall

$$
\text{Context Precision@K}=\frac{\sum_{k=1}^{K}\left(\text{Precision@k}\times v_k\right)}{\text{Total number of relevant items in the top }K\text{ results}},\quad \text{Precision@k}=\frac{\text{true positives@k}}{\text{true positives@k}+\text{false positives@k}},\quad \text{Context Recall}=\frac{\text{Number of claims in the reference supported by the retrieved context}}{\text{Total number of claims in the reference}}
$$

Current Ragas documentation definitions. v_k in {0,1} is the relevance indicator at rank k, so Context Precision@K is a rank-weighted precision (average-precision style); Context Recall is claim-level recall against a reference answer. Both need relevance judgements (LLM or non-LLM) and are evaluation metrics, not pruning rules.

Sources: ragas2026docs

### RAGChecker retriever and generator diagnostics

$$
\text{Claim Recall}=\frac{|\{c^{(gt)}_{i}\mid c^{(gt)}_{i}\in\{\text{chunk}_{j}\}\}|}{|\{c^{(gt)}_{i}\}|},\quad \text{Context Precision}=\frac{|\{\text{r-chunk}_{j}\}|}{k},\quad \text{Context Utilization}=\frac{|\{c^{(gt)}_{i}\mid c^{(gt)}_{i}\in\{\text{chunk}_{j}\}\text{ and }c^{(gt)}_{i}\in m\}|}{|\{c^{(gt)}_{i}\mid c^{(gt)}_{i}\in\{\text{chunk}_{j}\}\}|},\quad \text{Relevant Noise Sensitivity}=\frac{|\{c^{(m)}_{i}\mid c^{(m)}_{i}\notin gt\text{ and }c^{(m)}_{i}\in\{\text{r-chunk}_{j}\}\}|}{|\{c^{(m)}_{i}\}|}
$$

Claim-level entailment metrics. c^(gt) are ground-truth answer claims, c^(m) are claims in the model response m, an r-chunk is a retrieved chunk entailing at least one ground-truth claim, k is the number of retrieved chunks. Relevant noise sensitivity is the fraction of response claims that are incorrect and entailed by a relevant chunk, i.e. a direct measurement of harm done by noise that travels with useful context.

Sources: ru2024ragchecker

### Maximal Marginal Relevance (MMR)

$$
\mathrm{MMR}\;\stackrel{\mathrm{def}}{=}\;\arg\max_{D_i\in R\setminus S}\Bigl[\lambda\,\mathrm{Sim}_1(D_i,Q)-(1-\lambda)\max_{D_j\in S}\mathrm{Sim}_2(D_i,D_j)\Bigr]
$$

R is the ranked list retrieved for query Q, S the already selected subset, lambda in [0,1] trades relevance against novelty (lambda=1 gives the plain relevance ranking, lambda=0 a maximal-diversity ranking). Lin and Bilmes show that a set function whose marginal gain equals the bracketed term is submodular but non-monotone, so the greedy constant-factor guarantee does not apply to MMR.

Sources: carbonell1998mmr, goldstein1998mmr, lin2011submodular

### Submodularity and budgeted (knapsack) selection

$$
F(A\cup\{v\})-F(A)\;\ge\;F(B\cup\{v\})-F(B)\quad\forall A\subseteq B\subseteq V\setminus\{v\};\qquad S^{*}\in\arg\max_{S\subseteq V}F(S)\ \ \text{subject to}\ \ \sum_{i\in S}c_i\le b
$$

Diminishing-returns definition and 'Problem 1' of Lin and Bilmes (c_i = cost, e.g. token length of unit i; b = budget). As reported there: for monotone submodular F the greedy algorithm under a cardinality constraint is within (e-1)/e (about 0.63) of optimal (attributed to Nemhauser et al., 1978); with a knapsack constraint a greedy with partial enumeration achieves (1-1/e) (attributed to Sviridenko, 2004) and the practical cost-scaled greedy achieves (1-1/sqrt(e)) (attributed to Lin and Bilmes, 2010).

Sources: lin2011submodular

### Lin-Bilmes coverage plus diversity objective (and facility location)

$$
F(S)=L(S)+\lambda R(S),\qquad L(S)=\sum_{i\in V}\min\{C_i(S),\,\alpha\,C_i(V)\},\quad C_i(S)=\sum_{j\in S}w_{i,j},\qquad R(S)=\sum_{i=1}^{K}\sqrt{\sum_{j\in P_i\cap S}r_j},\qquad L_{\mathrm{FL}}(S)=\sum_{i\in V}\max_{j\in S}w_{i,j}
$$

Monotone submodular summary-quality objective: L is saturated coverage (w_ij >= 0 similarity, alpha in [0,1] saturation threshold), R rewards picking from different clusters P_1..P_K of the ground set (r_j >= 0 singleton reward; the square root gives diminishing returns inside a cluster). L_FL is the facility-location alternative the paper lists. Rewarding diversity instead of penalising redundancy keeps the objective monotone, unlike MMR.

Sources: lin2011submodular

### Determinantal point process (L-ensemble) with quality-diversity decomposition

$$
\mathcal{P}_L(\boldsymbol{Y}=Y)=\frac{\det(L_Y)}{\det(L+I)},\qquad L_{ij}=q_i\,\phi_i^{\top}\phi_j\,q_j,\qquad S_{ij}\equiv\phi_i^{\top}\phi_j=\frac{L_{ij}}{\sqrt{L_{ii}L_{jj}}}
$$

L is a real symmetric positive semidefinite kernel indexed by the items, L_Y its restriction to subset Y. q_i in R+ is per-item quality (here: relevance of a function to the query) and phi_i a unit-norm diversity feature vector (here: the function embedding), so det(L_Y) is the squared volume spanned by the quality-scaled vectors: redundant (near-parallel) items get low probability. Finding the mode (max det) is NP-hard (their Theorem 2.9).

Sources: kulesza2012dpp

### PageRank / LexRank recursive centrality and power iteration

$$
p(u)=\frac{d}{N}+(1-d)\sum_{v\in adj[u]}\frac{p(v)}{\deg(v)},\qquad \mathbf{p}=\bigl[d\,\mathbf{U}+(1-d)\,\mathbf{B}\bigr]^{T}\mathbf{p},\qquad \mathbf{p}_t=\mathbf{M}^{T}\mathbf{p}_{t-1}\ \text{until}\ \|\mathbf{p}_t-\mathbf{p}_{t-1}\|<\epsilon
$$

Classic recursively defined salience: the score of a unit is defined through the scores of its neighbours and computed as a fixed point. B is the row-normalised similarity/adjacency matrix, U has all entries 1/N, and in the LexRank paper's convention d is the jump probability ('typically chosen in the interval [0.1, 0.2]'). By the Perron-Frobenius theorem an irreducible aperiodic chain converges to a unique stationary distribution, which the power method computes.

Sources: erkan2004lexrank

### LLMLingua compression objective and token-retention rule

$$
\min_{\widetilde{\bm{x}},\tau}\ \mathrm{KL}\bigl(P(\widetilde{\bm{x}}_G\mid\widetilde{\bm{x}}),\,P(\bm{x}_G\mid\bm{x})\bigr),\qquad \tau=\widetilde{L}/L\in[0,1],\qquad \widetilde{\bm{s}}_j=\{s_{j,i}\mid p(s_{j,i})>\gamma_j\},\qquad p(\widetilde{\bm{s}}_j)=\prod_{i}p\bigl(\widetilde{s}_{j,i}\mid\widetilde{s}_{j,<i},\widetilde{\bm{s}}_{<j}\bigr)
$$

x is the original prompt of length L, x-tilde the compressed prompt of length L-tilde, x_G and x-tilde_G the generations they induce; tau is the compression rate and 1/tau the compression ratio. In the iterative token-level compression the prompt is cut into segments s_j, each segment is scored conditioned on the ALREADY COMPRESSED previous segments, and tokens whose perplexity-based score exceeds a segment-specific threshold gamma_j are retained. This is an established example of a score that is recomputed recursively on the output of the previous pruning step.

Sources: jiang2023llmlingua

### Token self-information and its additivity over lexical units (Selective Context)

$$
I(x_i)=-\log_2 P(x_i\mid x_0,x_1,\ldots,x_{i-1}),\qquad I(u)=\sum_{i=t}^{\alpha}I(x_i)
$$

Self-information (surprisal) of token x_i under a causal LM; the score of a lexical unit u (phrase or sentence made of tokens x_t..x_{t+alpha}) is the sum of the self-information of its tokens. Units are then filtered by a percentile of the score distribution rather than by a fixed threshold. This additivity is the simplest established token-to-unit aggregation rule.

Sources: li2023selective

### LongCodeZip approximated mutual information (function ranking) and conditional perplexity

$$
\mathrm{AMI}(c,q)=\mathrm{PPL}(q)-\mathrm{PPL}(q\mid c),\qquad \mathrm{PPL}(q\mid c)=\exp\Bigl(-\frac{1}{N}\sum_{i=1}^{N}\log P(q_i\mid q_{<i},c)\Bigr),\qquad R_{\mathrm{biased},i}=R_{\mathrm{base}}\bigl(1+\beta\,(2\,\mathrm{AMI}_{\mathrm{norm},i}-1)\bigr)
$$

Relevance of a code chunk c to instruction q is the reduction in perplexity of q when c is provided. Functions are ranked by AMI (coarse stage); inside each kept function, blocks are chosen by a 0/1 knapsack (value = normalised AMI, weight = token length) under a per-function budget whose retention ratio R_biased grows with the normalised AMI of the function. This is the closest published two-level (function then block) analogue of the hierarchy we target.

Sources: shi2025longcodezip

### Information-bottleneck objective for RAG noise filtering

$$
\min\ \mathcal{L}_{\mathrm{IB}}=\underbrace{I(\tilde{X};X\mid Q)}_{\textit{conciseness}}-\beta\,\underbrace{I(\tilde{X};Y\mid Q)}_{\textit{correctness}}
$$

X retrieved passages, X-tilde the compressed/filtered context, Y the ground output, Q the query. The first term penalises information kept about the raw retrieval, the second rewards information about the answer. The paper notes the conciseness term drives the filtered context to empty when retrieval is entirely irrelevant.

Sources: zhu2024ib

### Provence training loss and threshold inference

$$
\mathcal{L}=\sum_{n=1}^{N}\Bigl\{\sum_{k=1}^{L_n}\log P(y_{n,k}\mid z_{n,k})+\lambda\,\bigl(s_n-z_{n,0}\bigr)^2\Bigr\},\qquad z_n=\mathrm{Provence}(x_n)
$$

Formula reproduced as typeset in the paper. y_{n,k} are binary keep/drop labels for the L_n passage tokens, z_{n,0} is the reranking score predicted from the BOS position, s_n the teacher reranker score (MSE regulariser). At inference per-token keep probabilities are binarised with a threshold T (the paper uses T=0.1 and T=0.5), so the amount of pruning adapts to each query-passage pair instead of being a fixed compression ratio.

Sources: chirkova2025provence

### Split-conformal context filter with coverage guarantee

$$
\hat{\tau}_{\alpha}=\mathrm{Quantile}_{1-\alpha}\bigl(\{A(q,s):(q,s)\in\mathcal{D}_{\mathrm{cal}},\ r(q,s)=1\}\bigr),\qquad K_q=\{s:\ A(q,s)\le\hat{\tau}_{\alpha}\},\qquad P\bigl(s\in K_q\mid r(q,s)=1\bigr)\ \ge\ 1-\alpha
$$

A(q,s) is a nonconformity score (lower = more relevant), e.g. A_emb = 1 - cos(emb(q), emb(s)); the threshold is the empirical (1-alpha) quantile of the scores of relevant calibration snippets (equivalently the ceil((n+1)(1-alpha))-th smallest of n positives). Under exchangeability of calibration and test data a relevant snippet is retained with probability at least 1-alpha. This turns an arbitrary score threshold into one with a finite-sample recall guarantee.

Sources: chakraborty2026conformal

### RAPTOR soft clustering and model selection

$$
P(\mathbf{x})=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x};\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k),\qquad \mathrm{BIC}=\ln(N)\,k-2\ln(\hat{L})
$$

Text-chunk embeddings (after UMAP dimensionality reduction) are soft-clustered with a Gaussian mixture; the number of clusters is chosen by BIC (N = number of text segments, k = number of model parameters, L-hat = maximised likelihood) and parameters fitted by EM. Clusters are summarised and the embed-cluster-summarise step is applied recursively to build the tree.

Sources: sarthi2024raptor

### Recursive memory (summary) update as a Markov chain

$$
P(M_N\mid S)=\prod_{i=1}^{N}P(M_i\mid S_i,M_{i-1})
$$

M_i is the memory (summary) after session i and S_i the i-th session: each new summary depends only on the current content and the previous summary. This is the formal shape of recursive/rolling summarisation and of agent compaction.

Sources: wang2023recsum

### Context Selection Game (formal model of selection-based compaction/pruning)

$$
S\subseteq\mathcal{X},\qquad \sum_{x_i\in S}s(x_i)\le B,\qquad v_q:2^{\mathcal{X}}\to[0,1],\qquad \mathrm{val}(S,q):=v_q(S),\qquad \mathrm{err}(S,q):=1-\mathrm{val}(S,q)
$$

X = {x_1..x_N} item universe (items may be tokens, tool results, chunks: the framework is 'agnostic to the choice of granularity'), s(x_i) > 0 item size, B budget, v_q value of the retained subset for query q. Extractive pruning as done in programmersRAG is an instance of the SELECT class; the paper proves generation-based compaction is equivalent to one-way communication complexity and exhibits a query family where selection needs a Theta(log n) factor more budget than generation.

Sources: tirmazi2026compaction

### Empirical context-rot measures: normalized score and effective length (NoLiMa)

$$
\mathrm{NS}(\ell)=\frac{\mathrm{Acc}(\ell)}{\mathrm{Acc}_{\mathrm{base}}},\qquad \ell_{\mathrm{eff}}=\max\{\ell\ \text{tested}:\ \mathrm{Acc}(\ell)\ \ge\ 0.85\,\mathrm{Acc}_{\mathrm{base}}\}
$$

Symbolic rendering (ours) of definitions the paper gives in words: the base score is computed from the shortest contexts (250, 500 and 1K tokens); the 'normalized score' divides accuracy by the base score; the 'effective length' is the largest tested length whose score stays above 85% of the base score.

Sources: modarressi2025nolima

### Dependency-aware pruning utility (LaMR, Proposition 2)

$$
U(S)=\sum_{i\in S}r_i-\lambda\sum_{i\in S}c_i-M\sum_{(i,j)\in E}\mathbf{1}[\,i\in S,\ j\notin S\,]
$$

G=(V,E) is a dependency graph over code lines where (i,j) in E means line i requires line j; r_i relevance, c_i cost, M missing-support penalty. The paper states that if M > max_j (r_j - lambda c_j)_+ every optimal S* is dependency-closed (i in S*, (i,j) in E implies j in S*), and (Proposition 1) that the marginal value of a support line is M - lambda > 0 when its evidence line is kept and -lambda < 0 otherwise, so a single unconditional relevance score cannot represent code relevance.

Sources: wang2026lamr

### [OUR PROPOSAL] Recursive Context Quality (RCQ) over the hierarchy token, block, function, file, context

$$
M(t)=r(t),\ \ L(t)=1\ \ (\text{token } t);\qquad M(v)=\sum_{u\in\mathrm{ch}(v)}\bigl(1-\rho_v(u)\bigr)\,M(u),\qquad L(v)=\sum_{u\in\mathrm{ch}(v)}L(u),\qquad Q(v)=\frac{M(v)}{L(v)}=\sum_{u\in\mathrm{ch}(v)}\frac{L(u)}{L(v)}\bigl(1-\rho_v(u)\bigr)\,Q(u),\qquad \mathrm{CRI}(C)=1-Q(C)
$$

Own definition. r(t) in [0,1] is a token-level relevance/keep probability (e.g. a normalised self-information or a token-classifier probability); ch(v) are the children of node v one level down; rho_v(u_j) in [0,1] is the redundancy of child u_j with respect to the siblings kept before it, e.g. rho_v(u_j) = max_{i<j} max(0, cos(e_{u_i}, e_{u_j})) (the MMR redundancy term). M is effective relevant token mass, L token length, Q a density. Derivation of properties: (i) Q(v) in [0,1] by induction, because Q(v) is a convex combination (weights L(u)/L(v) sum to 1) of numbers in [0,1] each multiplied by a factor in [0,1]. (ii) Unrolling the recursion for M gives Q(C) = (1/L(C)) * sum over leaf tokens t of r(t) * product over the ancestors a of t strictly below the root of (1 - rho(a)): every token is discounted multiplicatively by the redundancy of each enclosing unit. (iii) Special cases: with rho = 0 and binary r, Q(C) is the fraction of relevant tokens; with sentences as leaves it equals RAGAS CR; with unit-length chunks as leaves and binary chunk relevance it equals RAGChecker context precision |r-chunk|/k. CRI(C) = 1 - Q(C) is our proposed structural context-rot index (share of token mass that is irrelevant or redundant); it is a property of the prompt, not a measured accuracy loss.

Sources: es2023ragas, ru2024ragchecker, carbonell1998mmr, li2023selective

### [OUR PROPOSAL] Density-improvement criterion and threshold guarantee for recursive pruning

$$
Q(v\setminus u)>Q(v)\iff\bigl(1-\rho_v(u)\bigr)Q(u)<Q(v);\qquad \text{keep }u\iff\bigl(1-\rho_v(u)\bigr)Q(u)\ge\theta\ \ \Longrightarrow\ \ Q(v')\ge\theta;\qquad \text{with support units }D:\ \ Q(v')\ \ge\ \theta\,\frac{L_K}{L_K+L_D}
$$

Own derivation. First statement (redundancies of the other children held fixed): Q(v minus u) = (M(v) - (1-rho_u) M(u)) / (L(v) - L(u)); cross-multiplying, Q(v minus u) > Q(v) iff (1-rho_u) M(u) L(v) < M(v) L(u) iff (1-rho_u) Q(u) < Q(v). So removing a child improves density exactly when its discounted quality is below the current quality of the parent; unconstrained density maximisation would therefore collapse to the single best leaf, which is why a threshold or a recall constraint is required. Second statement: process children greedily in descending Q and keep u iff its discounted quality is at least theta; redundancy is a max over previously kept siblings, so dropping siblings can only lower it; Q(v') is then a length-weighted mean of numbers each >= theta, hence Q(v') >= theta, and by induction over levels the final context satisfies Q(C') >= theta. With theta = 0.25 the final prompt is guaranteed to contain at least 25% effective relevant token mass under the chosen scorer. Third statement: if low-score support units D (includes, type definitions, callee signatures) are re-added for dependency closure with M(D) >= 0 and are excluded from the redundancy computation, then M(v') >= theta L_K and L(v') = L_K + L_D, giving the weaker bound. Checked numerically in this session on 300 random 4-level trees (0 violations of Q >= theta) and 2000 random instances of the criterion (0 mismatches).

Sources: wang2026lamr, chirkova2025provence, carbonell1998mmr

### [OUR PROPOSAL] Fixed-point relevance propagation over the function graph with contraction bound

$$
\mathbf{r}^{(k+1)}=T\bigl(\mathbf{r}^{(k)}\bigr)=(1-\alpha)\,\mathbf{p}_q+\alpha\,W^{\top}\mathbf{r}^{(k)},\qquad \mathbf{r}^{*}=(1-\alpha)\bigl(I-\alpha W^{\top}\bigr)^{-1}\mathbf{p}_q,\qquad \|\mathbf{r}^{(k)}-\mathbf{r}^{*}\|_1\le\alpha^{k}\,\|\mathbf{r}^{(0)}-\mathbf{r}^{*}\|_1\le 2\alpha^{k},\qquad k\ \ge\ \frac{\ln(2/\varepsilon)}{\ln(1/\alpha)}
$$

Own derivation in the PageRank / personalised-PageRank family. p_q is the query-conditioned prior over functions (normalised to sum 1), W a row-stochastic matrix built from call-graph and similarity edges (dangling rows get a self-loop or are redirected to p_q), alpha in [0,1) the continuation probability (1-alpha = restart; note that LexRank writes the jump probability as d, and HippoRAG calls the restart probability the damping factor and sets it to 0.5). Because W^T is column-stochastic, ||W^T x||_1 <= ||x||_1, hence ||T(r) - T(r')||_1 <= alpha ||r - r'||_1: T is a contraction on the complete metric space (R^N, l1), so by the contracting-mapping principle it has a unique fixed point reached from any start, with geometric error alpha^k. Two probability vectors differ by at most 2 in l1, which gives the sufficient iteration count: alpha=0.5, eps=1e-6 gives k = 21; alpha=0.85, eps=1e-6 gives k = 90 (in a numerical check on a random 40-node graph the geometric bound held and 8 and 11 iterations sufficed). Caveat: r* sums to 1 over all functions, so it is NOT a per-function probability of relevance; it must be calibrated (e.g. isotonic/logistic regression or the conformal threshold) before a 25% threshold is applied.

Sources: erkan2004lexrank, gutierrez2024hipporag, sobolev2011contracting

### [OUR PROPOSAL] Iterated prune-and-rescore operator and its termination

$$
K_\theta(S)=\{u\in S:\ s(u\mid S)\ge\theta\},\qquad F_\theta(S)=\mathrm{cl}_S\bigl(K_\theta(S)\bigr)\subseteq S,\qquad S_{k+1}=F_\theta(S_k),\qquad S_{k^*+1}=S_{k^*}\ \text{for some}\ k^*\le|S_0|
$$

Own derivation. s(u|S) is any context-dependent score (conditional perplexity given the currently kept context as in the iterative compression of LLMLingua, RCQ discounted quality, or calibrated propagated relevance); cl_S(K) is the smallest superset of K inside S that is closed under the dependency relation E (LaMR Proposition 2). Since F_theta(S) is a subset of S, |S_k| is a non-increasing sequence of non-negative integers that strictly decreases until S_{k+1} = S_k, so a fixed point is reached after at most |S_0| iterations. The fixed point is the stable context: every retained unit is either above threshold given the rest of the retained context or is required by such a unit. The procedure is conservative (a unit dropped for redundancy is not re-admitted if the unit it duplicated is dropped later); ordering candidates by descending score mitigates this.

Sources: jiang2023llmlingua, wang2026lamr

### [OUR PROPOSAL] Decision-theoretic reading of the 25% threshold and traversal depth bound

$$
\text{keep }u\iff p_u B-(1-p_u)C\ge 0\iff p_u\ge\theta=\frac{C}{B+C};\quad \theta=\tfrac14\iff B=3C;\qquad \mathbb{E}\bigl[\#\{\text{relevant in }K\}\bigr]=\sum_{u\in K}p_u\ \ge\ \theta\,|K|;\qquad \prod_{i=1}^{m}p_i\ge\theta\ \text{and}\ p_i\le p_{\max}<1\ \Rightarrow\ m\le\Bigl\lfloor\frac{\ln\theta}{\ln p_{\max}}\Bigr\rfloor
$$

Own derivation. p_u is a CALIBRATED probability that unit u is relevant, B the benefit of including a relevant unit and C the cost of including an irrelevant one (context-rot cost). Expected utility of inclusion is p_u B - (1-p_u) C, which is non-negative iff p_u >= C/(B+C); a 25% threshold therefore encodes the judgement that missing a relevant function is three times as costly as admitting an irrelevant one. If the p_u are calibrated, the expected number of relevant units in the kept set K is the sum of p_u over K, which is >= theta |K| (expected precision at least 25%), and the expected number of relevant units discarded is below theta times the number discarded. For chained tracing (caller to callee and onward), if the path probability is the product of step probabilities and every step probability is at most p_max < 1, tracing must stop after at most floor(ln theta / ln p_max) hops: theta=0.25 gives 2 hops for p_max=0.5 and 6 hops for p_max=0.8. None of these guarantees holds for uncalibrated similarity scores; the conformal quantile rule gives the complementary recall guarantee.

Sources: chakraborty2026conformal

### [OUR PROPOSAL] Empirical context-rot curve for evaluating the pruner

$$
\mathrm{Rot}(\ell)=1-\frac{\mathrm{Acc}(\ell)}{\mathrm{Acc}(\ell_0)},\qquad \Delta_{\mathrm{prune}}=\mathrm{Acc}\bigl(C'\bigr)-\mathrm{Acc}\bigl(C\bigr)
$$

Own definition obtained by taking one minus the NoLiMa normalized score: l_0 is the focused (oracle or minimal) context length, l the evaluated prompt length; Rot is the relative accuracy lost to context length and noise. Delta_prune compares answer accuracy with the pruned context C' against the unpruned retrieval C on the same mimalloc questions; together with the structural index CRI = 1 - Q it lets the paper test whether the structural metric actually predicts measured degradation.

Sources: modarressi2025nolima, hong2025contextrot

## Open questions

- Does the structural index CRI = 1 - Q actually predict measured accuracy loss for a small, quantised, fine-tuned local model on code questions? No opened source validates a structural context-quality score against accuracy for code; this must be tested.
- Which leaf scorer is best and affordable in C++ on this hardware: token self-information from the generator itself, conditional perplexity / AMI (needs extra forward passes), a small encoder classifier (LLMLingua-2 / Provence style), or embedding similarity? LLMLingua-2 argues entropy-based scores are suboptimal, while LongCodeZip reports conditional perplexity beats similarity for code.
- How large must the calibration set be for the conformal threshold on a single repository, and does exchangeability hold across question types (API usage, bug localisation, design questions)? The guarantee is marginal, not per-query.
- How should the fixed 25% threshold and a conformal threshold be reconciled when they disagree (precision-oriented versus recall-oriented operating points)?
- Is unrelated context harmful or harmless for the target model? Published results conflict (random documents improved accuracy in one study; pure noise was worse than no retrieval in another).
- Is there any benefit from LLM recursion (RLM) for inputs that already fit in the context window? Follow-up studies dispute that recursion is the driver of the reported gains and report degradation at depth 2.
- When does generated (summarised) context beat extractive selection for code? Theory shows a selection-versus-generation gap exists for some query families, but summaries can hallucinate and are expensive on a local model.
- How sensitive are results to the continuation probability alpha and to the relative weights of call edges, include edges and embedding-similarity edges in the propagation matrix W?
- How should redundancy be defined for code (embedding cosine, clone detection, shared callee sets), given that mimalloc contains many structurally similar platform-specific functions?
- What is the quantitative interaction between prompt length and density, i.e. the functional form of accuracy as a function of (L, Q)? Existing studies vary length or noise but none of the opened sources provides a fitted joint model.
- Does dependency closure reintroduce enough low-relevance text to negate the benefit of pruning in macro- and header-heavy C code, and is signature-only inclusion sufficient?
- Do positional effects (lost in the middle) persist for the chosen local model at the short prompt lengths that pruning produces, or only at long lengths?

## Sources

- **hong2025contextrot**: Kelly Hong, Anton Troynikov, Jeff Huber (2025). Context Rot: How Increasing Input Tokens Impacts LLM Performance. Chroma Technical Report (July 2025). https://www.trychroma.com/research/context-rot
  - evidence: 'we evaluate 18 LLMs, including the state-of-the-art GPT-4.1, Claude 4, Gemini 2.5, and Qwen3 models. Our results reveal that models do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows.' Also: 'Even a single distractor reduces performance relative to the baseline (needle only), and adding four distractors compounds this degradation further'; 'models perform better on shuffled haystacks than on logically structured ones'; focused prompts about 300 tokens vs full about 113k tokens; 'Whether relevant information is present in a model's context is not all that matters; what matters more is how that information is presented.'
- **liu2023lost**: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics (accepted 2023); arXiv:2307.03172. https://arxiv.org/abs/2307.03172
  - evidence: 'performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts'; 'in the worst case, performance in 20- and 30-document settings is lower than performance without any input documents (i.e., closed-book performance; 56.1%).'
- **shi2023distracted**: Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Scharli, Denny Zhou (2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML 2023; arXiv:2302.00093. https://arxiv.org/abs/2302.00093
  - evidence: Abstract: 'the model performance is dramatically decreased when irrelevant information is included.' Body: 'The drop on macro accuracy is especially large, showing that fewer than 30% of the base problems are consistently solved after adding distractors.'
- **cuconasu2024noise**: Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, Fabrizio Silvestri (2024). The Power of Noise: Redefining Retrieval for RAG Systems. SIGIR 2024 (47th International ACM SIGIR Conference); arXiv:2401.14887. https://arxiv.org/abs/2401.14887
  - evidence: 'the retriever's highest-scoring documents that are not directly relevant to the query (e.g., do not contain the answer) negatively impact the effectiveness of the LLM. Even more surprising, we discovered that adding random documents in the prompt improves the LLM accuracy by up to 35%.'
- **yoran2023robust**: Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant (2023). Making Retrieval-Augmented Language Models Robust to Irrelevant Context. arXiv:2310.01558 (v2, May 2024). https://arxiv.org/abs/2310.01558
  - evidence: 'recent work has shown that retrieval augmentation can sometimes have a negative effect on performance'; 'even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones.'
- **jin2025longrag**: Bowen Jin, Jinsung Yoon, Jiawei Han, Sercan O. Arik (2025). Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG. ICLR 2025; arXiv:2410.05983 (Oct 2024). https://arxiv.org/abs/2410.05983
  - evidence: 'for many long-context LLMs, the quality of generated output initially improves first, but then subsequently declines as the number of retrieved passages increases ... identifying the detrimental impact of retrieved hard negatives as a key contributor'; 'With a strong retriever, performance exhibits an inverted-U pattern'; retrieval reordering places higher-scored passages at the beginning and end.
- **du2025length**: Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Galstyan, Azton Wells, Roy Schwartz, Eliu A Huerta, Hao Peng (2025). Context Length Alone Hurts LLM Performance Despite Perfect Retrieval. Findings of EMNLP 2025; arXiv:2510.05381. https://arxiv.org/abs/2510.05381
  - evidence: 'even when models can perfectly retrieve all relevant information, their performance still degrades substantially (13.9%--85%) as input length increases ... This failure occurs even when the irrelevant tokens are replaced with minimally distracting whitespace, and, more surprisingly, when they are all masked.'
- **modarressi2025nolima**: Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, Hinrich Schutze (2025). NoLiMa: Long-Context Evaluation Beyond Literal Matching. ICML 2025; arXiv:2502.05167. https://arxiv.org/abs/2502.05167
  - evidence: 'At 32K, for instance, 11 models drop below 50% of their strong short-length baselines. Even GPT-4o ... experiences a reduction from an almost-perfect baseline of 99.3% to 69.7%.' 'we define the threshold as 85% of the base score. Thus, the effective length of a model is the largest tested length that exceeds this threshold'; 'normalized score, calculated by dividing the accuracy score by the base score.'
- **dai2026denoising**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong (2026). LLM-Oriented Information Retrieval: A Denoising-First Perspective. SIGIR 2026 (per arXiv comments); arXiv:2605.00505. https://arxiv.org/abs/2605.00505
  - evidence: 'When a single gold passage is buried among 9 noise passages (SNR = 0.10), EM falls to 26.6%, barely above the closed-book baseline of 23.6%. Pure noise contexts yield only 8.0% EM'; 'A single gold passage alone achieves 43.2% EM, but adding 9 noise passages collapses performance to 22.2-26.6% regardless of position.' Experiment: LLaMA-2-7B-Chat, 500 NQ samples.
- **xia2026rot**: Shijie Xia, Yikun Wang, Zhen Huang, Pengfei Liu (2026). Diagnosing and Mitigating Context Rot in Long-horizon Search. arXiv:2606.29718 (v2, Aug 2026). https://arxiv.org/abs/2606.29718
  - evidence: 'we define the premature termination rate as the proportion of give-up and uncertain incorrect outcomes. By controlling for query difficulty, we show that the premature termination rate is positively correlated with context length'; context management methods 'are inherently test-time scaling strategies'; 'for backbone models with strong agentic ability, context isolation using sub-agent calls can outperform other methods'.
- **martin2026monitor**: Sam Martin, Fabien Roger (2026). Classifier Context Rot: Monitor Performance Degrades with Context Length. arXiv:2605.12366. https://arxiv.org/abs/2605.12366
  - evidence: 'Opus 4.6, GPT 5.4, and Gemini 3.1 miss these actions 2x to 30x more often when they occur after 800K tokens of benign activity than when they occur on their own'; 'recall decreases from 98.6% to 88% when 800k tokens of benign actions are prepended.'
- **wang2026degradation**: Weiwei Wang, Jiyong Min, Weijie Zou (2026). Intelligence Degradation in Long-Context LLMs: Critical Threshold Determination via Natural Length Distribution Analysis. arXiv:2601.15300 (preprint). https://arxiv.org/abs/2601.15300
  - evidence: Abstract: 'intelligence degradation-defined as over 30% drop in task performance'; 'we identify the critical threshold for Qwen2.5-7B at 40-50% of maximum context length, where F1 scores drop from 0.55-0.56 to 0.3 (45.5% degradation)'. Unreviewed preprint; only the abstract-level claims are used.
- **tirmazi2026compaction**: Hayder Tirmazi, Sam Markelon, Allison Bishop, Michael Mitzenmacher (2026). Context Compaction Theory. arXiv:2608.01326 (preliminary version). https://arxiv.org/abs/2608.01326
  - evidence: Definition 1: selector 'chooses a subset S of X with sum of s(x_i) <= B ... awarded a score val(S,q) := v_q(S) and incurs error err(S,q) := 1 - val(S,q)'. 'We exhibit a family of queries on which selection needs a factor of Theta(log n) more budget than generation'. 'Due to context rot, there is often a cutoff point even smaller than the hard limit W ... This cutoff point is the effective context window.'
- **anthropic2025context**: Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, Jeremy Hadfield (Anthropic Applied AI team) (2025). Effective context engineering for AI agents. Anthropic Engineering blog (published 29 Sep 2025). https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
  - evidence: 'context rot: as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases'; 'Context, therefore, must be treated as a finite resource with diminishing marginal returns'; 'This results in n^2 pairwise relationships for n tokens'; 'finding the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome.'
- **mei2025survey**: Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu (2025). A Survey of Context Engineering for Large Language Models. arXiv:2507.13334. https://arxiv.org/abs/2507.13334
  - evidence: Abstract: 'This survey introduces Context Engineering, a formal discipline that transcends simple prompt design to encompass the systematic optimization of information payloads for LLMs ... foundational components: context retrieval and generation, context processing and context management ... systematic analysis of over 1400 research papers'.
- **zhang2025rlm**: Alex L. Zhang, Tim Kraska, Omar Khattab (2025). Recursive Language Models. arXiv:2512.24601 (v1 31 Dec 2025; v3 11 May 2026). https://arxiv.org/abs/2512.24601
  - evidence: 'a general inference paradigm that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt'; 'tend to exhibit context rot (Hong et al., 2025), a phenomenon ... where quality degrades steeply as prompts get longer'. Table 1 (GPT-5): base 24.0/0.0/44.0/0.1 vs RLM depth=1 62.0/91.3/56.0/58.0 on CodeQA/BrowseComp+/OOLONG/OOLONG-Pairs; Qwen3-Coder CodeQA: depth 0 66.0, depth 1 56.0, depth 2 54.0, depth 3 44.0. 'RLM-Qwen3-8B outperforms the underlying Qwen3-8B model by 28.3% on average'.
- **wang2026overthink**: Daren Wang (2026). Think, But Don't Overthink: Reproducing Recursive Language Models. arXiv:2603.02615. https://arxiv.org/abs/2603.02615
  - evidence: 'applying deeper recursion (depth=2) or using RLMs on simple retrieval tasks paradoxically degrades performance and exponentially inflates execution time (e.g., from 3.6s to 344.5s) and token costs'; 'For DeepSeek v3.2, accuracy dropped to 85.0% at Depth=1 and further plummeted to 70.0% at Depth=2' on S-NIAH (base 100.0%).
- **alizadeh2026srlm**: Keivan Alizadeh, Parshin Shojaee, Minsik Cho, Mehrdad Farajtabar (2026). Recursive Language Models Meet Uncertainty: The Surprising Effectiveness of Self-Reflective Program Search for Long Context. arXiv:2603.15653 (preprint). https://arxiv.org/abs/2603.15653
  - evidence: 'SRLM consistently outperforms state-of-the-art baselines, yielding up to 22% improvement over RLM under the same time budget. Our findings show that recursion itself is not the primary driver of performance in RLM'; 'for context lengths within the model's window, RLMs with recursion often degrade performance relative to the base model'.
- **sarthi2024raptor**: Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024; arXiv:2401.18059. https://arxiv.org/abs/2401.18059
  - evidence: 'BIC = ln(N)k - 2 ln(L-hat)'; 'we use the collapsed tree with 2000 maximum tokens, which approximately equates to retrieving the top-20 nodes'; 'The average ratio of the summary length to the sum of child node lengths across all datasets is 0.28, indicating a 72% compression rate'; 'between 18.5% to 57% of the retrieved nodes come from non-leaf nodes'.
- **edge2024graphrag**: Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130 (v2, Feb 2025). https://arxiv.org/abs/2404.16130
  - evidence: 'These summaries are generated in a bottom-up manner following the hierarchical structure of extracted communities, with summaries at higher levels of the hierarchy recursively incorporating lower-level summaries'; 'we use Leiden community detection in a hierarchical manner, recursively detecting sub-communities within each detected community'.
- **wang2023recsum**: Qingyue Wang, Yanhe Fu, Yanan Cao, Shuai Wang, Zhiliang Tian, Liang Ding (2023). Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. arXiv:2308.15022 (arXiv comment: accepted by Neurocomputing). https://arxiv.org/abs/2308.15022
  - evidence: 'P(M_N|S) = prod_{i=1}^{N} P(M_i|S_i, M_{i-1}) is a sequential or Markov process where each memory M_i of session i depends only on the current session and the previous memory M_{i-1}.'
- **sun2026svdrag**: Zhihui Sun (2026). SVD-RAG: Efficient Tree-Organized Retrieval-Augmented Generation via Singular Value Decomposition. arXiv:2607.10316 (single-author preprint). https://arxiv.org/abs/2607.10316
  - evidence: 'E(k) = sum_{i<=k} sigma_i / sum_{i<=r} sigma_i ... We select k as the smallest integer such that E(k) >= tau (default tau=0.95)'; 'SVD-RAG achieves retrieval quality within 1-5% of RAPTOR with LLM summarization (MRR 0.867 vs. 0.875 ...)'; 'On our small corpora, SVD summaries tend to be longer than the original text'. Corpora of 38 and 205 chunks only.
- **jiang2023llmlingua**: Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP 2023; arXiv:2310.05736. https://arxiv.org/abs/2310.05736
  - evidence: 'the compression rate is defined as tau = L-tilde/L, tau in [0,1], and the compression ratio is 1/tau'; objective 'min KL(P(x-tilde_G | x-tilde), P(x_G | x))'; 'The compressed prompt obtained from each segment is concatenated to the subsequent segment, enabling more accurate estimation of the conditional probability'; 'tokens in each s_j with the PPL greater than gamma_j are retained'; 'allows for up to 20x compression with little performance loss'.
- **pan2024llmlingua2**: Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Ruhle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. Findings of ACL 2024; arXiv:2403.12968. https://arxiv.org/abs/2403.12968
  - evidence: 'information entropy may be a suboptimal compression metric: (i) it only leverages unidirectional context ... (ii) it is not aligned with the prompt compression objective'; 'We formulate prompt compression as a token classification problem to guarantee the faithfulness of the compressed prompt to the original one'.
- **li2023selective**: Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin (2023). Compressing Context to Enhance Inference Efficiency of Large Language Models. EMNLP 2023; arXiv:2310.06201. https://arxiv.org/abs/2310.06201
  - evidence: 'I(x) = -log_2 P(x_t | x_0, x_1, ..., x_{t-1})'; 'we can calculate the self-information of a lexical unit by simply summing the self-information of the tokens in it'; 'we design a percentile-based filtering approach'; 'we achieve a 50% reduction in context cost, resulting in a 36% reduction in inference memory usage and a 32% reduction in inference time'.
- **xu2024recomp**: Fangyuan Xu, Weijia Shi, Eunsol Choi (2024). RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation. ICLR 2024; arXiv:2310.04408. https://arxiv.org/abs/2310.04408
  - evidence: 'If the retrieved documents are irrelevant to the input or offer no additional information to LM, our compressor can return an empty string, implementing selective augmentation'; 'compression rate of as low as 6% with minimal loss in performance'. Venue confirmed on the ICLR 2024 proceedings page.
- **chirkova2025provence**: Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stephane Clinchant (2025). Provence: efficient and robust context pruning for retrieval-augmented generation. ICLR 2025; arXiv:2501.16214. https://arxiv.org/abs/2501.16214
  - evidence: 'We simply use a threshold T to binarize the token probabilites (keep or not)'; 'for which T=0.1 and T=0.5, we observe that Provence pruning ratio automatically varies from 50% to 80%, depending on the dataset'; 'the optimal compression ratio (without losing performance) is specific to each particular question-context pair. Thus, using a threshold as a hyperparameter is more appropriate for this task.'
- **zhu2024ib**: Kun Zhu, Xiaocheng Feng, Xiyuan Du, Yuxuan Gu, Weijiang Yu, Haotian Wang, Qianglong Chen, Zheng Chu, Jingchang Chen, Bing Qin (2024). An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation. ACL 2024; arXiv:2406.01549. https://arxiv.org/abs/2406.01549
  - evidence: 'min L_IB = I(X-tilde, X | Q) [conciseness] - beta I(X-tilde; Y | Q) [correctness]'; 'this term guarantees that the filtered information will be rendered null when the retrieved content is entirely irrelevant to the output'; abstract reports a 2.5% compression rate.
- **chakraborty2026conformal**: Debashish Chakraborty, Eugene Yang, Daniel Khashabi, Dawn Lawrie, Kevin Duh (2026). Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction. ECIR 2026; arXiv:2511.17908. https://arxiv.org/abs/2511.17908
  - evidence: 'tau-hat_alpha = Quantile_{1-alpha}({A(q,s) : (q,s) in D_cal, r(q,s)=1}) ... At test time, a snippet s is retained iff A(q,s) <= tau-hat_alpha. Under the exchangeability assumption ... this guarantees P(s in K_q | r(q,s)=1) >= 1-alpha'; 'Conformal filtering consistently meets its target coverage ... and reduces retained context by 2-3x relative to unfiltered retrieval.'
- **shi2025longcodezip**: Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, Xiaodong Gu (2025). LongCodeZip: Compress Long Context for Code Language Models. ASE 2025; arXiv:2510.00446. https://arxiv.org/abs/2510.00446
  - evidence: 'AMI(c,q) = PPL(q) - PPL(q | c)'; 'LLMLingua and LLMLingua-2 exhibit poor performance because token-level compression corrupts code syntax and structure'; 'conditional perplexity-based ranking outperforms similarity-based approaches by 7.89% and random selection by 17.79% in ES score'; 'achieving up to a 5.6x compression ratio without degrading task performance'; block selection formulated 'as a classic 0/1 knapsack problem'.
- **zhang2025hcp**: Lei Zhang, Yunshui Li, Jiaming Li, Xiaobo Xia, Jiaxi Yang, Run Luo, Minzheng Wang, Longze Chen, Junhao Liu, Qiang Qu, Min Yang (2025). Hierarchical Context Pruning: Optimizing Real-World Code Completion with Repository-Level Pretrained Code LLMs. AAAI 2025 (Proc. AAAI Conf. on Artificial Intelligence 39(24), pp. 25886-25894); arXiv:2406.18294. https://arxiv.org/abs/2406.18294
  - evidence: 'maintaining the topological dependencies of files and increasing the code file content in the completion prompts can improve completion accuracy; pruning the specific implementations of functions in all dependent files does not significantly reduce the accuracy of completions'; 'the HCP successfully reduced the input from over 50,000 tokens to approximately 8,000 tokens'. Venue and pages confirmed on ojs.aaai.org.
- **wang2026lamr**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo (2026). Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning. COLM 2026 Workshop on Efficient Reasoning (per arXiv comments); arXiv:2605.15315. https://arxiv.org/abs/2605.15315
  - evidence: Proposition 2: utility 'U(S) = sum_{i in S} r_i - lambda sum_{i in S} c_i - M sum_{(i,j) in E} 1[i in S, j not in S]' ... 'every optimal S* is dependency-closed'; Proposition 1: 'A single unconditional relevance score cannot represent this conditional behavior'; abstract: 'saves up to 31% more tokens on multi-turn agent tasks'.
- **jia2026swezze**: Haoxiang Jia, Earl T. Barr, Sergey Mechtaev (2026). Compressing Code Context for LLM-based Issue Resolution. arXiv:2603.28119. https://arxiv.org/abs/2603.28119
  - evidence: 'noise floods the context window and distracts the model from the bug-fixing signal'; 'reduce an initial context to a minimal sufficient context (MSC), a one-minimal subsequence where every retained fragment is functionally necessary'; 'maintains a stable compression rate of about 6 times across models, reduces the total token budget by 51.8%-71.3% ... improves issue resolution rates by 5.0%-9.2%'.
- **zhang2025cast**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu (2025). cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree. Findings of EMNLP 2025, pp. 8106-8116; arXiv:2506.15655. https://arxiv.org/abs/2506.15655
  - evidence: 'a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits ... boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation'; 'we measure chunk size by the number of non-whitespace characters rather than by lines'; 'We set max_chunk_size = 2000'.
- **es2023ragas**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023). Ragas: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217 (v2, Apr 2025). https://arxiv.org/abs/2309.15217
  - evidence: 'The context c(q) is considered relevant to the extent that it exclusively contains information that is needed to answer the question. In particular, this metric aims to penalise the inclusion of redundant information'; 'CR = number of extracted sentences / total number of sentences in c(q)'; also F = |V|/|S| for faithfulness.
- **ragas2026docs**: Ragas project (official documentation) (2026). Ragas documentation: Context Precision and Context Recall (available metrics). docs.ragas.io (stable documentation, accessed 2026-09-29). https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
  - evidence: Context Precision@K = sum_{k=1}^{K} (Precision@k x v_k) / Total number of relevant items in the top K results, with 'v_k in {0,1} is the relevance indicator at rank k'; Context Recall = Number of claims in the reference supported by the retrieved context / Total number of claims in the reference (context recall page: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/).
- **ru2024ragchecker**: Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, Zheng Zhang (2024). RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. arXiv:2408.08067. https://arxiv.org/abs/2408.08067
  - evidence: 'The retriever's context precision is defined as |{r-chunk_j}|/k'; 'Retriever Recall Trades-off with Generator Noise Sensitivity'; 'claim recall 61.5 -> 77.6 with k 5 -> 20 ... faithfulness 88.1 -> 92.2 ... noise sensitivity 34.0 -> 35.4'.
- **joren2024sufficient**: Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, Cyrus Rashtchian (2024). Sufficient Context: A New Lens on Retrieval Augmented Generation Systems. arXiv:2411.06037 (v3, Apr 2025). https://arxiv.org/abs/2411.06037
  - evidence: 'Definition (Sufficient Context). An instance q' = (Q, C) has sufficient context if and only if there exists an answer A' such that A' is a plausible answer to the question Q given the information in C.'
- **carbonell1998mmr**: Jaime Carbonell, Jade Goldstein (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. SIGIR 1998 (Melbourne, Australia). https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
  - evidence: Abstract: 'The Maximal Marginal Relevance (MMR) criterion strives to reduce redundancy while maintaining query relevance in re-ranking retrieved documents and in selecting appropriate passages for text summarization.' Section 2 gives the MMR arg-max definition (the text layer of this scan is partly garbled; the clean formula was read from goldstein1998mmr and lin2011submodular).
- **goldstein1998mmr**: Jade Goldstein, Jaime Carbonell (1998). Summarization: (1) Using MMR for Diversity-Based Reranking and (2) Evaluating Summaries. ACL Anthology X98-1025. https://aclanthology.org/X98-1025.pdf
  - evidence: 'MMR(C,Q,R,S) = Argmax_{Di in R\\S} [lambda*Sim1(Di,Q) - (1-lambda) Max_{Dj in S} (Sim2(Di,Dj))]'; 'MMR computes incrementally the standard relevance-ranked list when the parameter lambda=1, and computes a maximal diversity ranking among the documents in R when lambda=0.'
- **lin2011submodular**: Hui Lin, Jeff Bilmes (2011). A Class of Submodular Functions for Document Summarization. ACL-HLT 2011. https://aclanthology.org/P11-1052.pdf
  - evidence: 'for any A subset of B subset of V minus v, a submodular function F must satisfy F(A+v) - F(A) >= F(B+v) - F(B)'; 'Theorem 2 ... F_MMR is non-monotone submodular ... so the greedy algorithm's constant-factor approximation guarantee does not apply'; 'F(S) = L(S) + lambda R(S)'; 'a practical greedy algorithm with a (1 - 1/sqrt(e))-approximation factor'; facility location 'L(S) = sum_{i in V} max_{j in S} w_{i,j}'.
- **kulesza2012dpp**: Alex Kulesza, Ben Taskar (2012). Determinantal point processes for machine learning. Foundations and Trends in Machine Learning 5(2-3), pp. 123-286; arXiv:1207.6083. https://arxiv.org/abs/1207.6083
  - evidence: 'P_L(Y = Y) = det(L_Y) / det(L + I)'; 'L_ij = q_i phi_i^T phi_j q_j ... We can think of q_i in R+ as measuring the intrinsic goodness of an item i, and phi_i^T phi_j in [-1,1] as a signed measure of similarity'; Theorem 2.9: 'dpp-mode is NP-hard'.
- **erkan2004lexrank**: Gunes Erkan, Dragomir R. Radev (2004). LexRank: Graph-based Lexical Centrality as Salience in Text Summarization. Journal of Artificial Intelligence Research 22, pp. 457-479; arXiv:1109.2128. https://arxiv.org/abs/1109.2128
  - evidence: 'p(u) = d/N + (1-d) sum_{v in adj[u]} p(v)/deg(v) (8) where N is the total number of nodes in the graph, and d is a damping factor, which is typically chosen in the interval [0.1, 0.2]'; 'By the Perron-Frobenius theorem (Seneta, 1981), an irreducible and aperiodic Markov chain is guaranteed to converge to a unique stationary distribution'; Algorithm 2 (power method).
- **gutierrez2024hipporag**: Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024; arXiv:2405.14831. https://arxiv.org/abs/2405.14831
  - evidence: 'runs the Personalized PageRank (PPR) algorithm on the KG, using the query concepts as the seeds'; 'the PPR damping factor at 0.5, which determines the probability that PPR will restart a random walk from the query nodes instead of continuing to explore the graph'; 'outperforms the state-of-the-art methods remarkably, by up to 20% ... 10-30 times cheaper and 6-13 times faster' than iterative retrieval like IRCoT.
- **sobolev2011contracting**: V.I. Sobolev (originator) (2011). Contracting-mapping principle. Encyclopedia of Mathematics (EMS Press / Springer), page revision of 7 Feb 2011. https://encyclopediaofmath.org/wiki/Contracting-mapping_principle
  - evidence: 'A theorem asserting the existence and uniqueness of a fixed point of a mapping of a complete metric space (or a closed subset of such a space) into itself' under a contraction inequality; 'starting from an arbitrary element, construct the sequence ... which converges to some element. This will be the unique solution'; 'also known as the contraction principle or Banach's fixed-point theorem. It was proved by S. Banach' (Fund. Math. 3 (1922) pp. 7-33).
- **ross2026redundancy**: Jonathan J Ross, Bevan Koopman, Anton van der Vegt, Guido Zuccon (2026). How retriever redundancy and diversity impact RAG effectiveness. arXiv:2608.13956. https://arxiv.org/abs/2608.13956
  - evidence: 'We show that duplicate redundancy and LLM paraphrasing does not significantly improve answer correctness. However, providing diverse documents is highly beneficial, improving answer correctness by 17%-47%.' Evaluated on FictionalQA.
