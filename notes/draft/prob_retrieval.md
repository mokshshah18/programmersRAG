# DRAFT (not yet citation-checked): Probabilistic relevance, adaptive cut-offs and graph tracing: evidence and formulas for a KDE/Bayes 'probabilistic trace' over a code call/similarity graph with stopping threshold tau = 0.25

This file is the raw research dossier. The checked version is `paper/notes/prob_retrieval.md` once it exists;
prefer that file when it is present.

## Recommendations

- State tau = 0.25 in the paper as a cost assumption, not as a magic number: it is the Bayes-optimal threshold exactly when omitting a relevant function costs three times as much as including an irrelevant one. Add a sensitivity sweep over tau in {0.10, 0.25, 0.50} and note that F-measure-style objectives do not admit a fixed probability threshold.
- Build a labelled calibration set on the mimalloc fork before any density estimation: (query, function, relevant?) triples, for example synthetic questions per function from the local model with positives = the function plus its verified callees. Keep a disjoint held-out split for conformal calibration and for reliability diagrams.
- Estimate f1 and f0 with a Gaussian KDE on a transformed score (atanh of cosine similarity) using a Scott or Silverman bandwidth chosen per class, then tabulate the posterior on a fixed grid (for example 256 to 1024 points) and apply pair-adjacent-violators to make it non-decreasing. A lookup table of this size is a few kilobytes, so the per-candidate posterior evaluation is branch-light and stays inside the 16 KB per-thread working set.
- Do all probability arithmetic in log-odds: posterior logit = prior logit + log-likelihood ratio. This makes the prior-shift fusion a single addition, avoids underflow in products, and turns path products into sums of -ln w so the trace is a bounded Dijkstra search with cut-off 1.386 nats.
- Estimate edge transfer probabilities per edge type from the calibration set (fraction of relevant callers whose callee is also relevant, and the reverse; similarity edges from the KDE grouping) and clamp them to w_max < 1. Report the implied hop bound floor(ln tau / ln w_max) so readers can see the trace is finite by construction.
- Protect against hubs. mimalloc has utility functions with very high in-degree (assertion, error-message, atomic and statistics helpers); down-weight them with a specificity factor 1/|callers| or a degree-normalised threshold as in approximate PPR, and decide expansion with max-product while using noisy-OR only to rank accepted nodes.
- Make edge weights query-aware where affordable: multiply w_uv by a gate max(cos(e_v, e_q), 0) as in query-aware spreading activation, since static transition weights are the documented cause of semantic drift in PPR-based retrieval.
- Add a distribution-free guard: calibrate a conformal-risk-control threshold for a target false-negative rate (for example alpha = 0.1 or 0.2) on held-out queries and use the more inclusive of tau and the conformal threshold. Report empirical coverage, mean set size and tokens.
- Evaluate against the right baselines: fixed top-k (k = 5, 10, 20), Adaptive-k largest gap with buffer 5, PPR top-k (damping 0.5) and k-hop ego-graph (k = 1, 2). Metrics: recall and precision of the accepted set, set size, tokens, expected calibration error or Brier score of the posterior, and downstream answer or edit correctness.
- Re-estimate the cost ratio after fine-tuning and quantisation. Training on mixed relevant and irrelevant contexts makes a model more tolerant of distractors, which lowers c_FP and therefore the optimal tau; the threshold should be a configurable parameter exposed next to the fine-tuning toggle, not a compile-time constant.
- Order the accepted functions so the highest-posterior items sit at the start and end of the prompt, and truncate by posterior rank if the accepted set exceeds the context budget; both choices follow from the lost-in-the-middle and conformal-retrieval caveats.
- Citations to avoid unless someone opens them first: Platt's original 1999/2000 chapter, Lewis (SIGIR 1995), Kanoulas et al. (SIGIR 2010), Collins and Loftus (1975), Silverman (1986), Scott (1992), Pearl (1988), Bahri et al. 'Surprise' (SIGIR 2023) and Saerens et al. (2002) were not opened in this session; they appear here only as cited by the opened sources. The web-search budget of the session was exhausted part-way, so later discovery used the arXiv and Semantic Scholar APIs.
- Local copies of every opened PDF and the extracted text are in C:\Users\CASHAMOK\AppData\Local\Temp\claude\C--Users-CASHAMOK-Documents-GitHub-programmersRAG\6fffbf87-56fc-4337-8f23-de8cab3f47af\scratchpad\lit (no repository files were modified).

## Findings

- [established] Probability Ranking Principle (PRP): ranking documents by decreasing probability of relevance gives the best effectiveness obtainable from the available data, and is Bayes-optimal (minimises expected loss) under 1/0 loss; for set retrieval the Bayes-optimal rule returns documents with P(R=1|d,q) > P(R=0|d,q). The optimality proof requires that all probabilities are known correctly, which 'is never the case in practice'. Robertson (1977) shows the principle is justified only under certain conditions and that it treats documents one at a time.  (robertson1977prp, manning2008ir)
- [established] With asymmetric costs the PRP becomes a cost rule: retrieve next the document minimising C0*P(R=0|d) - C1*P(R=1|d) (C1 = cost of missing a relevant document, C0 = cost of retrieving a non-relevant one). In general two-class cost-sensitive decision theory the optimal decision is 'positive iff p >= p*' with p* = (c10 - c00)/(c10 - c00 + c01 - c11), so any fixed probability threshold corresponds to exactly one cost ratio.  (manning2008ir, elkan2001cost)
- [our_proposal] OUR DERIVATION: the owner's threshold tau = 0.25 is decision-theoretically equivalent to declaring a false negative (omitting a relevant function) three times as costly as a false positive (including an irrelevant function): with c00 = c11 = 0, p* = c_FP/(c_FP + c_FN) = 0.25 gives c_FN/c_FP = (1 - tau)/tau = 3. Equivalently the likelihood-ratio test is f1(s)/f0(s) >= (1/3)*(1 - pi)/pi, e.g. LR >= 3 when the prior pi = 0.10 and LR >= 33 when pi = 0.01. The paper should state this cost ratio as the modelling assumption behind tau.  (elkan2001cost, manning2008ir)
- [contested] A fixed probability threshold is NOT optimal for every objective. 'While for some measures there exists an optimal fixed probability threshold, for others it does not'; Lewis's probability thresholding principle is satisfied by linear utility measures but not by the F measure, for which the optimal cut-off is a per-ranking rank/score threshold. The optimal threshold 'depends on the effectiveness measure being used - there is no single threshold suitable for all purposes'.  (arampatzis2008stop, arampatzis2011modeling)
- [established] Score-distribution (SD) models treat the scores of relevant and non-relevant documents for a query as a two-component mixture (classically normal for relevant, exponential for non-relevant), recover the components with EM even without relevance labels, and convert a score into P(rel|score) by Bayes' rule with the mixing weights as priors. This is the direct ancestor of the requested 'probability that the next function is relevant'.  (manmatha2001scoredist, arampatzis2011modeling, arampatzis2008stop)
- [established] Known defect of the normal-exponential model: because a normal falls faster than an exponential the two densities intersect twice, so the posterior is non-monotone at the top of the ranking (it violates the Recall-Fallout Convexity Hypothesis 'only (and always) at both ends of the score range'). Published patches (straight line to (1,1), randomisation within the offending range) are called 'theoretically unjustified'. The theoretical argument favours two-gamma as the more universal model; a Gaussian-mixture + Gamma model fitted by variational inference outperformed the exponential-Gaussian model.  (manmatha2001scoredist, arampatzis2011modeling, kanoulas2009scoredist)
- [established] Practical failure modes of any mixture/SD approach: relevance judgements are usually absent, sparse, incomplete or biased; EM is 'messy', hard to tune and sensitive to initial values; EM treats all data equally, producing a ranking-length bias ('the more the score data, the lower the estimated threshold'); monotone score transformations that change shape invalidate the chosen component family.  (arampatzis2011modeling)
- [established] Kernel density estimation (Parzen window): f_n(x) = (1/(n h)) sum_j K((x - X_j)/h) is asymptotically unbiased at continuity points when h(n) -> 0 under mild kernel conditions. In practice bandwidth selection 'strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel)'; standard rules of thumb are Scott's factor n^(-1/(d+4)) and Silverman's factor (n(d+2)/4)^(-1/(d+4)).  (parzen1962kde, scipy_gaussian_kde)
- [recent] A Bayes posterior built from class-conditional KDEs is an established calibrated classifier: P{Y=k|X=x0} = f_k(x0) pi_k / sum_k' f_k'(x0) pi_k' with f_k replaced by a kernel density estimate on a calibration set and pi_k by class proportions; KCal proves a full calibration guarantee for this construction as the per-class calibration sample grows.  (lin2022kcal)
- [our_proposal] OUR PROPOSAL: replace the parametric normal/exponential components of the SD model by class-conditional KDEs of the query-function similarity score, f1_hat (relevant) and f0_hat (non-relevant), and compute P(R=1|s) = pi f1_hat(s) / (pi f1_hat(s) + (1 - pi) f0_hat(s)). This removes the component-family selection problem but inherits (i) the possibility of a non-monotone posterior where the density estimates cross more than once and (ii) the label-sparsity problem for the relevant class. Monotonicity can be restored by projecting the posterior onto non-decreasing functions with pair-adjacent-violators isotonic regression.  (manmatha2001scoredist, parzen1962kde, lin2022kcal, zadrozny2002isotonic, arampatzis2011modeling)
- [established] Alternative score-to-probability calibrators: Platt scaling fits a two-parameter sigmoid 1/(1+exp(A f + B)) by regularised maximum likelihood with smoothed targets; isotonic regression (PAV) is a non-parametric intermediate between sigmoid fitting and binning that only assumes the classifier ranks correctly; logistic regression has long been used to calibrate IR scores into probabilities of relevance. Regression-style calibrators optimise overall fit, not necessarily quality at the top of the ranking.  (lin2007platt, zadrozny2002isotonic, arampatzis2011modeling)
- [established] A conventional KDE suffers 'well known boundary bias' when the variable has bounded support; transforming to an unconstrained variable (probit transformation) combined with local-likelihood density estimation gives estimators 'mostly free from boundary issues'.  (geenens2013probit)
- [our_proposal] OUR PROPOSAL: because cosine similarity is bounded in [-1, 1] and relevant scores pile up near the upper bound, estimate the class-conditional densities on a transformed score (e.g. z = atanh(s) or the probit of (s+1)/2) rather than on the raw score; the Jacobian cancels in the likelihood ratio f1/f0, so the posterior can be evaluated directly in the transformed space.  (geenens2013probit, parzen1962kde)
- [established] Neural rankers are not reliably calibrated: the PRP holds only if models are well calibrated and probabilities are reported with certainty, and BERT-based rankers 'are not robustly calibrated'. Raw similarity or re-ranker scores must therefore not be read as probabilities without a calibration stage.  (penha2021calibration)
- [established] Prior (base-rate) shift has a closed form: if p is the correct posterior under base rate b and class-conditional distributions are unchanged, the posterior under base rate b' is p' = b'(p - p b)/(b - p b + b' p - b b'). OUR NOTE: this is identical to the log-odds update logit p' = logit p - logit b + logit b' (verified numerically to 1e-15), which lets a graph-derived prior be fused with a score-derived likelihood ratio.  (elkan2001cost)
- [established] Learned ranked-list truncation: BiCut (Bi-LSTM, sequential keep/cut decisions with a loss weighting false positives vs false negatives), Choppy (Transformer over the relevance scores only, directly optimising a user-defined metric) and AttnCut (Transformer with reward-augmented maximum likelihood, a global rather than local decision, objectives F1 and best-F1-under-recall-constraint) all learn a per-query cut-off from the score list.  (lien2019bicut, bahri2020choppy, wu2021attncut)
- [contested] Evidence on learned truncation is mixed. In retrieve-then-re-rank pipelines 'supervised RLT methods do not show a clear advantage over using a fixed re-ranking depth', and with strong retrievers a fixed depth of 20 already gives an excellent trade-off; yet oracle per-query cut-offs are statistically significantly better than every fixed cut-off, so the headroom for adaptive cut-offs is real but current predictors do not capture it reliably.  (meng2024rlt)
- [recent] Adaptive-k (EMNLP 2025) chooses k at the largest gap of the descending-sorted query-passage similarities, adds a fixed buffer B = 5 and restricts the search to the top 90% of documents; it needs no tuning or extra LLM calls, uses up to 10x fewer tokens than full context while retrieving 70% of relevant passages. Stated limitation: a crafted chunk can create an artificial gap and mislead the cut-off.  (taguchi2025adaptivek)
- [recent] 2025-2026 adaptive-k work moves toward statistical modelling of the score list: TAA-k (ECML PKDD 2026) locates the onset of the noise-dominated tail by knee detection followed by Generalized-Pareto goodness-of-fit tests, assuming a two-component mixture with monotone likelihood ratio, reaching F1 within 2-3% of oracle; CAR (2025) detects the transition in clustered similarity distances (60% fewer tokens, 22% lower latency reported); pre-retrieval query clustering (CIKM 2026) assigns a depth per query cluster targeting ~95% coverage.  (song2026taak, xu2025car, xia2026preretrieval)
- [recent] Score-pattern cut-offs (largest gap, clustering) 'are efficient but assume the scorer is reliable. When the scorer itself is weak, score patterns offer little basis for choosing depth.' Adaptive shortlists can match the coverage of much longer fixed lists (90.3% at ~7 items vs 90.8% at 50 on BFCL) and improved downstream selection accuracy.  (repantis2026tools)
- [recent] Self-RAG, Adaptive-RAG and CRAG adapt whether or how to retrieve rather than producing a calibrated set size: Self-RAG trains the LM to emit reflection tokens; Adaptive-RAG routes queries with a small complexity classifier between no-retrieval, single-step and multi-step strategies; CRAG scores each document with a lightweight evaluator and triggers Correct/Incorrect/Ambiguous actions using upper and lower confidence thresholds that 'were set empirically' (e.g. (0.59, -0.99) on PopQA).  (asai2023selfrag, jeong2024adaptiverag, yan2024crag, repantis2026tools)
- [established] Split conformal prediction gives a distribution-free finite-sample guarantee P(Y_test in C(X_test)) >= 1 - alpha when the threshold is the ceil((n+1)(1-alpha))/n empirical quantile of calibration scores. The guarantee is marginal, not conditional on the input, assumes i.i.d./exchangeable calibration and test points, and 'the usefulness of the prediction sets is primarily determined by the score function'.  (angelopoulos2021gentle)
- [established] Conformal risk control extends this to the expected value of any bounded monotone loss, e.g. the false-negative rate of a retrieved set: choosing lambda_hat = inf{lambda : n/(n+1) R_n(lambda) + B/(n+1) <= alpha} guarantees E[L_{n+1}(lambda_hat)] <= alpha, and is tight up to O(1/n) (lower bound alpha - 2B/(n+1) for i.i.d. continuous losses). The guarantee fails if the loss is not monotone in lambda.  (angelopoulos2022crc)
- [recent] Conformal retrieval for RAG already exists: CONFLARE derives a similarity cut-off from a calibration set of answerable questions and a user error rate alpha and retrieves every chunk above it, 'ensuring the true answer is captured in the context with a (1-alpha) confidence level'; TRAQ gives an end-to-end correctness guarantee and shrinks set size by 16.2% with Bayesian optimisation. Caveat: low alpha or a weak embedding model can return more chunks than fit in the context window.  (rouzrokh2024conflare, li2023traq)
- [recent] Spreading activation is the classical associative-retrieval mechanism on networks; 2025-2026 RAG systems re-use it with explicit thresholds. SA-RAG updates a_j = min(1, a_j + sum_i a_i w_ij) in BFS order and keeps nodes whose activation exceeds tau_a. Query-aware spreading activation uses Delta r(v) = alpha * sigma(v) * sum over active higher-activation neighbours, with decay alpha = 0.7, depth T = 3 and threshold tau = 0.01, and reports that 'without the threshold ... propagation would cover the entire connected subgraph'; the semantic gate gave +3.6 to +7.4 F1 and 1.5-4.9x lower retrieval latency.  (crestani1997spreading, pavlovic2025sarag, makarov2026qasa)
- [established] Random walk with restart / Personalized PageRank define node relevance to a seed as the solution of a linear system (r = c W r + (1 - c) e; pr_alpha(s) = alpha s + (1 - alpha) pr_alpha(s) W). The Andersen-Chung-Lang push algorithm computes an epsilon-approximate PPR vector by pushing only from vertices whose residual satisfies r(u) >= epsilon d(u); the result has support volume <= 2/((1 - alpha) epsilon) and running time O(1/(epsilon alpha)), independent of graph size. This is the rigorous precedent for 'keep tracing until the mass falls below a threshold'.  (tong2006rwr, andersen2006local)
- [recent] HippoRAG runs PPR on an LLM-extracted knowledge graph seeded by query entities, weights seeds by node specificity s_i = |P_i|^-1, uses damping 0.5 and a synonymy-edge threshold 0.8 tuned on 100 examples, and improves multi-hop QA by up to 20% while being 10-20x cheaper than iterative retrieval. HippoRAG 2 adds passage nodes, query-to-triple seeding and an LLM triple filter, multiplies passage-node reset probability by 0.05, and reports a 7% gain on associative-memory tasks; its error analysis names triple filtering and graph search as the two main error sources.  (gutierrez2024hipporag, gutierrez2025hipporag2)
- [recent] GraphRAG builds an LLM-extracted entity graph, partitions it into a hierarchy of communities and pre-generates community summaries; it targets global 'sense-making' questions over a whole corpus rather than selecting a minimal evidence set, so it is a different objective from thresholded tracing.  (edge2024graphrag)
- [recent] Known failure mode of static-graph diffusion: fixed transition probabilities ignore the query, 'causing semantic drift where random walks are diverted into high-degree hub nodes before reaching critical downstream evidence'; remedies are query-aware edge weighting and anchoring of the start distribution (CatRAG), or a per-step query-similarity gate (query-aware spreading activation).  (lau2026catrag, makarov2026qasa)
- [recent] Repository/code-graph retrieval consistently limits expansion depth. RepoGraph (line-level definition/reference graph, +32.8% average relative improvement on SWE-bench) retrieves k-hop ego-graphs and restricts k <= 2 because of 'the potential introduction of noise or irrelevant nodes for k >= 3'. RepoHyper found that exhaustive search 'on all the directions and paths can include too many irrelevant contexts' and adds path-pattern filters plus link-prediction re-ranking. GraphCoder slices a statement-level code context graph to h hops with a maximum statement count and re-ranks with a decay-with-distance structural similarity. DyRetriever bounds multi-hop reasoning with maxHop.  (ouyang2024repograph, phan2024repohyper, liu2024graphcoder, liu2026dyretriever)
- [recent] Code retrieval benefits from combining structural (call/dependency) edges with similarity: RepoCoder iterates retrieval and generation (+10% over in-file completion); CodexGraph lets an agent query a code graph database because similarity-based retrieval 'often has low recall in complex tasks'; GraphCodeAgent (v1 title CodeRAG) traverses a requirement graph and a structural-semantic code graph for multi-hop retrieval; RepoScope predicts call chains from the observation that 'similar functions within the same repository often exhibit similar calling behaviors'; FeatLens (Sept 2026) applies personalized PageRank to a task-specific seed graph and then filters to a compact reasoning graph.  (zhang2023repocoder, liu2024codexgraph, li2025graphcodeagent, liu2025reposcope, li2026featlens)
- [established] Evidence combination along multiple edges: the noisy-OR model assigns each active cause an independent inhibitor probability q_i, giving P(effect false | causes) = product of q_i over active causes and P(effect true) = 1 - product; it needs only one parameter per parent instead of an exponential table. The independent/general cascade model on graphs has the same form, f_v(S) = 1 - prod_i (1 - p_v(u_i, S_{i-1})), valid when activation attempts are order-independent.  (srinivas1993noisyor, zhang1996causal, kempe2003influence)
- [our_proposal] OUR PROPOSAL (probabilistic trace): (1) seeds = functions whose KDE-Bayes posterior >= tau; (2) each edge (u,v) carries a transfer probability w_uv in [0, w_max], w_max < 1, estimated per edge type (caller->callee, callee->caller, similarity) from calibration data; (3) best-first expansion in order of max-product path probability rho(v) = max over paths of p_seed * prod w, i.e. Dijkstra on costs -ln w, halted when rho < tau (cost > -ln 0.25 = 1.386); (4) graph prior pi_v = 1 - (1 - pi_0) * prod over accepted parents (1 - w_uv P_u) (noisy-OR with leak pi_0); (5) final posterior logit P_v = logit pi_v + ln(f1_hat(s_v)/f0_hat(s_v)); (6) accept iff P_v >= tau. Because every w <= w_max < 1 the path probability is strictly decreasing with depth, so the procedure terminates without a separate hop limit.  (manmatha2001scoredist, lin2022kcal, elkan2001cost, srinivas1993noisyor, kempe2003influence, andersen2006local, makarov2026qasa)
- [our_proposal] OUR DERIVATION: under max-product propagation a node at hop distance k can pass the threshold only if w_max^k >= tau, i.e. k <= floor(ln tau / ln w_max). For tau = 0.25 this gives at most 2 hops at w_max = 0.5 or 0.6, 3 hops at w_max = 0.7 and 6 hops at w_max = 0.8. This reproduces from first principles the empirical depth limits used in the literature (k <= 2 in RepoGraph; T = 3 with decay 0.7 in query-aware spreading activation).  (ouyang2024repograph, makarov2026qasa)
- [our_proposal] OUR ANALYSIS of a noisy-OR failure mode: max_i a_i <= 1 - prod(1 - a_i) <= min(1, sum_i a_i), so several individually sub-threshold paths can push a node over tau (0.20, 0.15 and 0.10 combine to 0.388). When the paths share edges or all pass through one utility/hub function the independence assumption is violated and the noisy-OR over-counts, which is the probabilistic counterpart of the hub/semantic-drift problem. Mitigations: decide expansion with max-product and use noisy-OR only for ranking, count each distinct parent once, and down-weight high-degree nodes (degree-normalised residual threshold as in approximate PPR, or node specificity 1/|P_i| as in HippoRAG).  (srinivas1993noisyor, kempe2003influence, lau2026catrag, andersen2006local, gutierrez2024hipporag)
- [recent] False positives have a real cost, which justifies a non-zero c_FP: high-scoring retrieved documents that do not contain the answer 'negatively impact the effectiveness of the LLM'; retrieval augmentation can reduce accuracy when context is irrelevant, especially in multi-hop settings where it causes cascading errors; and model accuracy degrades when relevant information sits in the middle of a long context. Fine-tuning on a mix of relevant and irrelevant contexts (as few as 1,000 examples) makes a model robust, which means the appropriate cost ratio, and hence tau, changes after fine-tuning.  (cuconasu2024noise, yoran2023robust, liu2023lost)
- [our_proposal] OUR PROPOSAL (guarded threshold): since the Bayes threshold tau is optimal only if the posterior is calibrated, pair it with a conformal-risk-control threshold calibrated on held-out queries for a target false-negative rate alpha, using the set family C_lambda = {v : P_v >= 1 - lambda} (monotone in lambda, loss bounded by B = 1). Use the more inclusive of the two thresholds, t* = min(tau, 1 - lambda_hat), and report both the decision-theoretic and the distribution-free guarantee; if the accepted set exceeds the context budget, truncate by posterior rank.  (angelopoulos2022crc, angelopoulos2021gentle, rouzrokh2024conflare, elkan2001cost)

## Formulas

### PRP with retrieval costs

$$
C_0\cdot P(R=0\mid d) - C_1\cdot P(R=1\mid d) \le C_0\cdot P(R=0\mid d') - C_1\cdot P(R=1\mid d')
$$

Eq. (62) of the IR book: if the inequality holds for d against all not-yet-retrieved d', d is retrieved next. C_1 is the cost of not retrieving a relevant document, C_0 the cost of retrieving a non-relevant one. Under 1/0 loss the set rule is: d is relevant iff P(R=1|d,q) > P(R=0|d,q) (Eq. 61).

Sources: manning2008ir

### Expected cost of a decision

$$
L(x,i)=\sum_{j}P(j\mid x)\,C(i,j)
$$

Elkan Eq. (1): the optimal prediction for example x is the class i minimising expected cost, where C(i,j) is the cost of predicting i when the truth is j.

Sources: elkan2001cost

### Optimal probability threshold from a cost matrix

$$
p^{*}=\frac{c_{10}-c_{00}}{c_{10}-c_{00}+c_{01}-c_{11}}
$$

Elkan Eq. (2): predict positive iff p = P(j=1|x) >= p*. c_10 is the false-positive cost, c_01 the false-negative cost, c_00 and c_11 the costs of correct decisions. Requires the reasonableness conditions c_10 > c_00 and c_01 > c_11.

Sources: elkan2001cost

### Cost ratio implied by tau (our derivation)

$$
\tau=\frac{c_{FP}}{c_{FP}+c_{FN}}\;\Longleftrightarrow\;\frac{c_{FN}}{c_{FP}}=\frac{1-\tau}{\tau},\qquad \tau=0.25\;\Rightarrow\;c_{FN}=3\,c_{FP}
$$

Derivation: set c_00 = c_11 = 0 in Elkan Eq. (2), write c_10 = c_FP and c_01 = c_FN, and solve for the ratio. The same result follows from IR-book Eq. (62): retrieving d is cheaper than not retrieving it iff C_0 (1 - P) < C_1 P, i.e. P > C_0/(C_0 + C_1). OUR PROPOSAL.

Sources: elkan2001cost, manning2008ir

### Score-to-probability by Bayes' rule

$$
P(rel\mid score)=\frac{P(score\mid rel)\,P(rel)}{P(score\mid rel)\,P(rel)+P(score\mid nonrel)\,P(nonrel)}
$$

Manmatha et al. Eq. (8). In their model P(score|rel) is the Gaussian component and P(score|nonrel) the exponential component of a per-query mixture fitted by EM; P(rel) and P(nonrel) are the mixing weights.

Sources: manmatha2001scoredist

### Normal-exponential score densities

$$
P(s\mid 1)=\frac{1}{\sigma}\,\phi\!\left(\frac{s-\mu}{\sigma}\right),\qquad P(s\mid 0)=\psi(s-s_{\min};\lambda)
$$

Eqs. (7)-(8) of Arampatzis and Kamps (TREC 2008): phi is the standard normal density (sigma > 0), psi the exponential density with rate lambda > 0 defined for s >= s_min. Baseline parametric model against which a KDE variant should be compared.

Sources: arampatzis2008stop

### Parzen-Rosenblatt kernel density estimator

$$
f_n(x)=\frac{1}{nh}\sum_{j=1}^{n}K\!\left(\frac{x-X_j}{h}\right)
$$

Parzen Eq. (1.7). Asymptotically unbiased at continuity points of f when h = h(n) -> 0 and K satisfies Parzen's conditions (1.11)-(1.13) (bounded, absolutely integrable, |yK(y)| -> 0).

Sources: parzen1962kde

### Rule-of-thumb bandwidth factors

$$
h_{\text{Scott}}\propto n^{-1/(d+4)},\qquad h_{\text{Silverman}}\propto\left(\frac{n(d+2)}{4}\right)^{-1/(d+4)},\qquad n_{\text{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^{2}}
$$

Factors multiplying the data standard deviation/covariance as documented for scipy.stats.gaussian_kde; n is the number of points, d the dimension (d = 1 for a scalar score), n_eff the effective sample size for weighted samples.

Sources: scipy_gaussian_kde

### Bayes posterior from class-conditional KDEs

$$
\mathbb{P}\{Y=k\mid X=x_0\}=\frac{f_k(x_0)\,\pi_k}{\sum_{k'}f_{k'}(x_0)\,\pi_{k'}},\qquad \hat f_k(x_0)=\frac{1}{|\mathcal S^{k}_{cal}|}\sum_{(x,y)\in\mathcal S^{k}_{cal}}\hat\phi_b(x,x_0),\quad \hat\pi_k=\frac{|\mathcal S^{k}_{cal}|}{|\mathcal S_{cal}|}
$$

KCal Eq. (4) and the text below it: f_k is the density of class k, pi_k its proportion; plugging in the kernel density estimate with bandwidth b over the calibration set of class k yields the KDE classifier.

Sources: lin2022kcal

### KDE relevance posterior and likelihood-ratio test (our proposal)

$$
\hat P(R=1\mid s)=\frac{\pi\,\hat f_1(s)}{\pi\,\hat f_1(s)+(1-\pi)\,\hat f_0(s)}\;\ge\;\tau\;\Longleftrightarrow\;\Lambda(s)=\frac{\hat f_1(s)}{\hat f_0(s)}\;\ge\;\frac{\tau}{1-\tau}\cdot\frac{1-\pi}{\pi}
$$

Derivation: specialise the Bayes posterior to two classes (relevant/non-relevant) with Parzen estimates f1_hat, f0_hat of the score densities and prior pi = P(R=1); dividing numerator and denominator by (1-pi) f0_hat and rearranging P >= tau gives the likelihood-ratio form. With tau = 0.25 the right-hand side is (1/3)(1-pi)/pi. OUR PROPOSAL.

Sources: manmatha2001scoredist, parzen1962kde, lin2022kcal

### Platt scaling

$$
\Pr(y=1\mid x)\approx P_{A,B}(f)\equiv\frac{1}{1+\exp(Af+B)},\qquad \min_{A,B}\;-\sum_{i=1}^{l}\big[t_i\log p_i+(1-t_i)\log(1-p_i)\big],\quad t_i=\begin{cases}\dfrac{N_{+}+1}{N_{+}+2}&y_i=+1\\[4pt]\dfrac{1}{N_{-}+2}&y_i=-1\end{cases}
$$

Eqs. (1)-(2) of Lin, Lin and Weng restating Platt's method: f is the raw decision value, p_i = P_{A,B}(f_i), N+ and N- the numbers of positive and negative calibration examples. Parametric baseline for the KDE posterior.

Sources: lin2007platt

### Posterior under a changed base rate

$$
p'=b'\,\frac{p-pb}{\,b-pb+b'p-bb'\,}\qquad\big(\text{equivalently } \operatorname{logit}p'=\operatorname{logit}p-\operatorname{logit}b+\operatorname{logit}b'\big)
$$

Elkan Theorem 2: p is the posterior under base rate b, p' the posterior under base rate b' when class-conditional distributions are unchanged. The logit form in parentheses is our algebraic restatement (checked numerically) and is what the implementation should use.

Sources: elkan2001cost

### Split conformal threshold and coverage

$$
\hat q=\text{Quantile}\!\left(s_1,\dots,s_n;\;\frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right),\qquad P\big(Y_{test}\in\mathcal C(X_{test})\big)\ge 1-\alpha
$$

s_i are conformal (non-conformity) scores on n calibration points, alpha the user error rate. Theorem 1 of the tutorial (Vovk, Gammerman and Saunders). For retrieval, s_i can be the negative similarity of the lowest-scoring truly relevant item of calibration query i.

Sources: angelopoulos2021gentle

### Conformal risk control

$$
\hat\lambda=\inf\Big\{\lambda:\;\frac{n}{n+1}\hat R_n(\lambda)+\frac{B}{n+1}\le\alpha\Big\},\quad \hat R_n(\lambda)=\frac{1}{n}\sum_{i=1}^{n}L_i(\lambda)\;\Rightarrow\;\mathbb E\big[L_{n+1}(\hat\lambda)\big]\le\alpha
$$

Eq. (4) and Theorem 1: L_i are exchangeable losses, non-increasing and right-continuous in lambda and bounded by B. For a retrieved set use the false-negative proportion L_i(lambda) = 1 - |Y_i intersect C_lambda(X_i)|/|Y_i| with B = 1. For i.i.d. continuous losses E[L] >= alpha - 2B/(n+1).

Sources: angelopoulos2022crc

### Random walk with restart

$$
\vec r_i=c\,\tilde{\mathbf W}\,\vec r_i+(1-c)\,\vec e_i\;\;\Longrightarrow\;\;\vec r_i=(1-c)\,(\mathbf I-c\tilde{\mathbf W})^{-1}\vec e_i
$$

Tong et al. Eqs. (1)-(2): W-tilde is the normalised weighted adjacency matrix, e_i the indicator of seed i, r_ij the relevance of node j to seed i. Note: as printed, c multiplies the walk term and (1-c) the restart vector, although the paper's symbol table calls c the restart probability.

Sources: tong2006rwr

### Personalized PageRank vector

$$
\mathrm{pr}_\alpha(s)=\alpha s+(1-\alpha)\,\mathrm{pr}_\alpha(s)W,\qquad W=\tfrac12\big(I+D^{-1}A\big),\qquad \mathrm{pr}_\alpha(s)=\alpha s+\alpha\sum_{t=1}^{\infty}(1-\alpha)^{t}\,sW^{t}
$$

Andersen-Chung-Lang Eqs. (1) and (3): alpha in (0,1] is the teleport probability, s the starting (seed) vector, W the lazy random-walk matrix, D the degree matrix, A the adjacency matrix. The series shows that mass reaching hop t is discounted by (1-alpha)^t.

Sources: andersen2006local

### Approximate PPR by push with residual threshold

$$
\text{push}(u):\;p'(u)=p(u)+\alpha r(u),\;\;r'(u)=\tfrac{(1-\alpha)r(u)}{2},\;\;r'(v)=r(v)+\tfrac{(1-\alpha)r(u)}{2d(u)}\;\;\forall (u,v)\in E;\qquad r(v)\le\epsilon\,d(v)\;\forall v,\quad \mathrm{vol}(\mathrm{Supp}(p))\le\frac{2}{(1-\alpha)\epsilon},\quad \text{time }O\!\Big(\frac{1}{\epsilon\alpha}\Big)
$$

Definition 1, the push operation and Theorem 1: p is the approximation, r the residual, d(v) the degree. Pushing continues only while some vertex has residual at least epsilon times its degree, so the explored region is bounded independently of graph size; the degree normalisation is what protects against hubs.

Sources: andersen2006local

### Boolean noisy-OR

$$
P(f_x\mid\mathbf u)=\prod_{\{i\,\mid\,u_i=t_i\}}q_i,\qquad P(t_x\mid\mathbf u)=1-\prod_{\{i\,\mid\,u_i=t_i\}}q_i
$$

Srinivas Sec. 5.1: q_i is the inhibitor probability of cause i (probability that an active cause fails to produce the effect); the product runs over causes that are true. For the trace, q_i = 1 - w_{uv} P_u.

Sources: srinivas1993noisyor

### Cascade activation over a set of neighbours

$$
f_v(S)=1-\prod_{i=1}^{k}\big(1-p_v(u_i,S_{i-1})\big),\qquad p_v(u,S)=\frac{f_v(S\cup\{u\})-f_v(S)}{1-f_v(S)}
$$

Kempe, Kleinberg and Tardos Sec. 4: probability that node v is activated by neighbours S = {u_1..u_k}; the independent cascade model is the special case p_v(u,S) = p_{u,v}. Requires order-independence of the attempts.

Sources: kempe2003influence

### Spreading activation with threshold (SA-RAG)

$$
a_j\leftarrow\min\Big(1,\;a_j+\sum_{i\in N(j)}a_i\,w_{ij}\Big),\qquad \text{activated}(j)\iff a_j>\tau_a
$$

Update rule and Algorithm 1 of SA-RAG: seeds start at a = 1, all other nodes at 0, propagation proceeds breadth-first over weighted links w_ij, and nodes above the activation threshold are returned.

Sources: pavlovic2025sarag

### Query-aware spreading activation

$$
\Delta r^{(t)}(v)=\alpha\,\sigma(v)\sum_{u\in A^{(t)}(v)}r^{(t-1)}(u),\quad A^{(t)}(v)=\{u:(u,v)\in E,\;r^{(t-1)}(u)>\tau,\;r^{(t-1)}(u)>r^{(t-1)}(v)\},\quad \sigma(v)=\max(\cos(\mathbf e_v,\mathbf e_q),0)
$$

Eq. (1): alpha in (0,1] is a decay factor, tau a threshold, sigma the semantic gate between the node description embedding e_v and the question embedding e_q; the monotone condition r(u) > r(v) prevents cyclic reinforcement. The update is applied only when the increment exceeds tau. Defaults T = 3, alpha = 0.7, tau = 0.01.

Sources: makarov2026qasa

### Node specificity

$$
s_i=|P_i|^{-1}
$$

HippoRAG: P_i is the set of passages from which node i was extracted; a local IDF-like weight applied to seed probabilities before PPR. For code, P_i can be the set of callers or files referencing function i.

Sources: gutierrez2024hipporag

### Largest-gap cut-off (Adaptive-k)

$$
k=\arg\max_{i}\;g_i,\qquad g_i=s_{(i)}-s_{(i+1)},\quad s_{(1)}\ge s_{(2)}\ge\dots
$$

Algorithm 1 of Adaptive-k: s are query-passage cosine similarities sorted in descending order, g their first discrete differences; the top k items plus a buffer of B = 5 are retrieved and the search is limited to the top 90% of the list. Complexity O(n log n). Baseline for comparison.

Sources: taguchi2025adaptivek

### Probabilistic trace: path probability, graph prior and fused posterior (our proposal)

$$
\rho(v)=\max_{\gamma:\,u_0\leadsto v}\;P_{u_0}\prod_{(a,b)\in\gamma}w_{ab},\qquad \pi_v=1-(1-\pi_0)\prod_{u\in\mathrm{Pa}(v)}\big(1-w_{uv}P_u\big),\qquad \operatorname{logit}P_v=\operatorname{logit}\pi_v+\ln\frac{\hat f_1(s_v)}{\hat f_0(s_v)},\qquad \text{expand } v\iff\rho(v)\ge\tau,\;\;\text{keep } v\iff P_v\ge\tau
$$

Derivation: (a) product rule along a path assuming each edge transfers relevance independently with probability w_ab; maximising the product equals a shortest-path problem with edge cost -ln w_ab and source cost -ln P_u0, and the stopping rule rho < tau equals cost > -ln tau = 1.386 for tau = 0.25; (b) noisy-OR over already-accepted parents Pa(v) with a leak term equal to the global base rate pi_0; (c) Elkan's base-rate change in logit form replaces the global prior by pi_v and adds the KDE log-likelihood ratio of the node's own score s_v. OUR PROPOSAL.

Sources: srinivas1993noisyor, kempe2003influence, elkan2001cost, lin2022kcal, manmatha2001scoredist

### Hop bound implied by the threshold (our derivation)

$$
w_{\max}^{\,k}\ge\tau\;\Longrightarrow\;k\le\left\lfloor\frac{\ln\tau}{\ln w_{\max}}\right\rfloor
$$

Derivation: a node k edges from a seed has path probability at most P_seed * w_max^k <= w_max^k; requiring this to be at least tau and taking logarithms (both negative) gives the bound. tau = 0.25: k <= 2 for w_max = 0.5 or 0.6, k <= 3 for 0.7, k <= 6 for 0.8, k <= 13 for 0.9. OUR PROPOSAL.

Sources: ouyang2024repograph, makarov2026qasa

### Noisy-OR bounds (our derivation)

$$
\max_i a_i\;\le\;1-\prod_{i}(1-a_i)\;\le\;\min\Big(1,\sum_i a_i\Big),\qquad a_i=w_{u_i v}P_{u_i}
$$

Derivation: the lower bound holds because every factor (1 - a_j) <= 1; the upper bound is the union bound (Weierstrass product inequality). Max-product is therefore a conservative and noisy-OR an optimistic combination; the gap is largest at nodes with many weak parents (hubs). OUR PROPOSAL.

Sources: srinivas1993noisyor, kempe2003influence

### Guarded threshold (our proposal)

$$
\mathcal C_\lambda(q)=\{v:\;P_v\ge 1-\lambda\},\qquad t^{*}=\min\big(\tau,\;1-\hat\lambda\big)
$$

Derivation: C_lambda grows with lambda so the false-negative proportion is non-increasing in lambda and bounded by B = 1, satisfying the conditions of conformal risk control; lambda_hat from the CRC formula guarantees expected FNR <= alpha on exchangeable queries. Taking the smaller (more inclusive) threshold keeps the Bayes-optimal rule when calibration is good and the recall guarantee when it is not. OUR PROPOSAL.

Sources: angelopoulos2022crc, elkan2001cost

## Open questions

- Sample size: the mimalloc corpus has only a few hundred to low thousands of functions, and the relevant class per query is tiny. How many labelled positives are needed before the KDE of f1 is stable, and should positives be pooled across queries (one global f1) or estimated per query class?
- Are synthetic calibration queries exchangeable with real user queries? Both the KDE posterior and the conformal guarantees assume calibration and test scores come from the same distribution; a fork of memalloc/mimalloc used as a demo repository may not represent other codebases.
- How large is the noisy-OR over-estimate on a real call graph where paths share edges? Exact combination over overlapping paths is a network-reliability computation; an empirical comparison of max-product, noisy-OR and sampled reliability on the mimalloc graph is needed.
- What prior pi_0 should be used, and should it be per query? Options are a fixed base rate from calibration data, or a per-query mixing weight recovered by EM on the score list as in score-distribution models; the posterior and therefore the accepted set are sensitive to this choice.
- Is tau = 0.25 actually optimal for downstream answer or edit correctness with the fine-tuned, quantised local model? The 3:1 cost ratio is an assumption; the true ratio depends on how much distractors hurt that specific model and on the context budget.
- Does calibration survive changes to the embedding model, its quantisation, or the chunking? Any such change alters the score distribution and requires re-estimating f1, f0 and the conformal threshold; the cost and frequency of re-calibration are unknown.
- Should edge transfer probabilities be static per edge type or query-conditioned? Recent work attributes semantic drift to static weights, but query-aware gating adds an embedding comparison per visited edge, which interacts with the L1-sized per-thread working-set goal.
- How should the posterior be made monotone without discarding information: isotonic projection, a monotone-likelihood-ratio constrained density estimate, or a parametric family that satisfies the convexity hypothesis (two-gamma)?
- For objectives such as F1 on the retrieved set there is no optimal fixed probability threshold. Should the system expose a second mode that selects the cut-off maximising expected F1 from the estimated densities, as in score-distributional rank optimisation?
- Conformal guarantees are marginal over queries. Can coverage be made approximately conditional on query type (for example 'explain', 'find callers', 'edit') without an impractically large calibration set?

## Sources

- **robertson1977prp**: S. E. Robertson (1977). The Probability Ranking Principle in IR. Journal of Documentation 33(4):294-304. https://www.emerald.com/insight/content/doi/10.1108/eb026647/full/html
  - evidence: Publisher abstract page opened: 'The principle that, for optimal retrieval, documents should be ranked in order of the probability of relevance or usefulness has been brought into question by Cooper.' The abstract states the principle can be justified under certain assumptions and fails otherwise, because it handles documents one by one. Only the abstract page was accessible.
- **manning2008ir**: Christopher D. Manning, Prabhakar Raghavan, Hinrich Schuetze (2008). Introduction to Information Retrieval (Section 11.2: The Probability Ranking Principle; 'The 1/0 loss case' and 'The PRP with retrieval costs'). Cambridge University Press (online HTML edition). https://nlp.stanford.edu/IR-book/html/htmledition/the-prp-with-retrieval-costs-1.html
  - evidence: Verbatim: 'Let C_1 be the cost of not retrieving a relevant document and C_0 the cost of retrieval of a nonrelevant document ... C_0 P(R=0|d) - C_1 P(R=1|d) <= C_0 P(R=0|d') - C_1 P(R=1|d') (62) then d is the next document to be retrieved.' Companion page the-10-loss-case-1.html: 'Theorem. The PRP is optimal, in the sense that it minimizes the expected loss (also known as the Bayes risk) under 1/0 loss ... it requires that all probabilities are known correctly. This is never the case in practice.'
- **manmatha2001scoredist**: R. Manmatha, T. Rath, F. Feng (2001). Modeling Score Distributions for Combining the Outputs of Search Engines. ACM SIGIR 2001, pp. 267-275. https://ciir.cs.umass.edu/pubfiles/ir-224.pdf
  - evidence: Abstract: score distributions 'may be fitted using an exponential distribution for the set of non-relevant documents and a normal distribution for the set of relevant documents'. Sec. 3.2 Eq. (8) gives P(rel|score) by Bayes' rule with mixing parameters as priors, and notes the posterior can decrease with increasing score because 'the Gaussian density falls much more rapidly than the exponential and hence the two densities intersect twice'.
- **arampatzis2011modeling**: Avi Arampatzis, Stephen Robertson (2011). Modeling score distributions in information retrieval. Information Retrieval 14(1):26-46. https://link.springer.com/article/10.1007/s10791-010-9145-5
  - evidence: Full PDF read. 'the current theoretical argument points to the two gamma as the most-likely universal model, with the normal-exponential being a usable approximation'; the normal-exponential 'violates such conditions only (and always) at both ends of the score range'; Sec. 7: judgements are 'sparse, incomplete, or biased', EM 'was found to be messy and difficult to tune', 'the more the score data, the lower the estimated threshold'; 'the optimal threshold depends on the effectiveness measure being used'.
- **arampatzis2008stop**: Avi Arampatzis, Jaap Kamps (2008). Where to Stop Reading a Ranked List?. TREC 2008 Legal Track notebook/proceedings (NIST). https://trec.nist.gov/pubs/trec17/papers/uamsterdam-kamps.legal.rev.pdf
  - evidence: Sec. 2.2: 'While for some measures there exists an optimal fixed probability threshold, for others it does not. Lewis [13] formulates this in terms of whether or not a measure satisfies the probability thresholding principle, and proves that the F measure does not satisfy it.' Sec. 3.1 Eqs. (7)-(8) give the normal and exponential densities; Sec. 2.3 defines the optimal rank K = argmax_k M_k including k = 0.
- **kanoulas2009scoredist**: Evangelos Kanoulas, Virgil Pavlu, Keshi Dai, Javed A. Aslam (2009). Modeling the Score Distributions of Relevant and Non-relevant Documents. ICTIR 2009, Advances in Information Retrieval Theory (Springer LNCS), pp. 152-163. https://link.springer.com/chapter/10.1007/978-3-642-04417-5_14
  - evidence: Abstract on publisher page: 'we propose modeling the relevant documents' scores by a mixture of Gaussians and modeling the non-relevant scores by a Gamma distribution. Applying variational inference we automatically trade-off the goodness-of-fit with the complexity of the model ... In all experiments our model outperforms the dominant exponential-Gaussian model.' Abstract page only.
- **elkan2001cost**: Charles Elkan (2001). The Foundations of Cost-Sensitive Learning. IJCAI 2001 (Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence). https://cseweb.ucsd.edu/~elkan/rescale.pdf
  - evidence: Sec. 1.3: 'the optimal prediction is class 1 if and only if p >= p*' with Eq. (2) p* = (c10 - c00)/(c10 - c00 + c01 - c11). Sec. 2 special case: p*/(1 - p*) = c10/c01. Theorem 2: p' = b'(p - pb)/(b - pb + b'p - bb'), proved with Bayes' rule without independence assumptions. Pages rendered and read as images because the PDF text layer garbles the mathematics.
- **lin2007platt**: Hsuan-Tien Lin, Chih-Jen Lin, Ruby C. Weng (2007). A Note on Platt's Probabilistic Outputs for Support Vector Machines. Machine Learning 68(3):267-276. https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
  - evidence: Introduction restates Platt's method: 'approximating the posterior by a sigmoid function Pr(y=1|x) ~ P_{A,B}(f) = 1/(1+exp(Af+B))' with regularised maximum likelihood and targets t_i = (N+ + 1)/(N+ + 2) for positives and 1/(N- + 2) for negatives; the note proves the problem is convex and gives a numerically robust algorithm. Journal metadata confirmed on the Springer page.
- **zadrozny2002isotonic**: Bianca Zadrozny, Charles Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. ACM SIGKDD 2002. http://www.cs.columbia.edu/~djhsu/coms4771-f25/handouts/zadrozny2002kdd.pdf
  - evidence: 'We propose here an intermediary approach between sigmoid fitting and binning: isotonic regression ... If we assume that the classifier ranks examples correctly, the mapping from scores into probabilities is non-decreasing ... A commonly used algorithm for computing the isotonic regression is pair-adjacent violators (PAV)', which finds the stepwise-constant isotonic function minimising mean-squared error.
- **parzen1962kde**: Emanuel Parzen (1962). On Estimation of a Probability Density Function and Mode. The Annals of Mathematical Statistics 33(3):1065-1076. https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-3/On-Estimation-of-a-Probability-Density-Function-and-Mode/10.1214/aoms/1177704472.pdf
  - evidence: Scanned pages read as images. Eq. (1.7): f_n(x) = integral (1/h) K((x-y)/h) dF_n(y) = (1/(nh)) sum_j K((x - X_j)/h). Theorem 1A and Corollary 1A: the estimates are asymptotically unbiased at continuity points of the density when h(n) -> 0 and K satisfies (1.11)-(1.13). Cites Rosenblatt as prior work.
- **scipy_gaussian_kde**: SciPy developers (2026). scipy.stats.gaussian_kde (SciPy v1.18.0 Manual). Official SciPy documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
  - evidence: Notes section: 'Bandwidth selection strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel)'. 'Scott's Rule ... is: n**(-1./(d+4))'; Silverman's factor '(n * (d + 2) / 4.)**(-1. / (d + 4))'; 'neff = sum(weights)^2 / sum(weights^2)'.
- **lin2022kcal**: Zhen Lin, Shubhendu Trivedi, Jimeng Sun (2022). Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks. arXiv:2202.07679 (v3, Dec 2022). https://arxiv.org/abs/2202.07679
  - evidence: Abstract: KCal 'generates predictions using kernel density estimates on a calibration set' and 'enjoys a provable full calibration guarantee'. Eq. (4): P{Y=k|X=x0} = f_k(x0) pi_k / sum_k' f_k'(x0) pi_k', then 'replacing f_k with the kernel density estimate ... and the class proportion pi_k with |S_cal^k|/|S_cal|'. Bandwidth is chosen by cross-validation on the calibration set.
- **geenens2013probit**: Gery Geenens (2013). Probit transformation for kernel density estimation on the unit interval. arXiv:1303.4121. https://arxiv.org/abs/1303.4121
  - evidence: Abstract: kernel estimation of a density supported on the unit interval is difficult 'because of the well known boundary bias issues a conventional kernel density estimator would necessarily face'; 'combining the transformation idea with local likelihood density estimation produces viable density estimators, mostly free from boundary issues', with a cross-validation bandwidth rule.
- **penha2021calibration**: Gustavo Penha, Claudia Hauff (2021). On the Calibration and Uncertainty of Neural Learning to Rank Models. EACL 2021. https://arxiv.org/abs/2101.04356
  - evidence: Abstract: 'The PRP holds when two conditions are met: [C1] the models are well calibrated, and, [C2] the probabilities of relevance are reported with certainty ... BERT-based rankers are not robustly calibrated and that stochastic BERT-based rankers yield better calibration'.
- **lien2019bicut**: Yen-Chieh Lien, Daniel Cohen, W. Bruce Croft (2019). An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists. ACM ICTIR 2019. https://ciir-publications.cs.umass.edu/getpdf.php?id=1326
  - evidence: 'we implement an LSTM-based model, referred to as BiCut, over the ordered list and partition it to minimize a predefined cost function'; Sec. 3.2 loss has 'a hyperparameter that controls the impact of false positives and false negatives'; best F1 at alpha = 0.65, i.e. the loss works better when it favours precision.
- **bahri2020choppy**: Dara Bahri, Yi Tay, Che Zheng, Donald Metzler, Andrew Tomkins (2020). Choppy: Cut Transformer For Ranked List Truncation. SIGIR 2020. https://arxiv.org/abs/2004.13012
  - evidence: Abstract: 'Such truncation is a balancing act between the overall relevance, or usefulness of the results, with the user cost of processing more results ... Needing nothing more than the relevance scores of the results, the model uses a powerful multi-head attention mechanism to directly optimize any user-defined IR metric.'
- **wu2021attncut**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Yixing Fan, Yanyan Lan, Xueqi Cheng (2021). Learning to Truncate Ranked Lists for Information Retrieval. arXiv:2102.12793. https://arxiv.org/abs/2102.12793
  - evidence: Abstract: existing methods 'take some pre-defined loss as a proxy objective, which suffers from the limitation of local decision and non-direct optimization'; AttnCut uses a transformer 'to capture the global dependency within the ranked list' and RAML for direct optimisation; objectives F1 and 'best F1 under some minimal recall constraint'; evaluated on Robust04 and MQ2007.
- **meng2024rlt**: Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, Maarten de Rijke (2024). Ranked List Truncation for Large Language Model-based Re-Ranking. SIGIR 2024. https://arxiv.org/abs/2404.18185
  - evidence: 'we found supervised RLT methods do not show a clear advantage over using a fixed re-ranking depth'; 'with an effective retriever like SPLADE++ or RepLLaMA, a fixed re-ranking depth of 20 can already yield an excellent effectiveness/efficiency trade-off'; 'oracle cut-offs always perform statistically significantly ... better than all fixed cut-offs in terms of nDCG@10'.
- **taguchi2025adaptivek**: Chihiro Taguchi, Seiji Maekawa, Nikita Bhutani (2025). Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k. EMNLP 2025 (Main). https://arxiv.org/abs/2506.08479
  - evidence: 'Compute their first discrete differences g and choose the index k where the similarity drop is the largest'; 'we set B = 5'; 'we restrict the search for the largest gap to the top 90% of documents'; abstract: 'using up to 10x fewer tokens than full-context input, yet still retrieves 70% of relevant passages'; limitation: a crafted chunk could introduce 'a large gap in the similarity distribution and misleading the algorithm'.
- **song2026taak**: Ziyu Song, Jiaming Fang, Kuangyu Li, Tuo Xia, Chuanpeng Wang (2026). Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation. ECML PKDD 2026 (arXiv:2606.11907). https://arxiv.org/abs/2606.11907
  - evidence: Abstract: 'TAA-k exploits this geometric structure via knee detection to identify a compact candidate region, then applies EVT-based goodness-of-fit testing within this window ... Under mild monotone likelihood ratio assumptions, TAA-k yields a stable, query-adaptive cutoff corresponding to the earliest noise-dominated position ... near-oracle retrieval quality (F1 within 2-3% of oracle)'.
- **xu2025car**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari (2025). Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications. arXiv:2511.14769. https://arxiv.org/abs/2511.14769
  - evidence: Abstract: 'CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates ... CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance.'
- **xia2026preretrieval**: Ye Xia, Emre Yamangil, Haixun Wang (2026). Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems. CIKM 2026 Applied Research Track (arXiv:2609.13489). https://arxiv.org/abs/2609.13489
  - evidence: Abstract: fixed top-k 'is brittle: simple queries suffer over-retrieval (adding noise and cost) while complex queries are under-retrieved'; the method derives a per-query saturation point k* from the NDCG@k curve and summarises each query cluster 'with a recommended retrieval depth that targets high coverage (e.g., ~95%) using a mean-plus-variance rule'.
- **repantis2026tools**: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh, Joey Blackwell (2026). How Many Tools Should an LLM Agent See? A Chance-Corrected Answer. arXiv:2605.24660. https://arxiv.org/abs/2605.24660
  - evidence: Related work: gap and clustering methods 'are efficient but assume the scorer is reliable. When the scorer itself is weak, score patterns offer little basis for choosing depth.' Results: 'the learned policy nearly matches the coverage of showing 50 tools (90.3% vs 90.8%) while presenting only 7 on average'; Self-RAG/Adaptive-RAG/FLARE 'adapt whether to retrieve but treat depth K as fixed once triggered'.
- **asai2023selfrag**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv:2310.11511. https://arxiv.org/abs/2310.11511
  - evidence: Abstract: 'indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation'; the LM 'adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens'.
- **jeong2024adaptiverag**: Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. NAACL 2024. https://arxiv.org/abs/2403.14403
  - evidence: Abstract: the framework 'can dynamically select the most suitable strategy ... based on the query complexity ... operationalized with a classifier, which is a smaller LM trained to predict the complexity level of incoming queries with automatically collected labels', adapting between iterative, single-step and no-retrieval methods.
- **yan2024crag**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (2024). Corrective Retrieval Augmented Generation. arXiv:2401.15884. https://arxiv.org/abs/2401.15884
  - evidence: 'If the confidence score is higher than the upper threshold, the retrieved document is identified as Correct, while identified as Incorrect if below the lower threshold. Otherwise ... Ambiguous'; appendix: 'The two confidence thresholds ... were set empirically. Specifically, they were set as (0.59, -0.99) in PopQA, (0.5, -0.91) in PubQA and ArcChallenge, as well as (0.95, -0.91) in Biography.'
- **angelopoulos2021gentle**: Anastasios N. Angelopoulos, Stephen Bates (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511 (v6, Dec 2022). https://arxiv.org/abs/2107.07511
  - evidence: Theorem 1: for i.i.d. calibration and test points, P(Y_test in C(X_test)) >= 1 - alpha; q_hat is the ceil((n+1)(1-alpha))/n empirical quantile (code: np.ceil((n+1)*(1-alpha))/n). 'although the guarantee always holds, the usefulness of the prediction sets is primarily determined by the score function'; 'in the most general case, conditional coverage is impossible to achieve'.
- **angelopoulos2022crc**: Anastasios N. Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster (2022). Conformal Risk Control. arXiv:2208.02814 (v4, June 2025). https://arxiv.org/abs/2208.02814
  - evidence: Eq. (4): lambda_hat = inf{lambda : n/(n+1) R_hat_n(lambda) + B/(n+1) <= alpha}; Theorem 1: if L_i is non-increasing and right-continuous with sup L_i <= B then E[L_{n+1}(lambda_hat)] <= alpha; lower bound alpha - 2B/(n+1); 'if the risk is non-monotone, then this algorithm does not control the risk'. Example loss: false negative rate 1 - |Y intersect C|/|Y|.
- **rouzrokh2024conflare**: Pouria Rouzrokh, Shahriar Faghani, Cooper U. Gamble, Moein Shariatnia, Bradley J. Erickson (2024). CONFLARE: CONFormal LArge language model REtrieval. arXiv:2404.04287. https://arxiv.org/abs/2404.04287
  - evidence: Abstract: 'During inference, all chunks with similarity exceeding this threshold are retrieved to provide context to the LLM, ensuring the true answer is captured in the context with a (1-alpha) confidence level.' Limitation: weak embeddings or the chosen error rate can yield 'a set of retrieved document chunks too large to fit within the context window'.
- **li2023traq**: Shuo Li, Sangdon Park, Insup Lee, Osbert Bastani (2023). TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction. NAACL 2024 (arXiv:2307.04642). https://arxiv.org/abs/2307.04642
  - evidence: Abstract: TRAQ 'provides the first end-to-end statistical correctness guarantee for RAG ... leverages Bayesian optimization to minimize the size of the constructed sets ... reducing prediction set size by 16.2% on average compared to an ablation'.
- **crestani1997spreading**: F. Crestani (1997). Application of Spreading Activation Techniques in Information Retrieval. Artificial Intelligence Review 11(6):453-482. https://link.springer.com/article/10.1023/A:1006569829653
  - evidence: Publisher abstract: 'This paper surveys the use of Spreading Activation techniques on Semantic Networks in Associative Information Retrieval. The major Spreading Activation models are presented and their applications to IR is surveyed.' Only abstract and metadata were accessible; cited for the historical claim only.
- **tong2006rwr**: Hanghang Tong, Christos Faloutsos, Jia-Yu Pan (2006). Fast Random Walk with Restart and Its Applications. IEEE ICDM 2006 (author-hosted PDF). https://www.cs.cmu.edu/~htong/pdf/ICDM06_tong.pdf
  - evidence: Sec. 2.1 Eq. (1): r_i = c W~ r_i + (1 - c) e_i and Eq. (2): r_i = (1 - c)(I - c W~)^-1 e_i. 'The relevance score of node j wrt node i is defined as the steady-state probability r_ij that the particle will finally stay at node j.' The iterative method stops when the L2 norm of successive estimates is below a threshold or after m iterations.
- **andersen2006local**: Reid Andersen, Fan Chung, Kevin Lang (2006). Local Graph Partitioning using PageRank Vectors. FOCS 2006 (full version dated 24 Sept 2006). https://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
  - evidence: Eq. (1): pr_alpha(s) = alpha s + (1 - alpha) pr_alpha(s) W with W = (I + D^-1 A)/2. Definition 1: epsilon-approximate if r(v) <= epsilon d(v) for every vertex. Theorem 1: 'The support of p satisfies vol(Supp(p)) <= 2/((1-alpha) epsilon), and the running time of the algorithm is O(1/(epsilon alpha))', 'independent of the size of the graph'.
- **kempe2003influence**: David Kempe, Jon Kleinberg, Eva Tardos (2003). Maximizing the Spread of Influence through a Social Network. ACM SIGKDD 2003. https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
  - evidence: Independent cascade: when node v becomes active 'it is given a single chance to activate each currently inactive neighbor w; it succeeds with a probability p_{v,w} ... independently of the history thus far'. Sec. 4: 'the probability that v is not activated by this process is by definition prod_{i=1..k} (1 - p_v(u_i, S_{i-1}))', hence f_v(S) = 1 - prod(...).
- **srinivas1993noisyor**: Sampath Srinivas (1993). A Generalization of the Noisy-Or Model. UAI 1993 (Ninth Conference on Uncertainty in Artificial Intelligence), pp. 208-215; arXiv:1303.1479. https://arxiv.org/abs/1303.1479
  - evidence: 'Each cause is inhibited with probability q_i, i.e., there is a probability q_i that even when the cause U_i is active, it will not affect X.' Sec. 5.1: P(f_x|u) = prod over {i | u_i = t_i} of q_i and P(t_x|u) = 1 - P(f_x|u). 'The inhibitor probabilities are few in number (one associated with each predecessor U_i of X)'.
- **zhang1996causal**: Nevin Lianwen Zhang, David Poole (1996). Exploiting Causal Independence in Bayesian Network Inference. Journal of Artificial Intelligence Research 5:301-328. https://arxiv.org/abs/cs/9612101
  - evidence: 'Causal independence refers to the situation where multiple causes contribute independently to a common effect. A well-known example is the noisy OR-gate model (Good, 1961).' Their formulation covers 'noisy OR-gates, noisy MAX-gates, noisy AND-gates, and noisy adders as special cases' using an associative and commutative combination operator.
- **gutierrez2024hipporag**: Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024. https://arxiv.org/abs/2405.14831
  - evidence: PPR is 'a version of PageRank that distributes probability across a graph only through a set of user-defined source nodes'; 'We define the node specificity of node i as s_i = |P_i|^-1'; hyperparameters tuned on 100 MuSiQue examples: 'the synonymy threshold tau at 0.8 and the PPR damping factor at 0.5'; abstract: outperforms state of the art 'by up to 20%' and is '10-20 times cheaper and 6-13 times faster' than IRCoT (v3 PDF).
- **gutierrez2025hipporag2**: Bernal Jimenez Gutierrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su (2025). From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. ICML 2025. https://arxiv.org/abs/2502.14802
  - evidence: Seed nodes 'are used to assign the reset probabilities within PPR'; 'the reset probability of all passage nodes is multiplied by a weight factor ... we set the factor to be 0.05 by default'; query-to-triple 'improves Recall@5 by 12.5% compared to NER-to-node'; error analysis: 'Triple filtering and the graph search algorithm are the two main sources of errors'; abstract reports a 7% improvement on associative memory tasks.
- **edge2024graphrag**: Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130 (v2, Feb 2025). https://arxiv.org/abs/2404.16130
  - evidence: 'RAG fails on global questions directed at an entire text corpus, such as What are the main themes in the dataset?'; 'GraphRAG first uses an LLM to construct a knowledge graph ... Next, it partitions the graph into a hierarchy of communities of closely related entities' before summarising each community.
- **lau2026catrag**: Kwun Hang Lau, Fangyuan Zhang, Boyu Ruan, Yingli Zhou, Qintian Guo, Ruiyuan Zhang, Xiaofang Zhou (2026). Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation. arXiv:2602.01965. https://arxiv.org/abs/2602.01965
  - evidence: Abstract: PPR-based methods 'suffer from a Static Graph Fallacy: they rely on fixed transition probabilities determined during indexing. This rigidity ignores the query-dependent nature of edge relevance, causing semantic drift where random walks are diverted into high-degree hub nodes before reaching critical downstream evidence.' Remedies: symbolic anchoring, query-aware dynamic edge weighting, key-fact passage weighting.
- **makarov2026qasa**: Illia Makarov, Mykola Glybovets (2026). Query-Aware Spreading Activation for Multi-Hop Retrieval over Knowledge Graphs. arXiv:2606.30133 (accepted, Cybernetics and Systems Analysis). https://arxiv.org/abs/2606.30133
  - evidence: Eq. (1) read from the rendered page. 'Without the threshold, small increments would accumulate over T iterations, propagation would cover the entire connected subgraph, and top-K selection would gather many weakly activated entities.' Defaults: 'activation threshold tau = 0.01 ... T = 3, alpha = 0.7'. Abstract: the gate yields 'F1 gain of 3.6 to 7.4 points and a retrieval-latency reduction by a factor of 1.5 to 4.9'.
- **pavlovic2025sarag**: Jovan Pavlovic, Miklos Kresz, Laszlo Hajdu (2025). Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems. arXiv:2512.15922 (v3, May 2026). https://arxiv.org/abs/2512.15922
  - evidence: 'initially, activated nodes are assigned an activation value of a_i = 1 ... Activation then propagates outward ... in a BFS manner, through the weighted links w_ij ... nodes whose activation exceeds the threshold tau_a are marked as activated'; Algorithm 1 uses min(1, activation[target] + weight * activation[node]); documents below a cosine pruning threshold are removed; abstract reports up to 39% absolute improvement over naive RAG when combined with chain-of-thought iterative retrieval.
- **zhang2023repocoder**: Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, Weizhu Chen (2023). RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation. EMNLP 2023. https://arxiv.org/abs/2303.12570
  - evidence: Abstract: RepoCoder incorporates 'a similarity-based retriever and a pre-trained code language model in an iterative retrieval-generation pipeline' and 'significantly improves the In-File completion baseline by over 10% in all settings'; the generated completion is used as the next retrieval query.
- **ouyang2024repograph**: Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, Dong Yu (2024). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. ICLR 2025. https://arxiv.org/abs/2410.14684
  - evidence: 'Each node in the graph represents a line of code, and edges represent the dependencies of code definitions and references'; 'We retrieve the k-hop ego-graphs'; 'We limit our exploration to k values up to 2 due to the extensive context required for integration and the potential introduction of noise or irrelevant nodes for k >= 3'; 'average relative improvement of 32.8%' on SWE-bench.
- **liu2024graphcoder**: Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, Qianxiang Wang (2024). GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model. arXiv:2406.07003. https://arxiv.org/abs/2406.07003
  - evidence: A code context graph is 'a statement-level multi-graph' with control-flow and data/control-dependence edges; CCG slicing extracts vertices 'within h hops of control-flow neighbors ... limited to a maximum of' a fixed number of vertices; candidates 'are re-ranked by a decay-with-distance structural similarity measure'; reported +6.06 code match and +6.23 identifier match.
- **phan2024repohyper**: Huy N. Phan, Hoang N. Phan, Tien N. Nguyen, Nghi D. Q. Bui (2024). RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion. arXiv:2403.06095. https://arxiv.org/abs/2403.06095
  - evidence: 'We found that exhausted search on all the directions and paths can include too many irrelevant contexts to the query', so frequent path types are collected and added 'as filters for later BFS'; retrieved context is refined by 'Re-ranking as Link Prediction' on the Repo-level Semantic Graph.
- **liu2024codexgraph**: Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Shieh, Wenmeng Zhou (2024). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. arXiv:2408.03910. https://arxiv.org/abs/2408.03910
  - evidence: Abstract: 'Similarity-based retrieval often has low recall in complex tasks'; CodexGraph 'employs code graph databases as interfaces that allow LLM agents to interact seamlessly with code repositories' through a unified schema and graph queries; evaluated on CrossCodeEval, SWE-bench and EvoCodeBench.
- **li2025graphcodeagent**: Jia Li, Xianjie Shi, Kechi Zhang, Ge Li, Zhi Jin, Lei Li, Huangzhao Zhang, Jia Li, Fang Liu, Yuwei Zhang, Zhengwei Tao, Yihong Dong, Yuqi Zhu, Chongyang Tao (2025). GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation (v1 title: CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation). arXiv:2504.10046 (v2, Nov 2025). https://arxiv.org/abs/2504.10046
  - evidence: Constructs 'a Requirement Graph (RG) and a Structural-Semantic Code Graph (SSCG) of a repository with mapping relations', 'allowing the agent to perform multi-hop retrieval by traversing the SSCG'; motivation: LLMs' 'ability to understand code decays significantly as input length increases'.
- **liu2025reposcope**: Yang Liu, Li Zhang, Fang Liu, Zhuohang Wang, Donglin Wei, Zhishuo Yang, Kechi Zhang, Jia Li, Lin Shi (2025). RepoScope: Leveraging Call Chain-Aware Multi-View Context for Repository-Level Code Generation. ICSE 2026 (arXiv:2507.14791). https://arxiv.org/abs/2507.14791
  - evidence: Call chain prediction 'necessitates predicting the likelihood of each code element being called by the target function'; 'similar functions within the same repository often exhibit similar calling behaviors'; the method 'relies solely on static analysis' over a Repository Structural Semantic Graph with call, import and type-dependency relations.
- **liu2026dyretriever**: Zhongxin Liu, Zhonghao Jiang, Zhifan Ye, Haoye Wang, Jiakun Liu, Xiaoxue Ren (2026). Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation. ASE 2026 (arXiv:2608.01927). https://arxiv.org/abs/2608.01927
  - evidence: Abstract: similarity-based retrieval 'often fails to retrieve code snippets that the target function actually depends on'; DyRetriever works 'by first selecting a set of entry-point functions and then performing multihop reasoning along the code dependency graph', building a partial graph dynamically; Algorithm 1 takes 'maxHop: Maximum steps of multi-hop reasoning'.
- **li2026featlens**: Xutian Li, Bo Xiong, Yifeng Zhu, Kunze Li, Xianlin Zhao, Runbang Yan, Yanzhen Zou, Lu Zhang (2026). FeatLens: Feature-Guided Dynamic Code Graph Construction and Retrieval for Repository-Level Code Generation. arXiv:2609.26480. https://arxiv.org/abs/2609.26480
  - evidence: Abstract: 'it dynamically constructs a task-specific seed graph from the feature index and applies semantic-structural graph reasoning with personalized PageRank to select a compact reasoning graph'; 'By materializing only task-relevant graph regions and filtering weakly related nodes after expansion, FeatLens avoids maintaining a full repository graph'; best DR@15 of 0.501 (DevEval) and 0.460 (EvoCodeBench).
- **cuconasu2024noise**: Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, Fabrizio Silvestri (2024). The Power of Noise: Redefining Retrieval for RAG Systems. arXiv:2401.14887. https://arxiv.org/abs/2401.14887
  - evidence: Abstract: 'the retriever's highest-scoring documents that are not directly relevant to the query (e.g., do not contain the answer) negatively impact the effectiveness of the LLM. Even more surprising, we discovered that adding random documents in the prompt improves the LLM accuracy by up to 35%.'
- **yoran2023robust**: Ori Yoran, Tomer Wolfson, Ori Ram, Jonathan Berant (2023). Making Retrieval-Augmented Language Models Robust to Irrelevant Context. arXiv:2310.01558. https://arxiv.org/abs/2310.01558
  - evidence: Abstract: 'in multi-hop reasoning scenarios ... misuse of irrelevant evidence can lead to cascading errors'; NLI filtering prevents performance loss 'but at a cost of also discarding relevant passages'; 'even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones'.
- **liu2023lost**: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics (TACL). https://arxiv.org/abs/2307.03172
  - evidence: Abstract: 'performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts, even for explicitly long-context models.'
