# Probabilistic relevance, adaptive cut-offs and graph tracing

*Research note for the programmersRAG paper (feature 5: "keep tracing while P(relevant) >= 0.25"). Checked adversarially on 2026-09-30: every reference below was opened by the checker (arXiv abstract page, publisher/DOI record via Crossref, or author-hosted PDF). Body-text quotes were checked against PDFs downloaded in this session. Items marked **[PROPOSAL]** are our own derivations or designs, not published results.*

## Summary

- The owner's rule "keep every function with P(relevant) >= 0.25" is a **Bayes decision rule**. By the Probability Ranking Principle [1, 2] and two-class cost-sensitive decision theory [7], a fixed probability threshold tau minimises expected cost exactly when c_FN / c_FP = (1 - tau)/tau. So **tau = 0.25 means "leaving out a relevant function costs three times as much as including an irrelevant one"** [PROPOSAL: our reading of [7]]. The rule is optimal only if the probabilities are calibrated [2, 15]. It is also **not** optimal for set measures such as F1, which have no optimal fixed probability threshold [4, 5].
- The classical way to turn a similarity score into P(relevant | score) is the **score-distribution (SD) model**. It treats relevant and non-relevant scores as a two-component mixture and applies Bayes' rule [3, 4, 5, 6]. The requested "KDE-like probabilistic function" is a non-parametric form of this idea: a Bayes posterior built from class-conditional kernel density estimates. The same construction is used in KCal [13] and rests on the Rosenblatt-Parzen estimator [10, 11].
- The "keep tracing" part has rigorous graph precedents. These are random walk with restart and personalized PageRank [37, 38], where the push algorithm stops expanding when the residual mass per unit degree falls below epsilon, giving a size bound that does not depend on the graph. There is also thresholded spreading activation [36, 46, 47]. Recent graph-RAG work names **query-blind propagation into hub nodes ("semantic drift")** as the main failure [45, 46]. Code-graph retrievers cap expansion depth at 1-2 hops or with a maxHop parameter [49, 50, 51, 55].
- Fixed top-k is the baseline to beat. The evidence for adaptive cut-offs is **mixed**. Supervised truncation often fails to beat a well-chosen fixed depth [20], but oracle cut-offs are significantly better [20]. Gap, clustering and EVT heuristics [18, 21, 22, 23] assume a reliable scorer [25].
- **Distribution-free guarantees** are available and cheap to add. Split conformal prediction [29] and conformal risk control [30] can calibrate a threshold that bounds the expected **false-negative rate** of the retrieved set. RAG-specific versions exist [31, 32, 33, 34, 35].

## Established results

1. **PRP and 1/0 loss.** Ranking by decreasing probability of relevance minimises expected loss (Bayes risk) under 1/0 loss. The corresponding set rule is "d is relevant iff P(R=1|d,q) > P(R=0|d,q)" (IR book Eq. 61) [2]. The proof "requires that all probabilities are known correctly. This is never the case in practice." [2]. Robertson shows that the principle holds only under certain assumptions, and that its main weakness is that it considers each document independently of the rest [1].
2. **PRP with costs** (IR book Eq. 62) [2]. Let C_1 be the cost of not retrieving a relevant document and C_0 the cost of retrieving a non-relevant one. Then d is retrieved next if C_0 P(R=0|d) - C_1 P(R=1|d) is minimal among the documents not yet retrieved. In general two-class decision theory, predicting positive is optimal iff p >= p* (Elkan Eq. 2) [7]. So each fixed probability threshold corresponds to exactly one normalised cost ratio (see Formulas F3-F4).
3. **No universal threshold.** "While for some measures there exists an optimal fixed probability threshold, for others it does not". Lewis proved that the F measure does not satisfy the probability thresholding principle [5]. "The optimal threshold depends on the effectiveness measure being used" [4]. (Lewis 1995 is cited here second-hand, through [5].)
4. **Score-distribution models.** Manmatha et al. fit an exponential to non-relevant scores and a normal to relevant scores per query, using EM without relevance labels. They convert a score to P(rel|score) by Bayes' rule with the mixing weights as priors (their Eq. 8) [3]. Arampatzis and Kamps use the same normal-exponential model to choose where to stop a ranked list [5].
5. **Known defect of normal-exponential.** A Gaussian density falls faster than an exponential one, so "the two densities intersect twice" and the posterior is non-monotone at the extremes [3]. The model violates the Recall-Fallout Convexity Hypothesis "only (and always) at both ends of the score range". The ad-hoc fixes are "theoretically unjustified" [4]. The theoretical argument favours the **two-gamma** pair [4]. Kanoulas et al. model relevant scores with a Gaussian mixture and non-relevant scores with a Gamma, fitted by variational inference, and report that this beats the exponential-Gaussian model [6]. Arampatzis and Robertson note that Kanoulas et al. use the K-S test in a way that is "arguable", so their results are "indicative" [4].
6. **Practical failure modes of SD/EM.** Relevance judgements are sparse, incomplete or biased. EM "was found to be 'messy' and difficult to tune" and is very sensitive to initial values. EM treats all data equally, which causes a ranking-length bias: "the more the score data, the lower the estimated threshold". Regression-based calibration "optimizes the overall performance but not necessarily the retrieval quality at the top of the ranking" [4]. Arampatzis and Robertson also note that a Gaussian-kernel KDE can approximate any density, and that a Gaussian mixture is strictly more flexible [4, footnote 2].
7. **Kernel density estimation.** f_n(x) = (1/(nh)) sum_j K((x - X_j)/h) [10, 11]. Parzen proves asymptotic unbiasedness at continuity points when h(n) -> 0 [11]; consistency additionally needs n h(n) -> infinity. In practice "bandwidth selection strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel)" [12].
8. **KDE Bayes classifiers are calibrated classifiers.** KCal plugs class-conditional KDEs and class proportions into Bayes' rule (its Eq. 4) and proves a full-calibration guarantee [13]. KCal works on a learned embedding space and is a classifier-calibration method, not a retrieval method; the analogy to relevance scores is ours.
9. **Calibration maps.**
    - Platt scaling fits a sigmoid by regularised maximum likelihood with smoothed targets [8].
    - Isotonic regression (PAV) is "an intermediary approach between sigmoid fitting and binning". It assumes only that the scores rank examples correctly, and PAV finds the step-wise constant isotonic fit with least squared error [9].
10. **Boundary bias.** A conventional KDE on bounded support has "well known boundary bias issues". A probit transformation combined with local-likelihood estimation largely removes them [14].
11. **Neural rankers are not calibrated by default.** The PRP needs calibrated and certain probabilities, and "BERT-based rankers are not robustly calibrated" [15]. Raw cosine or re-ranker scores must not be read as probabilities.
12. **Learned truncation.**
    - BiCut is a Bi-LSTM whose loss weights false positives against false negatives. It worked best with alpha = 0.65, i.e. favouring precision [16].
    - Choppy is a Transformer over the scores only that optimises a user-defined metric [17].
    - Surprise converts the ranked scores into calibrated scores with a Generalized Pareto (EVT) model [18].
    - AttnCut uses global Transformer decisions with RAML, trained for F1 or best F1 under a recall constraint [19].
13. **Conformal prediction.** Split conformal prediction gives the finite-sample guarantee P(Y_test in C(X_test)) >= 1 - alpha under exchangeability. The guarantee is marginal, not conditional. "The usefulness of the prediction sets is primarily determined by the score function" [29]. Conformal risk control extends this to any bounded loss that is non-increasing in lambda, for example the false-negative rate of a set. It is tight up to 2B/(n+1), and "if the risk is non-monotone, then this algorithm does not control the risk" [30].
14. **Graph relevance by diffusion.**
    - RWR: r_i = c W r_i + (1 - c) e_i [37]. In the printed equation c multiplies the walk term, although the symbol table calls c "the restart probability".
    - Personalized PageRank: pr_alpha(s) = alpha s + (1 - alpha) pr_alpha(s) W, where W is the lazy walk.
    - The push algorithm ApproximatePR returns an epsilon-approximate vector with vol(Supp(p)) <= 2/((1 - alpha) epsilon) in time O(1/(epsilon alpha)), "independent of the size of the graph" [38].
15. **Evidence combination.** Noisy-OR: P(effect | active causes) = 1 - prod_i q_i, with one inhibitor probability per parent [40]. It is a special case of causal independence [41]. The independent/general cascade model has the same product form, provided activation attempts are order-independent [39].
16. **Spreading activation** is the classical associative-retrieval mechanism on networks [36]. Pavlovic et al. summarise the earlier IR evidence as follows. Spreading activation "needs to be constrained by limiting the number of steps to just a few hops ... and by using a decay factor smaller than one". Unconstrained spreading activation converges to a state in which results are "nearly independent of the input query" [47, citing its refs 40-41].

## Recent work (2024-2026)

**Adaptive-k and truncation**
- **RLT for LLM re-ranking (SIGIR 2024)** [20]. "Supervised RLT methods do not show a clear advantage over using a fixed re-ranking depth". With SPLADE++ or RepLLaMA, "a fixed re-ranking depth of 20 can already yield an excellent effectiveness/efficiency trade-off". Oracle cut-offs are statistically significantly better than every fixed cut-off (nDCG@10). There is headroom, but predictors do not capture it reliably.
- **Adaptive-k (EMNLP 2025)** [21]. Cut at the largest gap of the sorted similarities, add a buffer B = 5, and search only the top 90% of the list. It needs no tuning or extra LLM calls and uses up to 10x fewer tokens than full context while retrieving 70% of relevant passages.
- **TAA-k (ECML PKDD 2026)** [22]. Knee detection followed by Generalized-Pareto goodness-of-fit tests in a local window. Under a monotone-likelihood-ratio assumption it gives the earliest noise-dominated position, with F1 within 2-3% of the oracle.
- **CAR** [23] detects the transition in clustered similarity distances and reports 60% fewer tokens and 22% lower latency.
- **Pre-retrieval query clustering (CIKM 2026)** [24] assigns each query cluster a depth that targets about 95% coverage.
- **Tool-shortlist depth** [25]. Gap and clustering methods "are efficient but assume the scorer is reliable. When the scorer itself is weak, score patterns offer little basis for choosing depth." On BFCL the learned policy reaches 90.3% coverage with about 7 tools, against 90.8% with 50.
- **Self-RAG, Adaptive-RAG, CRAG** [26, 27, 28] adapt *whether or how* to retrieve (reflection tokens, a complexity classifier, and a retrieval evaluator with two confidence thresholds "set empirically", e.g. (0.59, -0.99) on PopQA). They do not produce a calibrated set size.

**Conformal retrieval for RAG**
- **CONFLARE** [31] sets a similarity cut-off from a calibration set so that the answer-bearing chunk is retrieved with confidence 1 - alpha. The paper warns that the retrieved set can become too large for the context window. Its text attributes this to "excessively high" error rates; logically, set size grows as alpha falls, so this looks like a wording slip in the source.
- **TRAQ** [32] gives an end-to-end conformal guarantee and shrinks sets by 16.2% with Bayesian optimisation.
- **C-RAG (ICML 2024)** [33] certifies an upper confidence bound on RAG generation risk, including under distribution shift.
- **Conformal score refinement** [34] applies a monotone transformation to retrieval scores that yields much smaller conformal sets with the same guarantee. This is directly relevant because our posterior is itself a monotone map of the score.
- **Conformal context engineering (ECIR 2026)** [35]. Conformal filtering meets its target coverage of relevant snippets and cuts retained context by 2-3x, with stable or better downstream accuracy.

**Graph tracing**
- **HippoRAG (NeurIPS 2024)** [42] runs PPR seeded by query entities, uses node specificity s_i = |P_i|^-1, damping 0.5 and a synonym threshold of 0.8 (both tuned on 100 examples). It gains up to 20% on multi-hop QA and is 10-20x cheaper than IRCoT.
- **HippoRAG 2 (ICML 2025)** [43] multiplies the passage-node reset probability by 0.05 and gains 7% on associative memory tasks. "Triple filtering and the graph search algorithm are the two main sources of errors."
- **GraphRAG** [44] targets global sense-making through community summaries, which is a different objective from selecting a minimal evidence set.
- **CatRAG** [45] names the "Static Graph Fallacy": fixed transition probabilities cause "semantic drift where random walks are diverted into high-degree 'hub' nodes". Its remedies are query-aware dynamic edge weighting and anchoring.
- **Query-aware spreading activation** [46] uses a per-step semantic gate sigma(v) = max(cos(e_v, e_q), 0), decay alpha = 0.7, T = 3 and threshold 0.01. "Without the threshold ... propagation would cover the entire connected subgraph." The gate gives +3.6 to +7.4 F1 and 1.5-4.9x lower latency.
- **SA-RAG** [47] seeds activation at 1, propagates breadth-first with a_j <- min(1, a_j + sum a_i w_ij), and keeps nodes above tau_a. It reports tau_a = 0.5 in practice.

**Code-graph retrieval**
- RepoCoder [48] iterates retrieval and generation (+10% over in-file completion).
- RepoGraph (ICLR 2025) [49] uses k-hop ego-graphs with k <= 2 "due to ... the potential introduction of noise or irrelevant nodes for k >= 3", and reports +32.8% average relative improvement on SWE-bench.
- GraphCoder [50] slices a statement-level graph to h hops and re-ranks with decay-with-distance.
- RepoHyper [51] found that exhaustive expansion "can include too many irrelevant contexts".
- CodexGraph [52] exists because "similarity-based retrieval often has low recall in complex tasks".
- GraphCodeAgent [53] traverses a requirement graph and a structural-semantic code graph.
- RepoScope (ICSE 2026) [54] observes that "similar functions within the same repository often exhibit similar calling behaviors". This is direct support for adding similarity edges beside call edges.
- DyRetriever (ASE 2026) [55] runs LLM-validated multi-hop search with a maxHop bound.
- FeatLens [56] applies PPR on a task-specific seed graph and then filters weakly related nodes.

**Why false positives cost something (c_FP > 0)**
- High-scoring irrelevant documents hurt LLM accuracy, while random documents can surprisingly help [57].
- Irrelevant evidence causes cascading errors in multi-hop settings, and about 1,000 mixed-context training examples make a model robust [58].
- Accuracy degrades when evidence sits in the middle of long contexts [59].

## Formulas

Every formula was re-derived or checked against the source by the checker. Notation: s is a similarity score, R in {0,1} is relevance, pi = P(R=1) is the prior, and tau is the threshold.

**F1. PRP with retrieval costs** (IR book Eq. 62) [2]
$$C_0\,P(R=0\mid d) - C_1\,P(R=1\mid d)\;\le\;C_0\,P(R=0\mid d') - C_1\,P(R=1\mid d')\quad\forall d'\text{ not yet retrieved}$$
C_1 is the cost of missing a relevant document and C_0 the cost of retrieving a non-relevant one. Under 1/0 loss the set rule is Eq. 61: d is relevant iff P(R=1|d,q) > P(R=0|d,q).

**F2. Expected cost** (Elkan Eq. 1) [7]
$$L(x,i)=\sum_{j}P(j\mid x)\,C(i,j)$$
C(i,j) is the cost of predicting i when the truth is j.

**F3. Optimal threshold** (Elkan Eq. 2) [7], re-derived by the checker
$$\text{predict }1\iff p\ge p^{*}=\frac{c_{10}-c_{00}}{c_{10}-c_{00}+c_{01}-c_{11}}$$
Derivation: (1-p)c_10 + p c_11 <= (1-p)c_00 + p c_01. Here c_10 is the false-positive cost and c_01 the false-negative cost. The formula assumes c_10 > c_00 and c_01 > c_11.

**F4. Cost ratio implied by tau** [PROPOSAL; derived from F3 with c_00 = c_11 = 0, consistent with F1]
$$\tau=\frac{c_{FP}}{c_{FP}+c_{FN}}\iff\frac{c_{FN}}{c_{FP}}=\frac{1-\tau}{\tau};\qquad \tau=0.25\Rightarrow c_{FN}=3\,c_{FP}$$

**F5. Score-to-probability by Bayes' rule** (Manmatha Eq. 8) [3]
$$P(\mathrm{rel}\mid s)=\frac{P(s\mid \mathrm{rel})P(\mathrm{rel})}{P(s\mid \mathrm{rel})P(\mathrm{rel})+P(s\mid \mathrm{nonrel})P(\mathrm{nonrel})}$$

**F6. Normal-exponential SD model** [5] (baseline)
$$f_1(s)=\frac{1}{\sigma}\phi\!\left(\frac{s-\mu}{\sigma}\right),\qquad f_0(s)=\lambda e^{-\lambda(s-s_{\min})},\;s\ge s_{\min}$$
The source writes f_0 as psi(s - s_min; lambda). We write out the exponential density explicitly.

**F7. Rosenblatt-Parzen KDE** (Parzen Eq. 1.7) [10, 11]
$$\hat f_n(x)=\frac{1}{nh}\sum_{j=1}^{n}K\!\left(\frac{x-X_j}{h}\right)$$
The estimator is asymptotically unbiased if h(n) -> 0 [11], and consistent if in addition n h(n) -> infinity.

**F8. Bandwidth factors as implemented in SciPy** [12]
$$h_{\text{Scott}}=n^{-1/(d+4)},\qquad h_{\text{Silverman}}=\left(\frac{n(d+2)}{4}\right)^{-1/(d+4)},\qquad n_{\text{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}$$
These are multiplied by the data covariance, and n is replaced by n_eff for weighted data. For d = 1 the Silverman factor is 1.06 n^(-1/5). SciPy notes that this "is not the same as 'Silverman's rule of thumb'" (0.9 min(sigma, IQR/1.34) n^(-1/5)).

**F9. KDE Bayes posterior** (KCal Eq. 4 and the text below it) [13]
$$P(Y=k\mid x_0)=\frac{f_k(x_0)\pi_k}{\sum_{k'}f_{k'}(x_0)\pi_{k'}},\qquad \hat f_k(x_0)=\frac{1}{|S^k_{cal}|}\sum_{(x,y)\in S^k_{cal}}\hat\phi_b(x,x_0),\qquad \hat\pi_k=\frac{|S^k_{cal}|}{|S_{cal}|}$$

**F10. Two-class KDE relevance posterior and LR test** [PROPOSAL; specialisation of F5/F9, re-derived]
$$\hat P(R=1\mid s)=\frac{\pi\hat f_1(s)}{\pi\hat f_1(s)+(1-\pi)\hat f_0(s)}\ge\tau\iff\Lambda(s)=\frac{\hat f_1(s)}{\hat f_0(s)}\ge\frac{\tau}{1-\tau}\cdot\frac{1-\pi}{\pi}$$
With tau = 0.25 this is Lambda >= (1-pi)/(3 pi): Lambda >= 3 at pi = 0.10 and Lambda >= 33 at pi = 0.01 (checked numerically).

**F11. Invariance under monotone transforms** [PROPOSAL; standard change of variables]
If z = g(s) with g strictly increasing (e.g. atanh s, or the probit of (s+1)/2), then f_k^Z(z) = f_k(s)/g'(s). The Jacobian cancels:
$$\frac{f^Z_1(z)}{f^Z_0(z)}=\frac{f_1(s)}{f_0(s)}$$
So the densities can be estimated in z-space, where there is no boundary [14], and the posterior can be used without back-transforming. This holds for the true densities. With KDEs the ratio of the estimates changes, because the smoothing happens in a different space.

**F12. Platt scaling** (Lin-Lin-Weng Eqs. 1-2) [8]
$$P(y=1\mid f)\approx\frac{1}{1+\exp(Af+B)},\quad \min_{A,B}-\sum_i\big[t_i\log p_i+(1-t_i)\log(1-p_i)\big],\quad t_i=\begin{cases}\frac{N_++1}{N_++2}&y_i=+1\\ \frac{1}{N_-+2}&y_i=-1\end{cases}$$

**F13. Posterior under a changed base rate** (Elkan Theorem 2) [7], plus our logit form
$$p'=b'\,\frac{p-pb}{b-pb+b'p-bb'}\quad\Longleftrightarrow\quad \operatorname{logit}p'=\operatorname{logit}p-\operatorname{logit}b+\operatorname{logit}b'$$
Checker's algebra: 1 - p' = b(1-p)(1-b')/D and p' = b'p(1-b)/D, so odds(p') = odds(p) * odds(b')/odds(b). The two forms agreed to 1e-15 on random inputs. The theorem assumes that the class-conditional distributions are unchanged.

**F14. Split conformal** [29]
$$\hat q=\mathrm{Quantile}\Big(s_1,\dots,s_n;\tfrac{\lceil (n+1)(1-\alpha)\rceil}{n}\Big),\qquad P\big(Y_{test}\in\mathcal C(X_{test})\big)\ge1-\alpha$$
The coverage is marginal and assumes exchangeable calibration and test points.

**F15. Conformal risk control** (Eq. 4, Theorems 1-2) [30]
$$\hat\lambda=\inf\Big\{\lambda:\tfrac{n}{n+1}\hat R_n(\lambda)+\tfrac{B}{n+1}\le\alpha\Big\}\;\Rightarrow\;\alpha-\tfrac{2B}{n+1}\le\mathbb E[L_{n+1}(\hat\lambda)]\le\alpha$$
Conditions for the upper bound:
- the losses are exchangeable, non-increasing and right-continuous in lambda;
- the losses satisfy L <= B;
- L(lambda_max) <= alpha, so that lambda-hat exists.

The lower bound additionally needs i.i.d. losses with no large jumps (Theorem 2). For a retrieved set, the false-negative proportion 1 - |Y ∩ C_lambda|/|Y| with B = 1 is a valid loss.

**F16. Random walk with restart** (Tong Eqs. 1-2) [37]
$$\vec r_i=c\tilde W\vec r_i+(1-c)\vec e_i\;\Rightarrow\;\vec r_i=(1-c)(I-c\tilde W)^{-1}\vec e_i$$

**F17. Personalized PageRank and push** [38]
$$\mathrm{pr}_\alpha(s)=\alpha s+(1-\alpha)\mathrm{pr}_\alpha(s)W=\alpha\sum_{t\ge0}(1-\alpha)^t sW^t,\qquad W=\tfrac12(I+D^{-1}A)$$
The push step moves alpha r(u) into p(u), keeps (1-alpha) r(u)/2 at u, and sends (1-alpha) r(u)/(2 d(u)) to each neighbour. The loop runs while some r(u) >= epsilon d(u). Theorem 1 then gives
$$\mathrm{vol}(\mathrm{Supp}(p))\le\frac{2}{(1-\alpha)\epsilon},\qquad\text{time }O\big(1/(\epsilon\alpha)\big)$$

**F18. Noisy-OR** (Srinivas Sec. 5.1) [40] and **cascade** (KKT Sec. 4) [39]
$$P(t_x\mid\mathbf u)=1-\prod_{i:u_i=t_i}q_i,\qquad f_v(S)=1-\prod_{i=1}^{k}\big(1-p_v(u_i,S_{i-1})\big)$$

**F19. Spreading activation** (SA-RAG, Algorithm 1) [47] and **query-aware SA** (Eq. 1) [46]
$$a_j\leftarrow\min\Big(1,a_j+\sum_{i\in N(j)}a_iw_{ij}\Big);\qquad \Delta r^{(t)}(v)=\alpha\,\sigma(v)\!\!\sum_{u\in A^{(t)}(v)}\!\!r^{(t-1)}(u)$$
- A^(t)(v) is the set of in-neighbours u with r^(t-1)(u) > tau and r^(t-1)(u) > r^(t-1)(v).
- sigma(v) = max(cos(e_v, e_q), 0).
- Defaults: T = 3, alpha = 0.7, tau = 0.01.

**F20. Node specificity** [42]
$$s_i=|P_i|^{-1}$$

**F21. Largest-gap cut-off** (Adaptive-k) [21]
$$k=\arg\max_{i}\,(s_{(i)}-s_{(i+1)}),\quad s_{(1)}\ge s_{(2)}\ge\cdots$$
Keep the top k plus a buffer of B = 5, and search only the top 90% of the list.

**F22. Probabilistic trace** [PROPOSAL]
$$\rho(v)=\max_{\gamma:u_0\leadsto v}P_{u_0}\!\!\prod_{(a,b)\in\gamma}\!\!w_{ab},\qquad \pi_v=1-(1-\pi_0)\!\!\prod_{u\in\mathrm{Pa}(v)}\!\!(1-w_{uv}P_u),\qquad \operatorname{logit}P_v=\operatorname{logit}\pi_v+\ln\frac{\hat f_1(s_v)}{\hat f_0(s_v)}$$
Rules: expand v iff rho(v) >= tau, and keep v iff P_v >= tau.
- Max-product path probability is a shortest-path problem on the non-negative costs -ln w, solved by Dijkstra with source cost -ln P_u0.
- For tau = 0.25 the stop condition is path cost > ln 4 = 1.386 nats.
- The logit fusion is F13 applied with b = pi_0 and b' = pi_v. It is exact only if s_v is conditionally independent of the parents' relevance given R_v (a naive-Bayes assumption). If that fails, graph evidence and score evidence are double-counted, because similar functions are both linked and similarly scored.
- With a leak term, pi_v >= pi_0 always: graph evidence can raise the prior but never lower it.

**F23. Hop bound** [PROPOSAL; checked numerically]
$$P_{u_0}w_{\max}^{k}\ge\tau\;\Rightarrow\;k\le\Big\lfloor\frac{\ln\tau}{\ln w_{\max}}\Big\rfloor$$
For tau = 0.25:

| w_max | hop bound k_max |
|---|---|
| 0.5 | 2 |
| 0.6 | 2 |
| 0.7 | 3 |
| 0.8 | 6 |
| 0.9 | 13 |

This bounds *expansion* depth. Frontier nodes one hop further can still be *scored* and accepted through F22.

**F24. Noisy-OR bounds** [PROPOSAL; elementary]
$$\max_i a_i\le1-\prod_i(1-a_i)\le\min\Big(1,\sum_ia_i\Big)$$
The upper bound is the Weierstrass product inequality. Example: 0.20, 0.15 and 0.10 combine to 0.388 > 0.25.

**F25. Guarded threshold** [PROPOSAL]
$$\mathcal C_\lambda(q)=\{v: P_v\ge1-\lambda\},\qquad t^*=\min(\tau,1-\hat\lambda)$$
Here lambda-hat comes from F15 with the false-negative-rate loss.

## What is novel in our proposal

All items below are **[PROPOSAL]**. None is an established result.

1. **Reading tau as a cost ratio** (F4). Choosing tau = 0.25 is equivalent to the modelling assumption c_FN = 3 c_FP. The paper should state it that way and report a sensitivity sweep.
2. **Non-parametric SD model for code retrieval** (F10-F11). The parametric normal/exponential or gamma components [3-6] are replaced by class-conditional KDEs of a transformed cosine score, followed by a PAV isotonic projection [9]. The projection removes the non-monotone posterior that any two-density model can show wherever the densities cross more than once. We found no prior work applying a KDE-Bayes posterior to function-level code retrieval, but we have not done a systematic search, so the paper should phrase this as "to our knowledge".
3. **Probabilistic trace on a call and similarity graph** (F22). Expansion is decided by max-product path probability, which is Dijkstra on -ln w. Acceptance is decided by the fused posterior (noisy-OR graph prior plus KDE likelihood ratio). The threshold is the same tau, so the procedure stops without a hand-set hop limit (F23). The design combines PPR/push [38], thresholded spreading activation [46, 47] and noisy-OR [40]. Its difference from them is that the stopping quantity is interpreted as a probability of relevance, which ties it to the cost ratio.
4. **Hop bound from tau and w_max** (F23). This is consistent with the small depths used empirically: k <= 2 in RepoGraph [49] and T = 3 in QASA [46]. It does **not** "reproduce" or explain them. RepoGraph chose k for context and noise reasons, and in QASA the threshold 0.01 with decay 0.7 alone would allow about 12 hops (ln 0.01 / ln 0.7 = 12.9), so T = 3 is an independent cap.
5. **Guarded threshold** (F25). The Bayes rule is paired with a conformal-risk-control threshold that bounds the expected false-negative rate, in the spirit of [31, 34, 35].

## Risks and failure modes

- **Calibration dependence.** Bayes optimality of tau holds only for calibrated posteriors [2, 15]. The KDE posterior is calibrated only asymptotically [13]. With few positives, f1 is poorly estimated in the upper tail, which is exactly where decisions are made [4].
- **Bandwidth over kernel.** Bandwidth dominates the estimate [12]. Oversmoothing f1 inflates posteriors for mid-range scores, and undersmoothing makes them spiky. Choose h by likelihood cross-validation per class and report the result.
- **Label sparsity and selection bias.** SD/EM methods suffer from sparse, biased judgements and a ranking-length bias [4]. Synthetic questions generated per function are a biased sample of real queries, which breaks the exchangeability that F14-F15 need.
- **Non-monotone posterior.** Any two-density model can cross twice [3, 4]. Apply PAV (isotonic regression) [9] and log how often it changes the posterior.
- **Hubs and double counting.** Noisy-OR assumes independent causes [39, 40, 41]. Call paths through shared utility functions violate this and inflate pi_v (F24). Static weights cause semantic drift into hubs [45]. The naive-Bayes fusion in F22 can double-count similarity.
- **Truncation is hard to learn.** Supervised cut-off predictors often fail to beat a fixed depth [20]. Score-pattern methods fail with weak scorers [25]. An adversarial chunk can create an artificial gap [21].
- **Conformal caveats.** The coverage is marginal only [29]. CRC needs a monotone loss and L(lambda_max) <= alpha [30]. Monotonicity breaks if the trace's expansion threshold is held at tau while the acceptance threshold varies, because the reachable set then differs from the set that was calibrated. With a low alpha or a weak embedder, the calibrated set can overflow the context window [31].
- **The cost ratio drifts.** Robustness training changes how much distractors hurt the model [58], so c_FP, and with it the optimal tau, changes after fine-tuning or quantisation.
- **Position effects.** Even a correct set can be under-used if key functions sit in the middle of the prompt [59].

## Design recommendations

1. **Make tau a cost parameter.** Expose `tau` (default 0.25, documented as c_FN : c_FP = 3 : 1) in the UI next to the fine-tuning toggle. Sweep tau over {0.10, 0.25, 0.50} in the evaluation, and state that F1-type objectives have no optimal fixed threshold [4, 5].
2. **Build a labelled calibration set first.** On the mimalloc checkout, collect (query, function, relevant?) triples. Keep three disjoint splits: KDE fitting, conformal calibration, and test. Draw reliability diagrams and report ECE and Brier score.
3. **Estimate the densities in transformed space.** Fit f1 and f0 by 1-D Gaussian KDE on z = atanh(clamp(s, -0.999, 0.999)), with h chosen by cross-validation (Scott or Silverman as the starting point [12]). Tabulate log Lambda(z) on a 256-1024-point grid, apply PAV to the resulting posterior, and store it as a lookup table of 1-4 KB. That fits the L1 budget of a single 16 KB work item.
4. **Work in log-odds everywhere.** Use logit P = logit pi + log Lambda (F13, F22) and path cost = sum of -ln w. The trace is then a bounded Dijkstra with cut-off ln(1/tau) = 1.386 nats. Products never underflow, and a prior change is a single addition.
5. **Estimate edge weights per edge type and clamp them.** Estimate w for caller->callee, callee->caller and similarity edges as conditional relevance frequencies on the calibration set. Clamp to w_max < 1, and print the implied hop bound (F23) in the paper.
6. **Protect against hubs.** For mimalloc's assertion, stats, atomic and error helpers, multiply w by the specificity 1/|callers| [42] or use a degree-normalised residual test r(u) >= epsilon d(u) [38]. Decide expansion by max-product, and use noisy-OR only to rank or score nodes that have already been reached.
7. **Gate edges by the query.** Multiply w_uv by max(cos(e_v, e_q), 0) [46], since static weights are the documented cause of drift [45]. The embeddings are already computed, so the extra cost is one dot product per edge visited.
8. **Add the conformal guard.** Calibrate lambda-hat for a target FNR alpha of 0.1-0.2 with CRC [30]. Apply the *same* threshold t* to both expansion and acceptance so that the set family is nested. Check that L(lambda_max) <= alpha, and report empirical FNR, mean set size and tokens.
9. **Compare against the right baselines.**
    - Baselines: fixed top-k (5, 10, 20) [20], Adaptive-k (largest gap plus B = 5) [21], PPR top-k with damping 0.5 [42], and k-hop ego-graphs with k = 1, 2 [49].
    - Metrics: set recall and precision, set size, tokens, calibration error of P_v, and downstream answer or edit correctness.
10. **Order and budget the prompt.** Put the highest-posterior functions at the start and end of the prompt [59]. If the accepted set exceeds the context budget, truncate by posterior rank and log it as a budget-induced false negative.
11. **Recalibrate after every model change.** Refit f1 and f0 and recalibrate lambda-hat whenever the embedder, its quantisation, the chunking or the fine-tuned generator changes [58]. Store the calibration version with the index.
12. **Report the proposals as proposals.** In the paper, label F4, F10-F11 and F22-F25 as contributions or derivations, and cite [1-7, 13, 30, 38, 40] as the foundations they build on.

## Corrections made to the incoming dossier

- **FeatLens [56].** The author list lacked the ninth author, **Bing Xie**. Added.
- **Repantis et al. [25].** The fourth author is **Joey Blackwell II**. Fixed.
- **Hop-bound claim (F23).** The dossier said the bound "reproduces from first principles" the depth limits of RepoGraph and QASA. This is overstated. RepoGraph's k <= 2 was chosen for context and noise reasons. With QASA's own threshold 0.01 and decay 0.7 the analogous bound is about 12 hops, so its T = 3 is an independent cap. Reworded to "consistent with".
- **SciPy "Silverman" factor (F8).** SciPy documents that its factor is not Silverman's rule of thumb. Added this caveat and the d = 1 value 1.06 n^(-1/5).
- **Parzen (F7).** h -> 0 gives asymptotic unbiasedness only. Consistency also needs n h -> infinity. Added.
- **CRC (F15).** Added the requirement L(lambda_max) <= alpha, without which lambda-hat may not exist. Clarified that the lower bound needs Theorem 2's extra conditions.
- **Guarded threshold (F25).** The CRC guarantee applies only if the whole pipeline, expansion included, is run at the calibrated threshold. The dossier kept expansion at tau. Corrected. Also removed the claim that taking min(tau, 1 - lambda-hat) "keeps the Bayes-optimal rule": when t* < tau the rule is no longer cost-optimal, it trades cost for recall.
- **Fusion (F22).** Stated the conditional-independence assumption behind the logit fusion, and the fact that noisy-OR with a leak can only raise the prior.
- **Invariance under transforms (F11).** The Jacobian cancels for the true densities, but KDE ratios are not invariant, because the estimates depend on the space in which smoothing happens. Clarified.
- **Exponential density (F6).** Written out explicitly as lambda exp(-lambda (s - s_min)).
- **CONFLARE [31].** The source text says the oversized-set problem arises when error rates are "excessively high". The dossier silently changed this to "low alpha". We now report the source's wording and flag it as an apparent slip.
- **Kanoulas et al. [6].** Added Arampatzis and Robertson's caveat that the K-S-test results are only "indicative".
- **Lost in the Middle [59].** Venue given as TACL, accepted 2023, as shown on arXiv. The volume number was not verified and was removed.
- **Elkan [7].** Page numbers were not verified and were left out.
- **Venues not confirmed on the pages opened** are cited as arXiv preprints: KCal, Self-RAG, CRAG, CodexGraph, Power of Noise, Yoran et al., AttnCut, Surprise.
- **Added after opening them:**
  - Rosenblatt 1956 [10], the foundational KDE paper.
  - Surprise [18], EVT-based truncation and the precursor of TAA-k.
  - C-RAG [33].
  - Conformal score refinement [34].
  - Conformal context engineering, ECIR 2026 [35].
- **Dropped:** none of the 54 dossier sources. All were confirmed by abstract or DOI record, and body quotes were checked against PDFs downloaded this session. Robertson 1977 was verified through its Crossref record and abstract, because the Emerald page returned HTTP 403.

## References

1. [robertson1977prp] Robertson, Stephen E. (1977). *The Probability Ranking Principle in IR*. Journal of Documentation 33(4):294-304. https://doi.org/10.1108/eb026647
2. [manning2008ir] Manning, Christopher D.; Raghavan, Prabhakar; Schütze, Hinrich (2008). *Introduction to Information Retrieval (Sec. 11.2, The Probability Ranking Principle)*. Cambridge University Press. https://nlp.stanford.edu/IR-book/html/htmledition/the-prp-with-retrieval-costs-1.html
3. [manmatha2001scoredist] Manmatha, R.; Rath, T.; Feng, F. (2001). *Modeling Score Distributions for Combining the Outputs of Search Engines*. Proc. 24th ACM SIGIR, pp. 267-275. https://ciir.cs.umass.edu/pubfiles/ir-224.pdf
4. [arampatzis2011modeling] Arampatzis, Avi; Robertson, Stephen (2011). *Modeling score distributions in information retrieval*. Information Retrieval 14(1):26-46 (online 2010). https://doi.org/10.1007/s10791-010-9145-5
5. [arampatzis2008stop] Arampatzis, Avi; Kamps, Jaap (2008). *Where to Stop Reading a Ranked List?*. Proc. TREC 2008 (TREC-17), Legal Track, NIST. https://trec.nist.gov/pubs/trec17/papers/uamsterdam-kamps.legal.rev.pdf
6. [kanoulas2009scoredist] Kanoulas, Evangelos; Pavlu, Virgil; Dai, Keshi; Aslam, Javed A. (2009). *Modeling the Score Distributions of Relevant and Non-relevant Documents*. ICTIR 2009, LNCS 5766, pp. 152-163. https://doi.org/10.1007/978-3-642-04417-5_14
7. [elkan2001cost] Elkan, Charles (2001). *The Foundations of Cost-Sensitive Learning*. IJCAI 2001. https://cseweb.ucsd.edu/~elkan/rescale.pdf
8. [lin2007platt] Lin, Hsuan-Tien; Lin, Chih-Jen; Weng, Ruby C. (2007). *A note on Platt's probabilistic outputs for support vector machines*. Machine Learning 68(3):267-276. https://doi.org/10.1007/s10994-007-5018-6
9. [zadrozny2002isotonic] Zadrozny, Bianca; Elkan, Charles (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. Proc. 8th ACM SIGKDD, pp. 694-699. https://doi.org/10.1145/775047.775151
10. [rosenblatt1956kde] Rosenblatt, Murray (1956). *Remarks on Some Nonparametric Estimates of a Density Function*. The Annals of Mathematical Statistics 27(3):832-837. https://doi.org/10.1214/aoms/1177728190
11. [parzen1962kde] Parzen, Emanuel (1962). *On Estimation of a Probability Density Function and Mode*. The Annals of Mathematical Statistics 33(3):1065-1076. https://doi.org/10.1214/aoms/1177704472
12. [scipy_gaussian_kde] SciPy developers (2026). *scipy.stats.gaussian_kde (SciPy v1.18.0 Manual)*. Official SciPy documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
13. [lin2022kcal] Lin, Zhen; Trivedi, Shubhendu; Sun, Jimeng (2022). *Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks*. arXiv:2202.07679. https://arxiv.org/abs/2202.07679
14. [geenens2013probit] Geenens, Gery (2013). *Probit transformation for kernel density estimation on the unit interval*. arXiv:1303.4121. https://arxiv.org/abs/1303.4121
15. [penha2021calibration] Penha, Gustavo; Hauff, Claudia (2021). *On the Calibration and Uncertainty of Neural Learning to Rank Models*. EACL 2021. https://arxiv.org/abs/2101.04356
16. [lien2019bicut] Lien, Yen-Chieh; Cohen, Daniel; Croft, W. Bruce (2019). *An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists*. ACM ICTIR 2019, pp. 79-82. https://doi.org/10.1145/3341981.3344234
17. [bahri2020choppy] Bahri, Dara; Tay, Yi; Zheng, Che; Metzler, Donald; Tomkins, Andrew (2020). *Choppy: Cut Transformer For Ranked List Truncation*. SIGIR 2020. https://arxiv.org/abs/2004.13012
18. [bahri2020surprise] Bahri, Dara; Zheng, Che; Tay, Yi; Metzler, Donald; Tomkins, Andrew (2020). *Surprise: Result List Truncation via Extreme Value Theory*. arXiv:2010.09797. https://arxiv.org/abs/2010.09797
19. [wu2021attncut] Wu, Chen; Zhang, Ruqing; Guo, Jiafeng; Fan, Yixing; Lan, Yanyan; Cheng, Xueqi (2021). *Learning to Truncate Ranked Lists for Information Retrieval*. arXiv:2102.12793. https://arxiv.org/abs/2102.12793
20. [meng2024rlt] Meng, Chuan; Arabzadeh, Negar; Askari, Arian; Aliannejadi, Mohammad; de Rijke, Maarten (2024). *Ranked List Truncation for Large Language Model-based Re-Ranking*. SIGIR 2024. https://arxiv.org/abs/2404.18185
21. [taguchi2025adaptivek] Taguchi, Chihiro; Maekawa, Seiji; Bhutani, Nikita (2025). *Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k*. EMNLP 2025 (Main). https://arxiv.org/abs/2506.08479
22. [song2026taak] Song, Ziyu; Fang, Jiaming; Li, Kuangyu; Xia, Tuo; Wang, Chuanpeng (2026). *Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation*. ECML PKDD 2026 (accepted; arXiv:2606.11907). https://arxiv.org/abs/2606.11907
23. [xu2025car] Xu, Yifan; Gupta, Vipul; Aggarwal, Rohit; Mahadevan, Varsha; Krishnamachari, Bhaskar (2025). *Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications*. arXiv:2511.14769. https://arxiv.org/abs/2511.14769
24. [xia2026preretrieval] Xia, Ye; Yamangil, Emre; Wang, Haixun (2026). *Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems*. CIKM 2026 Applied Research Track (accepted; arXiv:2609.13489). https://arxiv.org/abs/2609.13489
25. [repantis2026tools] Repantis, Vyzantinos; Gawde, Ameya; Singh, Harshvardhan; Blackwell II, Joey (2026). *How Many Tools Should an LLM Agent See? A Chance-Corrected Answer*. arXiv:2605.24660. https://arxiv.org/abs/2605.24660
26. [asai2023selfrag] Asai, Akari; Wu, Zeqiu; Wang, Yizhong; Sil, Avirup; Hajishirzi, Hannaneh (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511. https://arxiv.org/abs/2310.11511
27. [jeong2024adaptiverag] Jeong, Soyeong; Baek, Jinheon; Cho, Sukmin; Hwang, Sung Ju; Park, Jong C. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. NAACL 2024. https://arxiv.org/abs/2403.14403
28. [yan2024crag] Yan, Shi-Qi; Gu, Jia-Chen; Zhu, Yun; Ling, Zhen-Hua (2024). *Corrective Retrieval Augmented Generation*. arXiv:2401.15884. https://arxiv.org/abs/2401.15884
29. [angelopoulos2021gentle] Angelopoulos, Anastasios N.; Bates, Stephen (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511. https://arxiv.org/abs/2107.07511
30. [angelopoulos2022crc] Angelopoulos, Anastasios N.; Bates, Stephen; Fisch, Adam; Lei, Lihua; Schuster, Tal (2022). *Conformal Risk Control*. arXiv:2208.02814. https://arxiv.org/abs/2208.02814
31. [rouzrokh2024conflare] Rouzrokh, Pouria; Faghani, Shahriar; Gamble, Cooper U.; Shariatnia, Moein; Erickson, Bradley J. (2024). *CONFLARE: CONFormal LArge language model REtrieval*. arXiv:2404.04287. https://arxiv.org/abs/2404.04287
32. [li2023traq] Li, Shuo; Park, Sangdon; Lee, Insup; Bastani, Osbert (2024). *TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction*. NAACL 2024 (arXiv:2307.04642, 2023). https://arxiv.org/abs/2307.04642
33. [kang2024crag] Kang, Mintong; Gürel, Nezihe Merve; Yu, Ning; Song, Dawn; Li, Bo (2024). *C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models*. ICML 2024. https://arxiv.org/abs/2402.03181
34. [intrator2024refine] Intrator, Yotam; Kelner, Ori; Cohen, Regev; Goldenberg, Roman; Rivlin, Ehud; Freedman, Daniel (2024). *Streamlining Conformal Information Retrieval via Score Refinement*. arXiv:2410.02914. https://arxiv.org/abs/2410.02914
35. [chakraborty2025conformalctx] Chakraborty, Debashish; Yang, Eugene; Khashabi, Daniel; Lawrie, Dawn; Duh, Kevin (2026). *Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction*. ECIR 2026 (accepted; arXiv:2511.17908, Nov 2025). https://arxiv.org/abs/2511.17908
36. [crestani1997spreading] Crestani, F. (1997). *Application of Spreading Activation Techniques in Information Retrieval*. Artificial Intelligence Review 11(6):453-482. https://doi.org/10.1023/A:1006569829653
37. [tong2006rwr] Tong, Hanghang; Faloutsos, Christos; Pan, Jia-Yu (2006). *Fast Random Walk with Restart and Its Applications*. IEEE ICDM 2006, pp. 613-622. https://www.cs.cmu.edu/~htong/pdf/ICDM06_tong.pdf
38. [andersen2006local] Andersen, Reid; Chung, Fan; Lang, Kevin (2006). *Local Graph Partitioning using PageRank Vectors*. IEEE FOCS 2006, pp. 475-486. https://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
39. [kempe2003influence] Kempe, David; Kleinberg, Jon; Tardos, Éva (2003). *Maximizing the Spread of Influence through a Social Network*. Proc. 9th ACM SIGKDD, pp. 137-146. https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
40. [srinivas1993noisyor] Srinivas, Sampath (1993). *A Generalization of the Noisy-Or Model*. UAI 1993, pp. 208-215 (arXiv:1303.1479). https://arxiv.org/abs/1303.1479
41. [zhang1996causal] Zhang, Nevin Lianwen; Poole, David (1996). *Exploiting Causal Independence in Bayesian Network Inference*. Journal of Artificial Intelligence Research 5:301-328. https://arxiv.org/abs/cs/9612101
42. [gutierrez2024hipporag] Jiménez Gutiérrez, Bernal; Shu, Yiheng; Gu, Yu; Yasunaga, Michihiro; Su, Yu (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*. NeurIPS 2024. https://arxiv.org/abs/2405.14831
43. [gutierrez2025hipporag2] Jiménez Gutiérrez, Bernal; Shu, Yiheng; Qi, Weijian; Zhou, Sizhe; Su, Yu (2025). *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*. ICML 2025. https://arxiv.org/abs/2502.14802
44. [edge2024graphrag] Edge, Darren; Trinh, Ha; Cheng, Newman; Bradley, Joshua; Chao, Alex; Mody, Apurva; Truitt, Steven; Metropolitansky, Dasha; Ness, Robert Osazuwa; Larson, Jonathan (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130. https://arxiv.org/abs/2404.16130
45. [lau2026catrag] Lau, Kwun Hang; Zhang, Fangyuan; Ruan, Boyu; Zhou, Yingli; Guo, Qintian; Zhang, Ruiyuan; Zhou, Xiaofang (2026). *Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation*. arXiv:2602.01965. https://arxiv.org/abs/2602.01965
46. [makarov2026qasa] Makarov, Illia; Glybovets, Mykola (2026). *Query-Aware Spreading Activation for Multi-Hop Retrieval over Knowledge Graphs*. arXiv:2606.30133 (accepted, Cybernetics and Systems Analysis; not yet published). https://arxiv.org/abs/2606.30133
47. [pavlovic2025sarag] Pavlović, Jovan; Krész, Miklós; Hajdu, László (2025). *Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems*. arXiv:2512.15922. https://arxiv.org/abs/2512.15922
48. [zhang2023repocoder] Zhang, Fengji; Chen, Bei; Zhang, Yue; Keung, Jacky; Liu, Jin; Zan, Daoguang; Mao, Yi; Lou, Jian-Guang; Chen, Weizhu (2023). *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation*. EMNLP 2023. https://arxiv.org/abs/2303.12570
49. [ouyang2024repograph] Ouyang, Siru; Yu, Wenhao; Ma, Kaixin; Xiao, Zilin; Zhang, Zhihan; Jia, Mengzhao; Han, Jiawei; Zhang, Hongming; Yu, Dong (2025). *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph*. ICLR 2025 (arXiv:2410.14684, 2024). https://arxiv.org/abs/2410.14684
50. [liu2024graphcoder] Liu, Wei; Yu, Ailun; Zan, Daoguang; Shen, Bo; Zhang, Wei; Zhao, Haiyan; Jin, Zhi; Wang, Qianxiang (2024). *GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model*. arXiv:2406.07003. https://arxiv.org/abs/2406.07003
51. [phan2024repohyper] Phan, Huy N.; Phan, Hoang N.; Nguyen, Tien N.; Bui, Nghi D. Q. (2024). *RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion*. arXiv:2403.06095. https://arxiv.org/abs/2403.06095
52. [liu2024codexgraph] Liu, Xiangyan; Lan, Bo; Hu, Zhiyuan; Liu, Yang; Zhang, Zhicheng; Wang, Fei; Shieh, Michael; Zhou, Wenmeng (2024). *CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases*. arXiv:2408.03910. https://arxiv.org/abs/2408.03910
53. [li2025graphcodeagent] Li, Jia; Shi, Xianjie; Zhang, Kechi; Li, Ge; Jin, Zhi; Li, Lei; Zhang, Huangzhao; Li, Jia; Liu, Fang; Zhang, Yuwei; Tao, Zhengwei; Dong, Yihong; Zhu, Yuqi; Tao, Chongyang (2025). *GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation*. arXiv:2504.10046 (v1 titled 'CodeRAG: Supportive Code Retrieval on Bigraph for Real-World Code Generation'). https://arxiv.org/abs/2504.10046
54. [liu2025reposcope] Liu, Yang; Zhang, Li; Liu, Fang; Wang, Zhuohang; Wei, Donglin; Yang, Zhishuo; Zhang, Kechi; Li, Jia; Shi, Lin (2026). *RepoScope: Leveraging Call Chain-Aware Multi-View Context for Repository-Level Code Generation*. ICSE 2026 (accepted; arXiv:2507.14791, 2025). https://arxiv.org/abs/2507.14791
55. [liu2026dyretriever] Liu, Zhongxin; Jiang, Zhonghao; Ye, Zhifan; Wang, Haoye; Liu, Jiakun; Ren, Xiaoxue (2026). *Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation*. ASE 2026 (accepted; arXiv:2608.01927). https://arxiv.org/abs/2608.01927
56. [li2026featlens] Li, Xutian; Xiong, Bo; Zhu, Yifeng; Li, Kunze; Zhao, Xianlin; Yan, Runbang; Zou, Yanzhen; Zhang, Lu; Xie, Bing (2026). *FeatLens: Feature-Guided Dynamic Code Graph Construction and Retrieval for Repository-Level Code Generation*. arXiv:2609.26480. https://arxiv.org/abs/2609.26480
57. [cuconasu2024noise] Cuconasu, Florin; Trappolini, Giovanni; Siciliano, Federico; Filice, Simone; Campagnano, Cesare; Maarek, Yoelle; Tonellotto, Nicola; Silvestri, Fabrizio (2024). *The Power of Noise: Redefining Retrieval for RAG Systems*. arXiv:2401.14887. https://arxiv.org/abs/2401.14887
58. [yoran2023robust] Yoran, Ori; Wolfson, Tomer; Ram, Ori; Berant, Jonathan (2023). *Making Retrieval-Augmented Language Models Robust to Irrelevant Context*. arXiv:2310.01558. https://arxiv.org/abs/2310.01558
59. [liu2023lost] Liu, Nelson F.; Lin, Kevin; Hewitt, John; Paranjape, Ashwin; Bevilacqua, Michele; Petroni, Fabio; Liang, Percy (2023). *Lost in the Middle: How Language Models Use Long Contexts*. Transactions of the Association for Computational Linguistics (accepted 2023; arXiv:2307.03172). https://arxiv.org/abs/2307.03172
