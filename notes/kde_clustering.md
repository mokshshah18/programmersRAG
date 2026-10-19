# Kernel density estimation and mode-seeking clustering on the hypersphere

Verified research note for the programmersRAG C++20 rewrite.
Verification date: 2026-09-29. Evaluation corpus: a clone (fork) of microsoft/mimalloc (C, about 41 `.c` and 18 `.h` files, about 1.3 MB), i.e. on the order of 2000 functions.
Target machine: Intel i9-12950HX (8 P-cores with 48 KB L1d, 8 E-cores with 32 KB L1d, 24 logical CPUs), 32 GB RAM, RTX A1000 4 GB.

How this note was produced. A first researcher compiled a dossier of 36 findings, 28 formulas and 45 sources. Every source was then re-opened independently: the arXiv abstract page and, where available, the arXiv TeX source; the publisher, proceedings or Project Euclid page; the Crossref record for the DOI; or the PDF itself. Every formula was re-derived or checked numerically (mpmath at 60 digits, NumPy Monte Carlo). Anything that could not be confirmed was reworded or removed. Web search was not available during verification (the session search budget was exhausted), so the hunt for missing work used arXiv's own search pages, Crossref and OpenAlex; the novelty statements in this note are therefore "not found in these searches", not proofs of absence.

Status tags used below:

- **[EST]** established, peer-reviewed or widely replicated result.
- **[REC]** recent result (2024-2026), sometimes a preprint; treat as provisional.
- **[CON]** contested: sources disagree.
- **[OURS]** our own proposal, derivation or simulation. Not a published result. Must be labelled as such in the paper.

Notation. Ambient dimension $d$, sphere $S^{d-1} = \Omega_q \subset \mathbb{R}^{q+1}$ with intrinsic dimension $q = d-1$. Sample $X_1,\dots,X_n$ of unit vectors. Bandwidth $h$, concentration $\kappa = 1/h^2$. Cosine similarity $s = x^\top y$.

---

## Summary

1. **The object is well defined and classical.** On L2-normalised embeddings a kernel density estimate with the von Mises kernel $L(r)=e^{-r}$ is exactly an equal-weight mixture of $n$ von Mises-Fisher (vMF) densities with common concentration $\kappa = 1/h^2$ [7][8][20], and it is the same function as a Gaussian-kernel KDE restricted to the sphere because $\lVert x-y\rVert^2 = 2(1-x^\top y)$ [20][39]. All constants in the dossier's vMF, directional-KDE, AMISE and rule-of-thumb formulas were confirmed against the TeX sources and numerically.
2. **Directional mean shift is older than the dossier said.** It was first proposed by Oba, Kato and Ishii (2005) [18] and derived for the vMF kernel by Kobayashi and Otsu (2010) [19]. Zhang and Chen (JMLR 2021) [20] supplied the statistical and algorithmic theory: monotone ascent (Theorem 8), local convergence (Theorem 11) and linear convergence of Riemannian gradient ascent (Theorem 12). Their preprint [21] adds convergence from almost every starting point.
3. **The statistical rates are as bad in high dimension as the dossier claimed.** The radical signs that the first researcher could not read are confirmed from the TeX source: density, gradient and Hessian errors are $O(h^2)+O_P(\sqrt{1/(nh^{q})})$, $O_P(\sqrt{1/(nh^{q+2})})$ and $O_P(\sqrt{1/(nh^{q+4})})$ [20]. For $n = 2000$ the optimal-rate factor $n^{-2/(d+4)}$ is 0.28 at $d=8$, 0.66 at $d=32$ and 0.96 at $d=384$.
4. **Published evidence for density-based mode clustering stops at about 12 dimensions.** Zhang and Chen's simulation degrades from $q=3$ to $q=12$ [20]; Garcia-Portugues evaluates bandwidth selectors only up to $q=5$ [8]; Saavedra-Nieves and Fernandez-Perez simulate only the circle and the sphere [23]. Nothing in the literature validates vMF-KDE mode clustering between 8 and 384 dimensions.
5. **[OURS] Two closed-form diagnostics explain the breakdown.** Under an isotropic null the effective number of points contributing to a vMF-KDE is about $n e^{-\kappa^2/d}$, and at a data point the point's own kernel carries a share $[1+(n-1)e^{\kappa^2/(2d)-\kappa}]^{-1}$ of the density, which passes one half at $\kappa \approx \ln n$. In $d=384$, $n=2000$ every point becomes its own mode by $\kappa = 10$. Both were confirmed by simulation.
6. **[OURS] Dimension reduction widens the usable window of $\kappa$ rather than being strictly required.** On a synthetic 8-cluster corpus in $d=384$ directional mean shift recovered the clusters at the tested values $\kappa = 5, 10, 20$ but not at $\kappa = 2$ or at $\kappa \ge 40$; after PCA to 8 or 16 dimensions it recovered them at every tested $\kappa$ from 2 to 160. This is a best case for PCA (isotropic noise) and must be re-measured on the mimalloc fork.
7. **The Johnson-Lindenstrauss lemma cannot justify the reduction.** For $n=2000$ it needs $k \ge 365$ ($\epsilon=0.5$), 845 (0.3), 1755 (0.2), 6516 (0.1) [31], and the $\Omega(\epsilon^{-2}\log n)$ dependence is optimal [32]. Any reduction to 8-32 dimensions rests on data-dependent structure and empirical validation.
8. **The "25 percent probability of relevance" rule has published ancestors.** Combining vMF-kernel class-conditional densities by Bayes' theorem to get thresholdable posterior probabilities was published for document categorisation by Skabar and Memon (2010) [50]; posterior relevance from fitted score distributions is classical IR [51][52]. What we did not find is the query-seeded, background-corrected, trace-expanding variant over source code.
9. **[OURS] The rule reduces to a density-corrected cosine threshold.** $P(\mathrm{rel}\mid x)\ge 1/4$ is equivalent to $\kappa_R\, q^\top x - \log \hat f_B(x) \ge -\ln 3 - a$: a function in a dense region of the corpus (a hub) needs a higher cosine to be admitted. The three parameters can be fitted by logistic regression on a small labelled set.
10. **Recent adaptive-retrieval work (2024-2026) decides retrieval depth from one-dimensional score statistics, a learned calibration, query clusters or a conformal threshold** [53][54][55][56][57][58]. None estimates a density over the document embeddings. These are the baselines to beat.
11. **At this corpus size exact computation is the right choice.** One exact KDE evaluation is 768 000 multiply-adds at $d=384$; hashing-based estimators carry kernel-dependent factors that are large for the Gaussian kernel [35], and DEANN [38] only pays off at much larger $n$.

### Corrections made during verification

| # | First dossier said | Verified position |
|---|---|---|
| C1 | Yamasaki and Tanaka, "arXiv:2305.08463" | Published: IEEE TPAMI 46(10), 6688-6698, 2024, DOI 10.1109/TPAMI.2024.3385920 [17]. Their theorem gives convergence of Euclidean mean shift to a **critical point** of the KDE, not necessarily a mode. |
| C2 | McInnes and Healy, "arXiv" | Published: IEEE ICDM Workshops 2017, pp. 33-42, under the title "Accelerated Hierarchical Density Based Clustering" [25]. |
| C3 | Kong and Fowlkes, year 2017, arXiv | Published: CVPR 2018, pp. 9018-9028 [46]. |
| C4 | Chen tutorial, "arXiv" | Published: Biostatistics & Epidemiology 1(1), 161-187, 2017 [3]. |
| C5 | Bruch, "arXiv monograph" | Also published as a Springer book, 2024, DOI 10.1007/978-3-031-55182-6 [30]. |
| C6 | Xu et al. (CAR) has four authors | Five authors; Bhaskar Krishnamachari was missing [54]. |
| C7 | Choi et al., Ethayarajh, Kusupati et al., Xia et al. listed as arXiv only | CVPR 2024 pp. 23094-23104 [47]; EMNLP-IJCNLP 2019 pp. 55-65 [40]; NeurIPS 35 pp. 30233-30249 [42]; accepted at CIKM 2026 Applied Research Track [57]. |
| C8 | "Directional mean shift (Zhang and Chen, JMLR 2021) is the fixed-point iteration ..." | The algorithm predates them: Oba et al. 2005 [18], Kobayashi and Otsu 2010 [19]. Zhang and Chen re-derive it and prove its learning theory [20]. |
| C9 | Optimal Euclidean bandwidth constant "could not be verified" | The tutorial prints $4\mu_K$ in the numerator [3]. Differentiating its own MISE expansion gives $d\,\mu_K$, which is the dossier's value and reproduces the classical normal-reference rule; the printed 4 is correct only when $d=4$. The tutorial also labels $O(n^{-2/(d+4)})$ as the optimal MISE; substitution gives MISE $=O(n^{-4/(d+4)})$ and root-MISE $O(n^{-2/(d+4)})$. Finally $\sigma_K^2$ must be read as the per-coordinate second moment of $K$ when $d>1$ (F2). |
| C10 | JL bound: $k \ge 1754$ at $\epsilon=0.2$, 6515 at 0.1 | The bound values are 1754.05 and 6515.06, so the smallest integers are 1755 and 6516. The bound is sufficient, not necessary; optimality is due to Larsen and Nelson [32]. |
| C11 | Effective count 1557, 685, 273, 81, 13 at $\kappa = 0.5,1,1.5,2,3\times\sqrt d$ | Re-simulated over 1000 queries: 1560, 754, 265, 93, 20. The first dossier's values look like a single draw. Qualitative conclusion unchanged (F28). |
| C12 | "Multiplying probabilities ... guarantees termination because each factor is below 1" | False in general: a product of factors below 1 can stay above 0.25 forever. Termination follows from a visited set on a finite corpus, or from a bound $p_{\max}<1$ on every factor, which gives depth $\le \ln 4/\ln(1/p_{\max})$ (F30). |
| C13 | Refine the bandwidth by likelihood cross-validation (LCV) | Garcia-Portugues reports that LCV is competitive on the circle but that LSCV outperforms it on the sphere and in higher dimension ($q=2,\dots,5$), and that the exact-MISE mixture selector (EMI) is best overall; the rule of thumb "fails estimating densities with multimodality" [8]. Recommendation changed (D3). |
| C14 | HBE query time $O(d/(\sqrt\tau\epsilon^2))$; "every Gaussian-kernel algorithm applies without modification" | The clean bound holds for kernels with a constant-factor hashing property; Backurs et al. instantiate it for the Laplacian and exponential kernels [36]. For the Gaussian kernel the HBE constants are $e^{O(R^{4/3}\log\log n)}$ in the data radius $R$ [35]. The algebraic identity Gaussian = vMF on the sphere is exact; the performance guarantees are kernel-specific. |
| C15 | "Contextual embeddings are anisotropic ... so the background density is far from uniform" | Ethayarajh measured contextual **word** representations of BERT, ELMo and GPT-2 [40]. Contrastive training pushes sentence embeddings towards uniformity [39][41]. The degree of anisotropy of the chosen encoder on the corpus must be measured, not assumed. |
| C16 | The posterior-relevance formula is wholly "our proposal" | The construction (vMF-kernel densities + Bayes' theorem + probability threshold) is published for document categorisation [50]; score-distribution posteriors are classical [51][52]. Novelty claim narrowed (see "What is novel"). |
| C17 | Matryoshka: "Up to 14x faster yet accurate large-scale classification and retrieval" | Not a verbatim quotation. The abstract says "up to 14x smaller embedding size for ImageNet-1K classification at the same level of accuracy" and "up to 14x real-world speed-ups for large-scale retrieval" [42]. Prefix truncation is justified only for encoders trained with the Matryoshka loss. |
| C18 | RAPTOR: "number of components by BIC, so a chunk may belong to several clusters" | Multi-membership comes from soft (GMM) clustering; BIC only selects the number of components [45]. |
| C19 | "The Fisher kernel of the vMF model equals the cosine similarity"; spherical k-means "when all concentrations are equal (and tend to infinity) with hard assignments" | Banerjee et al. derive the Fisher-kernel identity for $\kappa=1$ assuming an identity Fisher information matrix. Spherical k-means is the special case of soft-moVMF with equal priors and equal $\kappa\to\infty$, and of hard-moVMF with equal priors and equal $\kappa$ [10]. |
| C20 | Radical signs and the contraction factor of Theorem 12 "did not survive text extraction" | Confirmed from the TeX source [20]: all stochastic terms carry square roots; $\Upsilon=\sqrt{1-\eta\lambda_*/2}$. Remark 3 of the paper prints the Euclidean comparison rates without the square root, an apparent typographical slip. |
| C21 | Kim et al.: "KDE error bounds" adapt to the volume dimension | Their bounds are for the stochastic term $\sup_x\lvert \hat p_h - \mathbb{E}\hat p_h\rvert$, uniformly over bandwidths; the bias is not covered [33]. |
| C22 | Density peaks: $\delta_i=\min_{j:\rho_j>\rho_i} d_{ij}$ | Incomplete: for the point of highest density the paper sets $\delta_i=\max_j d_{ij}$ [27]. |
| C23 | Stone (1980) gives "the optimal rate" | It is the optimal rate for estimating a functional at a point, uniformly over a smoothness class. The tutorial cites Stone (1982) and Tsybakov (2009) for the global statement; those two were not opened and are not cited here [3][4]. |
| C24 | "Kish effective number" | No source for that attribution was opened. The same quantity is defined in the SciPy documentation as `neff = sum(weights)^2 / sum(weights^2)` [5]; we cite that. |
| C25 | Sra (2012): truncated Newton gives "more accurate values for the same cost" | The comparison is with the fixed-point method of Tanabe et al. (2007), at the same amount of computation [11]. |

No source from the first dossier failed verification. Sixteen sources were added [9][18][19][22][32][41][48][49][50][51][52][55][56][58][60][61]. A second verification pass on 2026-09-30 re-opened every arXiv abstract page and every DOI through Crossref, re-ran the numerical checks (Banerjee Table 1, $\log c_{384}(100)$, JL bounds, $h_{\mathrm{ROT}}$ against a direct quadrature of the AMISE curvature term for $q=2$ and $q=5$, and the $n_{\mathrm{eff}}$ simulation) and found them consistent with this note; it added [60] and [61].

---

## Established results

### E1. Kernel density estimation and its rates

- **[EST]** The kernel density estimator is due to Rosenblatt (1956) and Parzen (1962) [1][2]. Rosenblatt's abstract states that all density estimates satisfying relatively mild conditions are biased, which is the origin of the bias-variance view of bandwidth choice [1].
- **[EST]** With a second-order kernel the pointwise error is $O(h^2)+O_P(\sqrt{1/(nh^d)})$ and the leading MISE terms are a squared bias of order $h^4$ and a variance of order $1/(nh^d)$ [3]. The optimal bandwidth scales as $n^{-1/(d+4)}$, giving MISE of order $n^{-4/(d+4)}$ and $L_2$ error of order $n^{-2/(d+4)}$ (F2; see C9).
- **[EST]** Stone proves that $r=(p-m)/(2p+d)$ is the optimal rate for estimating an $m$-th order derivative functional of a $p$-smooth function in $d$ dimensions at a point, and that an analogous result holds for density estimation [4]. With $p=2$, $m=0$ this is $n^{-2/(4+d)}$; the rate cannot be improved without extra smoothness [3].
- **[EST]** Normal-reference bandwidth factors (Scott $n^{-1/(d+4)}$; Silverman's multivariate suggestion $(n(d+2)/4)^{-1/(d+4)}$) scale the data covariance; the SciPy documentation warns that the estimate "works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed" and that the implementation does not support data lying in a lower-dimensional subspace [5]. A code corpus is multi-modal by construction, so these factors are a starting value only.

### E2. Directional data: the vMF distribution and the directional KDE

- **[EST]** The vMF density on $S^{d-1}$ is $c_d(\kappa)e^{\kappa\mu^\top x}$ with $c_d(\kappa)=\kappa^{d/2-1}/((2\pi)^{d/2}I_{d/2-1}(\kappa))$. It is uniform at $\kappa=0$, tends to a point mass as $\kappa\to\infty$, and is the maximum-entropy density on the sphere for a fixed mean [10]. The same constant appears in [8] and [20] in the $q=d-1$ notation.
- **[EST]** The maximum-likelihood mean direction is the normalised resultant; the concentration solves $I_{d/2}(\kappa)/I_{d/2-1}(\kappa)=\bar r$, which has no closed form. Banerjee et al. give $\hat\kappa=(\bar r d-\bar r^3)/(1-\bar r^2)$ [10]. We recomputed their Table 1: 10.1631, 60.0833, 300.084, 800.130 against true values 10, 60, 300, 800 (relative errors 1.63%, 0.14%, 0.028%, 0.016%); two Newton steps from the approximation recover the true value to at least 7 digits. Sra shows that, for the same amount of computation as the fixed-point method of Tanabe et al., truncated Newton steps are more accurate, and gives a fast routine for $I_s(x)$ [11].
- **[EST]** Cosine-similarity clustering has a generative reading. Spherical k-means is a special case of EM on a mixture of vMF distributions (see C19 for the exact conditions), and soft-moVMF yields per-point posterior memberships $p(h\mid x_i)$ [10].
- **[EST]** The directional KDE $\hat f_h(x)=\frac{c_{h,q}(L)}{n}\sum_i L((1-x^\top X_i)/h^2)$ goes back to Hall, Watson and Cabrera [6] and Bai, Rao and Zhao, who prove pointwise, uniform and $L_1$ strong consistency [7]. With the von Mises kernel it "can be viewed as a mixture of von Mises-Fisher densities" with common concentration $1/h^2$ [8].
- **[EST]** Bandwidth selection on the sphere. The AMISE-optimal bandwidth is proportional to $n^{-1/(4+q)}$; a rule of thumb with a vMF reference density ($h_{\mathrm{ROT}}$) and two mixture-based selectors (AMI, EMI) are due to Garcia-Portugues; likelihood and least-squares cross-validation (LCV, LSCV) are due to Hall et al. [6][8]. Garcia-Portugues warns that AMISE and exact MISE "may differ significantly for moderate and even large sample sizes", finds EMI the most competitive selector on average, finds LSCV better than LCV for $q\ge2$, and tests dimensions only up to $q=5$ with $n=1000$ [8]. Zhang and Chen use $h_{\mathrm{ROT}}$ with Banerjee's $\hat\kappa$ as their default and note that it "tends to be oversmoothing" when the density is not vMF [20].

### E3. Mean shift

- **[EST]** Mean shift was introduced by Fukunaga and Hostetler (1975) [13] and generalised by Cheng (1995), who shows that mean shift with kernel $K$ moves in the gradient direction of the density estimate built with its "shadow" kernel, and that the Gaussian kernel and its truncated version are the only kernels that are their own shadows, in which case the mean-shift vector is proportional to $\nabla\log q(x)$ [14]. (The publisher page of [13] was not accessible; its content is cited through [14].)
- **[EST]** Comaniciu and Meer's Theorem 1: for a kernel with convex, monotonically decreasing profile the density values along the trajectory are monotonically increasing; no step size is tuned [15].
- **[CON]** The same theorem also asserts convergence of the iterates. Yamasaki and Tanaka state that this proof "used an incorrect inequality evaluation to claim that the mode estimate sequence is a Cauchy sequence" (counterexamples by Li, Hu and Wu, 2007, cited second-hand), and that the EM argument alone is "not enough". They prove convergence to a critical point for $C^1$ subanalytic kernels, without assuming finitely many or isolated critical points, a non-degenerate Hessian, or any restriction on bandwidth or dimension, and give rate bounds; analytic kernels such as the Gaussian were covered by earlier results that theirs extend [17]. These statements concern the Euclidean algorithm.
- **[EST]** Directional mean shift was first proposed by Oba, Kato and Ishii, who introduce "mean-shift clustering based on the kernel density estimation for directional data" [18]; Kobayashi and Otsu derive the vMF version and a kernelised extension [19].
- **[EST]** Zhang and Chen write the algorithm as the fixed point $y_{s+1}=\nabla\hat f_h(y_s)/\lVert\nabla\hat f_h(y_s)\rVert$, which needs neither a step size nor the normalising constant [20]. Theorem 8: if $L$ is monotonically decreasing, differentiable and convex with $L(0)<\infty$, the density values increase monotonically. Theorem 11: under finitely many isolated modes (C1) and a non-vanishing radial gradient at the modes (C2), the iterates converge to a mode from any start within a radius $\hat r_k$. Theorem 12 proves linear convergence of Riemannian gradient ascent with a fixed step in a feasible range; the paper then argues, in the discussion that follows rather than as a separate theorem, that the adaptive step of directional mean shift falls in that range for a sufficiently small bandwidth.
- **[REC, preprint]** The EM perspective: every directional mean-shift iteration is a generalised EM step and an exact EM step for the von Mises kernel; under finitely many isolated modes and a non-increasing, differentiable, convex kernel with $0<L(0)<\infty$, the sequence converges to a local mode from almost every starting point (Theorem 4) [21]. This is an unrefereed arXiv preprint.
- **[EST]** Cost: $O(nq)$ per iteration for one query point, $O(n^2q)$ per iteration for all points, $O(n^2q\log(1/\epsilon))$ in total under linear convergence, $O(nq)$ memory [20].

### E4. What dimension does to densities

- **[EST]** Rates depend on the intrinsic sphere dimension $q$: bias $O(h^2)$; stochastic errors of order $\sqrt{1/(nh^q)}$, $\sqrt{1/(nh^{q+2})}$, $\sqrt{1/(nh^{q+4})}$ for the density, Riemannian gradient and Riemannian Hessian; estimated modes converge in Hausdorff distance at the gradient rate (Theorems 2, 4, 6) [20]. Mode estimation is therefore strictly harder than density estimation.
- **[EST]** In Zhang and Chen's simulation (four vMF components with $\nu'=10$ centred on orthogonal basis vectors, $n=1000$, $q=3,\dots,12$, rule-of-thumb bandwidth) the misclassification rate of mode clustering "gradually increase[s] to 1 (the worst case)". They attribute this to the untuned bandwidth and to the curse of dimensionality, and remark that real directional data "mostly lie on a (hyper)sphere with dimension $q\le3$" [20]. **[OURS]** A fixed $\nu'=10$ also makes the components themselves less concentrated as $q$ grows (mean resultant length $A_{q+1}(10)=0.854,\ 0.733,\ 0.634,\ 0.553$ at $q=3,6,9,12$), so this experiment mixes a harder problem with a poorer estimator.
- **[EST]** Comaniciu and Meer: kernel density estimation does "not scale well with the dimension of the space ... whenever the feature space has more than (say) six dimensions, the analysis should be approached carefully" [15].
- **[EST]** Under broad conditions the nearest and farthest neighbour distances converge as dimension grows, with effects visible at 10-15 dimensions, although the authors stress that this does not make high-dimensional search always meaningless [28]. The $k$-occurrence distribution becomes skewed and hubs emerge; hubness is tied to intrinsic dimensionality, so "dimensionality reduction may not constitute an easy mitigation"; in $\epsilon$-neighbourhood graphs "the hubness phenomenon does not occur" [29].
- **[EST]** For a unit query and spherically symmetric data, $\mathbb{E}\langle q,X\rangle=0$ and $\mathbb{E}\langle q,X\rangle^2=\lVert X\rVert^2/d$, so random directions become nearly orthogonal (Theorem 2.2); retrieval over clusters remains meaningful when "data points are clustered in (near) orthogonal subspaces" [30].
- **[EST]** Bounds that adapt to intrinsic dimension exist for the stochastic part of the error: Kim et al. define the volume dimension and show it equals the manifold dimension for a bounded density on a manifold with positive reach [33].
- **[EST]** Johnson-Lindenstrauss: any $n$ points embed into $k\ge4(\epsilon^2/2-\epsilon^3/3)^{-1}\ln n$ dimensions with squared distances preserved within $1\pm\epsilon$ [31]; the $\Omega(\epsilon^{-2}\lg n)$ dependence is optimal for nearly the full range of $\epsilon$ [32].

### E5. Adaptive bandwidths and neighbouring density-based methods

- **[EST]** The balloon estimator varies $h$ with the evaluation point; the sample-point estimator varies it with the data point [3]. With $h(x_i)$ reciprocal to the square root of the density the bias becomes proportional to $h^4$ while the variance order is unchanged, and the estimator is itself a density; balloon estimators "usually fail to integrate to one". Comaniciu, Ramesh and Meer build the variable-bandwidth mean shift on the sample-point estimator and prove its convergence [16].
- **[EST]** UMAP sets a per-point scale $\sigma_i$ by solving $\sum_{j=1}^{k}\exp(-\max(0,d(x_i,x_{i_j})-\rho_i)/\sigma_i)=\log_2 k$ [43]. This is a published precedent for calibrating a kernel width by a target neighbour mass.
- **[EST]** HDBSCAN*: core distance = distance to the $k$-th nearest neighbour (its reciprocal is a local density estimate); single linkage on the mutual reachability distance gives a hierarchy in $\lambda=1/\varepsilon$; flat clusters maximise stability [24][25]. The original algorithm is $O(N^2)$; the dual-tree Boruvka variant aims at "an average case complexity that grows approximately proportionally to $N\log N$" [25]. The library labels low-density points as noise and needs no cluster count [26].
- **[EST]** Density peaks: centres have high local density $\rho_i$ and large distance $\delta_i$ to any denser point; $d_c$ is chosen so that the average number of neighbours is 1-2% of the data; for small samples the density estimator "is unavoidably affected by large statistical errors" [27].
- **[EST]** Practice for text embeddings: BERTopic reduces with UMAP and clusters with HDBSCAN, citing distance concentration [44]; RAPTOR reduces with UMAP before Gaussian-mixture soft clustering for the same reason [45].
- **[EST]** Matryoshka representation learning trains nested embeddings whose prefixes are "at least as accurate and rich as independently trained low-dimensional representations" [42]. This applies only to encoders trained with that objective.

### E6. Fast approximate KDE

- **[EST]** Uniform random sampling needs $O(1/(\tau\epsilon^2))$ points for a $(1\pm\epsilon)$ estimate of a density that is at least $\tau$ [36][38], and this is tight up to a constant for worst-case input [38].
- **[EST]** Hashing-based estimators improve the dependence to $1/\sqrt\tau$ for scale-free kernels at the price of super-linear space [35]; Backurs, Indyk and Wagner keep the query time with linear space for the Laplacian and exponential kernels [36]; Siminelakis et al. add a diagnostic that chooses between sampling and hashing per data set [37]. See C14 for the Gaussian-kernel caveat.
- **[EST]** DEANN computes the contribution of $k$ approximate nearest neighbours exactly and samples the rest; it is unbiased "no matter the quality of the near neighbors", accepts arbitrary kernels, ships a C++ implementation with a Python interface, and has "no theoretically grounded way to choose the parameters except empirical grid search" [38].

### E7. Embedding geometry and the vMF kernel

- **[EST]** In Wang and Isola's analysis the second term of the asymptotic contrastive loss, for a finite data set, is a resubstitution entropy estimate built on a vMF-kernel KDE with $\kappa=\tau^{-1}$, where $\tau$ is the training temperature [39]. This motivates, but does not prove, $\kappa\approx1/\tau$ as a natural scale for a contrastively trained encoder.
- **[EST]** Contextual word representations of BERT, ELMo and GPT-2 are anisotropic, "occupying a narrow cone" [40]; the contrastive objective "regularizes pre-trained embeddings' anisotropic space to be more uniform" [41]. See C15.
- **[EST]** Non-parametric nearest-neighbour distance in deep feature space is effective for out-of-distribution detection and imposes no distributional assumption [48]. It is evidence that neighbour-distance scores remain informative in high-dimensional learned features where a full density estimate would not be.

### E8. Probability of relevance from scores (classical precedents)

- **[EST]** Skabar and Memon estimate class-conditional densities with vMF kernels and combine them "under Bayes' Theorem to arrive at posterior probabilities of class membership", so that one can ask "to which classes does a document belong with a minimum probability of 80%?"; they report well-calibrated outputs on Reuters-21578 [50]. Only the abstract and metadata were accessible.
- **[EST]** Per-query retrieval score distributions are well fitted by an exponential for non-relevant and a normal for relevant documents [51]; Arampatzis, Kamps and Robertson treat the rank cut-off as "a score-distributional threshold optimization problem" and repair two theoretical defects of the normal-exponential mixture with truncated models [52].

---

## Recent work (2024-2026)

- **[REC]** *Directional density-based clustering* (Saavedra-Nieves and Fernandez-Perez; online December 2024, issue dated March 2026) [23]. Clusters are connected components of highest-density regions $\{f\ge f_\tau\}$ with probability content $1-\tau$; the threshold "could be estimated as the $\tau$-quantile of the empirical distribution of $\{f_n(X_1),\dots,f_n(X_n)\}$"; the estimator is the vMF-kernel KDE. Connected components are found on a graph whose edge weights are the minimum density along the geodesic between sample points. Simulations cover the circle and the sphere only.
- **[REC, preprint]** *Kernel density estimation with polyspherical data* (Garcia-Portugues and Meilan-Vila, 2024) [9]: asymptotics, optimal bandwidths, plug-in and cross-validated selectors on products of spheres, and kernels "more efficient" than the von Mises-Fisher kernel. The application is to $(S^2)^{168}$, i.e. many low-dimensional spheres, not one high-dimensional sphere.
- **[REC]** *Mode and ridge estimation in Euclidean and directional product spaces* (Zhang and Chen, JCGS, online 2025) [22]: extends (subspace-constrained) mean shift to products of Euclidean and directional spaces with convergence results. Relevant if an embedding direction is ever combined with Euclidean features such as file position.
- **[REC]** *Convergence analysis of mean shift* (Yamasaki and Tanaka, TPAMI 2024) [17]: see E3.
- **[REC]** *Convergence and clustering analysis for mean shift with radially symmetric, positive definite kernels* (Pal, International Journal of Data Science and Analytics, 2026) [60]: for a *sufficiently large bandwidth*, convergence of the Euclidean mean-shift iterates holds in any dimension for any radially symmetric, strictly positive definite kernel; the author states that the result is partly more restrictive than [17] because of the bandwidth lower bound, but covers a kernel class that [17] does not. It also shows that with the Gaussian kernel accurate clustering at large bandwidths is generally impossible, while other kernels in the class may still cluster. Relevance for us: convergence guarantees and good clustering pull the bandwidth in opposite directions, so a convergence theorem is not evidence that the chosen $\kappa$ clusters well.
- **[REC, preprint]** *Unbiased estimation of squared concentration in the Fisher-von Mises-Langevin distribution* (Jabbar, Jiang and Popov, 2026) [61]: shows that no unbiased estimator of the vMF concentration $\kappa$ exists, reparameterises by the squared concentration ("intensity"), and gives (almost) unbiased U-statistic estimators, demonstrated on spherical word embeddings among other data. For us this is a caution that the $\hat\kappa$ fed into $h_{\mathrm{ROT}}$ (F6, F11) is biased by construction at small $n$; only the abstract was read.
- **[REC, preprint]** *Kernel smoothing on manifolds* (Bae and Polonik, 2026) [34]: finite-sample bounds for kernel smoothing and its first and second derivatives on a compact $d$-dimensional manifold using ambient Euclidean distances; after Theorem 3.6, "the upper bound depends on the dimension $d$ of $\mathcal{M}$, rather than the dimension of the ambient space".
- **[REC]** *RAPTOR* (ICLR 2024) [45]: recursive embed-cluster-summarise tree; UMAP then GMM soft clustering with BIC model selection; the authors concede that "the Gaussian assumption in GMMs may not perfectly align with the nature of text data". A vMF mixture is the spherical counterpart (**[OURS]** observation).
- **[REC]** *Contrastive Mean-Shift learning* (CVPR 2024) [47]: mean-shift steps inside contrastive training; radius neighbourhoods are replaced by $k$-NN because neighbour counts within a fixed radius vary strongly. Earlier, Kong and Fowlkes (CVPR 2018) ran a recurrent mean-shift module with the kernel $K\propto\exp(\delta x^\top x_i)$ on unit-norm pixel embeddings and set $1/\delta=(1-\alpha)/3$ from the training margin $\alpha$ [46]. Both are computer-vision results.
- **[REC, preprint]** *GLASS* (2026) [49]: anomaly scoring by vMF kernel density estimators in graph and text embedding spaces; the formulation "recovers angular k-nearest-neighbor scoring as a high-concentration limiting case", which matches our nearest-neighbour collapse analysis (F28).
- **[REC]** *Cosine Adapter* (CIKM 2024) [53]: cosine scores "should not be compared across different queries"; a query-dependent monotone map produces calibrated scores to which a global threshold is applied; evaluated on MS MARCO and Walmart product search.
- **[REC, preprint]** *CONFLARE* (2024) [58]: conformal prediction on a calibration set of questions yields a similarity cut-off such that the answer-bearing chunk is retrieved with probability $1-\alpha$.
- **[REC, preprint]** *Cluster-based Adaptive Retrieval (CAR)* (2025) [54]: the cut-off is the transition point found by clustering the ordered query-document distances; eight clustering back-ends give similar scores (0.58-0.63 on the Coinbase corpus, 0.30-0.34 on MultiHop-RAG).
- **[REC]** *Adaptive-k* (EMNLP 2025) [55]: single-pass selection of the number of passages "based on the distribution of the similarity scores", no tuning and no extra LLM calls.
- **[REC]** *Tail-Aware Adaptive-k* (ECML PKDD 2026) [56]: knee detection followed by an extreme-value goodness-of-fit test in a local window; the cut-off is justified "under mild monotone likelihood ratio assumptions".
- **[REC]** *Query-Clustering Adaptive Retrieval (QCAR)* (CIKM 2026) [57]: retrieval depth chosen per query cluster before retrieval, with a mean-plus-variance rule targeting about 95% coverage.
- **[REC, preprint]** Ricco, Cima and Di Pietro (2025) [59] fit separate Gaussian KDEs to the distributions of embedding distances for hallucinated and non-hallucinated responses and compare log-likelihoods, reaching 66% accuracy in one configuration. It is a precedent for class-conditional densities of scalar distances, with a modest effect size.

Search result (negative). arXiv metadata searches for `"kernel density" "retrieval-augmented"`, `"kernel density" "dense retrieval"`, `"kernel density" "source code"` (8 hits, none on code embeddings), `"mean shift" "source code"` (5 hits, none on code embeddings) and `"directional mean shift"` (only [20][21]) returned nothing that applies vMF-KDE or directional mean shift to source-code functions or to RAG retrieval depth.

---

## Formulas

Each formula lists its source and how it was checked. "TeX" means compared with the arXiv TeX source of the paper; "PDF text" means compared with text extracted from the PDF; "numeric" means evaluated with mpmath or NumPy.

### F1. Euclidean kernel density estimator

$$
\hat p_n(x)=\frac{1}{n h^{d}}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h}\right)
$$

Source: [3] eq. (1); origin [1][2]. Check: TeX.

### F2. MISE expansion and optimal bandwidth (Euclidean)

$$
\mathrm{MISE}(h)=\frac{h^{4}}{4}\,\sigma_K^{4}\int\lvert\nabla^{2}p(x)\rvert^{2}dx+\frac{\mu_K}{n h^{d}}+o(h^{4})+o\!\left(\frac{1}{n h^{d}}\right)
$$

$$
h_{\mathrm{opt}}=\left(\frac{d\,\mu_K}{\sigma_K^{4}\int\lvert\nabla^{2}p(x)\rvert^{2}dx}\cdot\frac{1}{n}\right)^{\frac{1}{d+4}},\qquad
\mathrm{MISE}(h_{\mathrm{opt}})=O\!\left(n^{-\frac{4}{d+4}}\right)
$$

with $\mu_K=\int K^2$, $\nabla^2$ the Laplacian and $\sigma_K^2=\int x_1^2K(x)\,dx$ the per-coordinate second moment of an isotropic kernel.
Source: expansion from [3] eq. (4). The constant in $h_{\mathrm{opt}}$ is **re-derived by us** (set $h^3\sigma_K^4R - d\mu_K/(nh^{d+1})=0$); the source prints 4 in place of $d$. Check: with the Gaussian kernel and a $N(0,\sigma^2 I_d)$ reference our constant gives $h=\sigma\,(4/((d+2)n))^{1/(d+4)}$, which is Silverman's factor in F4 and the classical $1.06\,\sigma n^{-1/5}$ for $d=1$.

### F3. Optimal nonparametric rate

$$
r=\frac{p-m}{2p+d},\qquad \text{error}\asymp n^{-r};\qquad p=2,\ m=0\ \Rightarrow\ r=\frac{2}{4+d}
$$

Source: [4] abstract. The substitution is ours. Numeric ($n=2000$): $n^{-2/(d+4)}=0.282$ ($d=8$), $0.656$ ($d=32$), $0.962$ ($d=384$). Halving the $L_2$ error multiplies the required $n$ by $2^{(d+4)/2}$: $2^{6}$, $2^{18}$, $2^{194}$.

### F4. Scott and Silverman factors; effective number of weighted points

$$
f_{\mathrm{Scott}}=n^{-\frac{1}{d+4}},\qquad f_{\mathrm{Silverman}}=\left(\frac{n(d+2)}{4}\right)^{-\frac{1}{d+4}},\qquad \Sigma_{\mathrm{kernel}}=f^{2}\,\widehat{\mathrm{Cov}}(X),\qquad n_{\mathrm{eff}}=\frac{\left(\sum_i w_i\right)^{2}}{\sum_i w_i^{2}}
$$

Source: [5] (SciPy v1.18.0 manual, which cites Scott 1992 and Silverman 1986; those books were not opened). Check: page text.

### F5. von Mises-Fisher density

$$
f_{\mathrm{vMF}}(x;\mu,\kappa)=c_d(\kappa)\,e^{\kappa\mu^{\top}x},\qquad c_d(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2}\,I_{d/2-1}(\kappa)},\qquad x,\mu\in S^{d-1},\ \kappa\ge0
$$

Equivalently $C_q(\kappa)=\kappa^{(q-1)/2}/((2\pi)^{(q+1)/2}I_{(q-1)/2}(\kappa))$ with $q=d-1$.
Source: [10] eqs. (2.1)-(2.2); [8] eq. (1). Check: TeX and numeric. $c_3(\kappa)=\kappa/(4\pi\sinh\kappa)$ to 12 digits; the density integrates to 1.000000000000 for $(d,\kappa)=(8,10),(32,30),(384,100)$; $\log c_{384}(100)=582.6318$.

### F6. vMF maximum likelihood

$$
\hat\mu=\frac{r}{\lVert r\rVert},\quad r=\sum_{i=1}^{n}x_i;\qquad A_d(\hat\kappa)=\frac{I_{d/2}(\hat\kappa)}{I_{d/2-1}(\hat\kappa)}=\bar r=\frac{\lVert r\rVert}{n};\qquad \hat\kappa\approx\frac{\bar r\,d-\bar r^{3}}{1-\bar r^{2}}
$$

$$
A_d'(\kappa)=1-A_d(\kappa)^{2}-\frac{d-1}{\kappa}A_d(\kappa)\qquad\text{(Newton refinement)}
$$

Source: [10] eqs. (2.4), (2.5), (4.4), footnote 3; [11]. Check: numeric (Table 1 reproduced).

### F7. EM for a mixture of vMF distributions (soft-moVMF)

$$
p(h\mid x_i,\Theta)=\frac{\alpha_h f_h(x_i\mid\theta_h)}{\sum_{l=1}^{k}\alpha_l f_l(x_i\mid\theta_l)},\qquad
\alpha_h=\frac1n\sum_{i=1}^{n}p(h\mid x_i,\Theta),\qquad
r_h=\sum_{i=1}^{n}x_i\,p(h\mid x_i,\Theta)
$$

$$
\hat\mu_h=\frac{r_h}{\lVert r_h\rVert},\qquad
\frac{I_{d/2}(\hat\kappa_h)}{I_{d/2-1}(\hat\kappa_h)}=\bar r_h=\frac{\lVert r_h\rVert}{\sum_{i}p(h\mid x_i,\Theta)},\qquad
\hat\kappa_h\approx\frac{\bar r_h d-\bar r_h^{3}}{1-\bar r_h^{2}}
$$

Source: [10] eqs. (3.3)-(3.7) and (4.4). Check: PDF text.

### F8. Directional kernel density estimator

$$
\hat f_h(x)=\frac{c_{h,q}(L)}{n}\sum_{i=1}^{n}L\!\left(\frac{1-x^{\top}X_i}{h^{2}}\right),\qquad
c_{h,q}(L)^{-1}=\int_{\Omega_q}L\!\left(\frac{1-x^{\top}y}{h^{2}}\right)\omega_q(dx)=O(h^{q})
$$

Source: [8] eq. (2), attributed there to [7]; see also [6][20]. Check: TeX.

### F9. vMF-kernel KDE and its Gaussian form

$$
\hat f_h(x)=\frac1n\sum_{i=1}^{n}f_{\mathrm{vMF}}\!\left(x;X_i,\tfrac{1}{h^{2}}\right)
=\frac{1}{n\,(2\pi)^{\frac{q+1}{2}}\,I_{\frac{q-1}{2}}(h^{-2})\,h^{q-1}}\sum_{i=1}^{n}\exp\!\left(\frac{x^{\top}X_i}{h^{2}}\right)
$$

$$
\exp\!\left(-\frac{\lVert x-y\rVert^{2}}{2\sigma^{2}}\right)=e^{-1/\sigma^{2}}\exp\!\left(\frac{x^{\top}y}{\sigma^{2}}\right)\quad(\lVert x\rVert=\lVert y\rVert=1),\qquad \kappa=\frac{1}{\sigma^{2}}=\frac{1}{h^{2}}
$$

Source: first line [8] Section 2 and [20]; the identity is elementary and is the basis of $\tilde f_h$ in [20] and of $G_t(u,v)=e^{-t\lVert u-v\rVert^2}=e^{2tu^\top v-2t}$ in [39] (so $t=\kappa/2$). Check: TeX and numeric.

### F10. Directional MISE and AMISE-optimal bandwidth

$$
\mathrm{MISE}(h)=b_q(L)^{2}R(\Psi(f,\cdot))\,h^{4}+\frac{c_{h,q}(L)}{n}\,d_q(L)+o\!\left(h^{4}+(nh^{q})^{-1}\right)
$$

$$
h_{\mathrm{AMISE}}=\left[\frac{q\,d_q(L)}{4\,b_q(L)^{2}\lambda_q(L)R(\Psi(f,\cdot))\,n}\right]^{\frac{1}{4+q}}
$$

$$
b_q(L)=\frac{\int_0^\infty L(r)r^{q/2}dr}{\int_0^\infty L(r)r^{q/2-1}dr},\quad
d_q(L)=\frac{\int_0^\infty L^2(r)r^{q/2-1}dr}{\int_0^\infty L(r)r^{q/2-1}dr},\quad
\lambda_q(L)=2^{q/2-1}\omega_{q-1}\int_0^\infty L(r)r^{q/2-1}dr
$$

$$
\Psi(f,x)=-x^{\top}\nabla f(x)+q^{-1}\left(\nabla^{2}f(x)-x^{\top}\mathcal{H}f(x)\,x\right),\qquad R(\Psi)=\int_{\Omega_q}\Psi(f,x)^{2}\,\omega_q(dx)
$$

Source: [8] Proposition 1 and Corollary 1. Check: TeX.
**[OURS]** For the von Mises kernel: $b_q=q/2$, $d_q=2^{-q/2}$, $\lambda_q=(2\pi)^{q/2}$, hence $h_{\mathrm{AMISE}}=[\,q\,(4\pi)^{q/2}R(\Psi)\,n\,]^{-1/(4+q)}$. Substituting the vMF curvature term of F11 reproduces $h_{\mathrm{ROT}}$ exactly.

### F11. Rule-of-thumb bandwidth (von Mises kernel, vMF reference)

$$
R\big(\Psi(f_{\mathrm{vMF}})\big)=\frac{\kappa^{\frac{q+1}{2}}\left[2q\,I_{\frac{q+1}{2}}(2\kappa)+(2+q)\,\kappa\,I_{\frac{q+3}{2}}(2\kappa)\right]}{2^{q+2}\pi^{\frac{q+1}{2}}\,I_{\frac{q-1}{2}}(\kappa)^{2}\,q}
$$

$$
h_{\mathrm{ROT}}=\left[\frac{4\pi^{1/2}\,I_{\frac{q-1}{2}}(\hat\kappa)^{2}}{\hat\kappa^{\frac{q+1}{2}}\left[2q\,I_{\frac{q+1}{2}}(2\hat\kappa)+(2+q)\,\hat\kappa\,I_{\frac{q+3}{2}}(2\hat\kappa)\right]n}\right]^{\frac{1}{4+q}},\qquad \kappa_{\mathrm{KDE}}=h_{\mathrm{ROT}}^{-2}
$$

Source: [8] Proposition 2, eq. (6); [20] eq. (31), which uses $\hat\nu=\bar R(q+1-\bar R^2)/(1-\bar R^2)$. Check: TeX; numeric. The curvature formula agrees with direct quadrature of the Laplace-Beltrami expression to 12 digits for $(q,\kappa)=(1,2),(2,3),(5,4),(15,20)$; the $q=2$ closed form $8\sinh^2\hat\kappa/(\hat\kappa[(1+4\hat\kappa^2)\sinh2\hat\kappa-2\hat\kappa\cosh2\hat\kappa]n)$ agrees to 12 digits.

### F12. Cross-validation selectors (directional)

$$
h_{\mathrm{LCV}}=\arg\max_{h>0}\sum_{i=1}^{n}\log\hat f_h^{-i}(X_i),\qquad
h_{\mathrm{LSCV}}=\arg\max_{h>0}\left[\frac{2}{n}\sum_{i=1}^{n}\hat f_h^{-i}(X_i)-\int_{\Omega_q}\hat f_h(x)^{2}\,\omega_q(dx)\right]
$$

Source: [8] Section 2.1, attributed to [6]. $\hat f_h^{-i}$ is the leave-one-out estimator. Check: TeX.

### F13. Euclidean mean-shift iteration

$$
y_{j+1}=\frac{\sum_{i=1}^{n}x_i\,g\!\left(\left\lVert\frac{y_j-x_i}{h}\right\rVert^{2}\right)}{\sum_{i=1}^{n}g\!\left(\left\lVert\frac{y_j-x_i}{h}\right\rVert^{2}\right)},\qquad g=-k'
$$

Source: [15] eq. (20); gradient interpretation [14]. **[OURS]** For $K(x)=e^{-\beta\lVert x\rVert^2}$ direct differentiation gives $m(x)-x=\frac{1}{2\beta}\nabla\log q(x)$; the constant in Cheng's eq. (21) was not legible in the scanned copy, so cite [14] for proportionality only.

### F14. Directional mean-shift update

$$
y_{s+1}=-\frac{\sum_{i=1}^{n}X_i\,L'\!\left(\frac{1-y_s^{\top}X_i}{h^{2}}\right)}{\left\lVert\sum_{i=1}^{n}X_i\,L'\!\left(\frac{1-y_s^{\top}X_i}{h^{2}}\right)\right\rVert_2}
=\frac{\nabla\hat f_h(y_s)}{\lVert\nabla\hat f_h(y_s)\rVert_2};
\qquad
L(r)=e^{-r}:\quad y_{s+1}=\frac{\sum_{i}X_i\,e^{\kappa\,y_s^{\top}X_i}}{\left\lVert\sum_{i}X_i\,e^{\kappa\,y_s^{\top}X_i}\right\rVert_2}
$$

Stopping rule $1-y_{s+1}^{\top}y_s\le\epsilon$ (the paper uses $\epsilon=10^{-7}$).
Source: [20] Algorithm 1, eqs. (20), (23); earlier derivations [18][19]. The vMF special case is a one-line substitution ($L'=-L$; the factor $e^{-\kappa}$ cancels). Check: TeX; numeric (monotone ascent observed in every run).

### F15. Directional mean shift as Riemannian gradient ascent

$$
y_{s+1}=\mathrm{Exp}_{y_s}\!\big(\hat\eta_s\,\mathrm{grad}\,\hat f_h(y_s)\big),\qquad
\hat\eta_s=\frac{\hat\theta_s}{\sin\hat\theta_s\;\lVert\nabla\hat f_h(y_s)\rVert_2},\qquad
\hat\theta_s=\angle\big(\nabla\hat f_h(y_s),\,y_s\big)
$$

$$
d(y_s,m_k)\le\Upsilon^{s}\,d(y_0,m_k),\qquad \Upsilon=\sqrt{1-\frac{\eta\lambda_*}{2}},\qquad
\eta\le\min\left\{\frac{2}{\lambda_*},\ \frac{1}{(q+1)\,C_3\,\zeta(1,r_0)}\right\},\qquad \zeta(1,c)=\frac{c}{\tanh c}
$$

Source: [20] Section 5.2 and Theorem 12(a); $\lambda_*$ bounds the Hessian eigenvalues at modes (M1), $C_3$ bounds derivatives up to order three. For the sample version (Theorem 12(b)) the bound gains $O(h^2)+O_P(\sqrt{\lvert\log h\rvert/(nh^{q+2})})$. Check: TeX.

### F16. Convergence rates of the directional KDE

$$
\hat f_h-f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q}}}\right),\qquad
\mathrm{grad}\,\hat f_h-\mathrm{grad}\,f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+2}}}\right)
$$

$$
\mathcal{H}\hat f_h-\mathcal{H}f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+4}}}\right),\qquad
\mathrm{Haus}(\mathcal M,\hat{\mathcal M}_n)=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+2}}}\right)
$$

Uniform versions replace $1$ by $\lvert\log h\rvert$ in the numerators; $\mathbb{P}(\hat K_n\ne K)\le B_3e^{-A_3nh^{q+4}}$.
Source: [20] Section 4.2, Theorems 2, 4, 6. Check: TeX (square roots confirmed).

### F17. Complexity of directional mean shift

$$
T_{\text{one query, one iteration}}=O(nq),\qquad T_{\text{all points}}=O\!\left(n^{2}q\log(1/\epsilon)\right),\qquad S=O(nq)
$$

Source: [20] Section 5.3. Check: TeX.

### F18. Balloon and sample-point estimators

$$
\hat f_{\mathrm{balloon}}(x)=\frac{1}{n\,h(x)^{d}}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h(x)}\right),\qquad
\hat f_{\mathrm{sp}}(x)=\frac{1}{n}\sum_{i=1}^{n}\frac{1}{h_i^{d}}K\!\left(\frac{x-X_i}{h_i}\right),\qquad
h_i=h_0\left[\frac{\lambda}{\tilde f(X_i)}\right]^{1/2}
$$

Source: [3] (remark on adaptive smoothing); [16] Section 1.2. Check: TeX and PDF text.
**[OURS]** Spherical analogue: $\hat f_{\mathrm{sp}}(x)=\frac1n\sum_i c_d(\kappa_i)\,e^{\kappa_i x^\top X_i}$ with $\kappa_i=\kappa_0\tilde f(X_i)/\lambda$. The normalisers $c_d(\kappa_i)$ now differ between points and no longer cancel; they must be evaluated in log space.

### F19. UMAP smooth nearest-neighbour calibration

$$
\rho_i=\min\{d(x_i,x_{i_j})\mid 1\le j\le k,\ d(x_i,x_{i_j})>0\},\qquad
\sum_{j=1}^{k}\exp\!\left(\frac{-\max(0,\ d(x_i,x_{i_j})-\rho_i)}{\sigma_i}\right)=\log_2 k
$$

Source: [43] Section 3.1. Check: TeX.

### F20. HDBSCAN* mutual reachability and stability

$$
d_{\mathrm{mreach}}(X_i,X_j)=\begin{cases}\max\{\kappa(X_i),\kappa(X_j),d(X_i,X_j)\}, & X_i\ne X_j\\ 0, & X_i=X_j\end{cases}
\qquad
\sigma(C_i)=\sum_{X_j\in C_i}\left(\lambda_{\max,C_i}(X_j)-\lambda_{\min,C_i}(X_j)\right),\quad \lambda=\frac{1}{\varepsilon}
$$

Here $\kappa(X)$ is the core distance (distance to the $k$-th nearest neighbour), not a concentration.
Source: [25] Section 2; [24]. Check: TeX. The source prints $d(X_j,X_j)$ inside the maximum, an evident typographical error for $d(X_i,X_j)$.

### F21. Density-peaks quantities

$$
\rho_i=\sum_{j}\chi(d_{ij}-d_c),\quad \chi(x)=\mathbb{1}[x<0];\qquad
\delta_i=\begin{cases}\min_{j:\rho_j>\rho_i}d_{ij}, & \text{if some }\rho_j>\rho_i\\ \max_j d_{ij}, & \text{otherwise}\end{cases};\qquad \gamma_i=\rho_i\delta_i
$$

Source: [27] eqs. (1)-(2); the convention for the densest point is stated in the main text and $\gamma_i$ in a figure caption. Check: PDF text.

### F22. Directional highest-density region

$$
L_f(\tau)=\{x\in S^{d-1}: f(x)\ge f_\tau\},\qquad f_\tau=\sup\{t:\ \mathbb{P}(X\in\{f\ge t\})\ge1-\tau\}
$$

$$
\hat L_f(\tau)=\{x\in S^{d-1}:\ f_n(x)\ge\hat f_\tau\},\qquad \hat f_\tau=\tau\text{-quantile of }\{f_n(X_i)\}_{i=1}^{n},\qquad f_n(x)=\frac1n\sum_{i=1}^{n}K_{vM}(x;X_i;1/h^{2})
$$

Source: [23] eqs. (4)-(6) (arXiv numbering). Check: TeX.

### F23. Johnson-Lindenstrauss bound

$$
k\ \ge\ 4\left(\frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{3}\right)^{-1}\ln n\ \Longrightarrow\ \exists f:\mathbb{R}^{d}\to\mathbb{R}^{k}:\ (1-\epsilon)\lVert u-v\rVert^{2}\le\lVert f(u)-f(v)\rVert^{2}\le(1+\epsilon)\lVert u-v\rVert^{2}\quad\forall u,v\in V
$$

Source: [31] Theorem 2.1; lower bound $m=\Omega(\epsilon^{-2}\lg n)$ in [32]. Numeric ($n=2000$): bound 364.84, 844.54, 1754.05, 6515.06, hence $k\ge365,\ 845,\ 1755,\ 6516$ for $\epsilon=0.5,0.3,0.2,0.1$.

### F24. DEANN estimator

$$
\widetilde{\mathrm{KDE}}_X(y)=\frac{k}{n}\cdot\frac{1}{k}\sum_{x\in X_1}K_h(x,y)+\frac{n-k}{n}\cdot\frac{1}{m}\sum_{x\in S}K_h(x,y),\qquad X_1=\mathrm{ANN}_X(y),\quad S\subset X\setminus X_1\ \text{uniform},\ \lvert S\rvert=m
$$

Source: [38] Algorithm 1, Lemma 3, Corollary 4. Check: TeX and PDF numbering.

### F25. Cost of approximate KDE

$$
m_{\mathrm{RS}}=O\!\left(\frac{1}{\epsilon^{2}\tau}\right),\qquad
T_{\mathrm{HBE}}=O\!\left(M^{3}\,\frac{1}{\epsilon^{2}}\,\frac{1}{\sqrt{\mu}}\,\log\frac1\delta\;T\right),\qquad
\text{stored hashes: }O\!\left(\frac{1}{\tau^{3/2}\epsilon^{4}}\right)\ \text{[35]}\quad\text{vs.}\quad O\!\left(\frac{1}{\tau\,\epsilon^{2}}\right)\ \text{[36]}
$$

$\tau$ is a lower bound on the density, $\mu\ge\tau$ the density at the query, $(M,T)$ the scale-free constant and evaluation cost of the hash family: $M=\sqrt e$, $T=O(dR^2)$ for the exponential kernel; $M,T=e^{O(R^{4/3}\log\log n)}$ for the Gaussian kernel.
Source: [38] Lemma 2; [36] Table 1; [35] Theorem 1.7 and its table of scale-free estimators. Check: TeX and PDF text.

### F26. Gaussian potential, uniformity loss and the vMF-KDE link

$$
G_t(u,v)=e^{-t\lVert u-v\rVert_2^{2}}=e^{2t\,u^{\top}v-2t},\qquad
\mathcal L_{\mathrm{uniform}}(f;t)=\log\mathbb{E}_{x,y\sim p_{\mathrm{data}}}\!\left[G_t(f(x),f(y))\right]
$$

$$
\frac{1}{N}\sum_{i=1}^{N}\log\!\Big(\frac{1}{N}\sum_{j=1}^{N}e^{f(x_i)^{\top}f(x_j)/\tau}\Big)=\frac{1}{N}\sum_{i=1}^{N}\log\hat p_{\mathrm{vMF\text{-}KDE}}(f(x_i))+\log Z_{\mathrm{vMF}}
$$

Source: [39], definition of the uniformity metric and the paragraph "Relation with feature distribution entropy estimation"; $\hat p_{\mathrm{vMF\text{-}KDE}}$ uses $\kappa=\tau^{-1}$ and $Z_{\mathrm{vMF}}$ is the vMF normalisation constant for that $\kappa$. Check: TeX.

### F27. Concentration of inner products

$$
\mathbb{E}\langle q,X\rangle=0,\qquad \mathrm{Var}\langle q,X\rangle=\mathbb{E}\langle q,X\rangle^{2}=\frac{\lVert X\rVert_2^{2}}{d},\qquad \lim_{d\to\infty}\mathbb{P}\big(\langle q,X\rangle>\epsilon\big)=0
$$

Source: [30] Theorem 2.2 and proof. Numeric: $d\cdot\mathrm{Var}=0.9988$ ($d=64$) and $1.0004$ ($d=384$) over $4\times10^5$ pairs.

### F28. [OURS] Effective number of contributing points and self-share

$$
w_i=e^{\kappa\,x^{\top}X_i},\qquad n_{\mathrm{eff}}(x)=\frac{(\sum_i w_i)^{2}}{\sum_i w_i^{2}}\ \approx\ n\,e^{-\kappa^{2}/d}\quad(\kappa^{2}\lesssim d)
$$

$$
\rho_{\mathrm{self}}(X_j)=\frac{e^{\kappa}}{e^{\kappa}+\sum_{i\ne j}e^{\kappa X_j^{\top}X_i}}\ \approx\ \left[1+(n-1)\,e^{\kappa^{2}/(2d)-\kappa}\right]^{-1},\qquad
\rho_{\mathrm{self}}=\tfrac12\iff\kappa-\frac{\kappa^{2}}{2d}=\ln(n-1)
$$

Derivation (ours): under the isotropic null $x^\top X_i$ has mean 0 and variance $1/d$ [30] and is approximately normal, so $\log w_i\approx N(0,\kappa^2/d)$, $\mathbb{E}w=e^{\kappa^2/(2d)}$, $\mathbb{E}w^2=e^{2\kappa^2/d}$. The definition of $n_{\mathrm{eff}}$ is the one in [5].
Simulation ($d=384$, $n=2000$, 20 corpora by 50 queries): $n_{\mathrm{eff}}=1560,\ 754,\ 265,\ 93,\ 20,\ 8$ at $\kappa/\sqrt d=0.5,1,1.5,2,3,4$, against the approximation $1558,\ 736,\ 211,\ 37,\ 0.2,\ 0$. The approximation is accurate for $\kappa\le\sqrt d$ and pessimistic beyond, where finite-sample maxima dominate. Self-share at a data point: $0.004,\ 0.067,\ 0.579,\ 0.906,\ 1.000$ at $\kappa=2,5,8,10,20$; the closed form gives $0.067,\ 0.578,\ 0.906$ at $\kappa=5,8,10$. Directional mean shift started from every point found 15 modes at $\kappa=5$ and 2000 modes (every point its own mode) at $\kappa=10$.
For clustered data the same argument with $m$ neighbours at similarity $s$ gives the working upper limit $\kappa\lesssim\ln m/(1-s)$. These are null-model references, not selection rules.

### F29. [OURS] Posterior relevance probability and the 25 percent rule

$$
P(\mathrm{rel}\mid x)=\frac{\pi\,\hat f_R(x)}{\pi\,\hat f_R(x)+(1-\pi)\,\hat f_B(x)}=\operatorname{logistic}\!\Big(\log\frac{\pi}{1-\pi}+\log\hat f_R(x)-\log\hat f_B(x)\Big)
$$

$$
P(\mathrm{rel}\mid x)\ge p^{*}\iff\frac{\hat f_R(x)}{\hat f_B(x)}\ge\frac{p^{*}}{1-p^{*}}\cdot\frac{1-\pi}{\pi};\qquad p^{*}=\tfrac14:\quad\frac{\hat f_R(x)}{\hat f_B(x)}\ge\frac{1-\pi}{3\pi}
$$

With a single query-centred kernel $\hat f_R(x)=c_d(\kappa_R)e^{\kappa_R q^\top x}$:

$$
\operatorname{logit}P(\mathrm{rel}\mid x)=a+\kappa_R\,q^{\top}x-\beta\log\hat f_B^{-x}(x),\qquad a=\log\frac{\pi}{1-\pi}+\log c_d(\kappa_R),\quad \beta=1
$$

$$
P\ge\tfrac14\iff q^{\top}x\ \ge\ \frac{\beta\log\hat f_B^{-x}(x)-a-\ln3}{\kappa_R}
$$

$\hat f_B^{-x}$ is the leave-one-out corpus vMF-KDE at $x$. Derivation: Bayes' rule; $4\pi f_R\ge\pi f_R+(1-\pi)f_B\iff3\pi f_R\ge(1-\pi)f_B$. Precedents for the construction: [50] (vMF kernels and Bayes' theorem), [51][52] (score-distribution posteriors), [53] (query-dependent calibration before a global threshold). The output is a probability only after $a$, $\kappa_R$ and (optionally) $\beta$ are fitted on labelled query-function pairs; without calibration it is a score.

### F30. [OURS] Path product and depth bound

$$
P_{\mathrm{path}}(x_m)=\prod_{j=1}^{m}P(\mathrm{rel}\mid x_j,\ x_{j-1}\ \text{relevant}),\qquad
P(\cdot)\le p_{\max}<1\ \Rightarrow\ P_{\mathrm{path}}\ge p^{*}\ \text{only if}\ m\le\frac{\ln(1/p^{*})}{\ln(1/p_{\max})}
$$

For $p^*=1/4$: $m\le13$ at $p_{\max}=0.9$, $m\le6$ at $0.8$, $m\le2$ at $0.5$. The product is a chain rule only under the modelling assumption that a function reached through an irrelevant predecessor is not counted as relevant. Termination of the trace is guaranteed by the visited set on a finite corpus or by the depth bound, not by the factors being below one.

---

## What is novel in our proposal

Everything in this section is **[OURS]**: a proposal assembled from published components. None of it is an established result, and the paper must say so.

1. **Application.** We found no published work that applies vMF-KDE, directional mean shift or highest-density-region clustering to embeddings of source-code functions. The closest precedents cluster text chunks with GMMs after UMAP [45], topics with HDBSCAN after UMAP [44], pixels with vMF mean shift [46], and documents with vMF kernels and Bayes' theorem [50].
2. **Trace expansion with a probability floor.** We found no retrieval procedure that expands a candidate set while a density-derived posterior stays above a fixed floor. Published adaptive-depth methods cut a ranked list using the one-dimensional score sequence [54][55][56], a learned query-dependent calibration [53], a per-query-cluster depth [57] or a conformal threshold [58].
3. **Background correction.** Dividing the query-centred kernel by the corpus density (F29) turns the 25 percent rule into a cosine threshold that rises in dense regions. This is a density-ratio form of hub correction [29]; we have not seen it stated for RAG.
4. **Diagnostics.** The closed forms for $n_{\mathrm{eff}}$ and the self-share (F28) and the resulting admissible window for $\kappa$ are our derivations, checked by simulation only.
5. **What is not novel.** The vMF-KDE, its bandwidth theory, directional mean shift and its convergence, the Bayes posterior from class-conditional kernel densities, and score-distribution thresholding are all published. The paper should present the method as a principled heuristic validated on one corpus (a fork of mimalloc).

---

## Risks and failure modes

1. **Rate of convergence.** With $n\approx2000$ the attainable accuracy of a density estimate degrades quickly with dimension: the optimal-rate factor is 0.28 at $d=8$, 0.66 at $d=32$ and 0.96 at $d=384$ (F3). Mode locations are harder still (F16).
2. **Self-dominance.** At data points the point's own kernel dominates once $\kappa\gtrsim\ln m/(1-s)$ (F28). Mean shift started from data points then returns every point as its own mode, and leave-in densities used for thresholds are inflated. Always use leave-one-out densities at sample points.
3. **Nearest-neighbour collapse.** For large $\kappa$ the KDE is a soft nearest-neighbour score [49] (F28). It still ranks, but it no longer estimates a density.
4. **Oversmoothing.** $h_{\mathrm{ROT}}$ assumes a single vMF population and oversmooths multimodal data [8][20]. Normal-reference rules have the same defect [5].
5. **Dimension reduction is not free.** JL gives no guarantee at 8-32 dimensions (F23). PCA can remove the direction that separates two small families. Hubness may persist after reduction [29]. Matryoshka prefixes are valid only for encoders trained that way [42].
6. **Exact and near duplicates.** A repository contains copied helpers, macros expanded per type and header/implementation pairs. Duplicates create arbitrarily sharp spurious modes and violate the isolated-modes condition (C1) of [20][21]. De-duplicate (cosine above about 0.999) before estimating densities and re-attach afterwards.
7. **Theory assumes an i.i.d. sample from a smooth density.** A repository is a small, fixed, structured population. Consistency theorems describe a limit that the corpus never approaches.
8. **Disputed convergence proofs.** Do not cite Comaniciu and Meer's Theorem 1 for convergence of the iterates [17]. For directional mean shift cite [20] (local) and, flagged as a preprint, [21] (almost everywhere). Convergence can be to a saddle point; perturb and re-run to test.
9. **Modality gap.** Natural-language queries and C functions need not occupy the same region of the sphere, so $q^\top x$ may live in a narrow band. $\kappa_R$ must be calibrated separately from the corpus $\kappa_B$, and then the normalisers do not cancel (absorbed in $a$ in F29).
10. **Query drift.** Re-centring $\hat f_R$ on accepted functions can pull the trace towards dense hub regions. The path product (F30) limits depth but not direction.
11. **Calibration data.** A posterior is only as good as its calibration set. Published calibrated cut-offs rely on large supervised sets [53] or a dedicated calibration set [58]. With a few dozen labelled queries report confidence intervals and call the output a score if calibration is poor.
12. **Numerics.** Direct Bessel evaluation underflows or overflows in high dimension [12][10]. $\log c_{384}(100)=+582.6$ cannot be exponentiated in double precision.
13. **Approximate KDE guarantees are kernel-specific.** HBE constants for the Gaussian kernel grow like $e^{O(R^{4/3}\log\log n)}$ [35]. **[OURS]** Writing the vMF kernel as $e^{-\lVert x'-y'\rVert^2}$ requires $x'=\sqrt{\kappa/2}\,x$, so the scaled data radius is $\sqrt{\kappa/2}$, which is about 3 at $\kappa=20$ and not small.
14. **Evidence gap.** No published experiment validates directional mode clustering between $q=12$ and $q=384$ [8][20][23]. Our synthetic experiment (D2) is favourable to PCA by construction.

---

## Design recommendations

These recommendations are **[OURS]**. Each cites the published result it relies on; numerical thresholds (for example $n_{\mathrm{eff}}\ge10$, cosine 0.99 for merging) are starting values to be tuned on the mimalloc fork, not published constants.

D1. **One kernel, one code path.** L2-normalise every function embedding. Use the kernel weight $w=\exp(\kappa(s-1))$ with $s$ the cosine similarity; it serves the KDE, mean shift and the relevance ratio (F9, F14, F29). Accumulate in log space with log-sum-exp and double-precision accumulators.

D2. **Working dimension.** Compute PCA on the mean-centred corpus embeddings, keep $q\in\{8,16,32\}$ components, re-normalise, and run all density computations there. Report an ablation over $q$ that includes the raw dimension. In our synthetic test (8 clusters, within-cluster cosine 0.67, $d=384$, $n=2000$) raw-space mean shift recovered the clusters for $\kappa\in\{5,10,20\}$ and failed at $\kappa=2$ (one mode) and $\kappa\ge40$ (2000 modes); after PCA to 8 or 16 dimensions it succeeded for every $\kappa\in\{2,\dots,160\}$ because within-cluster cosine rose to 0.99 and 0.97. State in the paper that JL does not justify the reduction and that it is validated empirically. Use Matryoshka prefixes only if the encoder was trained with that loss [42].

D3. **Choosing $\kappa$.** (a) Start from $h_{\mathrm{ROT}}$ (F11) with $\hat\kappa$ from F6 plus two Newton steps; expect the resulting $\kappa$ to be too small for multimodal data because the rule oversmooths. (b) Refine by least-squares cross-validation or an exact-MISE mixture selector, not by likelihood cross-validation, following the evidence for $q\ge2$ in [8]. (c) Reject any $\kappa$ for which the median leave-one-out $n_{\mathrm{eff}}$ is below about 10 or the median self-share exceeds 0.5 (F28). (d) Cross-check against the encoder's contrastive temperature, $\kappa\approx1/\tau$ [39], as a plausibility value only. Report $\kappa$, $n_{\mathrm{eff}}$ and self-share in the paper.

D4. **Adaptive concentration.** Because the corpus mixes dense families with one-off functions, compare a global $\kappa$ with per-point $\kappa_i$ set by a neighbour-mass calibration in the style of UMAP (solve $\sum_{j\le k}\exp(\kappa_i(s_{ij}-s_{i1}))=\log_2k$) [43] or by the sample-point rule $\kappa_i\propto\tilde f(X_i)$ [16]. Per-point normalisers $\log c_d(\kappa_i)$ are then required; compute them once per point in log space.

D5. **Grouping.** Run directional mean shift from every de-duplicated function (F14) with tolerance $10^{-7}$ and a cap of 200 iterations; merge end points with cosine above 0.99; treat groups smaller than a minimum size, and points below a highest-density-region threshold $\hat f_\tau$ computed from leave-one-out densities (F22), as noise rather than forcing them into a group.

D6. **Baselines and agreement.** On the same reduced vectors run HDBSCAN with angular distance [24][25], soft-moVMF with BIC over the number of components [10], and spherical k-means. Report adjusted Rand index against the mean-shift partition and bootstrap stability of the modes.

D7. **The 25 percent rule.** Implement F29 as a three-parameter logistic model in $(q^\top x,\ \log\hat f_B^{-x}(x))$. Fit $a$, $\kappa_R$ and $\beta$ on a labelled query set, report a reliability diagram, and apply $P\ge0.25$. Expand along call-graph or similarity edges with the path product of F30, a visited set, and a hard depth cap derived from $p_{\max}$.

D8. **Retrieval baselines.** Compare against fixed top-$k$, a raw cosine threshold, a score-gap cut-off (CAR [54], Adaptive-k [55]) and a conformal threshold at matched recall (CONFLARE [58]). Report precision, recall and tokens sent to the model.

D9. **Never evaluate Bessel functions directly in high dimension.** Mean shift and equal-$\kappa$ ratios need no normaliser. Where a normaliser is unavoidable (D4, reporting densities, $\kappa_R\ne\kappa_B$) use a log-domain routine validated against F5's check values: $\log c_{384}(100)=582.6318$, $\log c_{384}(20)=594.7283$, $\log c_{32}(30)=-2.0381$.

D10. **Exact sums at this scale.** One exact KDE evaluation costs $nd=768\,000$ multiply-adds at $d=384$ and $64\,000$ at $q=32$; one all-points mean-shift iteration costs $1.5\times10^9$ and $1.3\times10^8$. Add a DEANN-style estimator [38] only beyond roughly $10^5$ chunks, and prefer it to hashing for the Gaussian/vMF kernel (C14).

D11. **Blocks and ordering.** A 16 KiB block holds 128 float32 vectors at $q=32$, 64 at $q=64$ and only 10 at $d=384$. A block of that size plus the query vector and the accumulators fits in both the 32 KB and the 48 KB L1d caches at any dimension; what the reduction buys is the number of vectors processed per block. Combine per-block partial sums in fixed block order. Kernel sums are mathematically order-independent, but floating-point addition is not associative, so a deterministic reduction order is what makes results reproducible across runs and thread counts.

D12. **What to state in the paper.** The $n^{-2/(d+4)}$ rate; the published degradation between 3 and 12 dimensions; the absence of published validation at embedding dimension; the i.i.d. and isolated-mode assumptions; the fact that every experiment is on one corpus, a fork of mimalloc; and the label "proposal" on F28-F30.

---

## References

All entries were opened and checked on 2026-09-29. "Verified via" states what was actually read. Entries marked *added* were not in the first dossier.

1. Murray Rosenblatt (1956). *Remarks on Some Nonparametric Estimates of a Density Function.* The Annals of Mathematical Statistics 27(3), 832-837. DOI 10.1214/aoms/1177728190. <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-3/Remarks-on-Some-Nonparametric-Estimates-of-a-Density-Function/10.1214/aoms/1177728190.full>  
   Key: `rosenblatt1956`. Verified via: Project Euclid landing page (abstract) and Crossref record; full text is a scan and was not read.
2. Emanuel Parzen (1962). *On Estimation of a Probability Density Function and Mode.* The Annals of Mathematical Statistics 33(3), 1065-1076. DOI 10.1214/aoms/1177704472. <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-3/On-Estimation-of-a-Probability-Density-Function-and-Mode/10.1214/aoms/1177704472.full>  
   Key: `parzen1962`. Verified via: Project Euclid landing page and Crossref record; full text is a scan and was not read.
3. Yen-Chi Chen (2017). *A tutorial on kernel density estimation and recent advances.* Biostatistics & Epidemiology 1(1), 161-187. DOI 10.1080/24709360.2017.1396742 (arXiv:1704.03924v2). <https://arxiv.org/abs/1704.03924>  
   Key: `chen2017tutorial`. Verified via: arXiv abstract page, arXiv TeX source (equations read directly), Crossref record for the journal version.
4. Charles J. Stone (1980). *Optimal Rates of Convergence for Nonparametric Estimators.* The Annals of Statistics 8(6), 1348-1360. DOI 10.1214/aos/1176345206. <https://doi.org/10.1214/aos/1176345206>  
   Key: `stone1980`. Verified via: Full-text PDF (JSTOR copy hosted at sites.stat.washington.edu/courses/stat527/s14/readings/Stone_Annals_1980.pdf) and Crossref record.
5. The SciPy community (2026). *scipy.stats.gaussian_kde (SciPy v1.18.0 Manual, API reference).* Official SciPy documentation, v1.18.0 (accessed 2026-09-29). <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>  
   Key: `scipy_gaussian_kde`. Verified via: Documentation page fetched and its Notes section read.
6. Peter Hall, G. S. Watson, Javier Cabrera (1987). *Kernel density estimation with spherical data.* Biometrika 74(4), 751-762. DOI 10.1093/biomet/74.4.751. <https://academic.oup.com/biomet/article/74/4/751/280530>  
   Key: `hall1987`. Verified via: Publisher abstract page and Crossref record; formulas attributed to it are taken from Garcia-Portugues (2013).
7. Z. D. Bai, C. Radhakrishna Rao, L. C. Zhao (1988). *Kernel estimators of density function of directional data.* Journal of Multivariate Analysis 27(1), 24-39. DOI 10.1016/0047-259X(88)90113-3. <https://ideas.repec.org/a/eee/jmvana/v27y1988i1p24-39.html>  
   Key: `bai1988`. Verified via: RePEc record (abstract) and Crossref record.
8. Eduardo García-Portugués (2013). *Exact risk improvement of bandwidth selectors for kernel density estimation with directional data.* Electronic Journal of Statistics 7, 1655-1685. DOI 10.1214/13-EJS821 (arXiv:1306.0517v5). <https://arxiv.org/abs/1306.0517>  
   Key: `garciaportugues2013`. Verified via: arXiv abstract page, arXiv TeX source v5 (equations read directly), Crossref record.
9. Eduardo García-Portugués, Andrea Meilán-Vila (2024). *Kernel density estimation with polyspherical data and its applications.* arXiv:2411.04166v1 [stat.ME] (preprint, 6 November 2024). <https://arxiv.org/abs/2411.04166> *(added)*  
   Key: `garciaportugues2024polysphere`. Verified via: arXiv abstract page (metadata and abstract). Added during verification.
10. Arindam Banerjee, Inderjit S. Dhillon, Joydeep Ghosh, Suvrit Sra (2005). *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.* Journal of Machine Learning Research 6, 1345-1382. <https://jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf>  
   Key: `banerjee2005`. Verified via: JMLR PDF read in full text; Table 1 recomputed with mpmath.
11. Suvrit Sra (2012). *A short note on parameter approximation for von Mises-Fisher distributions: and a fast implementation of I_s(x).* Computational Statistics 27(1), 177-190 (online 3 February 2011). DOI 10.1007/s00180-011-0232-x. <https://link.springer.com/article/10.1007/s00180-011-0232-x>  
   Key: `sra2012`. Verified via: Springer article page (metadata and abstract) and Crossref record; full text not read.
12. Minyoung Kim (2021). *On PyTorch Implementation of Density Estimators for von Mises-Fisher and Its Mixture.* arXiv:2102.05340v1 [cs.LG] (preprint). <https://arxiv.org/abs/2102.05340>  
   Key: `kim2021vmfpytorch`. Verified via: arXiv abstract page and TeX source.
13. K. Fukunaga, L. Hostetler (1975). *The estimation of the gradient of a density function, with applications in pattern recognition.* IEEE Transactions on Information Theory 21(1), 32-40. DOI 10.1109/TIT.1975.1055330. <https://doi.org/10.1109/TIT.1975.1055330>  
   Key: `fukunaga1975`. Verified via: Crossref record only (publisher page not accessible); content claims rest on Cheng (1995), which was read.
14. Yizong Cheng (1995). *Mean Shift, Mode Seeking, and Clustering.* IEEE Transactions on Pattern Analysis and Machine Intelligence 17(8), 790-799. DOI 10.1109/34.400568. <https://doi.org/10.1109/34.400568>  
   Key: `cheng1995`. Verified via: Scanned full-text PDF (copy at members.loria.fr/MOBerger/Enseignement/Master2/Exposes/meanShiftCluster.pdf) and Crossref record.
15. Dorin Comaniciu, Peter Meer (2002). *Mean Shift: A Robust Approach Toward Feature Space Analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence 24(5), 603-619. DOI 10.1109/34.1000236. <https://doi.org/10.1109/34.1000236>  
   Key: `comaniciu2002`. Verified via: Full-text PDF (copy at courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf) and Crossref record.
16. Dorin Comaniciu, Visvanathan Ramesh, Peter Meer (2001). *The Variable Bandwidth Mean Shift and Data-Driven Scale Selection.* Proceedings of the Eighth IEEE International Conference on Computer Vision (ICCV 2001), vol. 1, 438-445. DOI 10.1109/ICCV.2001.937550. <https://doi.org/10.1109/ICCV.2001.937550>  
   Key: `comaniciu2001variable`. Verified via: Full-text PDF (copy at www.dgp.toronto.edu/~jay/MS%20bandwidth.pdf) and Crossref record.
17. Ryoya Yamasaki, Toshiyuki Tanaka (2024). *Convergence Analysis of Mean Shift.* IEEE Transactions on Pattern Analysis and Machine Intelligence 46(10), 6688-6698. DOI 10.1109/TPAMI.2024.3385920 (arXiv:2305.08463v3). <https://arxiv.org/abs/2305.08463>  
   Key: `yamasaki2023convergence`. Verified via: arXiv abstract page, TeX source v3, Crossref record for the journal version. Key kept from the first dossier; year corrected to 2024.
18. Shigeyuki Oba, Kikuya Kato, Shin Ishii (2005). *Multi-Scale Clustering for Gene Expression Profiling Data.* Fifth IEEE Symposium on Bioinformatics and Bioengineering (BIBE'05), 210-217. DOI 10.1109/BIBE.2005.41. <https://doi.org/10.1109/BIBE.2005.41> *(added)*  
   Key: `oba2005multiscale`. Verified via: Crossref record and OpenAlex abstract; cited by Zhang and Chen (2021) as the first generalisation of mean shift to directional data. Added during verification.
19. Takumi Kobayashi, Nobuyuki Otsu (2010). *Von Mises-Fisher Mean Shift for Clustering on a Hypersphere.* 2010 20th International Conference on Pattern Recognition (ICPR), 2130-2133. DOI 10.1109/ICPR.2010.522. <https://doi.org/10.1109/ICPR.2010.522> *(added)*  
   Key: `kobayashi2010vmf`. Verified via: Crossref record and OpenAlex abstract. Added during verification.
20. Yikun Zhang, Yen-Chi Chen (2021). *Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data.* Journal of Machine Learning Research 22(154), 1-92 (arXiv:2010.13523v2). <https://arxiv.org/abs/2010.13523>  
   Key: `zhang2021directional`. Verified via: arXiv abstract page, TeX source v2 (all equations and theorems read directly), JMLR PDF (theorem numbering).
21. Yikun Zhang, Yen-Chi Chen (2021). *The EM Perspective of Directional Mean Shift Algorithm.* arXiv:2101.10058v1 [math.ST] (preprint). <https://arxiv.org/abs/2101.10058>  
   Key: `zhang2021em`. Verified via: arXiv abstract page, TeX source and PDF (theorem numbering).
22. Yikun Zhang, Yen-Chi Chen (2025). *Mode and Ridge Estimation in Euclidean and Directional Product Spaces: A Mean Shift Approach.* Journal of Computational and Graphical Statistics 35(1), 101-110 (published online 2025; issue dated 2026). DOI 10.1080/10618600.2025.2505734 (arXiv:2110.08505v2). <https://arxiv.org/abs/2110.08505> *(added)*  
   Key: `zhang2025product`. Verified via: arXiv abstract page and Crossref record. Added during verification.
23. Paula Saavedra-Nieves, Martín Fernández-Pérez (2024). *Directional density-based clustering.* Advances in Data Analysis and Classification 20(1), 289-314 (published online 11 December 2024; issue dated March 2026). DOI 10.1007/s11634-024-00616-3 (arXiv:2302.09897v2). <https://link.springer.com/article/10.1007/s11634-024-00616-3>  
   Key: `saavedra2024directional`. Verified via: Springer article page (metadata and abstract), Crossref record, arXiv abstract page and TeX source v2.
24. Ricardo J. G. B. Campello, Davoud Moulavi, Joerg Sander (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* Advances in Knowledge Discovery and Data Mining (PAKDD 2013), Lecture Notes in Computer Science vol. 7819, Springer, 160-172. DOI 10.1007/978-3-642-37456-2_14. <https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14>  
   Key: `campello2013hdbscan`. Verified via: Springer chapter page (metadata and abstract) and Crossref record; algorithmic formulas taken from McInnes and Healy (2017).
25. Leland McInnes, John Healy (2017). *Accelerated Hierarchical Density Based Clustering.* 2017 IEEE International Conference on Data Mining Workshops (ICDMW), 33-42. DOI 10.1109/ICDMW.2017.12 (arXiv:1705.07321v2, titled 'Accelerated Hierarchical Density Clustering'). <https://arxiv.org/abs/1705.07321>  
   Key: `mcinnes2017accelerated`. Verified via: arXiv abstract page, TeX source v2, Crossref record for the workshop version.
26. Leland McInnes, John Healy, Steve Astels (2017). *hdbscan: Hierarchical density based clustering.* Journal of Open Source Software 2(11), 205. DOI 10.21105/joss.00205. <https://doi.org/10.21105/joss.00205>  
   Key: `mcinnes2017hdbscan`. Verified via: JOSS PDF (www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf) and Crossref record.
27. Alex Rodriguez, Alessandro Laio (2014). *Clustering by fast search and find of density peaks.* Science 344(6191), 1492-1496. DOI 10.1126/science.1242072. <https://doi.org/10.1126/science.1242072>  
   Key: `rodriguez2014density`. Verified via: Full-text PDF (copy at sites.psu.edu/mcnl/files/2017/03/9-2dhti48.pdf) and Crossref record.
28. Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, Uri Shaft (1999). *When Is "Nearest Neighbor" Meaningful?* Database Theory - ICDT'99, Lecture Notes in Computer Science vol. 1540, Springer, 217-235. DOI 10.1007/3-540-49257-7_15. <https://link.springer.com/chapter/10.1007/3-540-49257-7_15>  
   Key: `beyer1999nearest`. Verified via: Springer chapter page (metadata and full abstract) and Crossref record.
29. Miloš Radovanović, Alexandros Nanopoulos, Mirjana Ivanović (2010). *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* Journal of Machine Learning Research 11, 2487-2531. <https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf>  
   Key: `radovanovic2010hubs`. Verified via: JMLR PDF read in full text.
30. Sebastian Bruch (2024). *Foundations of Vector Retrieval.* Springer, Cham (monograph). DOI 10.1007/978-3-031-55182-6 (arXiv:2401.09350v1). <https://arxiv.org/abs/2401.09350>  
   Key: `bruch2024foundations`. Verified via: arXiv abstract page, TeX source (Chapter 2 read directly), Crossref record for the book.
31. Sanjoy Dasgupta, Anupam Gupta (2003). *An Elementary Proof of a Theorem of Johnson and Lindenstrauss.* Random Structures & Algorithms 22(1), 60-65. DOI 10.1002/rsa.10073. <https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf>  
   Key: `dasgupta2003jl`. Verified via: Author-hosted full-text PDF and Crossref record.
32. Kasper Green Larsen, Jelani Nelson (2017). *Optimality of the Johnson-Lindenstrauss Lemma.* 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), 633-638. DOI 10.1109/FOCS.2017.64 (arXiv:1609.02094v2). <https://arxiv.org/abs/1609.02094> *(added)*  
   Key: `larsen2017jl`. Verified via: arXiv abstract page and Crossref record. Added during verification.
33. Jisu Kim, Jaehyeok Shin, Alessandro Rinaldo, Larry Wasserman (2019). *Uniform Convergence Rate of the Kernel Density Estimator Adaptive to Intrinsic Volume Dimension.* Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 3398-3407 (arXiv:1810.05935v3). <https://proceedings.mlr.press/v97/kim19e.html>  
   Key: `kim2019uniform`. Verified via: PMLR page (metadata), arXiv abstract page and TeX source v3.
34. Eunseong Bae, Wolfgang Polonik (2026). *Kernel smoothing on manifolds.* arXiv:2601.16777v1 [math.ST] (preprint, 23 January 2026). <https://arxiv.org/abs/2601.16777>  
   Key: `bae2026manifolds`. Verified via: arXiv abstract page, TeX source and PDF (theorem numbering).
35. Moses Charikar, Paris Siminelakis (2017). *Hashing-Based-Estimators for Kernel Density in High Dimensions.* 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS), 1032-1043. DOI 10.1109/FOCS.2017.99 (full version arXiv:1808.10530v1). <https://arxiv.org/abs/1808.10530>  
   Key: `charikar2017hbe`. Verified via: arXiv abstract page, TeX source and PDF of the full version; Crossref record for the FOCS version.
36. Arturs Backurs, Piotr Indyk, Tal Wagner (2019). *Space and Time Efficient Kernel Density Estimation in High Dimensions.* Advances in Neural Information Processing Systems 32 (NeurIPS 2019). <https://proceedings.neurips.cc/paper_files/paper/2019/hash/a2ce8f1706e52936dfad516c23904e3e-Abstract.html>  
   Key: `backurs2019space`. Verified via: NeurIPS proceedings page (metadata) and full-text PDF.
37. Paris Siminelakis, Kexin Rong, Peter Bailis, Moses Charikar, Philip Levis (2019). *Rehashing Kernel Evaluation in High Dimensions.* Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 5789-5798. <https://proceedings.mlr.press/v97/siminelakis19a.html>  
   Key: `siminelakis2019rehashing`. Verified via: PMLR page (metadata) and full-text PDF.
38. Matti Karppa, Martin Aumüller, Rasmus Pagh (2022). *DEANN: Speeding up Kernel-Density Estimation using Approximate Nearest Neighbor Search.* Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 151, 3108-3137 (arXiv:2107.02736v2). <https://proceedings.mlr.press/v151/karppa22a.html>  
   Key: `karppa2022deann`. Verified via: PMLR page (metadata), PMLR PDF (lemma numbering), arXiv abstract page and TeX source v2.
39. Tongzhou Wang, Phillip Isola (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.* Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, 9929-9939 (arXiv:2005.10242v10). <https://arxiv.org/abs/2005.10242>  
   Key: `wang2020uniformity`. Verified via: arXiv abstract page, TeX source v10, PMLR page (metadata).
40. Kawin Ethayarajh (2019). *How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.* Proceedings of EMNLP-IJCNLP 2019, 55-65. DOI 10.18653/v1/D19-1006 (arXiv:1909.00512v1). <https://arxiv.org/abs/1909.00512>  
   Key: `ethayarajh2019contextual`. Verified via: arXiv abstract page, TeX source, Crossref record for the proceedings version.
41. Tianyu Gao, Xingcheng Yao, Danqi Chen (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6894-6910. DOI 10.18653/v1/2021.emnlp-main.552 (arXiv:2104.08821v4). <https://arxiv.org/abs/2104.08821> *(added)*  
   Key: `gao2021simcse`. Verified via: arXiv abstract page (abstract) and Crossref record. Added during verification.
42. Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi (2022). *Matryoshka Representation Learning.* Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 30233-30249 (arXiv:2205.13147v4). <https://arxiv.org/abs/2205.13147>  
   Key: `kusupati2022matryoshka`. Verified via: arXiv abstract page, TeX source v4 (abstract), NeurIPS proceedings page (metadata).
43. Leland McInnes, John Healy, James Melville (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* arXiv:1802.03426v3 [stat.ML] (preprint; v3 dated 18 September 2020). <https://arxiv.org/abs/1802.03426>  
   Key: `mcinnes2018umap`. Verified via: arXiv abstract page and TeX source v3.
44. Maarten Grootendorst (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794v1 [cs.CL] (preprint). <https://arxiv.org/abs/2203.05794>  
   Key: `grootendorst2022bertopic`. Verified via: arXiv abstract page and TeX source.
45. Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.* International Conference on Learning Representations (ICLR 2024) (arXiv:2401.18059v1). <https://arxiv.org/abs/2401.18059>  
   Key: `sarthi2024raptor`. Verified via: arXiv abstract page and TeX source (camera-ready header 'Published as a conference paper at ICLR 2024').
46. Shu Kong, Charless Fowlkes (2018). *Recurrent Pixel Embedding for Instance Grouping.* 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9018-9028. DOI 10.1109/CVPR.2018.00940 (arXiv:1712.08273v1). <https://arxiv.org/abs/1712.08273>  
   Key: `kong2018recurrent`. Verified via: arXiv abstract page, TeX source, Crossref record for the CVPR version.
47. Sua Choi, Dahyun Kang, Minsu Cho (2024). *Contrastive Mean-Shift Learning for Generalized Category Discovery.* 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 23094-23104. DOI 10.1109/CVPR52733.2024.02179 (arXiv:2404.09451v1). <https://arxiv.org/abs/2404.09451>  
   Key: `choi2024cms`. Verified via: arXiv abstract page, TeX source, CVF open-access page and Crossref record.
48. Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors.* Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162, 20827-20840 (arXiv:2204.06507v3). <https://proceedings.mlr.press/v162/sun22d.html> *(added)*  
   Key: `sun2022knnood`. Verified via: PMLR page (metadata) and arXiv abstract page (abstract). Added during verification.
49. Xudong Wang, Chris Ding, Tongxin Li, Jicong Fan (2026). *GLASS: Graph-Language Alignment with Spherical Scoring for Transferable Graph-Level Anomaly Detection.* arXiv:2609.05253v1 (preprint, 4 September 2026). <https://arxiv.org/abs/2609.05253> *(added)*  
   Key: `wang2026glass`. Verified via: arXiv abstract page (metadata and abstract). Added during verification.
50. Andrew Skabar, Shahmeer Memon (2010). *Density estimation-based document categorization using von Mises-Fisher kernels.* The 2010 International Joint Conference on Neural Networks (IJCNN), 1-8. DOI 10.1109/IJCNN.2010.5596595. <https://doi.org/10.1109/IJCNN.2010.5596595> *(added)*  
   Key: `skabar2010vmf`. Verified via: Crossref record and OpenAlex abstract; full text not read. Added during verification.
51. R. Manmatha, T. Rath, F. Feng (2001). *Modeling score distributions for combining the outputs of search engines.* Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 267-275. DOI 10.1145/383952.384005. <https://doi.org/10.1145/383952.384005> *(added)*  
   Key: `manmatha2001scores`. Verified via: Crossref record and OpenAlex abstract; full text not read. Added during verification.
52. Avi Arampatzis, Jaap Kamps, Stephen Robertson (2009). *Where to stop reading a ranked list? Threshold optimization using truncated score distributions.* Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 524-531. DOI 10.1145/1571941.1572031. <https://doi.org/10.1145/1571941.1572031> *(added)*  
   Key: `arampatzis2009stop`. Verified via: Crossref record and OpenAlex abstract; full text not read. Added during verification.
53. Nicholas Rossi, Juexin Lin, Feng Liu, Zhen Yang, Tony Lee, Alessandro Magnani, Ciya Liao (2024). *Relevance Filtering for Embedding-based Retrieval.* Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024), 4828-4835. DOI 10.1145/3627673.3680095 (arXiv:2408.04887v1). <https://arxiv.org/abs/2408.04887>  
   Key: `rossi2024relevance`. Verified via: arXiv abstract page, TeX source, Crossref record for the CIKM version.
54. Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari (2025). *Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications.* arXiv:2511.14769v1 [cs.IR] (preprint, dated 2 October 2025). <https://arxiv.org/abs/2511.14769>  
   Key: `xu2025car`. Verified via: arXiv abstract page (raw HTML meta tags) and TeX source. Author list corrected.
55. Chihiro Taguchi, Seiji Maekawa, Nikita Bhutani (2025). *Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k.* arXiv:2506.08479v3 (accepted at EMNLP 2025, main conference, per the arXiv comments field). <https://arxiv.org/abs/2506.08479> *(added)*  
   Key: `taguchi2025adaptivek`. Verified via: arXiv abstract page (metadata, abstract, comments). Added during verification.
56. Ziyu Song, Jiaming Fang, Kuangyu Li, Tuo Xia, Chuanpeng Wang (2026). *Tail-Aware Adaptive-k: Query-Adaptive Context Selection for Retrieval-Augmented Generation.* arXiv:2606.11907v1 (accepted at ECML PKDD 2026, per the arXiv comments field). <https://arxiv.org/abs/2606.11907> *(added)*  
   Key: `song2026taak`. Verified via: arXiv abstract page (metadata, abstract, comments). Added during verification.
57. Ye Xia, Emre Yamangil, Haixun Wang (2026). *Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems.* arXiv:2609.13489v2 [cs.IR] (accepted to the Applied Research Track of CIKM 2026, per the arXiv comments field). <https://arxiv.org/abs/2609.13489>  
   Key: `xia2026qcar`. Verified via: arXiv abstract page (raw HTML meta tags) and TeX source.
58. Pouria Rouzrokh, Shahriar Faghani, Cooper U. Gamble, Moein Shariatnia, Bradley J. Erickson (2024). *CONFLARE: CONFormal LArge language model REtrieval.* arXiv:2404.04287v1 (preprint). <https://arxiv.org/abs/2404.04287> *(added)*  
   Key: `rouzrokh2024conflare`. Verified via: arXiv abstract page (metadata and abstract). Added during verification.
59. Emanuele Ricco, Lorenzo Cima, Roberto Di Pietro (2025). *Hallucination Detection: A Probabilistic Framework Using Embeddings Distance Analysis.* arXiv:2502.08663v1 [cs.CL] (preprint). <https://arxiv.org/abs/2502.08663>  
   Key: `ricco2025hallucination`. Verified via: arXiv abstract page and TeX source.
60. Susovan Pal (2026). *Convergence and clustering analysis for mean shift with radially symmetric, positive definite kernels.* International Journal of Data Science and Analytics 22(1), article 144 (published online 20 April 2026). DOI 10.1007/s41060-026-01105-w (arXiv:2506.19837). <https://arxiv.org/abs/2506.19837> *(added)*  
   Key: `pal2026meanshift`. Verified via: arXiv abstract page and Crossref record; full text not read. Added during verification.
61. Zain Jabbar, Yuqin Jiang, Andrey A. Popov (2026). *Unbiased estimation of squared concentration in the Fisher-von Mises-Langevin distribution and the impossibility of unbiased concentration.* arXiv:2606.04267v1 (preprint, 2 June 2026). <https://arxiv.org/abs/2606.04267> *(added)*  
   Key: `jabbar2026intensity`. Verified via: arXiv abstract page (metadata and abstract); full text not read. Added during verification.
