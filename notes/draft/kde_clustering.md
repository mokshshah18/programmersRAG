# DRAFT (not yet citation-checked): Kernel density estimation and mode-seeking clustering on the hypersphere (von Mises-Fisher KDE, directional mean shift, bandwidth/concentration selection, convergence, high-dimensional breakdown and remedies) as the mathematical basis for grouping and probabilistically retrieving source-code function embeddings in programmersRAG

This file is the raw research dossier. The checked version is `paper/notes/kde_clustering.md` once it exists;
prefer that file when it is present.

## Recommendations

- L2-normalise every function embedding and treat the corpus as directional data; use the von Mises (vMF) kernel, which on unit vectors is identical to a Gaussian kernel with kappa = 1/h^2, so one code path serves KDE, mean shift and the relevance ratio.
- Do not run KDE or mean shift in the raw 384/768-dimensional embedding space. Reduce first to an intrinsic-scale dimension (start with q between 8 and 32 via PCA on the corpus or Matryoshka prefixes if the encoder supports them), re-normalise to unit length, and report the chosen q together with an ablation over q. State in the paper that the Johnson-Lindenstrauss bound does not justify this reduction at n of about 2000 and that it is validated empirically.
- Select the concentration in three cross-checked steps: (1) compute the directional rule of thumb h_ROT from the corpus-level kappa estimate (Banerjee approximation plus one or two Newton steps); (2) refine by likelihood cross-validation (leave-one-out log density); (3) sanity-check against the encoder's contrastive temperature (kappa = 1/tau) and against the null-model window sqrt(d) to sqrt(d ln n) in the working dimension. Report the effective number of contributing points n_eff as a diagnostic.
- Because a code corpus mixes dense families (near-duplicate helpers) with sparse one-off functions, prefer a sample-point adaptive concentration kappa_i tied to a pilot density or to the k-th neighbour similarity, and compare it against a single global kappa.
- Group functions with directional mean shift started from every function: iterate the normalised kernel-weighted mean until 1 - y_{s+1}^T y_s < 1e-7 (expect tens of iterations), then merge modes whose cosine similarity exceeds a tolerance. Cite monotone ascent and convergence from Zhang and Chen (2021) and the EM-perspective paper, not from Comaniciu and Meer's Theorem 1, whose sequence-convergence proof is disputed.
- Always report a baseline and an agreement score: run HDBSCAN on angular distance and soft-moVMF (or spherical k-means) on the same reduced embeddings and report adjusted Rand index against the mean-shift partition, plus bootstrap stability of the modes. Treat points below a highest-density-region threshold as noise instead of forcing them into a cluster.
- Implement the 25 percent rule as a calibrated posterior, not as a raw density or raw cosine threshold: compute the log likelihood ratio of a query-seeded vMF-KDE against the corpus background KDE, convert with a prior pi, and calibrate pi and kappa on a small labelled query set. Multiply probabilities along an expansion path and stop when the product drops below 0.25; also cap the number of hops. Compare against fixed top-k and against a score-gap cut-off such as CAR.
- Do all density arithmetic in log space with log-sum-exp and double-precision accumulators; never evaluate C_d(kappa) or Bessel functions directly in high dimension. Use the fact that mean shift and equal-kappa likelihood ratios need no normalising constant.
- Use exact O(n q) kernel sums for the mimalloc-scale corpus. Only if the index grows beyond roughly 1e5 chunks, add a DEANN-style estimator on top of the existing approximate-nearest-neighbour index, since it stays unbiased regardless of index quality and has a C++ reference implementation.
- For the multithreaded implementation, partition the reduced vectors into contiguous blocks (128 float32 vectors of dimension 32, or 64 of dimension 64, per 16 KB block) and combine per-block partial sums in a fixed block order. Kernel sums are mathematically order-independent, but floating-point addition is not associative, so a deterministic reduction order is what makes results reproducible across runs and thread counts.
- In the paper, state the limits explicitly: nonparametric density estimation converges at n^{-2/(d+4)}; published simulations show directional mean-shift clustering degrading between 3 and 12 dimensions; convergence theorems assume an i.i.d. sample from a smooth density with isolated modes, whereas a repository is a small fixed population. Present the method as a principled heuristic validated on one corpus (a fork of mimalloc).
- When quoting formulas in the paper, copy the radical signs and the contraction factor of Theorem 12 directly from the JMLR PDF of Zhang and Chen and the constant of eq. (5) from the Chen tutorial PDF, because these details did not survive text extraction in this research session.

## Findings

- [established] Kernel density estimation originates with Rosenblatt (1956) and Parzen (1962). For d-dimensional data with one bandwidth h the estimator is p_n(x) = (n h^d)^{-1} sum_i K((x - X_i)/h). Rosenblatt's abstract already states that density estimates satisfying mild conditions are necessarily biased, which is why bandwidth choice is a bias-variance trade-off.  (rosenblatt1956, parzen1962, chen2017tutorial)
- [established] For the Euclidean KDE the leading error terms are a squared bias of order h^4 and a variance of order 1/(n h^d); the optimal bandwidth scales as n^{-1/(d+4)} and the estimation error then decays only as n^{-2/(d+4)} (n^{-4/(d+4)} in squared error). Stone (1980) proves the optimal rate for nonparametric estimators is n^{-r} with r = (p-m)/(2p+d), and states an analogous result for density estimation, so the rate cannot be improved without assuming extra smoothness. The tutorial calls this the curse of dimensionality: the rate 'is very slow when d is large'.  (chen2017tutorial, stone1980)
- [established] Scott's and Silverman's rules are normal-reference rules: the bandwidth factors are n^{-1/(d+4)} and (n(d+2)/4)^{-1/(d+4)}, and the kernel covariance is the data covariance times factor^2. The official SciPy documentation warns that such estimation 'works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed' - which is exactly the multi-cluster regime of a code corpus, so these rules are only a starting point.  (scipy_gaussian_kde, garciaportugues2013)
- [established] The directional (spherical) KDE is f_h(x) = (c_{h,q}(L)/n) sum_i L((1 - x^T X_i)/h^2) for unit vectors on Omega_q in R^{q+1}; it goes back to Hall, Watson and Cabrera (1987) and Bai, Rao and Zhao (1988), who proved pointwise, uniform and L1 strong consistency. With the von Mises kernel L(r) = exp(-r) the estimator is exactly an equal-weight mixture of n von Mises-Fisher densities centred at the data points with common concentration kappa = 1/h^2.  (hall1987, bai1988, garciaportugues2013, zhang2021directional)
- [established] The von Mises-Fisher density on S^{d-1} is f(x|mu,kappa) = C_d(kappa) exp(kappa mu^T x) with C_d(kappa) = kappa^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(kappa)). It is the maximum-entropy density on the sphere for a fixed mean, reduces to the uniform density at kappa = 0 and tends to a point mass as kappa grows. The same constant appears in García-Portugués and in Zhang and Chen written with q = d-1.  (banerjee2005, garciaportugues2013, zhang2021directional, kim2021vmfpytorch)
- [established] The vMF maximum-likelihood mean direction is the normalised resultant; the concentration solves the implicit Bessel-ratio equation A_d(kappa) = I_{d/2}(kappa)/I_{d/2-1}(kappa) = r_bar, which has no closed form. Banerjee et al. give the approximation kappa = (r_bar d - r_bar^3)/(1 - r_bar^2), accurate to about 0.02-1.6 percent on their Table 1 (d = 10 to 1000); Sra (2012) shows truncated Newton steps give more accurate values for the same cost. We re-computed Table 1 in this session and reproduced 10.1631, 60.08, 300.08 and 800.13.  (banerjee2005, sra2012)
- [established] Cosine-similarity clustering has a generative justification: spherical k-means is the special case of EM on a mixture of vMF distributions when all concentrations are equal (and tend to infinity) with hard assignments, and the Fisher kernel of the vMF model equals the cosine similarity. Soft-moVMF yields per-point posterior membership probabilities p(h|x_i), which is a published probabilistic quantity that can be thresholded.  (banerjee2005)
- [established] For the directional KDE the MISE is b_q(L)^2 R(Psi) h^4 + c_{h,q}(L) d_q(L)/n + lower order, giving an AMISE-optimal bandwidth proportional to n^{-1/(4+q)}. García-Portugués derives a directional analogue of Silverman's rule (h_ROT) assuming a vMF reference density, plus mixture-based selectors (AMI, EMI); cross-validation selectors (LCV, LSCV) are due to Hall et al. He also warns that AMISE and exact MISE 'may differ significantly for moderate and even large sample sizes'. Zhang and Chen use h_ROT with Banerjee's kappa estimate as their default.  (garciaportugues2013, zhang2021directional, hall1987)
- [established] Mean shift was introduced by Fukunaga and Hostetler (1975) and generalised by Cheng (1995), who showed that mean shift with kernel K is gradient ascent on the density estimate built with its 'shadow' kernel, and that the Gaussian kernel is the only kernel (with its truncation) that is its own shadow, in which case the mean-shift vector is proportional to the gradient of log density. Comaniciu and Meer (2002) showed that for a kernel with convex, monotonically decreasing profile the density values along the trajectory increase monotonically and no step size needs tuning.  (fukunaga1975, cheng1995, comaniciu2002)
- [contested] The claim in Comaniciu and Meer's Theorem 1 that the sequence of mode estimates itself converges is disputed: Yamasaki and Tanaka state that 'the proof in [Comaniciu and Meer] used an incorrect inequality evaluation to claim that the mode estimate sequence is a Cauchy sequence' (counterexamples by Li, Hu and Wu 2007) and that EM-based arguments alone are also insufficient. They give a rigorous guarantee via the Lojasiewicz inequality for C^1 subanalytic kernels (including biweight) without assuming isolated critical points, a non-degenerate Hessian, or any restriction on bandwidth or dimension, plus convergence-rate bounds.  (yamasaki2023convergence, comaniciu2002)
- [recent] Directional mean shift (Zhang and Chen, JMLR 2021) is the fixed-point iteration y_{s+1} = -sum_i X_i L'((1 - y_s^T X_i)/h^2) / || the same sum ||, i.e. the normalised total gradient of the directional KDE. It needs neither a step size nor the normalising constant. If L is monotonically decreasing, differentiable and convex with L(0) finite (true for the von Mises kernel) the density values increase monotonically (Theorem 8); under isolated modes it converges to a mode from any start within a radius r_k (Theorem 11); and it is a Riemannian gradient ascent with adaptive step size that converges at least linearly near modes for a sufficiently small bandwidth (Theorem 12 and discussion).  (zhang2021directional)
- [recent] Directional mean shift is a generalised EM algorithm and an exact EM algorithm for the von Mises kernel; under (C1) finitely many isolated modes and (C2) a non-increasing, differentiable, convex kernel with 0 < L(0) < infinity the iterates converge to a local mode from almost all starting points on the sphere - a stronger (global) statement than the local result of the JMLR paper. Convergence to a saddle point or minimum is possible but unstable.  (zhang2021em)
- [recent] Statistical guarantees for the directional KDE depend on the intrinsic sphere dimension q: the density, Riemannian gradient and Hessian estimators have bias O(h^2) and stochastic error governed by n h^q, n h^{q+2} and n h^{q+4} respectively, and the set of estimated modes converges to the true modes in Hausdorff distance at the gradient rate. These match Euclidean rates with d replaced by q, so mode estimation is strictly harder than density estimation.  (zhang2021directional, chen2017tutorial)
- [established] Computational cost of directional mean shift: O(n q) per iteration for one query point, O(n^2 q) per iteration when all n points are shifted, O(n^2 q log(1/epsilon)) in total under linear convergence, and O(n q) memory.  (zhang2021directional)
- [established] KDE-based mode clustering demonstrably breaks down as dimension grows. In Zhang and Chen's simulations the misclassification rate of directional mean-shift clustering rises towards 1 (worst case) as q goes from 3 to 12 with the rule-of-thumb bandwidth; they attribute this to the untuned bandwidth and to the curse of dimensionality. Comaniciu and Meer advise that 'whenever the feature space has more than (say) six dimensions, the analysis should be approached carefully'. Running vMF-KDE or mean shift directly on raw 384- or 768-dimensional embeddings is therefore not supported by the literature.  (zhang2021directional, comaniciu2002, chen2017tutorial)
- [established] High-dimensional geometry undermines any distance-based density: under broad conditions the nearest and farthest neighbour distances converge (effects visible at 10-15 dimensions); k-nearest-neighbour lists become dominated by 'hub' points; and for spherically symmetric data the inner product with a unit query has mean 0 and variance ||X||^2/d, so random directions become nearly orthogonal. Bruch notes retrieval remains meaningful when data are clustered or have low intrinsic dimension.  (beyer1999nearest, radovanovic2010hubs, bruch2024foundations)
- [our_proposal] Derived consequence for vMF kernels (our derivation, checked by simulation in this session): for isotropic unit vectors the log kernel weights kappa x^T X_i have standard deviation kappa/sqrt(d), so the Kish effective number of points contributing to the KDE is about n exp(-kappa^2/d) while kappa^2 <= d. In simulation with d = 384, n = 2000 the effective count was 1557, 685, 273, 81 and 13 at kappa = 0.5, 1, 1.5, 2 and 3 times sqrt(d). Thus the KDE is nearly uniform for kappa well below sqrt(d) (about 20) and collapses to a nearest-neighbour rule beyond roughly sqrt(d ln n) (about 54); only a narrow window of concentrations carries density information under the null model.  (bruch2024foundations)
- [recent] The principal statistical remedy is low intrinsic dimension: KDE error bounds can be stated in terms of the intrinsic (volume or manifold) dimension of the data's support instead of the ambient dimension. Kim et al. define the volume dimension and obtain uniform bounds that adapt to it; Bae and Polonik (2026) give finite-sample bounds for kernel smoothing and its first and second derivatives on a compact d-dimensional manifold and note 'the upper bound depends on the dimension d of M, rather than the dimension of the ambient space'.  (kim2019uniform, bae2026manifolds, zhang2021directional)
- [contested] Current practice for clustering text embeddings reduces dimension before density-based clustering: BERTopic applies UMAP and then HDBSCAN explicitly because of distance concentration, and RAPTOR applies UMAP before Gaussian-mixture soft clustering for the same reason. However Radovanovic et al. caution that dimensionality reduction 'may not constitute an easy mitigation' of hubness because hubness is tied to intrinsic dimensionality, so the benefit must be validated empirically on the target corpus rather than assumed.  (grootendorst2022bertopic, sarthi2024raptor, mcinnes2018umap, radovanovic2010hubs)
- [established] The Johnson-Lindenstrauss lemma guarantees pairwise distances are preserved within a factor 1 +/- epsilon when projecting n points to k >= 4 (epsilon^2/2 - epsilon^3/3)^{-1} ln n dimensions. Matryoshka representation learning offers an alternative: nested embeddings whose low-dimensional prefixes remain accurate, enabling adaptive retrieval.  (dasgupta2003jl, kusupati2022matryoshka)
- [our_proposal] Computed consequence (ours): for n = 2000 points the JL bound requires k >= 365 at epsilon = 0.5, 845 at 0.3 and 1754 at 0.2, i.e. no useful guaranteed reduction from a 384-dimensional embedding. Random projection therefore cannot justify reducing to 8-32 dimensions; any such reduction must rely on data-dependent structure (PCA, Matryoshka prefixes, UMAP) and be validated empirically.  (dasgupta2003jl)
- [established] Adaptive bandwidths are the classical remedy for densities with both sharp peaks and sparse tails. The balloon estimator varies h with the evaluation point (k-NN choice, Loftsgaarden and Quesenberry); the sample-point estimator varies h with the data point and, with h_i proportional to pilot-density^{-1/2}, reduces the bias order from h^2 to h^4 while remaining a proper density. Comaniciu, Ramesh and Meer build a variable-bandwidth mean shift on it and prove convergence. Recent deep-embedding work replaces radius neighbourhoods with k-NN neighbourhoods for stability, and UMAP calibrates a per-point kernel width sigma_i so that the summed kernel weights over k neighbours equal log2(k).  (chen2017tutorial, comaniciu2001variable, choi2024cms, mcinnes2018umap)
- [established] HDBSCAN replaces a single density threshold with a hierarchy: the core distance of a point is its k-th nearest-neighbour distance (its reciprocal is a local density estimate), the mutual reachability distance is max(core_i, core_j, d_ij), single linkage on that metric yields a cluster tree in lambda = 1/distance, and clusters are selected by maximising stability (summed lambda persistence). The original algorithm is O(N^2); the dual-tree Boruvka variant is sub-quadratic and empirically approaches O(N log N). It labels low-density points as noise and needs no number of clusters.  (campello2013hdbscan, mcinnes2017accelerated, mcinnes2017hdbscan)
- [established] Density-peaks clustering (Rodriguez and Laio) characterises cluster centres as points with high local density rho_i and large distance delta_i to any point of higher density, read off a 'decision graph'; it needs only pairwise distances, and the cutoff d_c is chosen so that the average number of neighbours is about 1-2 percent of the data. The authors note that for very small samples 'the density estimator is unavoidably affected by large statistical errors'.  (rodriguez2014density)
- [recent] Density-based clustering has been extended to the hypersphere using highest-density regions: clusters are connected components of {x : f(x) >= f_tau} where f_tau is the largest level whose region has probability content at least 1 - tau, estimated as the tau-quantile of the KDE values at the sample points, with a vMF-kernel KDE. This gives a published, probability-calibrated way to turn a density into an inclusion threshold.  (saavedra2024directional)
- [recent] Fast approximate KDE in high dimension: uniform random sampling needs O(1/(tau epsilon^2)) kernel evaluations for a (1 +/- epsilon) estimate of densities >= tau; hashing-based estimators (Charikar and Siminelakis) improve the dependence to 1/sqrt(tau) at the price of super-linear space; Backurs, Indyk and Wagner restore linear space; Siminelakis et al. add diagnostics that choose between sampling and hashing per data set; DEANN computes the contribution of k approximate nearest neighbours exactly and estimates the remainder by sampling, remaining unbiased 'no matter the quality of the near neighbors' and ships a C++ implementation.  (charikar2017hbe, backurs2019space, siminelakis2019rehashing, karppa2022deann)
- [our_proposal] On unit vectors the Gaussian kernel and the vMF kernel coincide up to a constant, because ||x - y||^2 = 2(1 - x^T y): exp(-||x-y||^2/(2 sigma^2)) = exp(-1/sigma^2) exp(x^T y / sigma^2), i.e. kappa = 1/sigma^2 = 1/h^2. Hence every Euclidean Gaussian-kernel algorithm (HBE, DEANN, Gaussian mean shift) applies to vMF-KDE on normalised embeddings without modification.  (zhang2021directional, wang2020uniformity)
- [recent] Contrastively trained encoders are implicitly fitted to a vMF-KDE: Wang and Isola show that the uniformity term of the contrastive loss is a resubstitution entropy estimator based on a vMF kernel density estimate with kappa = 1/tau (tau the training temperature). At the same time contextual embeddings are anisotropic, occupying a narrow cone, so the background density of a real embedding corpus is far from uniform and must be estimated from the corpus.  (wang2020uniformity, ethayarajh2019contextual)
- [recent] Recent retrieval work (2024-2026) agrees that a fixed top-k is brittle and that raw cosine thresholds are not comparable across queries. Cluster-based Adaptive Retrieval finds the cut-off by clustering the ordered query-document distances (HDBSCAN, OPTICS and others; results are similar across clustering back-ends); QCAR chooses depth per query cluster before retrieval; the Cosine Adapter (CIKM 2024) learns a query-dependent monotone map from cosine score to a calibrated relevance score and only then applies a global threshold, improving precision on MS MARCO and production search.  (xu2025car, xia2026qcar, rossi2024relevance)
- [recent] RAPTOR (ICLR 2024) is a published precedent for recursive, probabilistic clustering in RAG: chunks are embedded, reduced with UMAP, soft-clustered with Gaussian mixtures (number of components by BIC, so a chunk may belong to several clusters), summarised, and the procedure is applied recursively to build a tree. The authors concede the Gaussian assumption 'may not perfectly align with the nature of text data'; a vMF mixture or vMF-KDE is the natural spherical counterpart.  (sarthi2024raptor, banerjee2005)
- [recent] Mean shift on learned hyperspherical embeddings has precedents in deep learning: Kong and Fowlkes use a vMF kernel K proportional to exp(delta x^T x_i) in a recurrent mean-shift grouping module on unit-norm embeddings, and Contrastive Mean-Shift learning (2024) performs k-NN-based mean-shift steps inside contrastive training. Both are in computer vision; neither addresses source code.  (kong2018recurrent, choi2024cms)
- [recent] Density estimation is statistically far better behaved on one-dimensional similarity or distance scores than in the embedding space. A 2025 hallucination-detection method fits separate Gaussian KDEs to the distributions of embedding distances for two classes and compares log-likelihoods; this is a direct precedent for estimating class-conditional densities of scalar scores and taking their ratio.  (ricco2025hallucination, chen2017tutorial)
- [our_proposal] In our searches we found no published work that applies vMF-KDE or directional mean shift to grouping source-code functions, nor a retrieval procedure that expands a trace while a KDE-derived relevance probability stays above a fixed threshold. Both must therefore be presented as new proposals that combine published components, and validated empirically on the mimalloc fork.  (zhang2021directional, xu2025car, sarthi2024raptor)
- [our_proposal] Proposed relevance probability: with a 'relevant' density f_R (vMF kernels centred on the query and already accepted functions) and a background density f_B (vMF-KDE of the whole corpus) and prior pi, Bayes' rule gives P(rel|x) = pi f_R(x) / (pi f_R(x) + (1-pi) f_B(x)). The 25 percent rule P >= 1/4 is equivalent to the likelihood-ratio test f_R(x)/f_B(x) >= (1-pi)/(3 pi). Multiplying probabilities along an expansion path and stopping when the product falls below 0.25 guarantees termination because each factor is below 1.  (banerjee2005, saavedra2024directional, rossi2024relevance)
- [our_proposal] Numerical practice: the vMF normaliser must be evaluated in log space. Bessel evaluation is the known failure point (SciPy underflows for high order; a vMF-based model became unstable above dimension 50), and Banerjee et al. report overflow problems for root finders in high dimension. In this session log C_d(kappa) for d = 384, kappa = 100 was +582.63, and the density integrated to 1.000000 when accumulated in log space. Mean shift and likelihood ratios with a common kappa need no normaliser at all.  (kim2021vmfpytorch, banerjee2005, zhang2021directional)
- [our_proposal] At the scale of the evaluation corpus exact computation is sufficient: with n about 2000 functions and d = 384 one exact KDE evaluation costs about 7.7e5 multiply-adds and one all-points mean-shift iteration about 1.5e9; after reduction to 32 dimensions a 16 KB block holds 128 float32 vectors. Sub-linear KDE structures only pay off at corpus sizes several orders of magnitude larger. Verification scripts are at C:/Users/CASHAMOK/AppData/Local/Temp/claude/C--Users-CASHAMOK-Documents-GitHub-programmersRAG/6fffbf87-56fc-4337-8f23-de8cab3f47af/scratchpad/kde/check.py and neff.py.  (zhang2021directional, karppa2022deann)

## Formulas

### Euclidean kernel density estimator

$$
\hat p_n(x)=\frac{1}{n h^{d}}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h}\right)
$$

Eq. (1) of the Chen tutorial; K is the kernel and h > 0 the bandwidth applied equally in every direction. Historical origin: Rosenblatt 1956, Parzen 1962.

Sources: chen2017tutorial, rosenblatt1956, parzen1962

### MISE expansion and optimal bandwidth (Euclidean)

$$
\mathrm{MISE}(h)=\frac{h^{4}}{4}\sigma_K^{4}\int\lvert\nabla^{2}p(x)\rvert^{2}dx+\frac{\mu_K}{n h^{d}}+o(h^{4})+o\!\left(\frac{1}{n h^{d}}\right),\qquad h_{\mathrm{opt}}=\left(\frac{d\,\mu_K}{\sigma_K^{4}\int\lvert\nabla^{2}p(x)\rvert^{2}dx}\cdot\frac{1}{n}\right)^{\frac{1}{d+4}}\propto n^{-\frac{1}{d+4}}
$$

The expansion is eq. (4) of the tutorial. The constant in h_opt is OUR derivation: setting the derivative of the two leading terms to zero gives h^3 sigma_K^4 R - d mu_K/(n h^{d+1}) = 0. The tutorial's eq. (5) gives the same n^{-1/(d+4)} scaling; its printed constant could not be verified from text extraction, so cite the tutorial for the scaling only. Substituting gives squared error of order n^{-4/(d+4)}, i.e. error of order n^{-2/(d+4)}, the rate the tutorial prints.

Sources: chen2017tutorial

### Optimal nonparametric rate (Stone)

$$
r=\frac{p-m}{2p+d},\qquad \text{error}\asymp n^{-r};\quad p=2,\ m=0\ \Rightarrow\ r=\frac{2}{4+d}
$$

Stone's abstract: r = (p-m)/(2p+d) is the optimal uniform rate for estimating an m-th order derivative functional of a p-smooth function in d dimensions, with an analogous result for densities. The special case p = 2, m = 0 is our substitution and reproduces n^{-2/(d+4)}. Our arithmetic: for n = 2000, n^{-2/(d+4)} is 0.28 at d = 8, 0.66 at d = 32 and 0.96 at d = 384; halving the error needs n multiplied by 2^{(d+4)/2}, i.e. 2^{194} at d = 384.

Sources: stone1980, chen2017tutorial

### Scott and Silverman bandwidth factors

$$
f_{\mathrm{Scott}}=n^{-\frac{1}{d+4}},\qquad f_{\mathrm{Silverman}}=\left(\frac{n(d+2)}{4}\right)^{-\frac{1}{d+4}},\qquad \Sigma_{\mathrm{kernel}}=f^{2}\,\widehat{\mathrm{Cov}}(X)
$$

As documented for scipy.stats.gaussian_kde (n data points, d dimensions), citing Scott (1992) and Silverman (1986). The kernel covariance is the data covariance multiplied by the squared factor.

Sources: scipy_gaussian_kde

### von Mises-Fisher density and normalising constant

$$
f_{\mathrm{vMF}}(x;\mu,\kappa)=C_d(\kappa)\,e^{\kappa\mu^{\top}x},\qquad C_d(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2}\,I_{d/2-1}(\kappa)},\qquad x,\mu\in S^{d-1}\subset\mathbb{R}^{d},\ \kappa\ge 0
$$

Banerjee et al. eqs. (2.1)-(2.2); I_r is the modified Bessel function of the first kind. In the q = d-1 notation of García-Portugués and Zhang-Chen: C_q(kappa) = kappa^{(q-1)/2} / ((2 pi)^{(q+1)/2} I_{(q-1)/2}(kappa)). Checked numerically here: for d = 3 it equals kappa/(4 pi sinh kappa), and the density integrates to 1 for d = 8, 32, 384.

Sources: banerjee2005, garciaportugues2013, zhang2021directional

### vMF maximum likelihood and concentration approximation

$$
\hat\mu=\frac{r}{\lVert r\rVert},\ r=\sum_{i=1}^{n}x_i;\qquad A_d(\hat\kappa)=\frac{I_{d/2}(\hat\kappa)}{I_{d/2-1}(\hat\kappa)}=\bar r=\frac{\lVert r\rVert}{n};\qquad \hat\kappa\approx\frac{\bar r\,d-\bar r^{3}}{1-\bar r^{2}};\qquad A_d'(\kappa)=1-A_d(\kappa)^{2}-\frac{d-1}{\kappa}A_d(\kappa)
$$

Banerjee et al. eqs. (2.4), (2.5), (4.4) and footnote 3 (derivative used for Newton refinement starting from the approximation). Sra (2012) analyses the accuracy of such approximations and recommends truncated Newton steps.

Sources: banerjee2005, sra2012

### EM for a mixture of vMF distributions (soft-moVMF)

$$
p(h\mid x_i,\Theta)=\frac{\alpha_h f_h(x_i\mid\theta_h)}{\sum_{l=1}^{k}\alpha_l f_l(x_i\mid\theta_l)},\quad \alpha_h=\frac{1}{n}\sum_{i}p(h\mid x_i,\Theta),\quad r_h=\sum_{i}x_i\,p(h\mid x_i,\Theta),\quad \hat\mu_h=\frac{r_h}{\lVert r_h\rVert},\quad \bar r_h=\frac{\lVert r_h\rVert}{n\alpha_h},\quad \hat\kappa_h=\frac{\bar r_h d-\bar r_h^{3}}{1-\bar r_h^{2}}
$$

E-step (3.7) and M-step (3.3)-(3.6) with the kappa approximation, as in Algorithm 1 of Banerjee et al. The posterior p(h|x_i) is the published per-point membership probability.

Sources: banerjee2005

### Directional kernel density estimator

$$
\hat f_h(x)=\frac{c_{h,q}(L)}{n}\sum_{i=1}^{n}L\!\left(\frac{1-x^{\top}X_i}{h^{2}}\right),\qquad c_{h,q}(L)^{-1}=\int_{\Omega_q}L\!\left(\frac{1-x^{\top}y}{h^{2}}\right)\omega_q(dy)=O(h^{q})
$$

Eq. (2) of García-Portugués and of Zhang-Chen, attributed to Hall et al. (1987) and Bai et al. (1988). Omega_q is the unit sphere in R^{q+1}, L a directional kernel, h the bandwidth. Bai et al.'s abstract gives the same form f_n(x) = (n h^{k-1})^{-1} C(h) sum K[(1 - x'X_i)/h^2].

Sources: garciaportugues2013, zhang2021directional, bai1988, hall1987

### vMF-kernel KDE (von Mises kernel L(r) = e^{-r})

$$
\hat f_h(x)=\frac{1}{n}\sum_{i=1}^{n}f_{\mathrm{vMF}}\!\left(x;X_i,\tfrac{1}{h^{2}}\right)=\frac{1}{n\,(2\pi)^{\frac{q+1}{2}}\,I_{\frac{q-1}{2}}(1/h^{2})\,h^{q-1}}\sum_{i=1}^{n}\exp\!\left(\frac{x^{\top}X_i}{h^{2}}\right)
$$

The directional KDE with the von Mises kernel is a mixture of vMF densities with means X_i and common concentration kappa = 1/h^2 (Zhang-Chen Section 2.1; García-Portugués Section 2).

Sources: zhang2021directional, garciaportugues2013

### Directional MISE and AMISE-optimal bandwidth

$$
\mathrm{MISE}(h)=b_q(L)^{2}R(\Psi(f,\cdot))h^{4}+\frac{c_{h,q}(L)}{n}d_q(L)+o\!\left(h^{4}+(nh^{q})^{-1}\right),\qquad h_{\mathrm{AMISE}}=\left[\frac{q\,d_q(L)}{4\,b_q(L)^{2}\lambda_q(L)R(\Psi(f,\cdot))\,n}\right]^{\frac{1}{4+q}}
$$

García-Portugués Proposition 1 and Corollary 1, with b_q(L) = int_0^inf L(r) r^{q/2} dr / int_0^inf L(r) r^{q/2-1} dr, d_q(L) = int L^2(r) r^{q/2-1} dr / int L(r) r^{q/2-1} dr, lambda_q(L) = 2^{q/2-1} omega_{q-1} int L(r) r^{q/2-1} dr, Psi(f,x) = -x^T grad f(x) + q^{-1}(Laplacian f(x) - x^T Hf(x) x), R(Psi) = int Psi^2. OUR evaluation for the von Mises kernel: b_q = q/2, d_q = 2^{-q/2}, lambda_q = (2 pi)^{q/2}, hence h_AMISE = [q (4 pi)^{q/2} R(Psi) n]^{-1/(4+q)}; this reproduces the printed rule of thumb exactly.

Sources: garciaportugues2013

### Directional rule-of-thumb bandwidth (von Mises kernel, vMF reference)

$$
h_{\mathrm{ROT}}=\left[\frac{4\pi^{1/2}\,I_{\frac{q-1}{2}}(\hat\kappa)^{2}}{\hat\kappa^{\frac{q+1}{2}}\left[2q\,I_{\frac{q+1}{2}}(2\hat\kappa)+(2+q)\,\hat\kappa\,I_{\frac{q+3}{2}}(2\hat\kappa)\right]n}\right]^{\frac{1}{4+q}},\qquad \kappa_{\mathrm{KDE}}=h_{\mathrm{ROT}}^{-2}
$$

García-Portugués Proposition 2 / eq. (6) and Zhang-Chen eq. (31). kappa-hat is the estimated concentration of the whole sample (Banerjee approximation or MLE). It is the AMISE-optimal bandwidth when the true density is vMF; it oversmooths multimodal data and performed poorly beyond about 10 dimensions in Zhang-Chen's simulation. We verified algebraically that q = 2 gives the printed closed form 8 sinh^2(k)/(k[(1+4k^2) sinh 2k - 2k cosh 2k] n).

Sources: garciaportugues2013, zhang2021directional

### Cross-validation bandwidth selectors (directional)

$$
h_{\mathrm{LCV}}=\arg\max_{h>0}\sum_{i=1}^{n}\log\hat f_h^{-i}(X_i),\qquad h_{\mathrm{LSCV}}=\arg\max_{h>0}\left[\frac{2}{n}\sum_{i=1}^{n}\hat f_h^{-i}(X_i)-\int_{\Omega_q}\hat f_h(x)^{2}\,\omega_q(dx)\right]
$$

Likelihood and least-squares cross-validation selectors of Hall et al. (1987) as restated by García-Portugués; f^{-i} is the leave-one-out estimator. LCV only needs leave-one-out log-densities, so it is easy to compute in log space for a modest corpus.

Sources: garciaportugues2013, hall1987

### Euclidean mean-shift iteration

$$
y_{j+1}=\frac{\sum_{i=1}^{n}x_i\,g\!\left(\left\lVert\frac{y_j-x_i}{h}\right\rVert^{2}\right)}{\sum_{i=1}^{n}g\!\left(\left\lVert\frac{y_j-x_i}{h}\right\rVert^{2}\right)},\qquad g=-k',\qquad m_{h,G}(y_j)=y_{j+1}-y_j
$$

Comaniciu-Meer eq. (20): k is the profile of kernel K, g the profile of its derivative kernel G. The mean-shift vector is proportional to the normalised density gradient (their eq. 19), and Cheng shows that for the Gaussian kernel it is proportional to grad log q(x).

Sources: comaniciu2002, cheng1995, fukunaga1975

### Directional mean-shift fixed-point update

$$
y_{s+1}=-\frac{\sum_{i=1}^{n}X_i\,L'\!\left(\frac{1-y_s^{\top}X_i}{h^{2}}\right)}{\left\lVert\sum_{i=1}^{n}X_i\,L'\!\left(\frac{1-y_s^{\top}X_i}{h^{2}}\right)\right\rVert_2}=\frac{\nabla\hat f_h(y_s)}{\lVert\nabla\hat f_h(y_s)\rVert_2};\qquad L(r)=e^{-r}:\ \ y_{s+1}=\frac{\sum_{i}X_i\,e^{\kappa\,y_s^{\top}X_i}}{\left\lVert\sum_{i}X_i\,e^{\kappa\,y_s^{\top}X_i}\right\rVert_2},\ \kappa=h^{-2}
$$

Zhang-Chen eqs. (20) and (23), Algorithm 1, with stopping rule 1 - y_{s+1}^T y_s <= epsilon. The vMF special case is our one-line substitution L'(r) = -e^{-r} (the constant e^{-1/h^2} cancels); it is a kernel-weighted mean of the data re-projected onto the sphere. In our pure-Python check (d = 16, n = 300, kappa = 30) the density increased monotonically and the iteration stopped after 10 steps.

Sources: zhang2021directional, zhang2021em

### Adaptive step size of directional mean shift

$$
y_{s+1}=\mathrm{Exp}_{y_s}\!\left(\eta_s\,\mathrm{grad}\,\hat f_h(y_s)\right),\qquad \eta_s=\frac{\theta_s}{\sin\theta_s\,\lVert\nabla\hat f_h(y_s)\rVert_2},\qquad \theta_s=\angle\left(\nabla\hat f_h(y_s),\,y_s\right)
$$

Zhang-Chen Section 5.2: the algorithm is Riemannian gradient ascent whose step is controlled by the norm of the total gradient, which is dominated by its radial part of order h^{-2}; a small bandwidth therefore places the step in the range where Theorem 12 gives linear convergence d(y_s, m_k) <= Upsilon^s d(y_0, m_k) with Upsilon < 1. The closed form of Upsilon should be copied from the JMLR PDF, as it did not survive text extraction.

Sources: zhang2021directional

### Convergence rates of the directional KDE, its derivatives and its modes

$$
\hat f_h-f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q}}}\right),\quad \mathrm{grad}\,\hat f_h-\mathrm{grad}\,f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+2}}}\right),\quad \mathcal{H}\hat f_h-\mathcal{H}f=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+4}}}\right),\quad \mathrm{Haus}(\mathcal M,\hat{\mathcal M}_n)=O(h^{2})+O_P\!\left(\sqrt{\tfrac{1}{nh^{q+2}}}\right)
$$

Zhang-Chen Section 4.2, Theorem 2 and Theorem 6(c), under smoothness (D1), kernel (D2') and mode regularity (M1-2) conditions; uniform versions (Theorem 4) replace 1 by |log h|. CAVEAT: the radical signs are vector graphics and were not present in our text extraction; they are restored from the paper's Remark 3 (rates align with the Euclidean ones with d replaced by q) and the Chen tutorial, where the square roots are printed. Verify against the PDF before publication.

Sources: zhang2021directional, chen2017tutorial

### Complexity of directional mean shift

$$
T_{\text{one query, one iteration}}=O(nq),\qquad T_{\text{all points}}=O\!\left(n^{2}q\log(1/\epsilon)\right),\qquad S=O(nq)
$$

Zhang-Chen Section 5.3, assuming linear convergence to reach an epsilon-error.

Sources: zhang2021directional

### Balloon and sample-point (adaptive) estimators

$$
\hat f_{\mathrm{balloon}}(x)=\frac{1}{n\,h(x)^{d}}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h(x)}\right),\qquad \hat f_{\mathrm{sp}}(x)=\frac{1}{n}\sum_{i=1}^{n}\frac{1}{h_i^{d}}K\!\left(\frac{x-X_i}{h_i}\right),\qquad h_i=h_0\left[\frac{\lambda}{\tilde f(X_i)}\right]^{1/2}
$$

Definitions from the Chen tutorial (remark on adaptive smoothing) and Comaniciu-Ramesh-Meer. With h_i reciprocal to the square root of a pilot density the bias becomes proportional to h^4 instead of h^2 while variance stays of order 1/(n h^d); h_0 is a fixed bandwidth and lambda a proportionality constant. The spherical analogue (our adaptation) uses per-point concentrations kappa_i proportional to the pilot density.

Sources: chen2017tutorial, comaniciu2001variable

### HDBSCAN mutual reachability distance and cluster stability

$$
d_{\mathrm{mreach}}(X_i,X_j)=\begin{cases}\max\{\kappa(X_i),\kappa(X_j),d(X_i,X_j)\}, & X_i\ne X_j\\ 0, & X_i=X_j\end{cases}\qquad \lambda=\frac{1}{\varepsilon},\qquad \sigma(C_i)=\sum_{X_j\in C_i}\left(\lambda_{\max,C_i}(X_j)-\lambda_{\min,C_i}(X_j)\right)
$$

McInnes-Healy Section 2: kappa(X) here is the core distance (distance to the k-th nearest neighbour), not a vMF concentration. Single linkage on d_mreach gives the hierarchy; flat clusters maximise total stability. (The paper prints d(X_j,X_j) in the max, an evident typo for d(X_i,X_j).)

Sources: mcinnes2017accelerated, campello2013hdbscan

### Density-peaks quantities

$$
\rho_i=\sum_{j}\chi(d_{ij}-d_c),\ \ \chi(x)=\mathbb{1}[x<0];\qquad \delta_i=\min_{j:\rho_j>\rho_i}d_{ij};\qquad \gamma_i=\rho_i\delta_i
$$

Rodriguez-Laio eqs. (1)-(2): local density, distance to the nearest point of higher density, and their product used to rank candidate centres. A Gaussian kernel density may replace the hard cutoff for small samples.

Sources: rodriguez2014density

### Directional highest-density region and its plug-in estimate

$$
L_f(\tau)=\{x\in S^{d-1}: f(x)\ge f_\tau\},\quad f_\tau=\sup\{t:\ P(X\in\{f\ge t\})\ge 1-\tau\};\qquad \hat L_f(\tau)=\{x:\hat f_n(x)\ge \hat f_\tau\},\ \ \hat f_\tau=\tau\text{-quantile of }\{\hat f_n(X_i)\}_{i=1}^{n}
$$

Saavedra-Nieves and Fernández-Pérez eqs. (4)-(6): the 100(1-tau) percent HDR and its estimate using the vMF-kernel KDE; clusters are the connected components of the region, and varying tau yields a cluster tree.

Sources: saavedra2024directional

### Johnson-Lindenstrauss dimension bound

$$
k\ \ge\ 4\left(\frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{3}\right)^{-1}\ln n\ \Longrightarrow\ \exists f:\mathbb{R}^{d}\to\mathbb{R}^{k}:\ (1-\epsilon)\lVert u-v\rVert^{2}\le\lVert f(u)-f(v)\rVert^{2}\le(1+\epsilon)\lVert u-v\rVert^{2}\ \ \forall u,v\in V
$$

Dasgupta-Gupta Theorem 2.1 for any set V of n points; the map is a scaled random projection found in randomised polynomial time. Our arithmetic: n = 2000 gives k >= 365 (epsilon 0.5), 845 (0.3), 1754 (0.2), 6515 (0.1).

Sources: dasgupta2003jl

### DEANN unbiased KDE estimator

$$
\widehat{\mathrm{KDE}}_X(y)=\frac{k}{n}\cdot\frac{1}{k}\sum_{x\in X_1}K_h(x,y)+\frac{n-k}{n}\cdot\frac{1}{m}\sum_{x\in S}K_h(x,y),\qquad X_1=\mathrm{ANN}_X(y),\ \ S\subset X\setminus X_1\ \text{uniform},\ |S|=m
$$

Karppa-Aumüller-Pagh Algorithm 1 and Lemma 3/Corollary 4: exact contribution of k approximate nearest neighbours plus a random-sampling estimate of the remainder; unbiased regardless of ANN quality, which only affects variance.

Sources: karppa2022deann

### Sample and time complexity of approximate KDE

$$
m_{\mathrm{RS}}=O\!\left(\frac{1}{\epsilon^{2}\tau}\right),\qquad T_{\mathrm{HBE}}=O\!\left(\frac{d}{\sqrt{\tau}\,\epsilon^{2}}\right),\qquad \text{hashes stored: }O\!\left(\frac{1}{\tau^{3/2}\epsilon^{4}}\right)\ (\text{HBE})\ \text{vs.}\ O\!\left(\frac{1}{\tau\,\epsilon^{2}}\right)\ (\text{Backurs et al.})
$$

tau is a lower bound on the density being estimated and epsilon the relative error. Random-sampling bound from DEANN Lemma 2 / Backurs introduction; HBE query time and storage from Backurs et al. Table 1 and Charikar-Siminelakis Theorem 1.7 (time O(M^3 epsilon^{-2} mu^{-1/2} log(1/delta) T)).

Sources: karppa2022deann, backurs2019space, charikar2017hbe

### Gaussian potential on the sphere, uniformity loss and the vMF-KDE link

$$
G_t(u,v)=e^{-t\lVert u-v\rVert_2^{2}}=e^{2t\,u^{\top}v-2t},\qquad \mathcal L_{\mathrm{uniform}}(f;t)=\log\mathbb{E}_{x,y}\!\left[G_t(f(x),f(y))\right],\qquad \frac{1}{N}\sum_{i=1}^{N}\log\!\Big(\frac{1}{N}\sum_{j=1}^{N}e^{f(x_i)^{\top}f(x_j)/\tau}\Big)=\frac{1}{N}\sum_{i=1}^{N}\log\hat p_{\mathrm{vMF\text{-}KDE}}(f(x_i))+\log Z_{\mathrm{vMF}}
$$

Wang-Isola Section 4: the Gaussian potential equals a vMF kernel on unit vectors, and the uniformity term of the contrastive loss is a resubstitution entropy estimate based on a vMF-KDE with kappa = 1/tau, where Z_vMF is the vMF normalisation constant for that kappa.

Sources: wang2020uniformity

### Concentration of inner products in high dimension

$$
\mathbb{E}[\langle q,X\rangle]=0,\qquad \mathrm{Var}[\langle q,X\rangle]=\mathbb{E}[\langle q,X\rangle^{2}]=\frac{\lVert X\rVert_2^{2}}{d},\qquad \lim_{d\to\infty}P\big(\langle q,X\rangle>\epsilon\big)=0
$$

Bruch Theorem 2.2 and its proof for a unit query and spherically symmetric data with bounded norm. Our simulation gave d times variance between 0.96 and 1.06 for d = 64 and d = 384.

Sources: bruch2024foundations

### Effective number of contributing points for a vMF kernel (our derivation)

$$
w_i=e^{\kappa\,x^{\top}X_i},\quad \log w_i\approx\mathcal N\!\left(0,\tfrac{\kappa^{2}}{d}\right)\ \Rightarrow\ n_{\mathrm{eff}}=\frac{(\sum_i w_i)^{2}}{\sum_i w_i^{2}}\approx n\,\frac{(\mathbb E w)^{2}}{\mathbb E w^{2}}=n\,e^{-\kappa^{2}/d}\quad(\kappa^{2}\lesssim d);\qquad \sqrt{d}\ \lesssim\ \kappa\ \lesssim\ \sqrt{d\ln n}
$$

OUR derivation: with inner-product variance 1/d the log weights are approximately normal with variance s^2 = kappa^2/d, and for a log-normal E[w] = e^{s^2/2}, E[w^2] = e^{2 s^2}. Simulation (d = 384, n = 2000): n_eff/n = 0.779, 0.342, 0.137 at kappa^2/d = 0.25, 1, 2.25 versus predicted 0.779, 0.368, 0.105; for larger kappa the finite-sample value exceeds the prediction but still collapses (13 points at kappa = 3 sqrt(d)). The window is a null-model reference for isotropic data, not a selection rule; real embeddings are clustered and anisotropic.

Sources: bruch2024foundations

### Posterior relevance probability and the 25 percent rule (our proposal)

$$
P(\mathrm{rel}\mid x)=\frac{\pi\,\hat f_R(x)}{\pi\,\hat f_R(x)+(1-\pi)\,\hat f_B(x)}\ \ge\ \tfrac14\iff\frac{\hat f_R(x)}{\hat f_B(x)}\ \ge\ \frac{1-\pi}{3\pi};\qquad P_{\mathrm{path}}(x_m)=\prod_{j=1}^{m}P(\mathrm{rel}\mid x_j)\ \ge\ \tfrac14
$$

OUR proposal from Bayes' rule. f_R is a vMF-KDE centred on the query and the functions accepted so far, f_B the corpus vMF-KDE, pi the prior fraction of relevant functions. Derivation: P >= 1/4 iff 4 pi f_R >= pi f_R + (1-pi) f_B iff 3 pi f_R >= (1-pi) f_B. With equal kappa in both densities the normalising constants cancel, so only log-sum-exp of kappa times cosine similarities is needed. The path product makes the trace terminate. The output is a probability only after calibration of pi and kappa on labelled queries.

Sources: banerjee2005, rossi2024relevance

## Open questions

- What is the intrinsic dimension of function embeddings for the mimalloc fork under the chosen local embedding model, and at which reduced dimension q does mode structure remain stable under bootstrap resampling? No source measures this for code embeddings.
- Which bandwidth selector is reliable between roughly 8 and 64 dimensions? The directional rule of thumb was shown to degrade by q = 12 in simulation, exact-MISE selectors were evaluated only in low dimension, and the temperature-based choice kappa = 1/tau is a heuristic we derive from Wang and Isola rather than a validated selector.
- Do modes of the vMF-KDE correspond to semantically meaningful function families in an allocator code base (for example page management, free lists, OS abstraction), or mainly to syntactic near-duplicates? This needs a labelled evaluation; we found no prior study of density-based grouping of source-code functions.
- How should the prior pi and the relevant-set density f_R be calibrated with very few labelled queries so that the 25 percent threshold is a genuine probability? Published calibrated cut-offs (Cosine Adapter) rely on large supervised training sets.
- Does path-product thresholding prevent query drift during iterative trace expansion, or does re-centring the relevant density on accepted functions still pull retrieval toward dense hub regions? Hubness theory suggests a risk that has not been tested in this setting.
- Embeddings are anisotropic: should the corpus be mean-centred or whitened before projecting to the sphere, and how does that change the vMF-KDE and its modes? The directional KDE theory assumes a smooth density on the sphere but says nothing about preprocessing.
- Published convergence and consistency theorems assume an i.i.d. sample from a smooth density with finitely many isolated modes; a repository is a small, fixed, highly structured population with exact duplicates. How much of the theory transfers is unknown and should be stated as a limitation.
- Three formula details could not be verified from text extraction and must be checked against the PDFs before publication: the radical signs in the Zhang-Chen rates, the closed form of the contraction factor in their Theorem 12, and the constant in eq. (5) of the Chen tutorial.
- Full texts of Fukunaga and Hostetler (1975), Hall, Watson and Cabrera (1987), Bai, Rao and Zhao (1988), Campello et al. (2013), Sra (2012), Rosenblatt (1956) and Parzen (1962) were not machine-readable in this session (paywall or scanned image); only abstracts or metadata were opened, and their formulas are quoted via later papers that were read in full. Primary-text verification is still outstanding.

## Sources

- **rosenblatt1956**: Murray Rosenblatt (1956). Remarks on Some Nonparametric Estimates of a Density Function. The Annals of Mathematical Statistics 27(3), 832-837. DOI 10.1214/aoms/1177728190. https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-3/Remarks-on-Some-Nonparametric-Estimates-of-a-Density-Function/10.1214/aoms/1177728190.full
  - evidence: Project Euclid abstract: 'all estimates of the density function satisfying relatively mild conditions are shown to be biased'; the asymptotic mean square error of a class of estimates is evaluated. The full text PDF is a scanned image and was not machine-readable here, so only the abstract and metadata are relied on.
- **parzen1962**: Emanuel Parzen (1962). On Estimation of a Probability Density Function and Mode. The Annals of Mathematical Statistics 33(3), 1065-1076. DOI 10.1214/aoms/1177704472. https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-3/On-Estimation-of-a-Probability-Density-Function-and-Mode/10.1214/aoms/1177704472.full
  - evidence: Project Euclid record confirms title, author, volume 33 issue 3, September 1962, pages 1065-1076 and DOI. Full text is a scanned image (not machine-readable here). Zhang and Chen (2021) cite it with Rosenblatt as the origin: 'the kernel density estimator (KDE; Rosenblatt 1956; Parzen 1962)'.
- **scipy_gaussian_kde**: The SciPy community (2026). scipy.stats.gaussian_kde (SciPy API reference). Official SciPy documentation (accessed 2026-09-29). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
  - evidence: Notes section: Scott's Rule 'n**(-1./(d+4))', Silverman's Rule '(n * (d + 2) / 4.)**(-1. / (d + 4))'; covariance attribute: 'the data covariance matrix multiplied by the square of the bandwidth factor'; 'The estimation works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed.' References Scott (1992) and Silverman (1986).
- **chen2017tutorial**: Yen-Chi Chen (2017). A Tutorial on Kernel Density Estimation and Recent Advances. arXiv:1704.03924v2 [stat.ME]. https://arxiv.org/pdf/1704.03924
  - evidence: Eq. (1) defines p_n(x) = (1/(n h^d)) sum K((x - X_i)/h); eq. (4) gives the MISE with bias term (h^4/4) sigma_K^4 int |Laplacian p|^2 and variance term mu_K/(n h^d). Open problems: 'KDE ... suffers severely from the so-called curse of dimensionality: The optimal convergence rate O(n^{-2/(d+4)}) is very slow when d is large, and this slow convergence rate cannot be improved (Stone, 1982; Tsybakov, 2009) unless we assume extra smoothness.' Also defines balloon and sample smoothing estimators.
- **stone1980**: Charles J. Stone (1980). Optimal Rates of Convergence for Nonparametric Estimators. The Annals of Statistics 8(6), 1348-1360. https://sites.stat.washington.edu/courses/stat527/s14/readings/Stone_Annals_1980.pdf
  - evidence: Abstract: 'It is shown that r = (p - m)/(2p + d) is the optimal (uniform) rate of convergence for a sequence of estimators ... An analogous result is obtained for nonparametric estimators of a density function.' Section 3 treats estimation of a density function with gamma = 1/(2p + d).
- **hall1987**: Peter Hall, G. S. Watson, Javier Cabrera (1987). Kernel density estimation with spherical data. Biometrika 74(4), 751-762. DOI 10.1093/biomet/74.4.751. https://academic.oup.com/biomet/article/74/4/751/280530
  - evidence: Abstract page: two classes of kernel density estimators for spherical data are studied; the second class includes estimators that 'out-perform all those in the first class, if loss is measured in either squared-error or Kullback-Leibler senses'; explicit bias, variance and loss formulas are derived. Only the abstract was accessible; the estimator formula and cross-validation selectors are taken from García-Portugués (2013), who attributes them to this paper.
- **bai1988**: Z. D. Bai, C. Radhakrishna Rao, L. C. Zhao (1988). Kernel estimators of density function of directional data. Journal of Multivariate Analysis 27(1), 24-39. DOI 10.1016/0047-259X(88)90113-3. https://ideas.repec.org/a/eee/jmvana/v27y1988i1p24-39.html
  - evidence: Abstract (RePEc record): 'The proposed estimator is of the form fn(x) = (nh^{k-1})^{-1} C(h) sum_{i=1}^n K[(1-x'Xi)/h^2] ... Conditions are imposed on K and f to prove pointwise strong consistency, uniform strong consistency, and strong L1-norm consistency of fn as an estimator of f.' Bibliographic data cross-checked with the Crossref record.
- **garciaportugues2013**: Eduardo García-Portugués (2013). Exact risk improvement of bandwidth selectors for kernel density estimation with directional data. Electronic Journal of Statistics 7, 1655-1685. DOI 10.1214/13-EJS821 (arXiv:1306.0517v5). https://arxiv.org/abs/1306.0517
  - evidence: Eq. (1) vMF density with C_q(kappa) = kappa^{(q-1)/2}/((2 pi)^{(q+1)/2} I_{(q-1)/2}(kappa)); eq. (2) directional KDE; 'the kernel estimator can be viewed as a mixture of von Mises-Fisher densities ... the common concentration parameter is given by 1/h^2'; Proposition 1 (MISE), Corollary 1 (h_AMISE), Proposition 2 (rule of thumb, eq. 6). 'the AMISE and MISE may differ significantly for moderate and even large sample sizes'. Exact LaTeX confirmed from the ar5iv rendering.
- **banerjee2005**: Arindam Banerjee, Inderjit S. Dhillon, Joydeep Ghosh, Suvrit Sra (2005). Clustering on the Unit Hypersphere using von Mises-Fisher Distributions. Journal of Machine Learning Research 6, 1345-1382. https://jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
  - evidence: Eqs. (2.1)-(2.2) density and c_d(kappa) = kappa^{d/2-1}/((2 pi)^{d/2} I_{d/2-1}(kappa)); (2.4)-(2.5) MLE; (3.3)-(3.7) EM updates; (4.4) kappa = (r_bar d - r_bar^3)/(1 - r_bar^2); Table 1 e.g. (d=10, r_bar=0.633668, kappa=10) gives 10.1631. 'for high dimensional data, problems of overflows and numerical instabilities plague such root-finders.' Section 7: the Fisher kernel of the vMF 'is exactly the cosine similarity'; spherical kmeans arises as a special case of EM on a mixture of vMFs.
- **sra2012**: Suvrit Sra (2012). A short note on parameter approximation for von Mises-Fisher distributions: and a fast implementation of I_s(x). Computational Statistics 27(1), 177-190. DOI 10.1007/s00180-011-0232-x. https://link.springer.com/article/10.1007/s00180-011-0232-x
  - evidence: Abstract: 'Maximum likelihood estimation for the vMF distribution turns out to be surprisingly hard because of a difficult transcendental equation that needs to be solved for computing the concentration parameter kappa ... one can achieve more accurate approximations using a truncated Newton method'; also presents a fast algorithm for the modified Bessel function for large arguments common to high-dimensional data. Only abstract and metadata were accessible.
- **fukunaga1975**: K. Fukunaga, L. Hostetler (1975). The estimation of the gradient of a density function, with applications in pattern recognition. IEEE Transactions on Information Theory 21(1), 32-40. DOI 10.1109/TIT.1975.1055330. https://api.crossref.org/works/10.1109/TIT.1975.1055330
  - evidence: Crossref record confirms title, authors, journal, volume 21, issue 1, pages 32-40, January 1975. The publisher pages were not accessible (HTTP 403 / empty), so content claims rest on Cheng (1995): 'The difference m(x) - x is called mean shift in Fukunaga and Hostetler' and 'Fukunaga and Hostetler is still the first work proposing mean shift explicitly as an iterative algorithm'.
- **cheng1995**: Yizong Cheng (1995). Mean Shift, Mode Seeking, and Clustering. IEEE Transactions on Pattern Analysis and Machine Intelligence 17(8), 790-799. https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/meanShiftCluster.pdf
  - evidence: Abstract: 'It is shown that mean shift is a mode-seeking process on a surface constructed with a "shadow" kernel. For Gaussian kernels, mean shift is a gradient mapping.' Theorem 1: H is a shadow of K iff the profiles satisfy h'(r) = -c k(r). Theorem 2: 'The only kernels that are their own shadows are the Gaussian kernel and its truncated version.'
- **comaniciu2002**: Dorin Comaniciu, Peter Meer (2002). Mean Shift: A Robust Approach Toward Feature Space Analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence 24(5), 603-619. https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf
  - evidence: Theorem 1: 'If the kernel K has a convex and monotonically decreasing profile, the sequences {y_j} and {f_{h,K}(j)} converge and {f_{h,K}(j)} is monotonically increasing.' Discussion: 'Kernel density estimation ... do not scale well with the dimension of the space ... whenever the feature space has more than (say) six dimensions, the analysis should be approached carefully.'
- **comaniciu2001variable**: Dorin Comaniciu, Visvanathan Ramesh, Peter Meer (2001). The Variable Bandwidth Mean Shift and Data-Driven Scale Selection. Proc. Eighth IEEE International Conference on Computer Vision (ICCV), vol. I, 438-445. https://www.dgp.toronto.edu/~jay/MS%20bandwidth.pdf
  - evidence: 'Employing the sample point estimator, we define the Variable Bandwidth Mean Shift, prove its convergence, and show its superiority over the fixed bandwidth procedure.' 'when h(x_i) is taken to be reciprocal to the square root of f(x_i) the bias becomes proportional to h^4, while the variance remains unchanged'; balloon estimators with k-NN bandwidth show insignificant improvement and 'usually fail to integrate to one'.
- **zhang2021directional**: Yikun Zhang, Yen-Chi Chen (2021). Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data. Journal of Machine Learning Research 22(154), 1-92 (arXiv:2010.13523). https://arxiv.org/abs/2010.13523
  - evidence: Eqs. (2)-(4), (20), (23), (31); Theorem 8 (ascending property for L 'monotonically decreasing, differentiable, and convex with L(0) < infinity'), Theorems 11-12; Section 5.3: 'the time complexity of Algorithm 1 is O(n x q) for one iteration ... total ... O(n^2 x q x log(1/epsilon))'. Section 6.1: 'As the dimension q of directional data becomes larger, the misclassification rates of mode clustering with Algorithm 1 also gradually increase to 1 (the worst case)'. 'the algorithm does not need to estimate the normalizing constant'. Code: github.com/zhangyk8/DirMS.
- **zhang2021em**: Yikun Zhang, Yen-Chi Chen (2021). The EM Perspective of Directional Mean Shift Algorithm. arXiv:2101.10058v1 [math.ST]. https://arxiv.org/abs/2101.10058
  - evidence: Abstract: 'any DMS iteration can be viewed as a generalized Expectation-Maximization (EM) algorithm; in particular, when the von Mises kernel is applied, it becomes an exact EM algorithm.' Theorem 4: under (C1) and (C2) the sequence 'converges to a local mode of f_h from almost all starting point'. (C2): L 'non-increasing, differentiable, and convex with 0 < L(0) < infinity'.
- **yamasaki2023convergence**: Ryoya Yamasaki, Toshiyuki Tanaka (2023). Convergence Analysis of Mean Shift. arXiv:2305.08463v3 [stat.ML]. https://arxiv.org/abs/2305.08463
  - evidence: 'the proof in [3] used an incorrect inequality evaluation to claim that the mode estimate sequence is a Cauchy sequence; see counterexamples given in [25]' ([3] = Comaniciu and Meer 2002, [25] = Li, Hu and Wu 2007). Conclusion: convergence proved for a C^1 subanalytic kernel; 'Our proof does neither presume that the KDE has a finite number of critical points or they are isolated, nor that its Hessian at a convergent point is non-degenerate, nor restriction on the size of the bandwidth or on the data dimension'.
- **saavedra2024directional**: Paula Saavedra-Nieves, Martín Fernández-Pérez (2024). Directional density-based clustering. Advances in Data Analysis and Classification 20(1), 289-314 (online 11 Dec 2024; issue dated March 2026). DOI 10.1007/s11634-024-00616-3; arXiv:2302.09897. https://link.springer.com/article/10.1007/s11634-024-00616-3
  - evidence: Abstract: 'directional density-based clustering methodology is fully established for the unit hypersphere by solving the computational problems associated to high dimensional spaces.' Eqs. (4)-(6) define the HDR L_f(tau) and its plug-in estimate with the vMF-kernel KDE; the threshold 'could be estimated as the tau-quantile of the empirical distribution of {f_n(X_1), ..., f_n(X_n)}'. Full text read from the arXiv version.
- **campello2013hdbscan**: Ricardo J. G. B. Campello, Davoud Moulavi, Joerg Sander (2013). Density-Based Clustering Based on Hierarchical Density Estimates. Advances in Knowledge Discovery and Data Mining (PAKDD 2013), Springer, 160-172. DOI 10.1007/978-3-642-37456-2_14. https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14
  - evidence: Abstract: 'We propose a theoretically and practically improved density-based, hierarchical clustering method ... we propose a novel cluster stability measure, formalize the problem of maximizing the overall stability of selected clusters, and formulate an algorithm that computes an optimal solution to this problem.' Only abstract and metadata were accessible; algorithmic formulas are taken from McInnes and Healy (2017).
- **mcinnes2017accelerated**: Leland McInnes, John Healy (2017). Accelerated Hierarchical Density Clustering. arXiv:1705.07321v2 [stat.ML]. https://arxiv.org/abs/1705.07321
  - evidence: Defines core distance ('the distance to the kth nearest neighbor'), mutual reachability distance max{kappa(Xi), kappa(Xj), d}, lambda = 1/epsilon and stability sigma(Ci). 'the HDBSCAN* algorithm on N data points has O(N^2) run-time'; with dual-tree Boruvka 'can be expected to approach O(N log N) performance for many data sets'.
- **mcinnes2017hdbscan**: Leland McInnes, John Healy, Steve Astels (2017). hdbscan: Hierarchical density based clustering. Journal of Open Source Software 2(11), 205. DOI 10.21105/joss.00205. https://www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf
  - evidence: 'Performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more robust to parameter selection.'
- **rodriguez2014density**: Alex Rodriguez, Alessandro Laio (2014). Clustering by fast search and find of density peaks. Science 344(6191), 1492-1496. DOI 10.1126/science.1242072. https://sites.psu.edu/mcnl/files/2017/03/9-2dhti48.pdf
  - evidence: 'cluster centers are surrounded by neighbors with lower local density and ... are at a relatively large distance from any points with a higher local density'; eqs. (1)-(2) define rho_i and delta_i; 'As a rule of thumb, one can choose d_c so that the average number of neighbors is around 1 to 2% of the total number of points'; for small sets 'the density estimator is unavoidably affected by large statistical errors'.
- **beyer1999nearest**: Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, Uri Shaft (1999). When Is "Nearest Neighbor" Meaningful?. Database Theory - ICDT'99, Springer, 217-235. DOI 10.1007/3-540-49257-7_15. https://link.springer.com/chapter/10.1007/3-540-49257-7_15
  - evidence: Abstract: 'under a broad set of conditions (much broader than independent and identically distributed dimensions), as dimensionality increases, the distance to the nearest data point approaches the distance to the farthest data point ... this effect can occur for as few as 10-15 dimensions. These results should not be interpreted to mean that high-dimensional indexing is never meaningful'.
- **radovanovic2010hubs**: Milos Radovanovic, Alexandros Nanopoulos, Mirjana Ivanovic (2010). Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data. Journal of Machine Learning Research 11, 2487-2531. https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf
  - evidence: 'as dimensionality increases, the distribution of N_k becomes considerably skewed to the right, resulting in the emergence of hubs'; the paper links hubness with intrinsic dimensionality, 'demonstrating that dimensionality reduction may not constitute an easy mitigation of the phenomenon'; 'in epsilon-neighborhood graphs ... the hubness phenomenon does not occur'.
- **bruch2024foundations**: Sebastian Bruch (2024). Foundations of Vector Retrieval. arXiv:2401.09350v1 [cs.DS] (monograph). https://arxiv.org/abs/2401.09350
  - evidence: Theorem 2.2: for a unit query and bounded-norm data drawn from a spherically symmetric distribution, lim_{d->inf} P(<q,X> > epsilon) = 0; proof shows E[<q,X>^2] = ||X||^2/d. 'if our data points are clustered in (near) orthogonal subspaces, then approximate MIPS over clusters makes sense'. Chapter 3: data 'actually lie on or near a low-dimensional manifold and, as such, have a low intrinsic dimensionality'.
- **dasgupta2003jl**: Sanjoy Dasgupta, Anupam Gupta (2003). An Elementary Proof of a Theorem of Johnson and Lindenstrauss. Random Structures and Algorithms 22, 60-65. DOI 10.1002/rsa.10073. https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf
  - evidence: Theorem 2.1: 'For any 0 < epsilon < 1 and any integer n, let k be a positive integer such that k >= 4(epsilon^2/2 - epsilon^3/3)^{-1} ln n. Then for any set V of n points in R^d, there is a map f: R^d -> R^k such that for all u, v in V, (1 - epsilon)||u - v||^2 <= ||f(u) - f(v)||^2 <= (1 + epsilon)||u - v||^2.'
- **kim2019uniform**: Jisu Kim, Jaehyeok Shin, Alessandro Rinaldo, Larry Wasserman (2019). Uniform Convergence Rate of the Kernel Density Estimator Adaptive to Intrinsic Volume Dimension. Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 3398-3407 (arXiv:1810.05935). https://proceedings.mlr.press/v97/kim19e.html
  - evidence: 'We first propose a novel concept, called the volume dimension, to measure the intrinsic dimension of the support of a probability distribution ... Our bounds depend on the volume dimension'; 'For a bounded density on a d_M-dimensional manifold we obtain, under appropriate conditions, that d_vol = d_M.' Full text read from arXiv v3.
- **bae2026manifolds**: Eunseong Bae, Wolfgang Polonik (2026). Kernel smoothing on manifolds. arXiv:2601.16777v1 [math.ST]. https://arxiv.org/abs/2601.16777
  - evidence: Abstract: 'we derive finite sample bounds for kernel smoothing and its (first and second) derivatives, and we establish asymptotic normality through Berry-Esseen type bounds. Special cases include kernel density estimation'. After Theorem 3.6: 'Note that the upper bound depends on the dimension d of M, rather than the dimension of the ambient space R^D.' Kernel weights use ambient Euclidean distance.
- **charikar2017hbe**: Moses Charikar, Paris Siminelakis (2017). Hashing-Based-Estimators for Kernel Density in High Dimensions. FOCS 2017 (preliminary version); full version arXiv:1808.10530. https://arxiv.org/abs/1808.10530
  - evidence: Abstract: 'We introduce a class of unbiased estimators for kernel density implemented through locality-sensitive hashing ... Our work is the first to provide data-structures with theoretical guarantees that improve upon simple random sampling in high dimensions.' Theorem 1.7: time O(M^3 (1/epsilon^2)(1/sqrt(mu)) log(1/delta) T).
- **backurs2019space**: Arturs Backurs, Piotr Indyk, Tal Wagner (2019). Space and Time Efficient Kernel Density Estimation in High Dimensions. 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). https://proceedings.neurips.cc/paper_files/paper/2019/file/a2ce8f1706e52936dfad516c23904e3e-Paper.pdf
  - evidence: 'a naive exact algorithm for this problem runs in a rectangular O(|X||Y|) time'; 'Tree-based techniques ... running times are exponential in d'; random sample of O(1/tau . 1/epsilon^2) points suffices; 'we present an improvement to their framework that retains the same query time, while requiring only linear space and linear preprocessing time.' Table 1 compares query time and stored hashes.
- **siminelakis2019rehashing**: Paris Siminelakis, Kexin Rong, Peter Bailis, Moses Charikar, Philip Levis (2019). Rehashing Kernel Evaluation in High Dimensions. Proceedings of the 36th International Conference on Machine Learning, PMLR 97. https://proceedings.mlr.press/v97/siminelakis19a/siminelakis19a.pdf
  - evidence: 'Despite worst-case theoretical guarantees, RS outperforms HBE on certain datasets in practice ... We propose a diagnostic procedure utilizing the variance bounds that, for a given dataset, chooses at runtime with minimal overhead the better of the two approaches'; 'up to 10x improvement in evaluation time'.
- **karppa2022deann**: Matti Karppa, Martin Aumüller, Rasmus Pagh (2022). DEANN: Speeding up Kernel-Density Estimation using Approximate Nearest Neighbor Search. Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 151, 3108-3137 (arXiv:2107.02736). https://proceedings.mlr.press/v151/karppa22a.html
  - evidence: 'The idea is to find points that have a large contribution to the KDE using ANN, compute their contribution exactly, and approximate the remainder with Random Sampling (RS).' Algorithm 1 returns (k/n) Z1 + ((n-k)/n) Z2; 'The estimate is unbiased no matter the quality of the near neighbors returned'; 'we provide a C++ implementation with a Python interface'; limitation: no theoretically grounded way to choose k and m except grid search.
- **wang2020uniformity**: Tongzhou Wang, Phillip Isola (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. Proceedings of the 37th International Conference on Machine Learning, PMLR 119 (arXiv:2005.10242). https://arxiv.org/abs/2005.10242
  - evidence: Defines G_t(u,v) = e^{-t||u-v||^2} = e^{2t u^T v - 2t} and L_uniform; 'the second term in Equation (2) can be alternatively viewed as a resubstitution entropy estimator of f(x) ... via a von Mises-Fisher (vMF) kernel density estimation (KDE)' where 'p_vMF-KDE is the KDE based on samples {f(x_j)} using a vMF kernel with kappa = tau^{-1}'.
- **ethayarajh2019contextual**: Kawin Ethayarajh (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. arXiv:1909.00512v1 [cs.CL]. https://arxiv.org/abs/1909.00512
  - evidence: 'In all layers of all three models, the contextualized word representations of all words are not isotropic: they are not uniformly distributed with respect to direction. Instead, they are anisotropic, occupying a narrow cone in the vector space.'
- **kusupati2022matryoshka**: Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi (2022). Matryoshka Representation Learning. 36th Conference on Neural Information Processing Systems (NeurIPS 2022); arXiv:2205.13147v4. https://arxiv.org/abs/2205.13147
  - evidence: 'MRL learns coarse-to-fine representations that are at least as accurate and rich as independently trained low-dimensional representations'; 'MRL learns representations of varying capacities within the same high-dimensional vector through explicit optimization of O(log(d)) lower-dimensional vectors in a nested fashion'; 'Up to 14x faster yet accurate large-scale classification and retrieval'.
- **mcinnes2018umap**: Leland McInnes, John Healy, James Melville (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426v3 [stat.ML]. https://arxiv.org/abs/1802.03426
  - evidence: Section 3.1: rho_i is the distance to the nearest neighbour and sigma_i is set so that sum_{j=1}^k exp(-max(0, d(x_i, x_ij) - rho_i)/sigma_i) = log2(k); 'this significantly improves the representation on very high dimensional data where other algorithms such as t-SNE begin to suffer from the curse of dimensionality.'
- **grootendorst2022bertopic**: Maarten Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794v1 [cs.CL]. https://arxiv.org/abs/2203.05794
  - evidence: 'As data increases in dimensionality, distance to the nearest data point has been shown to approach the distance to the farthest data point ... a more straightforward approach is to reduce the dimensionality of embeddings.' UMAP is used for reduction and 'The reduced embeddings are clustered using HDBSCAN', which models noise as outliers.
- **sarthi2024raptor**: Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR 2024 (arXiv:2401.18059). https://arxiv.org/abs/2401.18059
  - evidence: 'recursively embedding, clustering, and summarizing chunks of text'; 'soft clustering, where nodes can belong to multiple clusters'; 'Our clustering algorithm is based on Gaussian Mixture Models'; 'The high dimensionality of vector embeddings presents a challenge for traditional GMMs ... we employ ... UMAP'; BIC selects the number of clusters; 'the Gaussian assumption in GMMs may not perfectly align with the nature of text data'.
- **xu2025car**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan (2025). Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications. arXiv:2511.14769v1 [cs.IR]. https://arxiv.org/abs/2511.14769
  - evidence: 'CAR ... dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances'; clustering back-ends include HDBSCAN, OPTICS, BisectingKMeans chosen by silhouette score; Table 4 scores are similar across algorithms (0.58-0.63 on CDP, 0.30-0.34 on MultiHop-RAG). Corpus includes code and documentation.
- **xia2026qcar**: Ye Xia, Emre Yamangil, Haixun Wang (2026). Pre-retrieval Query Clustering for Adaptive Top-k Document Retrieval in RAG Systems. arXiv:2609.13489v2 [cs.IR]. https://arxiv.org/abs/2609.13489
  - evidence: 'RAG systems commonly retrieve a fixed number of documents (top-k) to ground generation, but this static approach is brittle: simple queries suffer over-retrieval ... while complex queries are under-retrieved'; 'we cluster a large set of queries in embedding space and summarize each cluster with a recommended retrieval depth that targets high coverage (e.g., ~95%) using a mean-plus-variance rule.'
- **rossi2024relevance**: Nicholas Rossi, Juexin Lin, Feng Liu, Zhen Yang, Tony Lee, Alessandro Magnani, Ciya Liao (2024). Relevance Filtering for Embedding-based Retrieval. CIKM 2024. DOI 10.1145/3627673.3680095 (arXiv:2408.04887). https://arxiv.org/abs/2408.04887
  - evidence: 'dense retrieval via ANN search has no natural cutoff'; 'the cosine similarity scores are usually not interpretable and should not be compared across different queries. Thus, applying a global threshold to the cosine similarity score is not an optimal approach'; the Cosine Adapter 'maps raw cosine similarity scores to interpretable scores using a query-dependent mapping function. We then apply a global threshold on the mapped scores'.
- **kong2018recurrent**: Shu Kong, Charless Fowlkes (2017). Recurrent Pixel Embedding for Instance Grouping. arXiv:1712.08273v1 [cs.CV]. https://arxiv.org/abs/1712.08273
  - evidence: 'we regress pixels into a hyper-spherical embedding space ... to group instances, we utilize a variant of mean-shift clustering, implemented as a recurrent neural network parameterized by kernel bandwidth'; 'we instead use the von Mises-Fisher distribution ... K(x, x_i) proportional to exp(delta x^T x_i)'; the kernel is recomputed each iteration (blurring mean shift).
- **choi2024cms**: Sua Choi, Dahyun Kang, Minsu Cho (2024). Contrastive Mean-Shift Learning for Generalized Category Discovery. arXiv:2404.09451v1 [cs.CV]. https://arxiv.org/abs/2404.09451
  - evidence: 'we revisit the mean-shift algorithm ... and incorporate it into a contrastive learning framework'; 'the number of neighbors within a fixed radius varies significantly during the update of the encoder ... we replace the distance-based NNs with kNNs, which greatly improves the stability'; the weight kernel 'can be interpreted as a simple approximation of a Gaussian kernel with adaptive bandwidth'.
- **ricco2025hallucination**: Emanuele Ricco, Lorenzo Cima, Roberto Di Pietro (2025). Hallucination Detection: A Probabilistic Framework Using Embeddings Distance Analysis. arXiv:2502.08663v1 [cs.CL]. https://arxiv.org/abs/2502.08663
  - evidence: 'fitting probability densities using Gaussian Kernel Density Estimation (KDE), separately for hallucinated and non-hallucinated data ... computed log-likelihoods under each KDE model'; reported accuracy 66% for a specific configuration, i.e. a modest effect size.
- **kim2021vmfpytorch**: Minyoung Kim (2021). On PyTorch Implementation of Density Estimators for von Mises-Fisher and Its Mixture. arXiv:2102.05340v1 [cs.LG]. https://arxiv.org/abs/2102.05340
  - evidence: 'implementation of vMF typically suffers from the notorious numerical issue of the Bessel function evaluation in the density normalizer, especially when the dimensionality is high'; scipy.special.iv underflows for order 100 at x = 0.03; footnote: with SciPy's Bessel 'the results become unstable when the dimension is greater than 50.'
