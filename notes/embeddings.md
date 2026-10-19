# Embedding models and vector compression for code retrieval

Verified research note for the programmersRAG C++20 rewrite.
Verification date: 2026-09-29. Evaluation corpus: a clone of microsoft/mimalloc (C, about 41 `.c` and 18 `.h` files, about 1.3 MB).
Target machine: Intel i9-12950HX (8 P-cores with 48 KB L1d, 8 E-cores with 32 KB L1d, 24 logical CPUs), 32 GB RAM, RTX A1000 4 GB.

How this note was produced. A first researcher compiled a dossier. Every source in it was then re-opened independently (arXiv abstract page plus the arXiv HTML full text, publisher or proceedings page, Hugging Face raw model card and file API, GitHub raw source). Every number quoted below was located in the opened text. Every formula was checked against the source and, where it is a derivation, re-derived and tested numerically. Square-bracket numbers refer to the reference list at the end.

Status tags used below:

- **[EST]** established, peer-reviewed or widely replicated result.
- **[REC]** recent result (2024-2026), often a preprint or a vendor model card; treat numbers as provisional.
- **[CON]** contested: sources disagree, or the claim depends on who evaluated it.
- **[OURS]** our own proposal, derivation or measurement. Not a published result. Must be labelled as such in the paper.

---

## Summary

1. **Model choice.** Qwen3-Embedding-0.6B [1][2][3] remains the recommended primary embedder, but for a narrower reason than the first dossier gave. It is the only sub-1B candidate that has all of: an Apache-2.0 licence, an official GGUF, a pooling type that llama.cpp implements natively (last-token), and 1024-d output. It is **not** the strongest small model on repository-level retrieval: in the full CORE-Bench tables jina-code-embeddings-0.5b beats it on both repository-level tiers [22], and in CoREB it is well behind code-specialised 0.5B models [24]. jina-code-embeddings-0.5b is CC-BY-NC-4.0 [5], which is the deciding constraint.
2. **Published C-language evidence exists and is sobering.** CORE-Bench reports per-language results that include C. On issue-to-edit localisation in C repositories, Qwen3-Embedding-0.6B scores NDCG@10 11.8, essentially equal to BM25 at 11.9 (the paper reports no significance test); in-domain fine-tuning of the same model raises it to 19.0 [22]. A lexical channel and an embedder fine-tuning option are therefore justified by published data, not only by intuition.
3. **Benchmark numbers are evaluator-dependent.** The same Qwen3-Embedding-0.6B checkpoint is reported at 75.41 [1], 75.42 [8], 74.69 [4] and 73.75 [9] on MTEB(Code). Differences below about 2 points between small models are not decisive. Several competing models train on the training splits of the same benchmark tasks [4][8][9].
4. **Compression.** RaBitQ [35] and its multi-bit extension [36] are the right family for this project because they need no codebook training and give a probabilistic error statement per vector. Product quantisation is a poor fit for a corpus of roughly 500 to 1,000 vectors [32][34].
5. **The error bound must be used correctly.** The half-width obtained by substituting RaBitQ's practical constant $\epsilon_0 = 1.9$ is a 1.9-standard-deviation interval, which covers about 94% of cases, not 99.9%. We derived the error variance in closed form and confirmed it by simulation. A 99.9% band at $D = 1024$ is about 0.078 for near-orthogonal pairs, consistent with the published conservative empirical constant 0.0898 [36].
6. **Geometry.** Anisotropy [43][44][45] and hubness [49][50] are real and measurable, but they are distinct effects [52], and the published hub-reduction result for centring applies to the centred inner product **without** re-normalisation [50]. RaBitQ's centring is an algebraic reparametrisation of the raw metric and does not by itself reduce hubs.
7. **Scale honesty.** At this corpus size the whole 1-bit index is 5 to 9 cache blocks. Quantisation and multi-threaded scanning cannot produce a user-visible speed-up here. They are justified as a design that scales and as a vehicle for the L1-resident block experiment, and the paper must say so.

### Corrections made during verification

| # | First dossier said | Verified position |
|---|---|---|
| C1 | Qwen3-Embedding-0.6B is "the best sub-1B model" on CORE-Bench | True only of main-text Table 3, which lists three small models. Appendix Tables 16-17 list jina-code-embeddings-0.5b at L1 63.5/93.0, L2 20.8/51.9, L3 36.7/45.5 versus Qwen3-0.6B at 66.9/94.7, 17.0/45.5, 32.6/40.2 [22]. Jina wins L2 and L3. |
| C2 | "The only C-family evidence found is C++ code-to-code" | CORE-Bench Tables 19-20 give per-language results including C [22]. |
| C3 | 1-bit RaBitQ score is an interval of half-width 0.045 usable as the rescoring band | 0.0446 is a 1.9 sigma band, about 94% two-sided coverage at $\rho = 0$. Use $z\,\sigma(\hat s)$ with an explicit $z$ (formula F10). |
| C4 | The half-width is a "cosine-estimate" error | The bound is on the inner product of centred, normalised vectors. For raw cosine the error is multiplied by $r_o r_q < 1$ (formula F12). |
| C5 | Centring "provably and empirically reduces hubs for inner-product/cosine similarity" | Suzuki et al. analyse and recommend the centred **inner product**, stating "we do not normalize these vectors" and "not distance or cosine" [50]. |
| C6 | PQ "has no theoretical error bound" | Jegou et al. prove $\mathrm{MSDE}(q) \le \mathrm{MSE}(q)$, a data-dependent bound in expectation [32]. What PQ lacks is a distribution-free per-vector probabilistic bound; that stronger statement is the RaBitQ authors' [35]. |
| C7 | Jina InfoNCE formula reproduced "as printed" | As printed it sums over all pairs $(i,j)$ with indices $0..n$, which is not a contrastive loss. Standard form given in F2. |
| C8 | All-but-the-top reproduced as printed | Algorithm 1 as printed projects the uncentred $v(w)$. The centred-consistent form differs by a constant vector, which matters if vectors are re-normalised afterwards (F18). |
| C9 | Calibrated int8 uses round-to-nearest then clip to $[-128,127]$ | The reference implementation uses floor, clips to $[0,255]$, then subtracts 128 [29] (F16). |
| C10 | Jina paper venue "arXiv" | Accepted at the NeurIPS 2025 Workshop on Deep Learning for Code; arXiv v2 dated 17 Aug 2026 [4]. |
| C11 | Mu, Bhat and Viswanath, ICLR 2018 | The arXiv record lists three authors; the ICLR 2018 camera-ready PDF (arXiv v2) lists Jiaqi Mu and Pramod Viswanath only [47]. |
| C12 | cAST "reports +4.3 Recall@5" | 4.3 is the maximum over retrievers; the paper's range is 1.8 to 4.3 [55]. |
| C13 | CoREB: short keyword queries collapse every model | True as reported, but the Search subtask has 13 queries (release v202603) and 25 (v202602) [24]. CoREB also shares four authors with C2LLM, the model it ranks highest among open weights. |
| C14 | EmbeddingGemma is "6-7 points lower" | 6.65 lower by the authors' numbers (68.76 vs 75.41), 4.94 lower in LightOn's re-evaluation (68.81 vs 73.75) [1][6][9]. |
| C15 | Scan uses "AND/XOR + popcount"; two side values per vector | RaBitQ uses bitwise-AND and popcount only, and needs the popcount of each code, which can be precomputed as a third side value [35]. |
| C16 | Huerga et al. measured int8 "without rescoring" | The paper does not evaluate rescoring at all; it tests two small models (384-d and 768-d) [30]. |
| C17 | Venuta et al. percentages | They are Precision@50 (Precision@20 on MultiPL-E) for functional-equivalence code-to-code retrieval; quoted ranges span the 600M and 8B Qwen3 models [23]. |
| C18 | RaBitQ versus TurboQuant read from one side only | TurboQuant's own abstract [39] and an independent third-party comparison [40] were opened and are summarised below. |

No source from the first dossier failed verification. Ten sources were added.

---

## Established results

### E1. Benchmarks for code retrieval

- **[EST]** CodeSearchNet contains about 6 million functions in six languages (Go, Java, JavaScript, PHP, Python, Ruby), of which 2 million have automatically derived natural-language queries, plus 99 expert queries with about 4k relevance annotations. C is not among its languages [21].
- **[EST]** CoIR (ACL 2025 main, pages 22074-22091) integrates 10 datasets, 8 sub-tasks, 4 task families and 14 programming languages, scored by nDCG@10. In the original evaluation the best mean was 56.26 (Voyage-Code-002), with E5-Mistral 55.18, E5-base 50.90, BGE-M3 39.31 and BM25 29.79. All open-source dense models were run with input length 512 [19].
- **[EST]** MTEB(Code) in MMTEB (ICLR 2025) consists of 12 retrieval tasks: AppsRetrieval, COIRCodeSearchNetRetrieval, CodeEditSearchRetrieval, CodeFeedbackMT, CodeFeedbackST, CodeSearchNetCCRetrieval, CodeSearchNetRetrieval, CodeTransOceanContest, CodeTransOceanDL, CosQA, StackOverflowQA and SyntheticText2SQL [20].

### E2. Matryoshka representation learning

- **[EST]** MRL (NeurIPS 2022) trains one $d$-dimensional vector so that each prefix $z_{1:m}$, for $m$ in a set $\mathcal{M}$ with $|\mathcal{M}| \le \lfloor \log d \rfloor$, is itself a usable representation. The loss is the sum of the task loss over the nested sizes with weights $c_m = 1$. The paper states it "imposes no additional cost during inference and deployment" and reports up to 14x smaller embeddings at equal ImageNet-1K accuracy [27].

### E3. Product quantisation

- **[EST]** PQ splits a vector into $m$ sub-vectors, quantises each with a $k^*$-centroid k-means codebook, gives code length $m \log_2 k^*$, and estimates distances asymmetrically from a per-query $m \times k^*$ table. The paper states that $k^* = 256$, $m = 8$ "is often a reasonable choice", and proves that the mean squared distance error is bounded by the quantiser's mean squared error [32].
- **[EST]** The Faiss k-means implementation warns when fewer than 39 training points per centroid are supplied (`min_points_per_centroid = 39`), which is 9,984 points for $k^* = 256$ [34]. The Faiss paper defines scalar and product quantisers as used here [33].

### E4. Geometry of transformer embeddings

- **[EST]** Anisotropy. Contextual representations occupy a narrow cone. For an isotropic space the mean cosine of random pairs would be 0; in GPT-2's last layer it is close to 1 [43]. The effect was first attributed to likelihood training with weight tying [44]. It is also observed with other objectives and in other modalities, which suggests it is inherent to Transformer self-attention [45].
- **[EST]** Cheap linear remedies. All-but-the-top subtracts the mean and removes the top $D \approx d/100$ principal components [47]. Whitening sets the mean to zero and the covariance to identity and can also reduce dimensionality [48]. Both were validated on word or sentence similarity tasks, not on code retrieval.
- **[EST]** Hubness. In high dimensions the distribution of $k$-occurrences $N_k$ becomes strongly right-skewed; points closer to the data mean tend to become hubs; the effect is "an inherent property of data distributions in high-dimensional vector space" and tracks intrinsic rather than embedding dimensionality [49].
- **[EST]** Centring reduces hubs when similarity is the inner product in the centred space, without re-normalisation. The mechanism is that vectors near the centroid become short and therefore score low [50].
- **[EST]** Contrastive learning asymptotically optimises alignment of positive pairs and uniformity on the hypersphere. The negative-sample term can be read as a resubstitution entropy estimator built on a von Mises-Fisher kernel density estimate with concentration $\kappa = \tau^{-1}$ [53]. The vMF density and its normalising constant are standard [54].

### E5. RaBitQ (1 bit per dimension)

- **[EST]** RaBitQ (SIGMOD 2024; Proc. ACM Manag. Data 2(3), DOI 10.1145/3654970) centres and normalises each vector, applies a random orthogonal rotation and stores the $D$ sign bits. The estimator $\langle \bar o, q\rangle / \langle \bar o, o\rangle$ is unbiased with error $O(1/\sqrt{D})$ with high probability, which is asymptotically optimal for $D$-bit codes. $\langle \bar o, o\rangle$ has expectation between 0.798 and 0.800 for $D \in [10^2, 10^6]$. In practice the authors fix $\epsilon_0 = 1.9$ and quantise the query to $B_q = 4$ bits, so the inner product reduces to $B_q$ bitwise-AND plus popcount operations over $D$-bit strings. The additive bound "holds regardless of the data distribution" [35].
- **[EST]** The authors note that they "have yet to extensively explore the normalization step itself" and that the multiplicative (relative-error) form of the bound assumes well-normalised data [35].

---

## Recent work (2024-2026)

### R1. Candidate embedding models under 1B parameters

All licences, dimensions and file sizes below were read from the model cards and the Hugging Face file API on 2026-09-29. File sizes are decimal megabytes.

| Model | Params | Dim (MRL) | Context | Pooling | Licence | MTEB(Code) as reported | GGUF |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B [1][2][3] | 0.6B, 28 layers | 1024 (32-1024) | 32K | last token (EOS) | Apache-2.0 | 75.41 [1]; 75.42 [8]; 74.69 [4]; 73.75 [9] | official: Q8_0 639.2 MB, f16 1197.6 MB |
| jina-code-embeddings-0.5b [4][5] | 494M | 896 (64/128/256/512/896) | 32768, recommended at most 8192 | last token | CC-BY-NC-4.0 | 78.72 [4] | official: F16 994.2 MB, Q8_0 531.1 MB, IQ1-IQ4 |
| EmbeddingGemma-300m [6][7] | 308M | 768 (512/256/128) | 2K | mean, then two linear projections | Gemma terms of use | 68.76 [6]; 68.81 [9] | ggml-org: Q8_0 333.6 MB; also QAT q4_0 and q8_0 repos [18] |
| C2LLM-0.5B [8] | 0.5B | not checked | not checked | PMA cross-attention | Apache-2.0 | 75.46 [8] | none listed [18] |
| voyage-4-nano [10] | 340M | 2048 (1024/512/256) | 32000 | mean | Apache-2.0 | 76.43 [9] | community only [18] |
| pplx-embed-v1-0.6b [11] | 596M [9] | 1024 | 32K | mean, native int8 | MIT | 75.18 [9] | community only [18] |
| CodeRankEmbed [13][14] | 137M | not checked | 8192 | not checked | MIT | not on this benchmark; CoIR 60.1 [13] | not checked |
| SFR-Embedding-Code-400M_R [15][16] | 400M | not checked | 8192 | not checked | CC-BY-NC-4.0, research only | not on this benchmark; CoIR 61.9 [16] | not checked |

Notes on the table.

- **[REC]** Qwen3 usage: query format `Instruct: {task_description}\nQuery:{query}`; documents get no instruction; tokenizer `padding_side` left; embeddings L2-normalised; omitting the query instruction costs about 1% to 5%; instructions should be written in English [2]. The embedding is the last-layer hidden state of an appended EOS token, input format `{Instruction} {Query}<|endoftext|>` [1]. The card's own examples are inconsistent about a space after `Query:`; pick one and freeze it.
- **[REC]** Jina: InfoNCE with $\tau = 0.05$, batch 512, training sequence length 512, Matryoshka-trained; last-token pooling beat mean and latent-attention pooling in their ablation. 78.41% is the mean over 25 benchmarks and 78.72% the mean over the 12 MTEB(Code) tasks; we recomputed both means from the table and they match. Training data include "the training splits of MTEB code tasks" [4]. llama.cpp always returns 896-d for this model, so Matryoshka slicing is done client-side; task prefixes are required [5].
- **[REC]** EmbeddingGemma: quantisation-aware-trained checkpoints score 68.70 (int8 per-block, card name Q8_0), 68.03 (mixed) and 67.99 (int4 per-block, Q4_0) against 68.76 at bf16 [6][7]. Code prompt `task: code retrieval | query: {content}`; document prompt `title: {title | "none"} | text: {content}` [7]. The Hugging Face source repository is gated (HTTP 401 without acceptance of terms); the ggml-org GGUF is not.
- **[REC]** nomic-embed-code is a 7B model (Qwen2.5-Coder-7B-Instruct base, Apache-2.0, last-token pooling) and is out of scope for a 4 GB GPU [12].
- **[REC]** pplx-embed is not instruction-tuned, outputs unnormalised int8 natively and must be compared by cosine [11].
- **[CON]** Pooling is not a free choice. Jina finds last-token best on a causal code backbone [4]; C2LLM argues EOS pooling is an information bottleneck for long code and uses cross-attention pooling [8]; EmbeddingGemma finds mean pooling beats attention pooling on a bidirectional encoder [6]; pplx-embed reports mean pooling with a diffusion-pretrained bidirectional backbone gains about 1 point over alternatives [11]. Use the pooling from the chosen model's card.

### R2. llama.cpp readiness

- **[REC]** `llama.h` (master, 2026-09-29) defines pooling types UNSPECIFIED, NONE, MEAN, CLS, LAST and RANK only, and `llama_get_embeddings_seq` returns `float[n_embd]` for a pooled sequence [17]. The embedding example documents `--embd-normalize 2` (Euclidean) as the default [17].
- **[OURS]** Inference: C2LLM's cross-attention pooling head is not among those types, so C2LLM cannot run through the stock embedding path without custom code. voyage-4-nano and pplx-embed load with `trust_remote_code=True` in their reference usage [10][11], so community GGUFs need output-parity verification before use.
- **[REC]** Reported defects of the official Qwen3 GGUF: the GGUF was converted before an upstream pooling-metadata change (discussion 5, June 2025); one user reports a 32k-token input requesting almost 50 GiB of scratch memory (discussion 9); users report that with the published file every input needs `<|endoftext|>` appended manually or matching quality is "terrible" (discussion 17, November 2025) [3]. These are user reports, not measurements by the authors, and must be re-tested against the llama.cpp version that is vendored.

### R3. Repository-level and C-language evidence

- **[REC]** CORE-Bench (EMNLP 2026 main) has three levels: L1 code understanding, L2 issue-to-edit localisation, L3 broader context retrieval, reported as NDCG@10/Recall@100 [22]. Four of its eight authors are also authors of Qwen3-Embedding [1][22].

  | Model | L1 | L2 | L3 | L1-L3 mean |
  |---|---|---|---|---|
  | BM25 | 34.7/67.0 | 13.0/38.8 | 20.5/30.1 | 22.7/45.3 (computed) |
  | CodeRankEmbed (137M) | 47.4/78.6 | 12.1/32.9 | 22.5/28.6 | 27.3/46.7 |
  | jina-embeddings-v5-text-small | 57.7/89.3 | 19.1/46.7 | 33.0/41.1 | 36.6/59.0 (computed) |
  | jina-code-embeddings-0.5b | 63.5/93.0 | 20.8/51.9 | 36.7/45.5 | 40.3/63.5 (computed) |
  | Qwen3-Embedding-0.6B | 66.9/94.7 | 17.0/45.5 | 32.6/40.2 | 38.8/60.1 |
  | Qwen3-Embedding-0.6B, in-domain fine-tuned | 58.1/90.3 | 26.5/59.4 | 44.5/54.4 | 43.0/68.1 |

  Rows marked "computed" are our arithmetic means of the three published level scores; the others are printed in the paper.

- **[REC]** CORE-Bench per-language results for **C** [22]:

  | Model | L2 on C | L3 on C |
  |---|---|---|
  | BM25 | 11.9/24.3 | 19.1/15.2 |
  | CodeRankEmbed | 12.4/32.2 | 26.3/20.4 |
  | jina-code-embeddings-0.5b | 13.2/35.9 | 32.3/27.8 |
  | Qwen3-Embedding-0.6B | 11.8/34.7 | 28.8/25.6 |
  | Qwen3-Embedding-8B | 17.0/40.3 | 35.9/30.1 |
  | Qwen3-Embedding-0.6B, in-domain fine-tuned | 19.0/44.7 | 40.4/34.0 |

  On C edit localisation the 0.6B dense model matches BM25 on NDCG@10 and beats it only on Recall@100. Fine-tuning helps repository-level retrieval substantially but lowered L1 from 66.9 to 58.1.

- **[REC]** Venuta, Tosoni and Ferragina (WISE 2026) benchmark 17 models on same-language functional-equivalence code-to-code retrieval with exact search. For C++: on CodeNet, Qwen3 Embedding and CodeXEmbed reach 99-100% Precision@50, Nomic 87%, CodeSage 75%, CodeRankEmbed 33%; on xCodeEval Qwen reaches 98%. They state that Qwen3-Embedder-600M "consistently dominates" the precision-throughput trade-off and that its 1024 dimensions give "faster distance computations and smaller index footprints" [23].
- **[CON]** CoREB reports the opposite for cross-lingual code-to-code: the Qwen3 family "collapses on code-to-code". Qwen3-Emb-0.6B scores nDCG@10 0.349 text-to-code, 0.617 code-to-text, 0.384 code-to-code, 0.477 overall, against 0.386, 0.755, 0.677 and 0.588 for Jina-code-emb-0.5b and 0.430, 0.753, 0.656 and 0.603 for C2LLM-0.5B [24]. The two studies measure different tasks, so both can be true; neither measures natural-language-to-C retrieval over a systems codebase.
- **[REC]** CoREB also reports that short keyword queries of about 19 tokens drive every model to near-zero nDCG@10, and suggests query expansion (HyDE, Query2doc) as the most promising mitigation. The subtask is small: 13 queries in release v202603 [24].
- **[REC]** ExecRetrieval (EMNLP 2026 main): when execution-verified buggy near-clones are planted in the pool, the best system reaches exec@10 = 1.00 but exec@1 = 0.331, and rank-1 misses are paired buggy variants 91.5-99.4% of the time. The authors conclude that current code embeddings "function as candidate-recall components" [25]. BM25 reaches exec@10 = 0.422. This supports retrieving a thresholded set rather than trusting rank 1, and it warns that embedding similarity groups near-identical functions regardless of correctness.
- **[REC]** A fully local RAG system over a C++ scientific codebase (IPPL) using `nomic-embed-text` through Ollama with exact FAISS search reports that model family and retrieval quality matter more than parameter count, and embeds a structured text (file, symbol, chunk type, generated explanation) rather than raw code [26]. Preprint; single codebase; LLM-judged.

### R4. Embedding quantisation and truncation

- **[REC]** Post-hoc quantisation on MTEB retrieval with mxbai-embed-large-v1 (1024-d): binary retains 92.53% without rescoring and 96.45% with float32-query rescoring; int8 with a rescore multiplier of 4-5 retains about 99%; exact-search CPU speed-ups averaged 3.66x (int8, USearch) and 24.76x (binary, Faiss) [28]. This is a vendor blog post, not a peer-reviewed paper.
- **[CON]** Calibrated int8 is model-dependent. A 2025 study of two small models measured a 1.5-3.5% drop for int8 and less than 0.3% for float8 at the same 4x compression, with PCA the best dimensionality reduction; PCA-50% plus float8 (8x) beat int8 alone (4x) [30].
- **[REC]** Measured MRL truncation on code retrieval, EmbeddingGemma on MTEB(Code, v1): 768-d 68.76, 512-d 68.48, 256-d 66.74, 128-d 62.96 [7]. This is the only published per-dimension curve on code retrieval that we found. No equivalent exists for Qwen3-Embedding-0.6B.
- **[REC]** Retrieval is the task family most sensitive to compression. For gte-Qwen2-7B (3584-d, not MRL-trained) retrieval degrades already at 1024 dimensions. For MRL-trained Qwen3-Embedding-8B (4096-d) the smallest setting that kept 99% of retrieval performance with prefix truncation was 2048 dimensions at 4 bits; 90% was kept at 256 dimensions at 2 bits [31]. These are English MTEB results for the 8B model with an equal-count scalar quantiser, not RaBitQ and not code.
- **[REC]** Quantisation-aware embedding training. pplx-embed trains with a tanh int8 quantiser and a straight-through estimator; its 0.6B/1024-d model loses roughly 2-4.4 points under binary quantisation and the 4B/2560-d model at most 1.6, which the authors attribute partly to output dimension [11]. voyage-4-nano states that it uses quantisation-aware training for int8 and binary outputs [10]. EmbeddingGemma adds a spread-out regulariser explicitly for robustness to embedding quantisation and ANN retrieval [6].

### R5. RaBitQ family and competitors

- **[REC]** Extended RaBitQ (Proc. ACM Manag. Data 3(3), article 202, June 2025, DOI 10.1145/3725413) generalises to $B$ bits per dimension with asymptotically optimal space-error trade-off. Empirically 4, 5 and 7 bits give over 90%, 95% and 99% recall without re-ranking on all tested datasets, and 3 bits suffice for more than 95% on OpenAI embedding datasets. The empirical constant $c_\epsilon = 5.75$ was fitted to the 99.9% quantile of errors on random Gaussian unit vectors. The paper also states that PQ-type methods at 32x or more compression "can only produce poor recall (e.g., <80%)" without re-ranking, and that PQ and OPQ usually fail to beat scalar quantisation at 4 or more bits per dimension [36].
- **[REC]** RaBitQ Library: Apache-2.0, C++17, 1-bit and multi-bit quantisers, IVF, HNSW; AVX2/FMA with optional AVX-512; C++ source builds validated on Windows x86-64; requires CMake 3.20 or newer and OpenMP; the README names Visual Studio 2026 for Windows builds [37].
- **[REC]** SAQ (Proc. ACM Manag. Data 3(6), December 2025, DOI 10.1145/3769824) allocates bits unevenly across segments of PCA-projected vectors and reports up to 80% lower quantisation error and over 80x faster encoding than extended RaBitQ [41]. It needs a PCA fit, so it is data-dependent. Abstract-level reading only.
- **[CON]** RaBitQ versus TurboQuant. TurboQuant (arXiv April 2025) claims near-optimal distortion within a factor of about 2.7 and better recall than product quantisation with near-zero indexing time [39]. The RaBitQ authors report that TurboQuant is worse than RaBitQ in most settings they tested and that several TurboQuant runtime and recall results could not be reproduced [38]. An independent third group concludes that the relative advantages "are criterion-dependent rather than absolute": TurboQuant and EDEN are favourable for mean-squared-error distortion, RaBitQ "provides strong high-probability control" [40]. High-probability control of a single estimate is exactly what a threshold decision needs, which is the reason to prefer RaBitQ here.
- **[REC]** Decision-level analysis. Xiao and Cao bound the probability that quantisation flips a comparison by the probability mass of exact margins near zero plus the tail probability of the calibrated residual, with no independence assumption. Across 24 configurations their margin-based predictor tracks held-out ranking flip rates with Spearman 0.970 against 0.739 for global rank correlation. They also report that a random rotation can improve global fidelity while raising the flip rate on the hardest comparisons [42]. Preprint, September 2026; the core inequality is elementary and we re-derived it (F13).

### R6. Geometry, 2024-2026

- **[REC]** A 2026 study gives a geometric account linking anisotropy to training dynamics that preferentially amplify tangent directions, across encoder and decoder models [46].
- **[REC]** Hubness as a RAG security concern: a few hub vectors can be retrieved for many unrelated queries. An admission-time global gate detects hub-like documents; a per-topic gate adds nothing (marginal +0.000), tested on five encoders of 384-1024 dimensions and 100k-document corpora [51].
- **[CON]** Hubness and anisotropy are statistically dissociable. On five production encoders, hub mass explains reciprocity failures (partial $R^2$ 0.302) and anisotropy does not (0.003); a hub-aware score correction (CSLS) closes 63.5% of the gap [52]. Preprint; multilingual idioms, not code. It contradicts any design that treats "fix anisotropy" and "fix hubness" as one step.

### R7. Chunking

- **[REC]** cAST (Findings of EMNLP 2025, pages 8106-8116) parses with tree-sitter, recursively splits AST nodes that exceed a budget, greedily merges adjacent small siblings, measures size in non-whitespace characters, and guarantees that concatenated chunks reproduce the file verbatim. Against fixed-size line chunking it reports Recall@5 gains of 1.8-4.3 points on RepoEval and Pass@1 +2.67 on SWE-bench, with `max_chunk_size = 2000` [55].
- **[CON]** A controlled study of 864 settings on two Python benchmarks finds pure function-level chunking worst by 3.57-5.64 points Exact Match on RepoEval and never Pareto-optimal, because functions are shorter than the budget and because it "discards module-level code outside function bodies". cAST, declaration and sliding-window chunking perform comparably with sliding window narrowly ahead. Context length dominates (up to 4.2 points); chunk size has a weaker non-monotonic effect (at most 1.9 points). Recommended default: chunk size 2,000, context at least 4,096 tokens [56]. Both studies use Python; neither covers C with macros and headers.

---

## Formulas

Notation: $D$ is the embedding dimension; $N$ the number of chunks; unit raw vectors $o_r, q_r$; centroid $c$.

### F1. Qwen3-Embedding training loss. Source: [1], Section 3.1.

$$
L_{\mathrm{embedding}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{s(q_i, d_i^{+})/\tau}}{Z_i}
$$

$$
Z_i = e^{s(q_i,d_i^{+})/\tau} + \sum_{k=1}^{K} m_{ik}\, e^{s(q_i,d_{i,k}^{-})/\tau} + \sum_{j\neq i} m_{ij}\, e^{s(q_i,q_j)/\tau} + \sum_{j\neq i} m_{ij}\, e^{s(d_i^{+},d_j)/\tau} + \sum_{j\neq i} m_{ij}\, e^{s(q_i,d_j)/\tau}
$$

$$
m_{ij} = \begin{cases} 0 & \text{if } s_{ij} > s(q_i,d_i^{+}) + 0.1 \ \text{ or } \ d_j = d_i^{+} \\ 1 & \text{otherwise} \end{cases}
$$

$s$ is cosine similarity, $K$ the number of hard negatives. The paper does not state the value of $\tau$.

### F2. InfoNCE in standard form. Source: [4], Equation 1, corrected.

$$
\mathcal{L}_{\mathrm{NCE}} = -\sum_{i=1}^{n} \ln \frac{e^{S_{i,i}/\tau}}{\sum_{k=1}^{n} e^{S_{i,k}/\tau}}, \qquad S_{i,k} = \cos(\mathbf{q}_i, \mathbf{d}_k)
$$

The equation printed in [4] sums $-\ln\sigma(S,\tau,i,j)$ over all $i,j$ from $0$ to $n$. That expression is minimised by a uniform softmax and contradicts the paper's own text, which treats $(q_i,d_i)$ as positive and $(q_i,d_j)$, $i\neq j$, as negative. The form above is the standard one and is our reading of the intent. Reported hyper-parameters: $\tau = 0.05$, $n = 512$ for the 0.5B model.

### F3. EmbeddingGemma spread-out regulariser. Source: [6], Equation 4.

$$
\mathcal{L}_S = \frac{1}{B(B-1)} \sum_{i \neq j} \left(\mathbf{q}_i^{\top}\mathbf{q}_j\right)^2 + \frac{1}{B(B-1)} \sum_{i \neq j} \left(\mathbf{p}_i^{+\top}\mathbf{p}_j^{+}\right)^2
$$

Second-moment term of the global orthogonal regulariser; $B$ is the batch size.

### F4. Matryoshka objective. Source: [27], Equation 1.

$$
\min_{\{\mathbf{W}^{(m)}\}_{m\in\mathcal{M}},\, \theta_F} \ \frac{1}{N} \sum_{i\in[N]} \sum_{m\in\mathcal{M}} c_m \cdot \mathcal{L}\!\left(\mathbf{W}^{(m)} \cdot F(x_i;\theta_F)_{1:m} \, ;\, y_i\right)
$$

with $|\mathcal{M}| \le \lfloor \log d \rfloor$ and $c_m = 1$. Here $N$ is the number of training examples.

### F5. Truncation at inference. Notation ours; operation stated in words in [7].

$$
\tilde z^{(m)} = \frac{z_{1:m}}{\lVert z_{1:m} \rVert_2}
$$

### F6. RaBitQ normalisation and distance decomposition. Source: [35], Section 3.1.1.

$$
\mathbf{o} = \frac{\mathbf{o}_r - \mathbf{c}}{\lVert \mathbf{o}_r - \mathbf{c} \rVert}, \qquad \mathbf{q} = \frac{\mathbf{q}_r - \mathbf{c}}{\lVert \mathbf{q}_r - \mathbf{c} \rVert}
$$

$$
\lVert \mathbf{o}_r - \mathbf{q}_r \rVert^2 = \lVert \mathbf{o}_r - \mathbf{c} \rVert^2 + \lVert \mathbf{q}_r - \mathbf{c} \rVert^2 - 2\, \lVert \mathbf{o}_r - \mathbf{c} \rVert\, \lVert \mathbf{q}_r - \mathbf{c} \rVert\, \langle \mathbf{q}, \mathbf{o} \rangle
$$

Checked: this is the law of cosines.

### F7. RaBitQ codebook and code. Source: [35], Section 3.1.

$$
\mathcal{C} = \left\{ +\tfrac{1}{\sqrt{D}},\, -\tfrac{1}{\sqrt{D}} \right\}^{D}, \qquad \bar{\mathbf{x}} = \arg\max_{\mathbf{x}\in\mathcal{C}} \langle P^{-1}\mathbf{o}, \mathbf{x} \rangle, \qquad \bar{\mathbf{x}} = \frac{2\bar{\mathbf{x}}_b - \mathbf{1}_D}{\sqrt{D}}, \qquad \bar{\mathbf{o}} = P\bar{\mathbf{x}}
$$

$P$ is a random orthogonal matrix, so $P^{-1} = P^{\top}$. The code $\bar{\mathbf{x}}_b \in \{0,1\}^D$ is the sign pattern of $P^{-1}\mathbf{o}$.

### F8. RaBitQ estimator and bound. Source: [35], Theorem 3.2, Equations 13-16.

$$
\mathbb{E}\left[ \frac{\langle \bar{\mathbf{o}}, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}, \mathbf{o} \rangle} \right] = \langle \mathbf{o}, \mathbf{q} \rangle
$$

$$
\mathbb{P}\left\{ \left| \frac{\langle \bar{\mathbf{o}}, \mathbf{q} \rangle}{\langle \bar{\mathbf{o}}, \mathbf{o} \rangle} - \langle \mathbf{o}, \mathbf{q} \rangle \right| > \sqrt{\frac{1 - \langle \bar{\mathbf{o}}, \mathbf{o} \rangle^2}{\langle \bar{\mathbf{o}}, \mathbf{o} \rangle^2}} \cdot \frac{\epsilon_0}{\sqrt{D-1}} \right\} \le 2 e^{-c_0 \epsilon_0^2}
$$

$$
\mathbb{E}\left[ \langle \bar{\mathbf{o}}, \mathbf{o} \rangle \right] = \sqrt{\frac{D}{\pi}} \cdot \frac{2\,\Gamma\!\left(\frac{D}{2}\right)}{(D-1)\,\Gamma\!\left(\frac{D-1}{2}\right)}
$$

The constant $c_0$ is not given a numerical value. We evaluated the expectation at $D = 1024$: 0.79808 (limit $\sqrt{2/\pi} = 0.79788$).

### F9. RaBitQ bitwise evaluation. Source: [35], Equations 20-22.

$$
\langle \bar{\mathbf{x}}, \bar{\mathbf{q}} \rangle = \frac{2\Delta}{\sqrt{D}} \langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u \rangle + \frac{2 v_l}{\sqrt{D}} \sum_{i=1}^{D} \bar{\mathbf{x}}_b[i] - \frac{\Delta}{\sqrt{D}} \sum_{i=1}^{D} \bar{\mathbf{q}}_u[i] - \sqrt{D}\, v_l
$$

$$
\langle \bar{\mathbf{x}}_b, \bar{\mathbf{q}}_u \rangle = \sum_{j=0}^{B_q - 1} 2^{j} \cdot \mathrm{popcount}\!\left( \bar{\mathbf{x}}_b \ \mathrm{AND} \ \bar{\mathbf{q}}_u^{(j)} \right)
$$

The rotated query is quantised as $\bar{\mathbf{q}} = \Delta\, \bar{\mathbf{q}}_u + v_l \mathbf{1}_D$ with $\bar{\mathbf{q}}_u \in \{0,\dots,2^{B_q}-1\}^D$ using randomised rounding; $\bar{\mathbf{q}}_u^{(j)}$ is its $j$-th bit-plane. $\sum_i \bar{\mathbf{x}}_b[i]$ is the popcount of the code and "can be pre-computed during the index phase". Checked numerically to $10^{-15}$.

### F10. Error variance and coverage of the 1-bit estimator. **[OURS]**, derived from [35] Equation 12.

Equation 12 of [35] writes the estimator as $\langle o,q\rangle + \sqrt{1-\langle o,q\rangle^2}\,\langle \bar o, e_1\rangle / \langle \bar o, o\rangle$, where $e_1$ is the unit vector orthogonal to $o$ in the plane of $o$ and $q$. Conditional on $a = \langle \bar o, o\rangle$, the component of the unit vector $\bar o$ orthogonal to $o$ has norm $\sqrt{1-a^2}$ and, by rotational symmetry of $P$, a direction uniform on the unit sphere of the $(D-1)$-dimensional complement. Its projection on a fixed unit vector has mean 0 and variance $(1-a^2)/(D-1)$. Hence, writing $\rho = \langle o, q\rangle$ and $\hat s$ for the estimate,

$$
\operatorname{Var}\!\left(\hat s - \rho \,\middle|\, a\right) = \left(1-\rho^2\right) \frac{1-a^2}{a^2\,(D-1)}, \qquad \sigma(\rho) \approx \frac{0.755\,\sqrt{1-\rho^2}}{\sqrt{D-1}} \quad (a \approx 0.798)
$$

So the bound of F8 is exactly $\epsilon_0$ standard deviations at $\rho = 0$, and a normal approximation gives two-sided coverage $2\Phi(\epsilon_0) - 1$.

| $\epsilon_0$ (or $z$) | Half-width at $D=1024$, $a=0.8$ | Two-sided miss probability (normal) |
|---|---|---|
| 1.9 | 0.0446 | 5.7% |
| 2.576 | 0.0604 | 1.0% |
| 3.0 | 0.0703 | 0.27% |
| 3.29 | 0.0771 | 0.10% |

Monte Carlo check (ours; $D = 1024$, $10^5$ random unit vectors, one random rotation, exact float query): standard deviation of the error 0.02363 at $\rho = 0$ against 0.02361 predicted; fraction outside the $\epsilon_0 = 1.9$ band 5.8% at $\rho = 0$, 2.9% at $\rho = 0.5$, 0.16% at $\rho = 0.8$; 99.9% quantile of the absolute error 0.0776 at $\rho = 0$. With the query quantised to 4 bits the error standard deviation rises by 4% to 11%. The same figures were obtained on a strongly anisotropic synthetic set, consistent with the distribution-free claim. This is a simulation on synthetic vectors, not a measurement on mimalloc embeddings.

### F11. Extended RaBitQ. Source: [36], Theorem 3.2 and the empirical-formula remark.

$$
B = \Theta\!\left( \log\!\left( \frac{1}{D} \cdot \frac{1}{\epsilon^2} \log\frac{1}{\delta} \right) \right) \quad \text{for } \frac{1}{\epsilon^2}\log\frac{1}{\delta} > D
$$

$$
\epsilon < 2^{-B} \cdot \frac{c_\epsilon}{\sqrt{D}}, \qquad c_\epsilon = 5.75 \quad \text{(with probability above 99.9\%, measured)}
$$

At $D = 1024$: 0.0898 ($B=1$), 0.0449 ($B=2$), 0.0225 ($B=3$), 0.0112 ($B=4$), 0.0056 ($B=5$), 0.0007 ($B=8$). The $B=1$ value 0.0898 upper-bounds our simulated 99.9% quantile 0.0776, so the two sources are consistent. $\epsilon$ is the absolute error of the inner product of unit vectors.

### F12. Raw cosine from centred quantities. **[OURS]**, from F6 with unit raw vectors.

Let $r_o = \lVert o_r - c\rVert$, $r_q = \lVert q_r - c \rVert$. Since $\lVert o_r - q_r\rVert^2 = 2 - 2\langle o_r, q_r\rangle$,

$$
\langle \mathbf{o}_r, \mathbf{q}_r \rangle = 1 - \frac{r_o^2 + r_q^2}{2} + r_o\, r_q\, \langle \mathbf{o}, \mathbf{q} \rangle, \qquad \Delta_{\mathrm{raw}} = r_o\, r_q\, \Delta_{\mathrm{centred}}
$$

Only $r_o$ needs to be stored per vector; $\langle o_r, c\rangle$ is not needed. Checked numerically to $10^{-15}$. Three different rankings are available from one code: raw cosine (above), centred inner product $r_o r_q \langle o,q\rangle$ (the quantity analysed in [50]), and centred cosine $\langle o, q\rangle$.

### F13. Decision flips under quantisation. Source: [42], Theorem 3; application to a threshold is **[OURS]**.

Let $\Psi$ be the exact decision functional, $\widehat\Psi_Q$ the approximate one, $R = \widehat\Psi_Q - c_Q \Psi$ with calibration constant $c_Q > 0$, $\Gamma = |\Psi|$, and the flip event $\mathcal{E} = \{\Psi \widehat\Psi_Q \le 0,\ \Psi \ne 0\}$. For every $\tau > 0$,

$$
\mathbb{P}(\mathcal{E}) \ \le\ \mathbb{P}(0 < \Gamma \le \tau) \ +\ \mathbb{P}(|R| \ge c_Q \tau)
$$

Proof check: on $\mathcal{E}$ the residual opposes the sign of $\Psi$, so $|R| \ge c_Q \Gamma$; split on $\Gamma \le \tau$. For the 25% relevance cut take $\Psi_i = s_i - s_{\mathrm{thr}}$ and $\widehat\Psi_i = \hat s_i - s_{\mathrm{thr}}$ with $c_Q = 1$ (the estimator is unbiased). The rescoring rule is then

$$
\text{rescore } i \iff \left| \hat s_i - s_{\mathrm{thr}} \right| \le z\, \hat\sigma_i, \qquad \hat\sigma_i = \sqrt{1 - \hat s_i^{\,2}} \ \sqrt{\frac{1 - a_i^2}{a_i^2\,(D-1)}}
$$

with $a_i = \langle \bar o_i, o_i\rangle$ stored per vector and $z$ chosen from the table in F10.

### F14. Product quantisation. Source: [32], Equations 13 and 14-18.

$$
\ell = m \log_2 k^{*}, \qquad \tilde d(x,y)^2 = d\big(x, q(y)\big)^2 = \sum_{j=1}^{m} d\big(u_j(x),\, q_j(u_j(y))\big)^2, \qquad \mathrm{MSDE}(q) \le \mathrm{MSE}(q)
$$

### F15. Native int8 embedding quantisation. Source: [11], Section 2.2.

$$
\mathbf{e} = \left\lfloor 127 \cdot \tanh\!\left( \frac{1}{L} \sum_{l=1}^{L} \mathbf{v}_l \right) + \frac{1}{2} \right\rfloor \in \{-127, \dots, 127\}^{d}
$$

### F16. Calibrated int8 scalar quantisation. Source: procedure in [28], exact form from the reference implementation [29].

$$
q_i = \operatorname{clip}\!\left( \left\lfloor \frac{255\,(x_i - \min_i)}{\max_i - \min_i} \right\rfloor,\ 0,\ 255 \right) - 128
$$

$\min_i$ and $\max_i$ are per-dimension extremes over a calibration set. The implementation computes the step $(\max_i - \min_i)/255$ and replaces a zero step by 1 to avoid division by zero. Binary quantisation sets bit $i$ to $[x_i > 0]$ and packs 8 bits per byte (stored as packed value minus 128 in the `binary` precision).

### F17. Alignment, uniformity and the vMF kernel. Sources: [53] Section 4.1 and Section 4.2; [54] Equations 2.1-2.2.

$$
\mathcal{L}_{\mathrm{align}}(f;\alpha) = \mathbb{E}_{(x,y)\sim p_{\mathrm{pos}}} \left[ \lVert f(x) - f(y) \rVert_2^{\alpha} \right], \qquad \mathcal{L}_{\mathrm{uniform}}(f;t) = \log \ \mathbb{E}_{x,y \,\overset{\mathrm{i.i.d.}}{\sim}\, p_{\mathrm{data}}} \left[ G_t\big(f(x), f(y)\big) \right]
$$

$$
G_t(u,v) = e^{-t \lVert u - v \rVert_2^2} = e^{2t\, u^{\top} v - 2t}
$$

$$
f_{\mathrm{vMF}}(x \mid \mu, \kappa) = c_d(\kappa)\, e^{\kappa\, \mu^{\top} x}, \qquad c_d(\kappa) = \frac{\kappa^{d/2 - 1}}{(2\pi)^{d/2}\, I_{d/2-1}(\kappa)}
$$

On unit vectors $G_t$ is a vMF kernel with $\kappa = 2t$. In [53] the contrastive denominator corresponds to a vMF kernel density estimate with $\kappa = \tau^{-1}$, so $\tau = 0.05$ gives $\kappa = 20$. **[OURS]**: $c_d(\kappa)$ is identical for every kernel when one $\kappa$ is used, so it cancels in any ratio of densities; the Bessel function at order 511 need not be evaluated.

### F18. All-but-the-top. Source: [47], Algorithm 1.

As printed:

$$
\mu = \frac{1}{|V|}\sum_{w\in V} v(w), \quad \tilde v(w) = v(w) - \mu, \quad u_1,\dots,u_d = \mathrm{PCA}\big(\{\tilde v(w)\}\big), \quad v'(w) = \tilde v(w) - \sum_{i=1}^{D_{\mathrm{top}}} \big( u_i^{\top} v(w) \big)\, u_i
$$

Centred-consistent form (**[OURS]**, recommended):

$$
v'(w) = \tilde v(w) - \sum_{i=1}^{D_{\mathrm{top}}} \big( u_i^{\top} \tilde v(w) \big)\, u_i
$$

The two differ by the constant vector $\sum_i (u_i^{\top}\mu)\, u_i$. The constant is irrelevant to pairwise differences but changes directions after re-normalisation. Rule of thumb $D_{\mathrm{top}} \approx d/100$, about 10 for $d = 1024$.

### F19. Anisotropy. Measure described in words in [43]; identity is **[OURS]**.

$$
A = \frac{1}{N(N-1)} \sum_{i \ne j} \frac{x_i^{\top} x_j}{\lVert x_i \rVert\, \lVert x_j \rVert} = \frac{N \lVert c \rVert^2 - 1}{N - 1} \ \text{ for unit vectors}, \qquad \frac{1}{N}\sum_{i} \lVert x_i - c \rVert^2 = 1 - \lVert c \rVert^2
$$

Derivation: $\sum_{i,j} x_i^{\top}x_j = \lVert \sum_i x_i \rVert^2 = N^2 \lVert c\rVert^2$; subtract the $N$ diagonal terms. Anisotropy is therefore obtained in $O(ND)$ from the centroid that RaBitQ needs anyway, and the mean squared residual norm is approximately $1 - A$. Checked numerically.

### F20. Hubness. Sources: [49] Section 3; [50] Equations 1-3.

$$
S_{N_k} = \frac{\mathbb{E}\left[ (N_k - \mu_{N_k})^3 \right]}{\sigma_{N_k}^3}, \qquad K_{\mathrm{cent}} = H K H, \quad H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^{\top}, \quad (K_{\mathrm{cent}})_{ij} = \langle x_i - \bar x,\ x_j - \bar x \rangle
$$

$N_k(x)$ counts how often $x$ is among the $k$ nearest neighbours of other points; [49] and [50] use $k = 10$ in tables.

### F21. Vectors per cache block. **[OURS]**.

$$
n_{\mathrm{blk}} = \left\lfloor \frac{16384}{\lceil D B / 8 \rceil + s} \right\rfloor
$$

$B$ bits per dimension, $s$ bytes of side information per vector.

| $D$ | $B$ | $s$ | Bytes per vector | Vectors per 16 KB |
|---|---|---|---|---|
| 1024 | 1 | 8 ($r_o$, $a$) | 136 | 120 |
| 1024 | 1 | 12 (plus popcount) | 140 | 117 |
| 1024 | 4 | 8 | 520 | 31 |
| 1024 | 8 (int8) | 0 | 1024 | 16 |
| 1024 | 32 (float32) | 0 | 4096 | 4 |
| 256 | 1 | 8 | 40 | 409 |

Working set per thread: one 16,384-byte block plus four 128-byte query bit-planes is 16,896 bytes. Two hyper-threads on one P-core need 33,792 of 49,152 bytes; one E-core thread needs 16,896 of 32,768 bytes. The $1024 \times 1024$ float32 rotation matrix is 4 MiB and is used once per query, outside the scan.

### F22. nDCG@10. Source: [22], Appendix D.

$$
\mathrm{DCG@10}(q) = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{\log_2(i+1)}, \qquad \mathrm{NDCG@10}(q) = \frac{\mathrm{DCG@10}(q)}{\mathrm{IDCG@10}(q)}
$$

---

## What is novel in our proposal

Everything in this section is **[OURS]**. None of it is an established result.

1. **L1-sized block layout for binary codes.** Packing 117 to 120 one-bit RaBitQ codes with their side values into 16 KB blocks so that a scan thread's working set (block plus query bit-planes, about 16.9 KB) fits the L1 data cache of both core types. The published RaBitQ work packs codes for SIMD batches [35]; it does not size blocks to a cache level. Whether this produces a measurable benefit is an empirical question that the paper must answer with measurements.
2. **Variance-calibrated threshold rescoring.** Using the closed-form error standard deviation of F10 and a chosen $z$ to decide which candidates near the relevance threshold need a higher-precision rescore (F13). RaBitQ uses its bound to decide re-ranking in nearest-neighbour search [35]; using it to protect a probability threshold is our adaptation, and the decomposition of [42] gives it a citable framing.
3. **One centroid, three uses, kept separate.** The corpus centroid yields (i) the RaBitQ normalisation, (ii) the anisotropy measure via F19 and (iii) the centred inner product of [50]. Our contribution is the explicit statement that these give three different rankings (F12) and must be compared empirically, instead of assuming that centring fixes anisotropy and hubness together.
4. **Kernel bandwidth from the training temperature.** Initialising the vMF concentration of the density estimator at $\kappa = 1/\tau$ of the embedder's contrastive training [53], then tuning on the target corpus. For Qwen3-Embedding the temperature is unpublished [1], so this gives only a starting value borrowed from comparable models ($\tau = 0.05$ in [4] and [8]).
5. **Evaluation on a C systems codebase.** A hand-labelled query set over mimalloc. Public benchmarks contain little C, and the one published C breakdown [22] concerns issue localisation, not question answering over an allocator.

---

## Risks and failure modes

1. **Scale mismatch.** About 1.3 MB of source at 2,000 non-whitespace characters per chunk gives roughly 500 to 1,000 chunks (our estimate). The 1-bit index is then about 70 to 140 KB, which is 5 to 9 blocks. An exact float32 scan of 1,000 vectors is about $10^6$ multiply-adds. Retrieval time will be dominated by the transformer forward pass that embeds the query. Multi-threaded block scanning will not show a speed-up at this size and thread start-up may make it slower.
2. **Dense retrieval may not beat BM25 on C.** See the C rows in R3 [22]. If the mimalloc evaluation shows the same, the paper must report it.
3. **Bound misuse.** Applying the 0.0446 band as if it were a 99.9% band would silently misclassify several percent of near-threshold candidates (F10). The band also applies to the centred inner product, not to raw cosine (F12).
4. **Small residual norms.** For an anisotropic single-repository corpus, $1 - \lVert c\rVert^2$ may be small. Centred cosine then amplifies small raw differences, and gaps between relevant and irrelevant chunks may be comparable to the quantisation error. The RaBitQ authors say the normalisation step is not yet well studied [35].
5. **Random rotation can hurt the hardest comparisons.** Reported in [42]: rotation improved global fidelity while raising flip rates on anchor comparisons. Near-threshold candidates are exactly such comparisons.
6. **Centring may not reduce hubs under cosine.** The published result is for the unnormalised centred inner product [50]; hubness and anisotropy are dissociable [52]. Hubness may also be immaterial at $N \approx 10^3$; it has to be measured before any claim is made.
7. **GGUF correctness.** The EOS and pooling-metadata problems are user-reported for the official Qwen3 GGUF [3]. A silent mismatch produces plausible but poor vectors.
8. **Unknown cost of weight quantisation.** No published table gives the retrieval cost of post-hoc GGUF quantisation (Q8_0 or 4-bit) for Qwen3-Embedding-0.6B or any code embedder. The only published quantised-checkpoint scores are EmbeddingGemma's, and those are quantisation-aware-trained [6][7].
9. **Unknown truncation curve.** No per-dimension code-retrieval scores exist for Qwen3-Embedding-0.6B. The EmbeddingGemma curve [7] and the 8B English result [31] do not transfer automatically.
10. **Benchmark contamination and conflicts of interest.** Jina, C2LLM and LateOn train on the training splits of MTEB(Code) tasks [4][8][9]. CORE-Bench shares authors with Qwen3-Embedding [1][22]; CoREB shares authors with C2LLM [8][24]. CoREB reports contamination and label noise in existing benchmarks [24].
11. **Near-clone grouping.** Embedding similarity places buggy single-edit variants next to the correct function [25]. Density-based grouping of "similar functions" will merge them. In mimalloc this affects families of near-identical functions that differ in one flag or size class.
12. **Short queries.** Keyword-style queries are reported to fail for all models [24], on a small sample. Identifier look-ups such as a bare function name should be routed to a lexical index.
13. **Chunking evidence is Python-only.** Neither [55] nor [56] tested C with macros, conditional compilation and headers. tree-sitter parses unpreprocessed C, so macro-heavy regions can produce error nodes.
14. **Licences.** jina-code-embeddings-0.5b and SFR-Embedding-Code are non-commercial [5][16]. EmbeddingGemma is under Gemma terms and its source repository is gated [7].
15. **Preprint status.** References [1], [6], [8], [9], [11], [15], [21], [24], [26], [30], [31], [33], [38], [39], [40], [42], [46], [48], [51], [52] and [56] are arXiv preprints or technical reports for which no peer-reviewed venue was confirmed. Model cards [2][3][5][7][10][12][13][16] and the blog post [28] are vendor documentation. Source files [17][29][34][37] and the hub listing [18] are point-in-time snapshots.

---

## Design recommendations

1. **Primary embedder.** Qwen3-Embedding-0.6B from `Qwen/Qwen3-Embedding-0.6B-GGUF`, run in-process through llama.cpp with `LLAMA_POOLING_TYPE_LAST`, 1024-d, L2-normalised. Use f16 (1197.6 MB) for the reference index and Q8_0 (639.2 MB) as the measured alternative, and report the difference, because no published number exists. State in the paper that the choice is driven by licence and runtime support, and that jina-code-embeddings-0.5b scores higher on repository-level retrieval [22] but is CC-BY-NC.
2. **Prompt format, frozen and recorded.** Query: `Instruct: <one English sentence>\nQuery:<question>`. Document: chunk text with no instruction. Record the exact instruction sentence in the paper, since the card reports 1-5% sensitivity [2].
3. **Parity test in the build.** Embed a fixed set of strings with and without an appended `<|endoftext|>` and compare against vectors from the reference Python implementation. Fail the build if cosine falls below 0.999. Set `n_ctx` to what the chunks need (2,048 to 8,192 tokens), not 32K.
4. **Lexical channel.** Add BM25 over identifiers and comments and fuse it with the dense score. Justification: BM25 equals the dense model on C edit localisation [22], and short keyword queries defeat dense models [24]. Report dense-only, lexical-only and fused results.
5. **Fine-tuning toggle should cover the embedder.** In-domain fine-tuning raised Qwen3-Embedding-0.6B from 17.0 to 26.5 (L2) and from 32.6 to 44.5 (L3) NDCG@10, at the cost of L1 [22]. If the embedder is fine-tuned, re-embed the corpus and recompute the centroid, rotation-independent side values and codes.
6. **Compression: two tiers.** Tier 1: 1-bit RaBitQ codes (128 bytes) with $r_o$, $a$ and optionally the popcount, in 16 KB blocks, scanned with bitwise-AND and popcount against a 4-bit query. Tier 2: float32 originals (4 MB per 1,000 chunks) for rescoring. Skip int8 as a middle tier; at this size it adds a calibration step and an error source for no benefit.
7. **Threshold rescoring with explicit confidence.** Apply F13 with $z = 3.29$ for a nominal 0.1% two-sided miss rate per candidate, or the published conservative band 0.0898 at $B = 1$ [36]. Report the fraction of candidates rescored and the measured flip rate against the float32 decision.
8. **Always report the float32 exact baseline.** Give nDCG@10, Recall@k and the size of the thresholded set for float32, 1-bit without rescoring and 1-bit with rescoring, with retention as a percentage of float32.
9. **Do not use product quantisation; do not truncate below 512 dimensions.** Faiss asks for at least 9,984 training vectors for 256 centroids [34], about ten times what this corpus provides, and PQ gives no per-vector bound. The only code truncation curve loses about 2 points at 256-d and about 6 at 128-d [7].
10. **Geometry diagnostics before any geometric preprocessing.** Report anisotropy via F19, mean residual norm, and $S_{N_{10}}$ for raw cosine, centred inner product and centred cosine. Adopt centring or all-but-the-top (centred form, F18) only if retrieval quality on the labelled set improves. Initialise the vMF concentration at $\kappa = 20$ and tune.
11. **Chunking.** cAST-style split-then-merge over a tree-sitter C parse with a budget of 2,000 non-whitespace characters, retaining module-level code (macros, typedefs, static inline helpers, globals) and guaranteeing byte-exact reconstruction of each file, which also gives the inline-edit component exact offsets [55][56]. Attach enclosing function names as metadata. Include a sliding-window baseline.
12. **Scope the cache claim.** The 16 KB argument applies to the code scan, density evaluation and rescoring. Embedding inference holds hundreds of megabytes of weights and cannot be L1-resident. Measure with and without block packing and with 1, 2, 8 and 24 threads, and report the result even if it shows no gain at this corpus size.

Alternates for A/B testing, in order: jina-code-embeddings-0.5b GGUF (896-d, `--pooling last`, task prefixes) only if the project stays non-commercial; EmbeddingGemma-300m QAT Q8_0 (768-d, 2K context) as the small fallback. Watch list without verified GGUF: voyage-4-nano, pplx-embed-v1-0.6b. Reference implementation for unit tests: the RaBitQ Library [37].

---

## References

All URLs opened on 2026-09-29. "Read" states how much of the source was checked.

1. Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, Jingren Zhou. "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models." arXiv:2506.05176, v3, 11 June 2025. https://arxiv.org/abs/2506.05176 . Read: full text.
2. Qwen team, Alibaba Cloud. "Qwen3-Embedding-0.6B" model card. Hugging Face, 2025. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B . Read: raw README.
3. Qwen team, Alibaba Cloud; community contributors. "Qwen3-Embedding-0.6B-GGUF" model card, file listing and discussions 5, 9 and 17. Hugging Face, 2025. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF . Read: raw README, file API, discussion API.
4. Daria Kryvosheieva, Saba Sturua, Michael Guenther, Han Xiao. "Efficient Code Embeddings from Code Generation Models." NeurIPS 2025 Workshop on Deep Learning for Code; arXiv:2508.21290, v2, 17 August 2026. https://arxiv.org/abs/2508.21290 . Read: full text.
5. Jina AI. "jina-code-embeddings-0.5b-GGUF" model card. Hugging Face, 2025. https://huggingface.co/jinaai/jina-code-embeddings-0.5b-GGUF . Read: raw README, file API.
6. Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, and 83 others. "EmbeddingGemma: Powerful and Lightweight Text Representations." arXiv:2509.20354, v3, 1 November 2025. https://arxiv.org/abs/2509.20354 . Read: full text.
7. Google DeepMind. "EmbeddingGemma model card." Google AI for Developers, last updated 25 September 2025. https://ai.google.dev/gemma/docs/embeddinggemma/model_card . Read: full page.
8. Jin Qin, Zihan Liao, Ziyin Zhang, Hang Yu, Peng Di, Rui Wang. "C2LLM Technical Report: A New Frontier in Code Retrieval via Adaptive Cross-Attention Pooling." arXiv:2512.21332, 24 December 2025. https://arxiv.org/abs/2512.21332 . Read: full text and model card.
9. Raphael Sourty, Antoine Chaffin, Paulo Roberto Moura Junior, Amelie Chatelain. "DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search." arXiv:2607.27178, v2, 31 July 2026. https://arxiv.org/abs/2607.27178 . Read: full text.
10. Voyage AI. "voyage-4-nano" model card. Hugging Face, 2026. https://huggingface.co/voyageai/voyage-4-nano . Read: raw README.
11. Sedigheh Eslami, Maksim Gaiduk, Markus Krimmel, Louis Milliken, Bo Wang, Denis Bykov. "Diffusion-Pretrained Dense and Contextual Embeddings." arXiv:2602.11151, v2, 13 February 2026. https://arxiv.org/abs/2602.11151 . Read: full text and model card.
12. Nomic AI. "nomic-embed-code" model card. Hugging Face, 2025. https://huggingface.co/nomic-ai/nomic-embed-code . Read: raw README.
13. Nomic AI. "CodeRankEmbed" model card. Hugging Face, 2024. https://huggingface.co/nomic-ai/CodeRankEmbed . Read: raw README.
14. Tarun Suresh, Revanth Gangi Reddy, Yifei Xu, Zach Nussbaum, Andriy Mulyar, Brandon Duderstadt, Heng Ji. "CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking." ICLR 2025; arXiv:2412.01007. https://arxiv.org/abs/2412.01007 . Read: abstract and key sections.
15. Ye Liu, Rui Meng, Shafiq Joty, Silvio Savarese, Caiming Xiong, Yingbo Zhou, Semih Yavuz. "CodeXEmbed: A Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieval." arXiv:2411.12644, v3, 8 August 2025. https://arxiv.org/abs/2411.12644 . Read: abstract and key sections. Title spelling as published.
16. Salesforce AI Research. "SFR-Embedding-Code-400M_R" model card. Hugging Face, 2025. https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R . Read: raw README.
17. ggml-org and llama.cpp contributors. "llama.cpp: include/llama.h and examples/embedding/README.md." GitHub, master branch, accessed 2026-09-29. https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h . Read: both files.
18. Hugging Face. "Hub model search API listings for candidate embedding models." Queried 2026-09-29. https://huggingface.co/api/models?search=voyage-4-nano . Point-in-time listing.
19. Xiangyang Li, Kuicai Dong, Yi Quan Lee, Wei Xia, Hao Zhang, Xinyi Dai, Yasheng Wang, Ruiming Tang. "CoIR: A Comprehensive Benchmark for Code Information Retrieval Models." Proceedings of ACL 2025 (Volume 1: Long Papers), pages 22074-22091. DOI 10.18653/v1/2025.acl-long.1072. https://aclanthology.org/2025.acl-long.1072/ ; arXiv:2407.02883. Read: full text.
20. Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Marton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, and 78 others. "MMTEB: Massive Multilingual Text Embedding Benchmark." ICLR 2025; arXiv:2502.13595. https://arxiv.org/abs/2502.13595 . Read: abstract, Section on benchmarks, Table 17.
21. Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, Marc Brockschmidt. "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search." arXiv:1909.09436, v3, 8 June 2020. https://arxiv.org/abs/1909.09436 . Read: abstract and Section 2.
22. Fuwei Zhang, Yanzhao Zhang, Mingxin Li, Dingkun Long, Lexiang Hu, Pengjun Xie, Zhao Zhang, Fuzhen Zhuang. "CORE-Bench: A Comprehensive Benchmark for Code Retrieval in the Era of Agentic Coding." Accepted at EMNLP 2026 (main); arXiv:2606.11864, v3, 24 August 2026. https://arxiv.org/abs/2606.11864 . Read: full text including Tables 3, 16, 17, 19, 20.
23. Leonardo Venuta, Francesco Tosoni, Paolo Ferragina. "Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval." Accepted at WISE 2026; arXiv:2606.27401, v2, 18 September 2026 (preliminary version). https://arxiv.org/abs/2606.27401 . Read: full text.
24. Siqiao Xue, Zihan Liao, Jin Qin, Ziyin Zhang, Yixiang Mu, Fan Zhou, Hang Yu. "Beyond Retrieval: A Multitask Benchmark and Model for Code Search." arXiv:2605.04615, v2, 8 May 2026. https://arxiv.org/abs/2605.04615 . Read: full text.
25. Aaryan Kapoor, Md Abdullah Al Hafiz Khan. "ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval." Accepted at EMNLP 2026 (main); arXiv:2609.01865, 1 September 2026. https://arxiv.org/abs/2609.01865 . Read: abstract, Sections 1 and 5.
26. Aaron Nobile, Andreas Adelmann, Mohsen Sadr. "Retrieval-Augmented Generation for Scientific Code Understanding." arXiv:2609.12190, 10 September 2026. https://arxiv.org/abs/2609.12190 . Read: abstract and Section 3.
27. Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi. "Matryoshka Representation Learning." Advances in Neural Information Processing Systems 35 (NeurIPS 2022); arXiv:2205.13147. https://arxiv.org/abs/2205.13147 . Read: Sections 3-4.
28. Aamir Shakir, Tom Aarsen, Sean Lee. "Binary and Scalar Embedding Quantization for Significantly Faster and Cheaper Retrieval." Hugging Face Blog, 22 March 2024. https://huggingface.co/blog/embedding-quantization . Read: full post.
29. UKPLab and sentence-transformers contributors. "sentence_transformers/util/quantization.py" (function `quantize_embeddings`). GitHub, master branch, accessed 2026-09-29. https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util/quantization.py . Read: function body.
30. Naaman Huerga-Perez, Ruben Alvarez, Ruben Ferrero-Guillen, Alberto Martinez-Gutierrez, Javier Diez-Gonzalez. "Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques." arXiv:2505.00105, 30 April 2025. https://arxiv.org/abs/2505.00105 . Read: full text.
31. Riku Kisako, Hayato Tsukagoshi, Ryohei Sasano. "When Is 0.1% Enough? Analyzing the Combined Effects of Dimensionality Reduction and Quantization on Text Embedding Compression." arXiv:2606.01074, 31 May 2026. https://arxiv.org/abs/2606.01074 . Read: full text.
32. Herve Jegou, Matthijs Douze, Cordelia Schmid. "Product Quantization for Nearest Neighbor Search." IEEE Transactions on Pattern Analysis and Machine Intelligence 33(1):117-128, 2011. DOI 10.1109/TPAMI.2010.57. Author version: https://www.irisa.fr/texmex/people/jegou/papers/jegou_searching_with_quantization.pdf . Read: Sections II-III.
33. Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria Lomeli, Lucas Hosseini, Herve Jegou. "The Faiss library." arXiv:2401.08281, v4, 23 October 2025. https://arxiv.org/abs/2401.08281 . Read: Sections on quantisers.
34. Meta Platforms and Faiss contributors. "faiss/Clustering.h and faiss/Clustering.cpp." GitHub, main branch, accessed 2026-09-29. https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h . Read: `ClusteringParameters` and the training-size warning.
35. Jianyang Gao, Cheng Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 2(3), article 167, 27 pages, May 2024 (SIGMOD 2024). DOI 10.1145/3654970. arXiv:2405.12497. https://arxiv.org/abs/2405.12497 . Read: Sections 3-5.
36. Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, Raymond Chi-Wing Wong. "Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search." Proceedings of the ACM on Management of Data 3(3), article 202, 26 pages, June 2025 (SIGMOD 2025). DOI 10.1145/3725413. arXiv:2409.09913. https://arxiv.org/abs/2409.09913 . Read: Sections 2, 3, 5.
37. VectorDB-NTU. "RaBitQ Library" (README). GitHub, accessed 2026-09-29. https://github.com/VectorDB-NTU/RaBitQ-Library . Read: README and repository metadata.
38. Jianyang Gao, Yutong Gou, Yuexuan Xu, Jifan Shi, Yongyi Yang, Shuolin Li, Raymond Chi-Wing Wong, Cheng Long. "Revisiting RaBitQ and TurboQuant: A Symmetric Comparison of Methods, Theory, and Experiments." arXiv:2604.19528, v2, 30 April 2026. https://arxiv.org/abs/2604.19528 . Read: abstract and Sections 1-5.
39. Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv:2504.19874, 28 April 2025. https://arxiv.org/abs/2504.19874 . Read: abstract and first page. Reference [38] states it was accepted at ICLR 2026; the OpenReview page could not be opened, so that venue is not confirmed first-hand.
40. Heesang Ann, Joongkyu Lee, Min-hwan Oh. "Block-Sphere Vector Quantization." arXiv:2605.19972, 19 May 2026. https://arxiv.org/abs/2605.19972 . Read: abstract.
41. Hui Li, Shiyuan Deng, Xiao Yan, Xiangyu Zhi, James Cheng. "SAQ: Pushing the Limits of Vector Quantization through Code Adjustment and Dimension Segmentation." Proceedings of the ACM on Management of Data 3(6), 25 pages, December 2025. DOI 10.1145/3769824. arXiv:2509.12086. https://arxiv.org/abs/2509.12086 . Read: abstract.
42. Wenxuan Xiao, Xu Cao. "When Does Low-Bit Quantization Preserve the Decisions of Vector Search?" arXiv:2609.09854, 9 September 2026. https://arxiv.org/abs/2609.09854 . Read: Sections 1-3 and experimental summary.
43. Kawin Ethayarajh. "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." Proceedings of EMNLP-IJCNLP 2019, pages 55-65. DOI 10.18653/v1/D19-1006. https://aclanthology.org/D19-1006/ ; arXiv:1909.00512. Read: full text.
44. Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. "Representation Degeneration Problem in Training Natural Language Generation Models." ICLR 2019; arXiv:1907.12009. https://arxiv.org/abs/1907.12009 . Read: abstract and Sections 1-2.
45. Nathan Godey, Eric de la Clergerie, Benoit Sagot. "Anisotropy Is Inherent to Self-Attention in Transformers." Proceedings of EACL 2024 (Volume 1: Long Papers), pages 35-48. DOI 10.18653/v1/2024.eacl-long.3. https://aclanthology.org/2024.eacl-long.3/ ; arXiv:2401.12143. Read: abstract and Sections 1-3.
46. Raphael Bernas, Fanny Jourdan, Antonin Poche, Celine Hudelot. "Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics." arXiv:2604.08764, 9 April 2026. https://arxiv.org/abs/2604.08764 . Read: abstract and Section 1.
47. Jiaqi Mu, Pramod Viswanath. "All-but-the-Top: Simple and Effective Postprocessing for Word Representations." ICLR 2018. arXiv:1702.01417 (the arXiv record additionally lists Suma Bhat as second author). https://arxiv.org/abs/1702.01417 . Read: Sections 1-2 and Algorithm 1.
48. Jianlin Su, Jiarun Cao, Weijie Liu, Yangyiwen Ou. "Whitening Sentence Representations for Better Semantics and Faster Retrieval." arXiv:2103.15316, 29 March 2021. https://arxiv.org/abs/2103.15316 . Read: abstract and Section 1.
49. Milos Radovanovic, Alexandros Nanopoulos, Mirjana Ivanovic. "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data." Journal of Machine Learning Research 11(86):2487-2531, 2010. https://jmlr.org/papers/v11/radovanovic10a.html . Read: Sections 1-5.
50. Ikumi Suzuki, Kazuo Hara, Masashi Shimbo, Marco Saerens, Kenji Fukumizu. "Centering Similarity Measures to Reduce Hubs." Proceedings of EMNLP 2013, pages 613-623. https://aclanthology.org/D13-1058/ . Read: full text.
51. Prashant Kumar Pathak, Tarun Kumar Sharma. "When Global Gating Is Enough: Admission-Time Hubness Control in Anisotropic Vector Retrieval Systems." arXiv:2606.19692, 18 June 2026. https://arxiv.org/abs/2606.19692 . Read: abstract and Section 1.
52. Adib Sakhawat, Fardeen Sadab, Atik Shahriar. "Hubness, Not Anisotropy, Drives Cross-Lingual Retrieval Asymmetry in Multilingual Embedding Models." arXiv:2605.26575, 26 May 2026. https://arxiv.org/abs/2605.26575 . Read: abstract.
53. Tongzhou Wang, Phillip Isola. "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119:9929-9939. https://proceedings.mlr.press/v119/wang20k.html ; arXiv:2005.10242. Read: Sections 4.1-4.2.
54. Arindam Banerjee, Inderjit S. Dhillon, Joydeep Ghosh, Suvrit Sra. "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions." Journal of Machine Learning Research 6(46):1345-1382, 2005. https://jmlr.org/papers/v6/banerjee05a.html . Read: Section 2.1.
55. Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu. "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree." Findings of the Association for Computational Linguistics: EMNLP 2025, pages 8106-8116. DOI 10.18653/v1/2025.findings-emnlp.430. https://aclanthology.org/2025.findings-emnlp.430/ ; arXiv:2506.15655. Read: full text.
56. Xinjian Wu, Jingzhi Gong, Gunel Jahangirova, Jie Zhang. "How Does Chunking Affect Retrieval-Augmented Code Completion? A Controlled Empirical Study." arXiv:2605.04763, 6 May 2026. https://arxiv.org/abs/2605.04763 . Read: full text.
