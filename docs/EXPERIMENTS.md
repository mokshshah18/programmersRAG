# Experiments

Every number reported in the paper is produced by a driver in `bench/` and
stored under `paper/results/`. A driver never prints a number that it did not
measure in that run. Each result file starts with a `meta` object:

```json
{ "experiment": "e1_retrieval", "started": "...", "finished": "...", "git_commit": "...",
  "corpus": { "path": "...", "commit": "...", "files": 0, "units": 0, "functions": 0 },
  "machine": { "cpu": "...", "logical_cpus": 0, "l1d_bytes": [0], "ram_gb": 0.0, "os": "..." },
  "models": { "embedding": "...", "chat": "..." },
  "settings": { "...": "every setting that influences the result" },
  "seed": 0 }
```

The corpus is the demo repository: a fork of the mimalloc allocator checked out
at `Demo repo/mimalloc`. Drivers record its commit hash. Drivers never modify
the demo repository; experiments that change files work on a copy made under
`build/`.

All drivers are sub-commands of one executable, `prag-bench`:

```
prag-bench <experiment> --settings prag.json --out paper/results [--quick]
```

`--quick` runs a reduced version for smoke testing; its output is marked
`"quick": true` and must never be used in the paper.

## Query sets

Built once by `prag-bench queries` and stored in `paper/results/queries.json`.

* **Q-doc**: for every function whose documentation has at least 6 words, the
  documentation is the question and the function is the relevant unit. Because
  the documentation is part of the indexed unit, retrieval for Q-doc is run
  against embeddings of the unit with its leading comment removed.
* **Q-gen**: for a random sample of functions, a question written by the local
  chat model from the function body, instructed not to use the function's name.
  Questions that contain the function name anyway are discarded.
* Both sets are split by function into calibration (50 %), held-out (25 %) and
  test (25 %), with a fixed seed. Calibration fits the score densities; held-out
  is used for the reliability diagram and the conformal threshold; test is used
  for every reported retrieval number.
* **Utility labels**: for a subset of test queries, a candidate unit `v` is
  labelled useful when adding it to the context lowers the loss of the
  reference answer under the serving model by at least 0.02 nats per token:
  `nll(answer | question, oracle) - nll(answer | question, oracle + v) >= 0.02`.
  Candidates are callees, callers and same-group neighbours of the oracle and an
  equal number of random units.

## E1 retrieval (`e1_retrieval.json`)

Methods: fixed top-k for k in {1, 3, 5, 10, 20}; largest-gap cut-off; personalised
PageRank top-k; trace with tau in {0.10, 0.25, 0.50}; trace without graph edges;
trace without hub damping; the split-conformal threshold at alpha in {0.1, 0.2}.
For each method and query set: recall of the relevant unit, mean and median set
size, precision, mean tokens, mean reciprocal rank of the underlying ranking,
and on the utility-labelled subset the recall and precision against useful
units. Report 95 % bootstrap confidence intervals (1000 resamples over queries).
Also: float32 exact scoring versus Int8 and RaBitQ with rescoring (recall
retention, number of rescored units).

## E2 calibration (`e2_calibration.json`)

Reliability diagram on the held-out split (10 equal-width bins), expected
calibration error, Brier score and log loss for: the KDE posterior, the KDE
posterior without the isotonic step, the logistic fallback, and Platt scaling
fitted on the calibration split. Class-conditional densities on a grid. The z
value at which the posterior reaches 0.25.

## E3 density groups (`e3_groups.json`)

Anisotropy and hubness before and after centring and reduction. Explained
variance versus reduced dimension in {8, 16, 32, 64}. Selected kappa (rule of
thumb and cross-validated) per dimension. Number of modes versus kappa.
Agreement (adjusted Rand index, normalised mutual information) between the
groups and (a) source files, (b) top-level directories, (c) spherical k-means
with the same number of clusters, (d) groups from 20 bootstrap resamples
(stability). Mean-shift iterations per point. Share of call-graph edges whose
end points share a group, against the same share under random assignment with
the same group sizes.

## E4 context rot (`e4_context_rot.json`)

For test queries with reference answers (the documentation of the oracle
function), the loss of the reference answer under the serving model
(nats per token) with these contexts:

| context | content |
|---|---|
| none | question only |
| oracle | the relevant unit only |
| oracle + k hard | plus the k most similar non-relevant units, k in {2, 4, 8, 16} |
| oracle + k random | plus k random units |
| top-k | the k best-scoring units, k in {5, 10, 20} |
| trace | the units accepted by the trace, unpruned |
| trace + RCQ | after recursive context quality pruning |

For every context also: tokens, quality Q, rot index, prompt evaluation time.
Reported: mean loss per condition with confidence intervals; the rot curve
(loss versus number of distractors and versus tokens); Spearman correlation
between the rot index of a context and its loss increase over the oracle
context; latency versus tokens.

## E5 execution (`e5_execution.json`)

Block-size sweep {1, 2, 4, 8, 16, 32, 64, 256, 1024} KiB and thread sweep
{1, 2, 4, 8, 16, 24} for: (a) a single-query scan of the corpus index, (b) a
single-query scan of a synthetic index of 1,000,000 vectors, (c) a batch of 64
queries processed block by block, (d) mean shift on the corpus. Median of at
least 7 repetitions with the inter-quartile range. Overhead of ordered
execution against an unordered parallel loop doing the same work. Gate waiting
time as a share of total time. Determinism: hash of the results for every
thread count and block size (all must agree where the mathematics says so) and
the number of admission and commit order violations (must be 0). The detected
cache topology.

No hardware performance counters are read; cache behaviour is inferred from
timing only and the paper says so.

## E6 fine-tuning and quantisation (`e6_tuning.json`)

Training set: examples built from the calibration and held-out functions of
the corpus (never from test functions). Base: the float32 conversion of the
tuning base model. For base and fine-tuned model: loss on the held-out domain
examples, loss on the general probe text, training and validation loss per
epoch, wall time, peak memory, trainable parameter count. Quantisation of the
fine-tuned model to Q8_0, Q6_K, Q5_K_M and Q4_K_M, each without and with the
repository importance matrix: file size, domain loss, mean KL divergence per
token from the float32 model, agreement of the most probable token, prompt and
generation speed. The promotion gate decision with all its inputs.

## E7 inline edits (`e7_edits.json`)

(a) Matching engine, no model: the known-target protocol on functions of the
corpus. For each case the intended result is constructed first, the edit is
rendered as a block, and the location surface is perturbed (re-indentation,
trailing blanks, line-ending conversion, unrelated insertions above the
target, drifted SEARCH text). Per rung and per perturbation: clean success,
clean failure, silent misapplication, with Wilson 95 % intervals.

(b) With the serving model: mechanical tasks with a constructible target
(insert a given comment as the first line of a function body; rename a local
variable; change a numeric literal; replace a given call by another). Per
task type, with and without the envelope grammar: share of answers that parse,
share that apply, share whose result equals the target, syntax regressions
caught, tokens generated, time.

(c) Build check: the corpus copy with all successfully applied edits of (b)
that equal their target is configured and compiled with the platform compiler;
report whether it builds.
