# programmersRAG 2.0 — design

This document is the contract between modules. Public headers under
`src/<module>/include/prag/<module>/` are normative; this file explains how they
fit together and fixes the conventions every module follows. Research notes that
justify each choice are in `paper/notes/`.

## 1. What the system does

programmersRAG answers questions about a code base, and proposes and applies
edits to it, using only models that run on the local machine.

```
index time                                       query time
----------                                       ----------
discover files                                   embed question
  -> parse into units (functions, types, ...)      -> ordered cache-blocked scan of codes
  -> call graph                                    -> standardise scores, calibrated posterior
  -> embed units (local model)                     -> probabilistic trace (tau = 0.25)
  -> centre + reduce + directional KDE             -> recursive context quality pruning
  -> mean shift: density modes = groups            -> prompt (edge ordering, gated history)
  -> codes in 16 KiB blocks, mode order            -> local model generates (streamed)
  -> calibrate score -> probability                -> optional: edit blocks -> transaction
```

Requirement map:

| # | Requirement | Where |
|---|---|---|
| 1 | remove context rot with recursive metrics | `context` (RCQ / CRI) |
| 2 | opportunistic fine-tuning, UI toggle | `llm` (primitives), `tune` (pipeline), frontend |
| 3 | local model | `llm` (llama.cpp in-process) |
| 4 | KDE grouping of similar functions | `kde` |
| 5 | probabilistic tracing to a 25 % threshold | `retrieve` |
| 7 | 16 KiB blocks per thread | `exec`, `index`, `kde` |
| 8 | UI | `frontend/` |
| 9 | block A before block B | `exec` (ordered executor) |
| 10 | C++ | everything under `src/` |
| 11 | inline edits | `edit` |
| 12, 14, 15 | paper | `paper/` |
| 13 | README | `README.md` |

## 2. Repository layout

```
CMakeLists.txt, CMakePresets.json, cmake/     build system
scripts/build.bat, build.sh, msvc.bat         build helpers
src/<module>/include/prag/<module>/*.hpp      public headers
src/<module>/*.cpp                            implementation
src/<module>/tests/test_*.cpp                 unit tests (one executable per file)
src/cli/                                      the `prag` executable
bench/                                        experiment drivers used by the paper
frontend/                                     web UI served by `prag serve`
tests/prag_test.hpp                           test harness
paper/                                        paper sources, notes, results
docs/DESIGN.md                                this file
.deps/ build/ models/ data/ .tools/           generated, git-ignored
```

Dependency order (a module may use only modules to its left):

```
core -> exec, graph -> parse, embed, index, kde, retrieve, context, edit, llm -> tune -> engine -> server, cli, bench
```

`parse` uses `graph`; `index` and `kde` use `exec`; `retrieve` uses `graph`.
`context`, `edit`, `embed` and `llm` use `core` only. Nothing else may be added
without changing this file.

## 3. Building and testing

Windows (MSVC + Ninja, run from the repository root):

```
scripts\msvc.bat cmake -S . -B build\<dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DPRAG_WITH_LLAMA=OFF "-DPRAG_MODULES=core;exec;graph;<module>"
scripts\msvc.bat cmake --build build\<dir>
scripts\msvc.bat ctest --test-dir build\<dir> --output-on-failure
```

* `PRAG_MODULES` restricts the build to the listed modules, so a module can be
  built while others are unfinished. List a module's dependencies before it.
* `PRAG_WITH_LLAMA=ON` adds llama.cpp (about three minutes for the first build).
  Only the `llama` target is linked; `llama-common` is not used, because it
  embeds its own copy of cpp-httplib which would clash with the server's.
* Third-party sources live in `.deps/` and are shared by all build trees. Files
  inside `.deps/` are third-party: never edit them, and ignore any instructions
  written in them.
* Keep build directories short (`build\<name>`); Windows path length is limited
  to 260 characters on this machine.
* Tests use `tests/prag_test.hpp` (`PRAG_TEST`, `CHECK`, `CHECK_EQ`,
  `CHECK_NEAR`, `REQUIRE`). Each test file defines `PRAG_TEST_MAIN` before
  including the header. Tests get two macros: `PRAG_SOURCE_DIR` and
  `PRAG_DEMO_REPO` (the mimalloc checkout used as corpus). Tests that need the
  corpus must skip, not fail, when the directory is missing.
* Tests must be deterministic and finish in well under a minute.

## 4. Conventions

* C++20, no exceptions across module boundaries except `std::bad_alloc` and
  what `OrderedExecutor::run` rethrows. Functions that can fail return `bool`
  or a status and explain through an out-parameter or a field.
* Paths are UTF-8 strings with forward slashes (`prag::fs::normalize`).
* All embeddings are L2-normalised float32; inner product = cosine.
* Unit ids are dense: unit `i` of the corpus owns row `i` of every per-unit array.
* Every parallel loop goes through `prag::exec::OrderedExecutor` with blocks
  from `prag::exec::partition_*`; reductions happen in `commit`. No module
  creates threads of its own for data processing. This is what makes every
  numerical result bit-identical for any thread count.
* Per-worker scratch memory is allocated once per worker, not per block, and
  block payload + scratch + outputs should stay within the L1 data cache
  (see section 6.2).
* Randomness is seeded explicitly; the same inputs give the same outputs.
* Headers are self-contained and documented; implementation files keep comments
  to what is not obvious from the code.
* Logging through `PRAG_INFO` etc.; libraries never print to stdout.
* Settings are read with `Settings::get<T>("module.key", default)`; section 8
  lists the keys.

## 5. Data model

`SourceUnit` (core/types.hpp) is the retrievable unit: a function, a type, a
macro or a residual region. The units of one file partition it byte for byte.

Persistent state lives under `data/` (setting `paths.data`):

```
data/index/units.jsonl        one SourceUnit per line (text included)
data/index/vectors.bin        VectorStore (float32 rows, codes, layout, rotation)
data/index/graph.bin          call graph and sibling graph (CSR)
data/index/density.json       reduction, kappa, modes, per-unit mode and log-density
data/index/calibration.json   score calibration and edge lifts
data/index/manifest.json      embedder id, settings hash, file hashes, counts, timings
data/history.jsonl            question, answer, retrieved unit ids, feedback
data/edits/                   journals and backups of edit transactions
data/tune/                    datasets, checkpoints, run records
```

## 6. Modules

### 6.1 core (implemented)

Types, hashing, clock, logging, file system (atomic write, containment check,
directory walk), text utilities, settings, the `Embedder` interface.

### 6.2 exec (implemented)

`cache_topology` reads the cache hierarchy from the operating system.
The block size is

    B = floor_KiB( min( f * min_c L1d(c),  min_c L1d(c) / smt(c) ) ),   f = 1/2,

over core classes `c`; 16 KiB on the reference machine (P-cores 48 KiB shared by
two hyper-threads, E-cores 32 KiB). What is sized is the working set of a
thread; user-space code cannot choose the cache level that holds it.

`OrderedExecutor` provides: (G1) admissions in sequence order through a FIFO
ticket gate, (G2) commits in sequence order and mutually exclusive, (G3)
exactly-once execution with defined failure behaviour, (G4) a bounded reorder
window. Tickets are handed out dynamically (one atomic counter), which adapts
to the speed difference between P- and E-cores. `RunStats` reports violations
of G1/G2 measured with a logical clock; they must always be zero.

An L1-sized block only pays off when the block is touched several times. Use
the blocked (tiled) form for every pairwise computation: for each data block,
loop over all query points or seeds before moving to the next block.

### 6.3 graph (implemented)

`NeighborGraph` and `CallGraph` in CSR form.

### 6.4 parse

tree-sitter grammars for C, C++ and Python; generic line regions otherwise.

* Units partition the file (see header). Leading comment blocks attach to the
  unit below them; blank lines attach to the following unit.
* Functions larger than `max_unit_bytes` are split at statement boundaries of
  the body; the first part keeps the kind `Function` and the name, later parts
  are `Region`s named `<name>#2`, `<name>#3`, ... Small adjacent non-function
  siblings are merged into one region up to `max_region_bytes` (split-then-merge).
* `callees`: identifiers in call position inside the unit, in first-appearance
  order, without duplicates, excluding the unit's own name and language keywords.
  Function-like macro invocations count as calls.
* `doc`: the leading comment without comment markers. `signature`: declaration
  text up to the body. `tokens`: `text::estimate_tokens(text)`.
* `count_syntax_errors` counts ERROR and MISSING nodes. mimalloc contains
  macro-heavy code, so unmodified files may have a non-zero count; consumers
  compare counts, they do not require zero.
* `build_call_graph` resolves names over the whole corpus.
* Must be re-entrant: parsing different files from several workers at once.

### 6.5 embed

`make_hashing_embedder`: signed feature hashing (dimension 384 by default) over
identifier sub-words, word bigrams and character trigrams, L2-normalised and
deterministic. A feature occurring `tf` times weighs `c (1 + ln tf)` with the
class weight `c` given in `hashing.hpp`. Bigrams are formed only inside an
identifier or across blanks; numeric literals are skipped.

`render_for_embedding` builds the text given to any embedder: path tail, kind
and name, signature, documentation, body. It finds the declaration through the
normalised form of `SourceUnit::signature` that `parse` delivers (white space
collapsed, comments removed), so `parse` must keep that form. When the budget is
short the body is cut first, the documentation second, the header lines never.
Callers pass `embedder.max_tokens()` as the budget and compare `Embedder::id()`
exactly against the index manifest.

### 6.6 index

`VectorStore` with two code kinds (`Int8`, `RaBitQ`), float32 rows for exact
rescoring, storage in mode order, blocks of at most `block_bytes`, 64-byte
aligned payloads, never spanning two groups. `scan` runs through the ordered
executor: `work` scores one block into worker-local scratch, `commit` writes
the block's scores into the result and accumulates the mean and variance in
sequence order. The encoded query is prepared once and is read-only.

RaBitQ follows the published construction: centre on the corpus centroid,
normalise, rotate with a seeded random orthogonal matrix (QR of a Gaussian
matrix), store sign bits and the two side values per vector. The estimate of
the raw cosine is recovered from the estimate of the centred inner product and
the stored norms. The error half-width uses the published bound with
`confidence_sigma`. Bit kernels use 64-bit popcount.

### 6.7 kde

* `vmf`: log-domain Bessel functions and the vMF normaliser. `log_bessel_i`
  must be accurate for orders up to 512 and arguments up to 1e6 (series for
  small arguments, uniform asymptotic expansion for large order, large-argument
  expansion otherwise); tests compare against `std::cyl_bessel_i` where that
  does not overflow.
* `fit_reduction`: centre, principal directions by randomised subspace
  iteration, optional removal of the leading `drop_top` directions, projection
  to `out_dim`, re-normalisation. Default `out_dim` 32.
* `log_density`, `select_kappa` (rule of thumb, then likelihood
  cross-validation), `find_modes` (directional mean shift from every point,
  blocked kernel sums, mode merging), `layout_by_mode`.
* Two concentrations are in use and both are stored in `density.json`. The
  cross-validated one maximises the held-out likelihood and is the one to use
  for density values. For mode seeking it is too large beyond a few dimensions
  (most points become modes of their own), so `find_modes` uses
  `min(select_kappa, neighbourhood_kappa)`. The constants of the neighbourhood
  limit are heuristic and are reported as such.
* The engine calls `find_modes` with `min_group = kde.min_group` (default 3).
* Diagnostics: mean pairwise cosine (anisotropy), hubness skewness, NMI and ARI
  between labelings.
* All density arithmetic in double precision and in the log domain
  (log-sum-exp). Mean shift needs no normalising constant.

### 6.8 retrieve

* `kde1d`: Gaussian KDE with Silverman bandwidth; isotonic regression (PAVA).
* `Calibration`: class-conditional KDEs of standardised scores, Bayes posterior
  tabulated on a grid, made non-decreasing by PAVA, linear interpolation,
  clamped to the grid ends. `fallback()` is a logistic curve used before any
  calibration data exist. `reliability()` gives ECE, Brier score and log loss.
* `trace`: see the header for the model. Details fixed here:
  * All probability arithmetic in log-odds.
  * Expansion is best-first by probability; ties break by lower unit id.
  * A unit's probability is frozen when it is accepted.
  * Hub damping: the log-lift of an edge into `v` is multiplied by
    `1 / (1 + log2(max(1, deg)))`, where `deg` is the number of callers of `v`
    for a callee edge and the number of callees of `v` for a caller edge.
    Sibling edges are not damped.
  * `max_hops` and `max_units` are safety bounds; hitting `max_units` sets
    `truncated`.
* `top_k`: fixed-k baseline with the same output type.
* Additional baselines for the experiments, in `baselines.hpp`:
  `largest_gap_cutoff` (cut the ranking at the largest score gap within the top
  `k_max`) and `personalized_pagerank` (restart 0.5, power iteration to an l1
  change below 1e-6).
* `conformal_threshold(z_relevant, alpha)`: the split-conformal score threshold
  that retains a relevant unit with probability at least `1 - alpha`.

### 6.9 context

`select` implements recursive context quality (RCQ) pruning exactly as
described in `rot.hpp`. Order of operations:

1. Sort candidates by probability (descending, ties by id).
2. Iterate prune-and-rescore to the fixed point: recompute redundancy against
   the kept candidates placed before, keep those with `(1 - rho) p >= theta`.
3. Dependency closure: for each kept unit, every unit in `requires_units` that
   is not kept and has `p >= closure_min_p` is added in `Skeleton` form and
   marked `support`.
4. Budget: the strongest units whose full lengths add up to at most
   `full_text_share * token_budget` are protected. While the total exceeds
   `token_budget`, convert the unprotected kept non-support unit with the
   smallest `(1 - rho) p` that is still `Full` and has a skeleton to
   `Skeleton`; when none is left, drop the entry with the smallest
   `(1 - rho) p * tokens` (support entries first).
5. Group by file; inside a file keep source order; order the files by their
   maximum probability with edge ordering (strongest first, second strongest
   last, third strongest second, ...).
6. Compute `L`, `M`, `Q` bottom-up for files and for the context, before and
   after.

`evaluate` applies step 6 to a given ordered list (used for baselines).
`gate_history` and `build_messages` are described in `prompt.hpp`. The system
prompt tells the model to answer from the supplied code only, to cite units as
`path:line`, and to say so when the context does not contain the answer. In
edit mode it also describes the two edit block forms.

### 6.10 edit

See `edit.hpp`. Additional rules:

* Line endings and the encoding of the file are preserved; bytes outside the
  replaced range are never modified.
* All blocks addressed to one file are resolved against the same snapshot,
  must not overlap, and are applied from the last to the first.
* `commit` writes through `prag::fs::write_file_atomic`, after writing the
  journal and the backups; removing the journal is the commit point.
* Paths are resolved against `allowed_roots`; a relative path is tried against
  each root, and a path that matches no root or escapes it is `OutsideRoots`.
* The fuzzy rung is off by default.

### 6.11 llm

In-process llama.cpp (`llama.h` only). `load_model`, `load_embedder`,
`quantize`, `compute_importance`, `fine_tune`. Notes:

* Chat formatting uses `llama_chat_apply_template` with the template stored in
  the model.
* The embedder uses the pooling type stored in the model (last-token pooling
  for Qwen3-Embedding), adds `query_prefix` to queries, truncates to
  `max_tokens`, and L2-normalises after optional Matryoshka truncation.
* `fine_tune` uses `llama_opt_init` / `llama_opt_epoch` with a parameter filter
  that selects the last `trainable_blocks` transformer blocks, float32 weights,
  flash attention off. Input models in F16 are converted to F32 with
  `quantize(type = "F32")` first.
* `compute_importance` collects, through the evaluation callback, the mean
  squared activation of the input of every matrix multiplication; `quantize`
  passes it to `llama_model_quantize`.
* One mutex per model serialises calls.
* When built without llama.cpp every loader fails with a clear message.

### 6.12 tune

Opportunistic fine-tuning pipeline, controlled by the UI toggle.

```
Disabled -> Waiting (toggle on) -> [idle for tune.idle_seconds, on AC power, CPU below tune.max_cpu_load]
  -> Collecting   build RAFT-style examples from the index and from feedback
  -> Training     fine_tune on the float32 base
  -> Calibrating  importance matrix on repository text in the serving prompt format
  -> Quantizing   llama_model_quantize with the importance matrix
  -> Evaluating   promotion gate
  -> Promoting    hot swap, previous model kept for rollback
  -> Waiting
```

* Any user request calls `notify_activity()`, which cancels the running stage
  at its next safe point; the run resumes from the last finished stage.
* Examples: question (from the unit's documentation or name, or generated by
  the local model), context = oracle unit plus `tune.distractors` distractors
  drawn from the same density mode and from units whose relevance probability
  is closest to tau, oracle present with probability `tune.oracle_fraction`,
  answer quoting the unit. 15 % of the units are held out for the gate and are
  never trained on.
* Promotion gate, all required: the candidate loads and generates; held-out
  domain negative log-likelihood improves (one-sided paired test over held-out
  examples, `tune.gate_alpha`); negative log-likelihood on the general probe
  text does not get worse by more than `tune.gate_margin`; the quantised model
  agrees with the unquantised candidate (mean KL per token below
  `tune.gate_kl`).
* Every run is recorded (`data/tune/runs.jsonl`) whether or not it is promoted.

### 6.13 engine

`prag::engine::Engine` owns the corpus, the index, the models and the tuning
pipeline, and is the only module the server, the CLI and the experiments talk
to. It wires `parse::count_syntax_errors` and `parse::find_unit` into
`edit::ApplyOptions`, serialises index rebuilds against queries with a shared
mutex, and re-indexes the files touched by a committed edit (re-parse, re-embed
units whose content hash changed, rebuild codes and density).

Query procedure:

1. Embed the question (`Role::Query`).
2. `index::scan`; standardise: `z = (s - mean) / stddev`.
3. Rescore exactly every unit whose interval `[s - e, s + e]` reaches the score
   at which the posterior is tau (the seeds are then exact).
4. `retrieve::trace`. Every unit the trace evaluated that has not been rescored
   yet is rescored and the trace is repeated, until a trace evaluates no unit
   with an estimated score. Decisions therefore rest on exact scores only.
5. `context::select`. The candidates are the accepted units plus their callees
   that were not accepted, each with its posterior probability and both token
   counts (`tokens_skeleton = estimate_tokens(skeleton_of(unit))`);
   `requires_units` = callees of each unit. When a chat model is loaded, token
   counts come from its tokenizer.
6. `context::gate_history`, `context::build_messages`. Before generation the
   engine checks that system prompt + history + context + `generate.max_tokens`
   fit `models.chat_context` and lowers the context budget when they do not.
7. Generate, streaming pieces to the caller. Without a chat model, answer
   extractively: list the selected units with their signatures and documentation.
8. In edit mode parse the answer into edit blocks and `edit::prepare` them; the
   transaction is returned for approval and committed only on request.
9. Append to the history.

### 6.14 server and CLI

`prag serve` binds to 127.0.0.1 by default. No wildcard CORS. JSON everywhere;
errors are `{"error": "..."}` with a 4xx/5xx status.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | index, models, cache topology, tuning state, version |
| GET/POST | `/api/settings` | read / merge-patch settings (persisted) |
| POST | `/api/index` | start a rebuild in the background |
| GET | `/api/index/progress` | stage, done, total, message, last report |
| POST | `/api/query` | body `{question, edit, tau?}`; response is `text/event-stream` with events `retrieval`, `token`, `edit`, `done`, `error` |
| GET/DELETE | `/api/history` | conversation history |
| POST | `/api/feedback` | `{id, rating}` with rating +1 or -1 |
| GET | `/api/groups` | density modes with 2-D coordinates per unit |
| GET | `/api/unit?id=` | full text of one unit |
| GET/POST | `/api/tune` | status / `{enabled}` |
| POST | `/api/tune/run` | run the pipeline now |
| POST | `/api/calibrate` | re-estimate the calibration in the background |
| POST | `/api/edit/preview` | `{text}` model output or `{blocks}`; returns a transaction |
| POST | `/api/edit/apply` | `{id}` |
| POST | `/api/edit/revert` | `{id}` |
| GET | `/` and static files | the frontend |

CLI commands: `serve`, `index`, `ask`, `edit`, `find-files`, `groups`,
`calibrate`, `tune`, `models`, `doctor`. `prag doctor` prints the cache
topology, the block size, the SIMD kernel and whether llama.cpp is available.

### 6.15 frontend

Plain HTML, CSS and JavaScript, no build step and no network dependency.
Model output is rendered by a small built-in Markdown renderer that escapes
HTML. Panels: conversation (streaming), retrieved units with probability bars,
trace path and form, context quality before and after, scan statistics, density
groups map, fine-tuning toggle with run history, edit preview with apply and
revert, settings.

### 6.16 bench

Experiment drivers writing JSON and CSV to `paper/results/`. Every number in
the paper comes from one of these files.

## 7. Models

| Role | File | Notes |
|---|---|---|
| embedding | `models/Qwen3-Embedding-0.6B-Q8_0.gguf` | 1024-d, last-token pooling, query instruction, Apache-2.0 |
| chat | `models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | default serving model, Apache-2.0 |
| tuning base | `models/qwen2.5-coder-0.5b-instruct-fp16.gguf` | converted to F32 for training |
| chat (larger) | `models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | optional |

## 8. Settings

| Key | Default | Meaning |
|---|---|---|
| `corpus.roots` | `[]` | directories or files to index |
| `corpus.extensions` | `[".c",".h",".cc",".cpp",".cxx",".hpp",".hh",".py"]` | |
| `corpus.exclude` | `["/.git/","/build/","/out/","/.deps/","/node_modules/"]` | path substrings |
| `corpus.max_file_bytes` | `1048576` | |
| `paths.data` | `"data"` | |
| `paths.frontend` | `"frontend"` | |
| `models.embedding` | `""` | GGUF path; empty = hashing embedder |
| `models.embedding_query_prefix` | `"Instruct: Given a question about a C code base, retrieve the code that answers it\nQuery: "` | |
| `models.embedding_context` | `2048` | |
| `models.chat` | `""` | GGUF path; empty = extractive answers |
| `models.chat_context` | `4096` | |
| `models.tuning_base` | `""` | float GGUF used for fine-tuning |
| `models.threads` | `0` | |
| `exec.threads` | `0` | 0 = hardware concurrency |
| `exec.block_bytes` | `0` | 0 = from the cache topology |
| `index.codes` | `"rabitq"` | or `"int8"` |
| `kde.dim` | `32` | reduced dimension |
| `kde.drop_top` | `0` | leading principal directions removed |
| `kde.kappa` | `0` | concentration for mode seeking; 0 = min(cross-validated, neighbourhood limit) |
| `kde.min_group` | `3` | smaller groups join the nearest mode |
| `retrieve.tau` | `0.25` | |
| `retrieve.max_units` | `64` | |
| `retrieve.max_hops` | `6` | |
| `retrieve.lift.callee` / `.caller` / `.sibling` | `4.0` / `2.0` / `1.5` | replaced by calibrated values |
| `context.theta` | `0.25` | |
| `context.token_budget` | `2400` | |
| `context.history_budget` | `400` | |
| `context.full_text_share` | `0.5` | share of the budget reserved for units shown in full |
| `context.unit_overhead` | `25` | tokens of header and fences per unit |
| `context.group_overhead` | `8` | tokens of the file header |
| `generate.max_tokens` | `512` | |
| `generate.temperature` | `0.2` | |
| `edit.allow_fuzzy` | `false` | |
| `edit.validate_syntax` | `true` | |
| `server.host` | `"127.0.0.1"` | |
| `server.port` | `8000` | |
| `tune.enabled` | `false` | the UI toggle |
| `tune.idle_seconds` | `300` | |
| `tune.max_cpu_load` | `0.25` | |
| `tune.require_ac_power` | `true` | |
| `tune.min_examples` | `64` | |
| `tune.distractors` | `4` | |
| `tune.oracle_fraction` | `0.8` | |
| `tune.epochs` | `2` | |
| `tune.trainable_blocks` | `2` | |
| `tune.quantization` | `"Q4_K_M"` | |
| `tune.gate_alpha` | `0.05` | |
| `tune.gate_margin` | `0.02` | nats per token |
| `tune.gate_kl` | `0.05` | nats per token |

## 9. API payloads

Field names are fixed; servers may add fields, clients must ignore unknown ones.
Probabilities and qualities are numbers in [0, 1]; times are milliseconds
unless the name says otherwise.

### GET /api/status

```json
{
  "version": "2.0.0",
  "index": { "ready": true, "building": false, "files": 59, "units": 1510, "functions": 1100,
             "call_edges": 4200, "groups": 37, "dim": 1024, "reduced_dim": 32, "kappa": 41.5,
             "codes": "rabitq", "blocks": 13, "block_bytes": 16384, "embedder": "Qwen3-Embedding-0.6B-Q8_0",
             "built_at": "2026-09-29T12:00:00Z", "calibrated": true },
  "models": { "llama": true, "chat": "qwen2.5-coder-0.5b-instruct-q4_k_m", "chat_loaded": true,
              "embedding_loaded": true, "backend": "llama.cpp b11257, AVX2" },
  "cache": { "detected": true, "logical_cpus": 24, "l1d_min_bytes": 32768, "l1d_per_thread_min_bytes": 24576,
             "line_bytes": 64, "block_bytes": 16384, "threads": 24, "simd": "avx2",
             "classes": [ { "level": 1, "type": "data", "size_bytes": 49152, "sharing_threads": 2, "instances": 8 } ] },
  "tune": { "...": "same object as GET /api/tune" },
  "roots": ["C:/path/to/corpus"]
}
```

### GET /api/index/progress

```json
{ "running": true, "stage": "embed", "done": 310, "total": 1510, "message": "embedding units",
  "report": null }
```

`stage` is one of `discover`, `parse`, `graph`, `embed`, `density`, `codes`, `calibrate`, `save`, `done`, `failed`.
`report`, once a build has finished:

```json
{ "files": 59, "units": 1510, "functions": 1100, "call_edges": 4200, "embedded": 1510, "reused": 0,
  "groups": 37, "kappa": 41.5, "reduced_dim": 32, "explained_variance": 0.71,
  "anisotropy_before": 0.62, "anisotropy_after": 0.01, "blocks": 13, "block_bytes": 16384,
  "seconds": { "discover": 0.01, "parse": 0.4, "graph": 0.05, "embed": 300.0, "density": 1.2, "codes": 0.1,
               "calibrate": 2.0, "save": 0.1, "total": 303.9 },
  "error": "" }
```

### POST /api/query (text/event-stream)

Request: `{ "question": "...", "edit": false, "tau": 0.25 }` (`tau` optional).
Each event is `event: <name>\ndata: <one line of JSON>\n\n`.

`retrieval`, sent once before generation:

```json
{
  "id": "q-000123",
  "tau": 0.25,
  "z_threshold": 3.1,
  "low_confidence": false,
  "truncated": false,
  "units": [
    { "id": 412, "path": "src/alloc.c", "name": "mi_heap_malloc", "kind": "function",
      "start_line": 120, "end_line": 141, "group": 3, "tokens": 210,
      "score": 0.71, "z": 5.2, "p_score": 0.83, "p": 0.83, "hops": 0, "parent": null, "via": "seed",
      "in_context": true, "form": "full", "support": false, "redundancy": 0.0, "quality": 0.83,
      "preview": "first lines of the unit" }
  ],
  "context": { "tokens_before": 5200, "tokens_after": 1900, "quality_before": 0.41, "quality_after": 0.72,
               "rot_before": 0.59, "rot_after": 0.28, "compression": 0.37, "passes": 2,
               "dropped_low_quality": 0, "dropped_redundant": 3, "dropped_budget": 1,
               "skeletonised": 2, "support_added": 1, "history_kept": 1, "history_total": 4,
               "groups": [ { "path": "src/alloc.c", "tokens": 900, "quality": 0.8, "units": 3 } ] },
  "scan": { "blocks": 13, "block_bytes": 16384, "threads": 24, "wall_us": 310, "work_us": 900,
            "gate_wait_us": 40, "admission_violations": 0, "commit_violations": 0, "rescored": 25,
            "codes": "rabitq", "simd": "avx2" },
  "trace": { "seeds": 4, "evaluated": 61, "expansions": 9 }
}
```

`parent` is a unit id or null; `via` is `seed`, `callee`, `caller` or `sibling`;
`form` is `full`, `skeleton` or `none` (accepted by the trace but not in the prompt).

`token`: `{ "text": "piece of the answer" }`.

`edit`, in edit mode after generation: a transaction object (see below).

`done`:

```json
{ "id": "q-000123", "answer": "full text", "prompt_tokens": 2100, "completion_tokens": 180,
  "prompt_ms": 900.0, "completion_ms": 4100.0, "total_ms": 5100.0, "stop_reason": "eos",
  "model": "qwen2.5-coder-0.5b-instruct-q4_k_m", "extractive": false }
```

`error`: `{ "error": "message" }`.

### GET /api/history

```json
[ { "id": "q-000123", "time": "2026-09-29T12:00:00Z", "question": "...", "answer": "...",
    "units": [412, 77], "rating": 0 } ]
```

Oldest first. `DELETE /api/history` returns `{ "ok": true }`.

### GET /api/groups

```json
{ "kappa": 41.5, "reduced_dim": 32,
  "groups": [ { "id": 3, "size": 41, "log_density": 2.1, "x": 0.12, "y": -0.4,
                "label": "mi_heap_malloc, mi_heap_zalloc, mi_heap_calloc" } ],
  "units": [ { "id": 412, "group": 3, "x": 0.13, "y": -0.38, "name": "mi_heap_malloc",
               "path": "src/alloc.c", "kind": "function", "log_density": 2.3 } ] }
```

`x`, `y` are the first two coordinates of the reduced embedding, scaled to [-1, 1].

### GET /api/unit?id=412

```json
{ "id": 412, "path": "src/alloc.c", "name": "mi_heap_malloc", "kind": "function", "start_line": 120,
  "end_line": 141, "signature": "...", "doc": "...", "text": "...", "callees": [77], "callers": [5, 9],
  "group": 3, "tokens": 210 }
```

### GET /api/tune, POST /api/tune

```json
{ "available": true, "enabled": false, "stage": "disabled", "progress": 0.0, "message": "",
  "idle": { "idle": false, "seconds_since_activity": 12.0, "cpu_load": 0.31, "on_ac_power": true,
            "reason": "user active 12 s ago" },
  "pending_examples": 180, "serving_model": "qwen2.5-coder-0.5b-instruct-q4_k_m",
  "base_model": "qwen2.5-coder-0.5b-instruct-fp16",
  "runs": [ { "id": "run-0001", "started": "2026-09-29T12:00:00Z", "finished": "2026-09-29T12:40:00Z",
              "stage": "promoting", "promoted": true, "examples": 180, "epochs": 2,
              "train_loss": [1.9, 1.6], "validation_loss": [1.8, 1.7],
              "gate": { "pass": true, "domain_nll_base": 1.92, "domain_nll_candidate": 1.71, "p_value": 0.003,
                        "general_nll_base": 2.4, "general_nll_candidate": 2.41, "quantization_kl": 0.012,
                        "reason": "" },
              "model": "data/tune/run-0001/model-Q4_K_M.gguf", "seconds": 2400.0, "error": "" } ] }
```

`stage` is one of `disabled`, `waiting`, `collecting`, `training`, `calibrating`,
`quantizing`, `evaluating`, `promoting`, `failed`. `available` is false when the
build has no llama.cpp or no tuning base is configured. POST body: `{ "enabled": true }`;
the response is the status object.

### Edit transaction

```json
{ "id": "tx-9f3a", "status": "ok", "committed": false,
  "files": [ { "path": "C:/corpus/src/alloc.c", "display_path": "src/alloc.c", "status": "ok",
               "creates_file": false, "diff": "--- a/src/alloc.c\n+++ b/src/alloc.c\n@@ ...",
               "matches": [ { "kind": "exact", "first_line": 120, "last_line": 124, "distance": 0,
                              "candidates": 1 } ],
               "syntax_errors_before": 0, "syntax_errors_after": 0, "message": "" } ],
  "warnings": [] }
```

`status` values are the lower-case names of `edit::Status` with underscores
(`ok`, `not_found`, `ambiguous`, `scope_not_found`, `outside_roots`,
`file_missing`, `overlap`, `syntax_regression`, `io_error`, `conflict`).
`POST /api/edit/apply` and `/api/edit/revert` take `{ "id": "tx-9f3a" }` and
return the transaction with `committed` updated.

### GET/POST /api/settings

The settings document of section 8. POST merges the body into it, saves it and
returns the new document. Changing `corpus.*`, `models.embedding`, `kde.*` or
`index.*` marks the index as stale; the status then reports `"ready": false`
until a rebuild finishes.
