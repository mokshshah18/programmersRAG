# Engineering facts: llama.cpp API, dependencies and model files

Verified research note for programmersRAG (features 2, 3, 5, 7, 9 and 11: opportunistic local fine-tuning, a fully local model, a custom-quantised generator, thread placement, ordering, and machine-parseable edit proposals).

- Verification date: 2026-09-29.
- Status: second-pass, adversarial check of a first-pass dossier. Every source in the reference list was opened by the checker. llama.cpp files were fetched as raw files at tag `v0.5.0` and read; pull requests, issues and releases were read through the GitHub API; model repositories were read through the Hugging Face API and their GGUF headers were parsed from a ranged download; arXiv items were checked against the arXiv API record. Where only the abstract was read, the reference says so.
- Independent re-verification, 2026-09-30 (third pass): `git ls-remote` again resolves `v0.5.0^{}` to `7fe450e19305b828c199d602c23a8337aaa1f03b` (newest nightly now `b11259`); the raw `include/llama.h` at the tag has the same SHA-256 (`0199c9f1...8694`, 85,393 bytes) and all quoted declarations match after whitespace normalisation; all 25 model/asset URLs plus the two gated Google URLs were HEAD-checked again with `curl -sIL --ssl-no-revoke` (25 x final 200 with the listed Content-Length, 2 x 401); tags `tree-sitter v0.27.0`, `tree-sitter-c v0.24.2`, `tree-sitter-cpp v0.23.4`, `cpp-httplib v0.58.0`, `nlohmann/json v3.12.0` and `tectonic@0.17.0` exist; every arXiv reference was re-opened on its abs page; and the source-level corrections below (OpenMP affinity, cross-entropy normalisation `-1/nr`, `graph_max_nodes` branches, unsigned data-set size, LoRA shape checks, memory-model intercept) were re-derived from the files at the tag. No further errors were found.
- Labels: **[E]** established, **[R]** recent (2024-2026), **[C]** contested, **[P]** our own proposal, derivation or measurement (not a published result).
- Evaluation context: all experiments planned for the paper run on the demo repository, which is a fork of `microsoft/mimalloc` (59 C source and header files, 1,273,275 bytes). First-party timings in this note are single runs on one laptop (Windows 11, Intel i9-12950HX, 32 GB RAM, CPU-only build) and carry no variance estimate.
- Scope: this note is the engineering ground truth (what the library actually does at the pinned tag). The scientific discussion of fine-tuning and quantisation methods is in the companion note `finetune_quant.md`; cache and ordering questions are in `cache_parallel.md`.

## Summary

1. The pinned dependency set exists and is current: llama.cpp `v0.5.0` (commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, released 2026-09-23, bundling ggml 0.25.1), tree-sitter `v0.27.0`, tree-sitter-c `v0.24.2`, tree-sitter-cpp `v0.23.4`, cpp-httplib `v0.58.0`, nlohmann/json `v3.12.0`, Tectonic `0.17.0`. Each tag was confirmed with `git ls-remote --tags` and is the highest version tag of its repository [1, 63-68].
2. Every function signature the first pass listed exists in `include/llama.h` at the tag (54 declarations checked; file hash identical to the first-pass value). The quotations were whitespace-normalised, not byte-exact [3].
3. All 23 model URLs answer HTTP 302 then 200 with exactly the listed Content-Length; the two Google upstream repositories answer 401 (gated) [43-62].
4. Upstream llama.cpp can train, but only full F32 tensors with AdamW or SGD; there is no LoRA trainer upstream, the module maintainer calls the code unusable, and there is no test coverage for it [9, 10, 36, 38, 42]. A downstream project adds native LoRA training on top of llama.cpp [95].
5. Two first-pass claims were wrong in substance and are corrected here: (a) CPU affinity through a ggml thread pool does **not** require an OpenMP-free build; (b) the guard "train only if T > n_ctx + 1" does not prevent an empty data set.
6. One first-pass proposal has a gap: a low-rank adapter extracted by SVD cannot represent the bias and normalisation vectors that the measured partial fine-tuning run also trained.
7. On Qwen2.5-Coder-0.5B the preset called `Q4_K_M` is mostly **not** a K-quant: 145 of 291 tensors fall back to `q5_0` or `q8_0` because 896 is not a multiple of 256, giving 6.16 bits per weight. This is confirmed in source and in the retained log [15, 104].

### Verification status and corrections to the first-pass dossier

All 59 first-pass sources were located and none had to be removed as unverifiable. One (a Wikipedia article) was replaced by the primary papers; one had an unusable local `file://` URL and was re-pointed to the public source; ten composite entries were split so that each reference has one URL, giving 70. A further 34 sources were added by the checker after opening them. Total: 104 verified references.

| # | First-pass statement | Correction |
|---|---|---|
| 1 | "Explicit affinity through cpumask requires building with GGML_OPENMP=OFF (the threadpool worker code is compiled under `#ifndef GGML_USE_OPENMP`)" | Wrong. With OpenMP, `ggml_threadpool_new_impl` still computes a mask per worker and `ggml_graph_compute` applies priority and affinity inside `#pragma omp parallel` on every graph evaluation. Only dedicated worker threads, pause/resume and polling are OpenMP-free features [7]. |
| 2 | Signatures "quoted" from `llama.h` | Tokens are correct, spacing is not verbatim. Two single-line declarations fail a byte match (`llama_pooling_type`, `llama_sampler_chain_add`) and about fifteen are multi-line in the header [3]. |
| 3 | `llama_sampler_init_penalties(...)` elided | Full signature at the tag has five parameters, the first being `int32_t n_vocab` [3]. |
| 4 | "A caller must enforce T > n_ctx + 1" | Insufficient. That only prevents unsigned wrap-around. N >= 1 needs T >= n_ctx + 1 + s, and a non-empty training **and** validation split needs N >= 2. See F4. |
| 5 | Memory model "M_0 about 3.31 GiB" | Arithmetic error. The fit gives M_0 = 3306 MiB = 3.23 GiB (3.47 GB decimal). Slope 12.87 bytes per parameter is correct. See F10. |
| 6 | Node budget N_max = max(1024, 8 n_tensors) + LoRA nodes | Incomplete. The function also adds backend-sampler nodes, and the LoRA term is exactly 6 nodes per adapted tensor. See F9. |
| 7 | "The exact normalisation inside ggml_cross_entropy_loss was not inspected" | Now inspected: the result is multiplied by -1/n_r, n_r being the number of rows (token positions). The per-position mean in F5 is confirmed [6]. |
| 8 | AdamW formula attributed to Loshchilov and Hutter without qualification | The code multiplies the decay by the learning rate (factor 1 - alpha lambda). Algorithm 2 of the paper multiplies it by the schedule multiplier only. The two agree when eta_t = 1 and lambda_paper = alpha lambda_ggml. See F1 [6, 75]. |
| 9 | Learning-rate schedule stated without domain | The decay branch is initialised only when 0 < lr_min < lr0. See F3 [11]. |
| 10 | Issue #18499 dated 2026 | Opened 2025-12-30. The quoted maintainer comments are from January 2026 [38]. |
| 11 | LoRA cited as arXiv 2021 | Published at ICLR 2022 [76]. |
| 12 | CodeRankEmbed source dated 2025 | Base model repository created 2024-11-03; paper at ICLR 2025; the two GGUF conversions date from 2025-05-14 and 2026-06-22 [56-58, 85]. |
| 13 | Eckart-Young theorem cited from Wikipedia | Replaced by Eckart and Young (1936) and Mirsky (1960) [77, 78]. |
| 14 | `core.longpaths` cited from a local `file://` path | Re-pointed to the Git for Windows source at tag `v2.53.0.windows.1` [74]. |
| 15 | Tectonic statements all cited to the installation page | The first-run download and automatic-rerun sentences are on the "Build Your First Document" page [69, 70]. |
| 16 | "Quantise took 7.4 s", "F32 conversion 4.4 s" | The retained logs report 6.40 s and 4.13 s of quantisation time [104]. |
| 17 | "The resulting model generated correct C" | One smoke prompt (maximum of two integers, 95 tokens). It is not an evaluation [104]. |
| 18 | "The same effect explains the 491 MB size of the official q4_k_m file" | Unverified inference. Reworded to "is consistent with". |
| 19 | "BUILD_SHARED_LIBS defaults ON except on MinGW" | Also OFF on Emscripten [22]. |
| 20 | "Newest nightly tag b11256" | b11258 on 2026-09-29 and b11259 on 2026-09-30 [1]. Nightly numbering moves daily; cite the release tag, not a nightly. |
| 21 | Pin "tag v0.5.0 (commit 7fe450e...)" with `GIT_SHALLOW TRUE` | CMake documents that with `GIT_SHALLOW` the `GIT_TAG` must be a branch or tag name, not a commit hash [72]. Pin the tag name and check the resolved commit afterwards. |
| 22 | ggml-org conversions listed under "Gemma licence" | The two ggml-org repositories declare no licence in their metadata; the base models are under the Gemma terms [49, 60]. |

## Established results

### Release and versioning [E]

- `refs/tags/v0.5.0` is an annotated tag (`c13fcbf6...`) that peels to commit `7fe450e19305b828c199d602c23a8337aaa1f03b`. The release is not a pre-release and was published 2026-09-23T20:50:06Z [1].
- Release policy: nightly builds use tags `b[NUM]` and `-DLLAMA_BUILD_IS_DEV=ON`; official releases use `vX.Y.Z` and `-DLLAMA_BUILD_IS_DEV=OFF`; release tags are created only when the internal ggml copy equals a released ggml [2]. `LLAMA_BUILD_IS_DEV` defaults to ON, so a consumer must set it OFF to get version string `0.5.0` rather than `0.5.0-dev` [22].
- The tag bundles ggml 0.25.1 (`ggml/CMakeLists.txt`), cpp-httplib 0.57.1 and nlohmann/json 3.12.0 [22, 23]. The release text says "ggml 0.25.0"; the build file is authoritative.

### Public C API at the tag [E]

All of the following exist in `include/llama.h` [3]. Spacing is normalised.

| Area | Declarations |
|---|---|
| Lifecycle | `llama_backend_init(void)`, `llama_backend_free(void)`, `llama_model_load_from_file(const char * path_model, struct llama_model_params params)`, `llama_model_free`, `llama_init_from_model(struct llama_model * model, struct llama_context_params params)`, `llama_free`, `llama_model_save_to_file(const struct llama_model * model, const char * path_model)` |
| Deprecated | `llama_load_model_from_file`, `llama_new_context_with_model`, `llama_free_model` (wrapped in `DEPRECATED(...)`) |
| Absent (0 matches) | `use_mmap`, `use_mlock`, `llama_kv_self`, `llama_kv_cache`, `llama_set_adapter_lora`, `llama_rm_adapter_lora`, `llama_clear_adapter_lora` |
| Loading | `enum llama_load_mode { AUTO = -1, NONE = 0, MMAP = 1, MLOCK = 2, MMAP_MLOCK = 3, DIRECT_IO = 4 }`, `enum llama_lazy_mode { OFF = 0, AUTO = 1, ON = 2 }` |
| Context | `enum llama_flash_attn_type { AUTO = -1, DISABLED = 0, ENABLED = 1 }`, `enum llama_context_type { DEFAULT = 0, MTP = 1 }`, fields `n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`, `n_threads`, `n_threads_batch`, `type_k`, `type_v`, `embeddings`, `abort_callback` |
| Memory | `llama_memory_t llama_get_memory(const struct llama_context * ctx)`, `llama_memory_clear(mem, bool data)`, `llama_memory_seq_rm`, `llama_memory_seq_cp`, `llama_memory_seq_keep`, `llama_memory_seq_add`, `llama_memory_seq_div`, `llama_memory_seq_pos_min`, `llama_memory_seq_pos_max`, `llama_memory_can_shift` |
| Tokenisation | `llama_model_get_vocab`, `llama_vocab_n_tokens`, `llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)`, `llama_token_to_piece(vocab, token, buf, length, lstrip, special)`, `llama_detokenize(vocab, tokens, n_tokens, text, text_len_max, remove_special, unparse_special)`, `llama_vocab_is_eog` |
| Batches | `llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)`, `llama_batch_free`, `llama_batch_get_one(llama_token * tokens, int32_t n_tokens)`, `llama_decode`, `llama_encode` |
| Embeddings | `llama_set_embeddings`, `llama_get_embeddings`, `llama_get_embeddings_ith`, `llama_get_embeddings_seq`, `llama_model_n_embd`, `llama_model_n_embd_inp`, `llama_model_n_embd_out`, `llama_pooling_type(ctx)` |
| Sampling | `llama_sampler_chain_init`, `llama_sampler_chain_add`, `llama_sampler_init_greedy(void)`, `_dist(uint32_t seed)`, `_top_k(int32_t k)`, `_top_p(float p, size_t min_keep)`, `_min_p(float p, size_t min_keep)`, `_temp(float t)`, `_penalties(int32_t n_vocab, int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present)`, `_grammar(vocab, grammar_str, grammar_root)`, `llama_sampler_sample(smpl, ctx, idx)`, `llama_sampler_free` |
| Chat | `llama_model_chat_template(model, name)`, `llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length)`, `struct llama_chat_message { const char * role; const char * content; }` |
| Adapters | `llama_adapter_lora_init(model, path_lora)`, `llama_adapter_lora_init_from_file_ptr(model, FILE * file)`, `llama_set_adapters_lora(ctx, adapters, n_adapters, scales)`, `llama_adapter_lora_free` |
| Quantisation | `uint32_t llama_model_quantize(fname_inp, fname_out, const llama_model_quantize_params * params)`, `llama_model_quantize_default_params(void)` |
| Training | `llama_opt_param_filter`, `llama_opt_param_filter_all`, `struct llama_opt_params`, `llama_opt_init(lctx, model, lopt_params)`, `llama_opt_epoch(lctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)` |
| Threads | `llama_attach_threadpool(ctx, threadpool, threadpool_batch)`, `llama_detach_threadpool`, `llama_set_n_threads(ctx, n_threads, n_threads_batch)`, `llama_set_abort_callback` |

Behavioural facts read from the header and sources:

- `llama_tokenize` returns the token count, a negative count when the buffer is too small, and `INT32_MIN` on overflow. The tokenisation section is marked "The API is thread-safe." The logging functions are marked not thread safe. The header makes **no** thread-safety statement for `llama_decode` on a shared context [3].
- `llama_decode` return codes: 0 success; 1 no KV slot; 2 aborted; -1 invalid batch; below -1 fatal [3].
- The header recommends querying actual values after context creation because requested values "may differ from the actual values used by the context" [3]. The first-party log shows a request of 128 becoming 256 [104].
- `llama_get_embeddings_seq` returns NULL when pooling is NONE. `llama_get_embeddings` carries a TODO comment to deprecate it in favour of `llama_get_embeddings_ith`. The reference example clears memory, decodes, reads per-sequence embeddings and normalises [3, 21].
- `llama_sampler_chain_add` takes ownership of the added sampler [3].
- `llama_chat_apply_template` does not use a Jinja parser. A NULL template means `"chatml"`; an unrecognised template returns -1. ChatML is detected by the substring `<|im_start|>` and rendered as `<|im_start|>role\ncontent<|im_end|>\n` plus `<|im_start|>assistant\n` [3, 25, 26].
- `struct llama_model_quantize_params` has 15 fields in this order: `nthread`, `ftype`, `output_tensor_type`, `token_embedding_type`, `allow_requantize`, `quantize_output_tensor`, `only_copy`, `pure`, `keep_split`, `dry_run`, `imatrix`, `kv_overrides`, `tt_overrides`, `prune_layers`, `max_buf_size`. Default `ftype` is `LLAMA_FTYPE_MOSTLY_Q8_0`. Pointer arrays are sentinel-terminated (`name == nullptr`, `pattern == nullptr`, `-1`, `key[0] == 0`). String overrides are limited to `char val_str[128]` [3, 15].

### Training implementation [E]

- `examples/training/finetune.cpp` builds target `llama-finetune`, links `llama-common` and `llama`, passes `llama_opt_param_filter_all`, and writes a complete GGUF with `llama_model_save_to_file`. It forces `load_mode = none` and F32 K/V cache types [9].
- README, verbatim: "So far finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP." and "Flash attention is disabled during training because `FLASH_ATTN_EXT` has no backward pass." [9].
- Only F32 tensors become parameters; `token_embd.weight` and `rope_freqs.weight` are skipped with FIXME markers. `opt_init` overwrites `model->hparams.n_ctx_train` and asserts `n_ctx_train % n_batch == 0` and `n_batch % n_ubatch == 0` [10].
- Defaults: learning rate 1e-5, 2 epochs, validation split 0.05, AdamW; ggml defaults alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight decay 0 [5, 11].
- `ggml-opt.h`, which `llama.h` includes, supplies the data-set and result objects used by the training API: `ggml_opt_dataset_init(type_data, type_label, ne_datapoint, ne_label, ndata, ndata_shard)`, `ggml_opt_result_init/reset/free/loss/accuracy`, and the optimiser types `GGML_OPT_OPTIMIZER_TYPE_ADAMW` and `GGML_OPT_OPTIMIZER_TYPE_SGD` [4].
- The model saver writes the context length from `hparams.n_ctx_train`, leaves the file-type key commented out, writes no chat template, and has a TODO for LoRA [12].

### LoRA at run time [E]

- The loader requires `general.type == "adapter"`, `adapter.type == "lora"`, an architecture equal to the base model, and reads `adapter.lora.alpha` as a float [13].
- Tensors are paired by the suffixes `.lora_a` and `.lora_b`. For a weight with ggml shape `ne = [d_in, d_out]` the checks are `a->ne[0] == d_in`, `b->ne[1] == d_out`, `a->ne[1] == b->ne[0]`. For the token embedding the roles of A and B are flipped [13].
- Application is in one place, `build_lora_mm`: the output is `W x + scale * B (A x)`. An adapter therefore affects only weights that enter the graph through that matrix product [14].
- Merging is available offline: `llama-export-lora -m base.gguf --lora adapter.gguf -o merged.gguf`, with `--lora-scaled FNAME:SCALE` for a user scale [20].

### Importance matrix [E]

- The tool accumulates the sum of squared inputs per column (`acc[j] += x[j] * x[j]`) and a call count, and stores them as F32 tensors `<name>.in_sum2` and `<name>.counts` in a GGUF file [19].
- The design rationale is the diagonal of the second moment of the layer input (F6) [28]. How the importance enters each quantiser is analysed in `finetune_quant.md` and was not re-verified here.

### Build system [E]

- Targets: `llama` (C++17, comment "don't bump", links ggml publicly), `llama-common`, `llama-common-base`, `ggml`, tools `llama-quantize` (executable from `main.cpp`) with library `llama-quantize-impl` (links `llama-common` publicly), `llama-imatrix`, `llama-export-lora`; example `llama-finetune` [17, 22].
- As a sub-project `LLAMA_STANDALONE` is OFF, so `LLAMA_BUILD_COMMON`, `_TESTS`, `_TOOLS`, `_EXAMPLES`, `_SERVER`, `_APP` default OFF. Tools and examples are added only if `LLAMA_BUILD_COMMON` is also ON. `llama-export-lora` additionally requires `NOT GGML_BACKEND_DL AND GGML_CPU` [22].
- `BUILD_SHARED_LIBS` defaults ON except on MinGW and Emscripten; the build guide states "For static builds, add `-DBUILD_SHARED_LIBS=OFF`" and that the project builds without SSL support when OpenSSL is absent [22, 24].
- `LLAMA_CURL` is deprecated and ignored. `LLAMA_OPENSSL` defaults ON. Executables go to `${CMAKE_BINARY_DIR}/bin`. MSVC gets `/utf-8` and `/bigobj`; WIN32 gets `_CRT_SECURE_NO_WARNINGS` [22].
- The vendored cpp-httplib is compiled as a static library with four non-default `CPPHTTPLIB_*` definitions, only when `LLAMA_BUILD_COMMON` is ON. `llama-common` exposes the `vendor/` directory as a public include path through `vendor::nlohmann` [22].
- Minimum CMake versions: 3.14 (llama.cpp, ggml, cpp-httplib), 3.13 (tree-sitter and both grammars), 3.5 (nlohmann/json). CMake 4.0 removed compatibility below 3.5 [22, 23, 63-67, 73].

### Thread pool [E]

- `struct ggml_threadpool_params { bool cpumask[GGML_MAX_N_THREADS]; int n_threads; enum ggml_sched_priority prio; uint32_t poll; bool strict_cpu; bool paused; }` with `GGML_MAX_N_THREADS = 512`; an all-zero mask means default affinity. `ggml_threadpool_new`, `_free`, `_pause` and `_resume` are declared in `ggml-cpu.h` [8].
- With `strict_cpu` each worker is assigned one CPU of the mask in round-robin order; without it every worker gets the whole mask. When no pool is attached, ggml creates a disposable pool with default parameters (no affinity) for every graph evaluation [7].
- On Windows the mask is applied with `SetThreadAffinityMask` and covers the first 64 logical CPUs only. Under OpenMP, pause and resume are no-ops [7].

### Other dependencies [E]

- tree-sitter: language version 15, minimum compatible 13. tree-sitter-c parser is version 15 (3,872,012 bytes), tree-sitter-cpp is version 14 (17,299,078 bytes). Both grammar build files define the same custom target `ts-test` and look for the tree-sitter CLI. tree-sitter-cpp exposes no public include directory [63-65].
- cpp-httplib: blocking socket I/O, HTTP/1.1 only, 32-bit platforms not supported, thread pool base size max(8, hardware_concurrency - 1) scaling to four times, Server-Sent Events and WebSocket documented, and a documented delay of up to 2 s per request for `localhost` on Windows systems with misconfigured IPv6 [66].
- nlohmann/json: target `nlohmann_json::nlohmann_json`; the release tarball is 114,576 bytes [67].
- Tectonic 0.17.0: one executable; Windows archive 21,060,223 bytes, SHA-256 `f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f` [68-70].
- CMake: `EXCLUDE_FROM_ALL` in FetchContent needs 3.28; `SOURCE_SUBDIR` may name a missing directory to suppress a project's own build file; `GIT_CONFIG` passes `--config` to `git clone`; the documentation itself advises a short `FETCHCONTENT_BASE_DIR` on Windows [71, 72].

### Classical results used [E]

- Decoupled weight decay for Adam [75]. Low-rank adaptation [76]. Best rank-r approximation by truncated SVD, in the Frobenius norm [77] and in every unitarily invariant norm [78] (bibliographic data of both verified through Crossref; the statement is the standard attribution, the full texts were not read). Second-order post-training quantisation [79]. Matryoshka representations, which justify truncating an embedding [87].

## Recent work (2024-2026)

### Training support in llama.cpp [R]

- The LoRA-producing examples were removed on 2024-07-25 because they "are no longer working and require too much efforts to maintain" [29].
- The current trainer arrived with PR #10544 (merged 2025-05-12) and gained SGD and command-line options with PR #13873 (merged 2025-08-14) [30, 31].
- Issue #13485 (open, by the module maintainer): "LoRA training should in principle be achievable with the same tools as the full finetune, it's simply not implemented." [42].
- Two native QLoRA pull requests were closed without merge, on 2026-05-04 and 2026-09-24 [32, 33]. QLoRA itself, training adapters on a frozen quantised base, is published [81]; it is simply not available in the pinned library.
- PR #27199 "finetune: fix no KV cache" was merged 2026-09-02 and is contained in the tag (GitHub compare: the tag is 372 commits ahead of the merge commit and 0 behind) [34]. PR #28763, which skips precompiled headers for the server on MSVC, is contained as well and matters only if the server is built [41]. PR #25428 (training graph for Qwen2 with a larger node reservation) and PR #27690 (signed data-set size) are open [35, 37].
- Downstream, QVAC Fabric (Tether Data, blog of 2025-12-01) reports native LoRA training added to llama.cpp through new API functions, with adapters saved as GGUF. Vendor report, not peer reviewed, not reproduced by us [95].
- MobileFineTuner is an independent C++ framework for full and LoRA fine-tuning on phones, evaluated with GPT-2, Gemma 3 and Qwen2.5 (abstract) [94].

### Contested: is the upstream trainer usable? [C]

- Against: maintainer comment of 2026-01-16, "The code is not in a usable state for basically anything." [38]; issue #27648 reports a crash on MSVC and notes that there is no CI or test coverage for `llama-finetune` [36]; PR #27690 adds that training "may still be blocked" by a separate backward-pass issue [37].
- For: the README calls it "technically functional" [9], and first-party runs on the tag succeeded for a 5-layer test model and for block-restricted training of a 24-layer model [104].
- Our reading: usable for a guarded, block-restricted experiment; not a dependable product feature.

### Memory and layer selection [R]

- Parameter efficiency is not memory efficiency: LoRA and IA3 "remain bound by intermediate tensors that scale linearly with sequence length" (abstract) [93]. This supports bounding the sequence length and the number of trained blocks rather than only the parameter count.
- Training a subset of layers is a published strategy: surgical fine-tuning "matches or outperforms commonly used fine-tuning approaches" on distribution shifts (vision experiments) [90]; LISA "randomly freezes most middle layers" with memory cost comparable to LoRA on language models [91]. Neither paper tests our rule (train the last k blocks); they support the family, not the specific choice.
- LoRA and full fine-tuning produce structurally different updates ("intruder dimensions") [92]. The reverse operation we propose, compressing a full-rank update by SVD, is not covered by that paper.

### Quantisation and model artefacts [R]

- A unified evaluation of llama.cpp formats exists for one 8B model (3 to 8 bit K-quants and legacy formats) [98]. Code-specific evidence exists for Qwen2.5-Coder and CodeLlama with six methods including GGUF; methods "differ meaningfully" in correctness and code quality (abstract) [99]. Neither abstract mentions a 0.5B model or the fallback mixture we observe; the bodies were not read.
- Supply-chain risk is documented. A quantised GGUF can behave maliciously while its full-precision source looks benign (ICML 2025) [96]. An audit of 327 artefacts found silently defective conversions in an official library, invisible to surface statistics (preprint) [97].
- GGUF metadata predicts throughput only approximately and coefficients are not universal across systems (preprint) [101].
- Serving-stack behaviour confounds tool-use measurements, and "constrained decoding removes parse failures but can induce non-termination" (workshop paper) [100]. This applies directly to grammar-constrained edit proposals.
- Weight loading: memory-mapped, file-backed weights avoid a second copy (preprint) [103]. Training cannot use this path because it needs writable weights [9].
- Keeping model weights cache-resident is reported for server CPUs with gigabyte-scale last-level caches (preprint, abstract) [102]. A 0.5B model does not fit in a 48 KB L1 cache, so the 16 KB block rule of feature 7 can only govern our own index and retrieval data, not llama.cpp inference.

### Models [R]

Verified from repositories, GGUF headers and papers. Sizes in bytes.

| Model | File | Size | Licence | Header facts |
|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct [43, 83] | `qwen2.5-coder-0.5b-instruct-fp16.gguf` | 1,266,425,728 | apache-2.0 | qwen2, 291 tensors, 24 blocks, width 896, feed-forward 4864, context 32768, template 2,507 chars |
| same | `...-q4_k_m.gguf` / `...-q8_0.gguf` | 491,400,064 / 675,710,848 | apache-2.0 | |
| same, community [44] | `Qwen2.5-Coder-0.5B-Instruct-f16.gguf` | 994,156,928 | apache-2.0 | |
| Qwen2.5-Coder-1.5B-Instruct [45, 46] | `...-q4_k_m.gguf` / `...-q8_0.gguf` / community f16 | 1,117,320,768 / 1,894,532,160 / 3,093,669,376 | apache-2.0 | no official fp16 |
| Qwen3-0.6B [47, 48] | Q8_0 / community BF16 | 639,446,688 / 1,198,182,848 | apache-2.0 | |
| gemma-3-270m-it [49-51] | Q8_0 / community F16 | 291,545,600 / 542,835,488 | Gemma terms | upstream gated (401) |
| stories260K [52] | `stories260K-f32.gguf` | 1,185,376 | not declared | smoke test |
| Qwen3-Embedding-0.6B [53, 54, 84] | Q8_0 / f16 | 639,150,592 / 1,197,629,632 | apache-2.0 | qwen3, 28 blocks, dim 1024, context 32768, pooling 3 (LAST) |
| nomic-embed-text-v1.5 [55, 86] | Q8_0 / f16 / f32 | 146,146,432 / 274,290,560 / 547,664,768 | apache-2.0 | nomic-bert, dim 768, context 2048, pooling 1 (MEAN) |
| CodeRankEmbed [56-58, 85] | community Q8_0 / f16 / second Q8_0 | 146,033,152 / 274,181,632 / 146,029,792 | mit | nomic-bert, dim 768, context 2048, pooling 2 (CLS) |
| jina-code-embeddings-0.5b [59, 88] | Q8_0 | 531,065,120 | cc-by-nc-4.0 | dim 896, last-token pooling |
| EmbeddingGemma-300M [60-62, 89] | Q8_0 (two conversions) | 333,590,944 / 328,577,056 | Gemma terms | upstream gated (401) |

Usage requirements read from the cards: Qwen3-Embedding uses `Instruct: {task}\nQuery:{query}` on the query side and the card reports a drop of "approximately 1% to 5%" without it [54]; nomic-embed-text requires task prefixes and defaults to 2048 tokens in llama.cpp [55]; CodeRankEmbed requires the query prefix "Represent this query for searching relevant code" and reports CSN MRR 77.9 and CoIR NDCG@10 60.1 [56]; jina-code-embeddings requires last-token pooling and task prefixes [59].

Two details that matter for the design:

- The CodeRankEmbed community card states that pooling is a serve-time setting the file does not carry. The header contradicts this: `nomic-bert.pooling_type = 2` is stored. Only the query prefix is absent from the file [57].
- The official Qwen fp16 file stores `output.weight` and `token_embd.weight` separately (630,167,424 weights against 0.49B on the card) and is named "Qwen2.5 Coder 0.5B Instruct AWQ". Whether it derives from AWQ-scaled weights [80] is unresolved [43].

## Formulas

Each formula was re-derived or checked against the source. "Checked" lists what was verified.

### F1. AdamW step as implemented in ggml [E]

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{2},\qquad
\hat m_t = \frac{m_t}{1-\beta_1^{\,t}},\qquad
\hat v_t = \frac{v_t}{1-\beta_2^{\,t}}
$$

$$
w_t = (1-\alpha\lambda)\, w_{t-1} - \alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Source: `ggml_compute_forward_opt_step_adamw_f32` and the factors `beta1h = 1/(1 - beta1^iter)`, `beta2h = 1/(1 - beta2^iter)` [5, 6]. Checked: signs, the position of epsilon outside the square root, and that t counts optimiser steps (it starts at 1 and advances only after an optimiser-graph evaluation), so with gradient accumulation g_t is the accumulated gradient of k micro-batches.

Relation to the paper. Algorithm 2, line 12 of [75] reads

$$
\theta_t = \theta_{t-1} - \eta_t\left(\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
$$

The two coincide for eta_t = 1 and lambda_paper = alpha lambda_ggml. In ggml the effective decay therefore shrinks when the learning rate is lowered.

### F2. SGD step as implemented in ggml [E]

$$
w_t = (1-\alpha\lambda)\, w_{t-1} - \alpha\, g_t
$$

Source: `ggml_compute_forward_opt_step_sgd_f32` [6]. No momentum.

### F3. Learning-rate schedule of llama-finetune [E]

For 0 < alpha_min < alpha_0, with E epochs and requested decay length D:

$$
E_d = \begin{cases} D & 0 < D < E \\ E & \text{otherwise} \end{cases},\qquad
s = \frac{\log_2(\alpha_0/\alpha_{\min})}{E_d},\qquad
\alpha(e) = \begin{cases} \alpha_{\min} & e \ge E_d \\ \alpha_0\, 2^{-e\,s} & e < E_d \end{cases}
$$

If alpha_min <= 0 (the default, -1) the rate is constant, alpha(e) = alpha_0. Source: `lr_opt::init`, `lr_opt::get_lr` [11]. Checked: continuity at e = E_d, since alpha_0 2^{-E_d s} = alpha_min. The rate is constant within an epoch. If alpha_min >= alpha_0 the initialisation is skipped and the default E_d = -1 makes every epoch return alpha_min.

### F4. Next-token training set built from a token stream [E], with corrected guard [P]

$$
N = \left\lfloor \frac{T - n_{ctx} - 1}{s} \right\rfloor,\qquad s = \left\lfloor \frac{n_{ctx}}{2} \right\rfloor
$$

$$
x^{(i)} = (t_{is}, \dots, t_{is+n_{ctx}-1}),\qquad
y^{(i)} = (t_{is+1}, \dots, t_{is+n_{ctx}}),\qquad i = 0,\dots,N-1
$$

$$
N_{train} = \lfloor N(1-v) \rfloor,\qquad N_{val} = N - N_{train}
$$

Source: `common_opt_dataset_init`, `finetune.cpp` [9, 11]. Checked:

- Index range. The largest index read is (N-1)s + n_ctx <= T - 1 - s, so no read is out of bounds; the rule discards one usable window.
- Numerical examples. T = 2037, n_ctx = 256 gives N = 13 (matches our log). T = 819, n_ctx = 256 gives 4 and T = 513, n_ctx = 512 gives 0 (match the regression test of PR #27690) [37, 104].
- Unsigned arithmetic. At the tag the numerator is evaluated in `size_t`, so T < n_ctx + 1 wraps to about 2^64 and the first allocation fails [36, 37].
- Corrected guard. N >= 1 if and only if T >= n_ctx + 1 + s. Both splits are non-empty if and only if N >= 2 (for 0 < v <= 1/2), that is T >= n_ctx + 1 + 2s, which is 2 n_ctx + 1 for even n_ctx.
- Each token appears in about n_ctx / s = 2 windows per epoch.

### F5. Training objective, accumulation and reported uncertainty [E]

$$
\mathcal{L}(\theta) = -\frac{1}{n_r}\sum_{r=1}^{n_r}\sum_{c=1}^{V} y_{r,c}\,\log\frac{\exp z_{r,c}(\theta)}{\sum_{c'=1}^{V}\exp z_{r,c'}(\theta)}
= -\frac{1}{n_r}\sum_{r=1}^{n_r}\log p_\theta\!\left(t_{r+1}\mid t_{\le r}\right)
$$

$$
\mathcal{L}_{graph} = \frac{1}{k}\,\mathcal{L},\qquad k = \frac{n_{batch}}{n_{ubatch}}
$$

y is one-hot, V is the vocabulary size, n_r = n_ubatch is the number of token positions in a micro-batch. Source: `opt_init`, `opt_epoch_iter`, `ggml_compute_forward_cross_entropy_loss_f32` (final line multiplies by -1/nr), and the scaling by 1/opt_period [5, 6, 10].

The epoch loss and its uncertainty over B micro-batch losses l_b (rescaled by k) are

$$
\bar\ell = \frac{1}{B}\sum_{b=1}^{B}\ell_b,\qquad
u = \sqrt{\frac{\tfrac{1}{B}\sum_{b}\ell_b^{2} - \bar\ell^{\,2}}{B-1}}
$$

Source: `ggml_opt_result_loss` [5]. u is the standard error of the mean under independence. Our windows overlap by half, so u understates the true uncertainty [P].

### F6. Importance-matrix objective [E]

For a weight row w, quantised row q and input column a:

$$
F = \Big[\sum_j (q_j - w_j)\, a_j\Big]^2,\qquad
\frac{\partial F}{\partial q_i} = 2\sum_j a_i a_j\,(q_j - w_j)
$$

$$
\mathbb{E}[F] = (q-w)^{\top} H\,(q-w),\qquad H = \mathbb{E}\big[a a^{\top}\big]
\;\approx\; \sum_i h_i\,(q_i - w_i)^2,\qquad h_i = \mathbb{E}\big[a_i^2\big]
$$

$$
\hat h_i = \frac{1}{n}\sum_{t=1}^{n} a_{t,i}^{2}
$$

Source: PR #4861 for the argument, `imatrix.cpp` for the estimator (in_sum2 divided by counts) [19, 28]. Checked: the PR omits the factor 2 and writes `<a_j a_j>` where `<a_i a_j>` is meant; neither affects the minimiser. H is the layer-wise Hessian used by GPTQ up to a constant factor; GPTQ keeps the off-diagonal terms [79].

### F7. Statistics printed by `llama-imatrix --show-statistics` [E]

$$
p_i = \frac{\hat h_i}{\sum_{j=1}^{N}\hat h_j},\qquad
S = -\sum_{i=1}^{N} p_i \log_2 p_i,\qquad
E_{norm} = \frac{S}{\log_2 N},\qquad
ZD = \frac{1}{N}\,\Big|\{\, i : (\hat h_i - \mu)/\sigma > 1 \,\}\Big|
$$

Source: `compute_statistics` and the README [18, 19]; ZD follows [82]. Checked: 0 <= S <= log2 N, so E_norm lies in [0, 1]. The README warns that these are computed on squared activations.

### F8. LoRA update and the scale llama.cpp applies [E]

$$
y = W_0 x + \gamma\,\frac{\alpha}{r}\, B\,(A x),\qquad
B \in \mathbb{R}^{d_{out}\times r},\quad A \in \mathbb{R}^{r\times d_{in}},\quad r \ll \min(d_{in}, d_{out})
$$

Source: [76] for the method; `get_scale` and `build_lora_mm` for the implementation [13, 14]. gamma is the per-adapter scale passed to `llama_set_adapters_lora`, alpha is `adapter.lora.alpha`, r = `b->ne[0]`. If alpha = 0 the scale is gamma. In ggml shape notation A has `ne = [d_in, r]` and B has `ne = [r, d_out]`.

### F9. Compute-graph node budget [E], and the limit it puts on training [P]

$$
N_{\max} = \max(1024,\; 8\, n_{tensors}) \;+\; 6\sum_{a \in \mathcal{A}} |M_a| \;+\; n_s \;+\; \max(o_{\max}-1,\,0)\; n_{s,\max}
$$

A is the set of adapters loaded on the model, |M_a| the number of tensor pairs in adapter a, n_s the total and n_{s,max} the largest node count of backend samplers, o_max = min(min(n_tokens, n_outputs_max), n_seq_max n_outputs_max_per_seq). Source: `graph_max_nodes` for architectures without a special case, and `get_n_nodes` [10, 13]. Several other architectures have special-case budgets in the same function; qwen2 is not among them.

For Qwen2.5-Coder-0.5B with no adapter and no backend sampler, N_max = 8 x 291 = 2328. Forward plus backward graph must fit. First-party result [P]: training the last 16, 12, 8, 4 or 1 of 24 blocks fits; the last 20, 22 or 24 do not. The boundary between 16 and 20 was not resolved [104].

### F10. Empirical memory model for partial fine-tuning [P]

$$
M_{peak} \approx M_0 + c\, P_{train},\qquad M_0 \approx 3306\ \text{MiB} = 3.23\ \text{GiB},\qquad c \approx 12.87\ \text{bytes per parameter}
$$

Own fit. Reported points: (14,912,384; 3489 MiB), (119,299,072; 4770 MiB), (238,598,144; 6234 MiB). Slope from the outer points: 2745 x 2^20 / 223,685,760 = 12.87. Intercept: 3489 - 183 = 3306 MiB. The middle point is reproduced (3306 + 1464 = 4770). Interpretation: three F32 tensors per trainable parameter (one accumulated gradient and two moments, `tensors_per_param` in [5]) give 12 bytes; the remainder is transient.

Caveats: three points and two parameters leave one degree of freedom; the parameter counts are confirmed in the logs but the three memory values are **not** in the retained logs; the micro-batch was 128 tokens and [93] shows that activation memory grows with sequence length.

### F11. Post-hoc LoRA extraction from a partially fine-tuned model [P]

$$
\Delta W = W_{ft} - W_0 = U\Sigma V^{\top},\qquad
B = U_r\Sigma_r^{1/2},\quad A = \Sigma_r^{1/2}V_r^{\top},\quad \alpha = r
$$

$$
\lVert \Delta W - BA \rVert_F = \sqrt{\sum_{i>r}\sigma_i^2},\qquad
\rho(r) = \frac{\sum_{i\le r}\sigma_i^2}{\sum_{i}\sigma_i^2}
$$

Own proposal, not implemented or tested. Optimality of the truncation is the Eckart-Young theorem [77, 78]. alpha = r makes the scale of F8 equal to gamma. rho(r) is the fraction of squared update norm retained and is the quantity to report.

Storage per block of Qwen2.5-Coder-0.5B at rank r, from the verified shapes (896 x 896 twice, 896 x 128 twice, 896 x 4864 three times):

$$
P_{LoRA}(r) = r\,\big[\,2(896+896) + 2(896+128) + 3(896+4864)\,\big] = 22{,}912\, r
$$

At r = 16 this is 366,592 parameters per block, 2.46% of the 14,912,384 in the block.

Gap found in checking: each block has 12 trainable tensors of which 5 are vectors (three attention biases, two RMSNorm weights; 2,944 parameters). A LoRA pair cannot represent a vector and `build_lora_mm` applies adapters only to matrix products. If those vectors are trained, the adapter cannot reproduce the fine-tuned model. The parameter filter must accept only two-dimensional tensors.

### F12. Bits per weight [E]

$$
\mathrm{BPW} = \frac{8\cdot\text{bytes of tensor data}}{\text{number of weights}}
$$

Nominal values from the block layouts in `ggml-common.h`: q4_0 and iq4_nl 18 bytes per 32 weights (4.5); q5_0 22 per 32 (5.5); q8_0 34 per 32 (8.5); q4_K 144 per 256 (4.5); q6_K 210 per 256 (6.5625); iq4_xs 136 per 256 (4.25). Checked against the log: 462.96 MiB x 2^20 x 8 / 630,167,424 = 6.163 [104]. The README lists 4.8944 for Q4_K_M on Llama-3.1-8B [16].

Fallback rule [15]: a tensor whose column count `ne[0]` is not a multiple of the block size of the target type is demoted: q4_K to q5_0, q5_K to q5_1, q6_K to q8_0; q2_K, q3_K and the ternary types to q4_0; every IQ type with 256-weight blocks to iq4_nl. If the fallback type does not fit either, the tensor is stored as F16. 896 = 3.5 x 256, so every tensor with 896 columns falls back; `ffn_down` has 4864 = 19 x 256 columns and does not. 1536 = 6 x 256, so the 1.5B model is unaffected.

## What is novel in our proposal

Stated as proposals. None of these is an established result.

1. **Block-restricted fine-tuning under a fixed graph budget [P].** Using only the public C API with a parameter filter on `blk.<i>.` names, without patching the library. The idea of training a subset of layers is published [90, 91]; the contribution is the characterisation of what the unpatched llama.cpp graph budget allows (F9) and a linear memory model (F10).
2. **Post-hoc low-rank extraction [P].** Because upstream has no LoRA trainer [42], train selected matrices in F32, then store the change as a rank-r adapter (F11) so that the base GGUF, its chat template and its context length stay untouched and the fine-tune toggle becomes a run-time switch. Untested.
3. **Corpus-specific importance matrix [P].** Computing the second moments on the indexed repository itself and reporting the true bit rate including fallbacks (F12). The method is upstream [18, 28]; the observation that a common preset silently becomes a 5 to 8 bit mixture on a width-896 model is ours and confirmed in source.
4. **Metadata-preserving model promotion [P].** Identifying that the upstream saver loses context length, file type and chat template after training [12], and routing around it.

What is not novel and must not be presented as such: AdamW, cross-entropy training, LoRA, truncated SVD, importance-weighted quantisation, grammar-constrained sampling.

## Risks and failure modes

1. **Upstream training code is unstable.** No tests, open crash reports on MSVC, a maintainer who disowns it [36-38]. A tag bump can break the feature silently.
2. **Hard abort instead of error.** Exceeding the node budget triggers `GGML_ASSERT`, which terminates the process. If training runs in the server process, a wrong k kills the server.
3. **Model mutation.** `opt_init` overwrites `n_ctx_train` on the model object [10]. Training on the instance that serves queries corrupts its context length.
4. **Metadata loss.** A saved model reports the training context as its context length and has no chat template [12, 104]. The template is 2,507 characters; the override mechanism allows 127 [3, 43].
5. **Template divergence.** The Qwen Jinja template inserts a default system prompt when none is given; the built-in ChatML renderer does not [25, 43]. Prompts differ between our code and tools that use Jinja unless a system message is always sent.
6. **Adapter cannot represent vectors.** See F11.
7. **Quantisation is not what its name says.** See F12. A size or quality claim for "Q4_K_M" on the 0.5B model would be wrong.
8. **Evidence gap for small code models.** The llama.cpp evaluation we verified uses an 8B model [98]; the code-generation study covers Qwen2.5-Coder and CodeLlama but its abstract does not state model sizes [99]. Nothing verified here shows that a 0.5B model survives 4 to 6 bit quantisation on code.
9. **Third-party GGUF files.** The default embedding file comes from an individual account created in 2026 [57]. Quantised artefacts can be malicious or silently broken [96, 97].
10. **Licences.** jina-code-embeddings is non-commercial [59]. Gemma derivatives carry the Gemma terms even where the conversion repository declares nothing [49, 60].
11. **Measurement quality.** Single runs. The same configuration took 56.8 s in one run and 78.1 s in another [104]. The training text was 2,037 tokens with two validation windows, so validation losses have uncertainties near 0.46 and no quality conclusion is possible.
12. **Overfitting and forgetting.** An estimated 330,000 tokens of one repository (extrapolated from 36,000 characters) is a small corpus for up to 119 million trainable parameters. See `finetune_quant.md`.
13. **Constrained decoding can fail to terminate** [100]. A grammar guarantees syntax, not a finite answer.
14. **Thread placement on Windows.** Affinity covers the first 64 logical CPUs only, and with OpenMP the mask is re-applied by whichever runtime thread has a given index [7]. MSVC builds do not enable AVX-VNNI [39].
15. **Build fragility on Windows.** A global C++20 standard breaks `llama-chat.cpp` under MSVC [25, 104]; the longest path in the tree is 161 characters, which for this checkout location and the default dependency directory (72-character prefix) gives 233 characters, 27 below the 260 limit [27]; Schannel-based curl fails against huggingface.co without `--ssl-no-revoke` (reproduced by the checker: exit code 35, `CRYPT_E_NO_REVOCATION_CHECK`).
16. **Two HTTP libraries in one binary.** Linking `llama-common` together with header-only cpp-httplib 0.58.0 gives two definitions of the same namespace with different settings [22, 66].
17. **Pre-tokeniser regular expressions on MSVC.** Patterns with `\uHHHH` escapes abort model loading (open issue) [40]. The four models we load were unaffected [104].

## Design recommendations

1. **Pin by tag, verify by hash.** `GIT_TAG v0.5.0`, `GIT_SHALLOW TRUE`, `GIT_CONFIG core.longpaths=true`, `EXCLUDE_FROM_ALL`. A commit hash is not allowed with a shallow clone [72], so after population run `git rev-parse HEAD` and fail the configure step unless it equals `7fe450e19305b828c199d602c23a8337aaa1f03b`.
2. **Options before the declaration.** `BUILD_SHARED_LIBS=OFF`, `LLAMA_BUILD_IS_DEV=OFF`, `LLAMA_BUILD_COMMON=ON`, `LLAMA_BUILD_TOOLS=ON`, `LLAMA_BUILD_EXAMPLES=ON`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_BUILD_SERVER=OFF`, `LLAMA_BUILD_APP=OFF`, `LLAMA_OPENSSL=OFF`, `GGML_CUDA=OFF`, `GGML_NATIVE=ON`. This is the recipe that was built on the target machine. Examples are needed only for `llama-finetune`; our trainer uses the C API, so `LLAMA_BUILD_EXAMPLES` can probably be switched OFF, but that variant was not built. Add `add_dependencies(app llama-quantize llama-imatrix)` so that the two tools are built despite `EXCLUDE_FROM_ALL`.
3. **Never set `CMAKE_CXX_STANDARD` globally.** Use `target_compile_features(<target> PRIVATE cxx_std_20)` on our own targets.
4. **Short paths.** `GIT_CONFIG core.longpaths=true` is the actual fix for the checkout failure [72, 74]. In addition keep `FETCHCONTENT_BASE_DIR` short; the CMake documentation advises a short path just below the top of the build tree [71]. Quote the corpus path, which contains a space.
5. **Link only `llama`.** Keep `llama-common` out of the server binary. Run `llama-imatrix` and `llama-quantize` as child processes; the importance-matrix reader lives in `llama-common`, so calling `llama_model_quantize` directly with an importance matrix would require parsing the `.in_sum2` and `.counts` tensors ourselves [19].
6. **One owner thread per context.** The header promises thread safety only for tokenisation [3]. Route every `llama_decode` for a context through a single thread with a FIFO queue; this also gives the A-before-B ordering of feature 9 for model calls. Use `llama_set_abort_callback` for cancellation.
7. **Thread placement.** Keep the default OpenMP build. To restrict inference to P-cores create a `ggml_threadpool` with `cpumask` set for logical CPUs 0 to 15, `strict_cpu` as required, and attach it with `llama_attach_threadpool`. The numbering 0 to 15 for the hyper-threaded P-cores is an assumption; read the efficiency class of each logical processor from the operating system before building the mask.
8. **Fine-tuning toggle.** Run training in a **separate process** on a **separate F32 copy**: `load_mode = LLAMA_LOAD_MODE_NONE`, flash attention disabled, F32 K/V types, `n_batch = n_ubatch = 128`, `n_ctx` a multiple of it. Accept only two-dimensional tensors named `blk.<i>.` with i >= 24 - k, k <= 16, default k between 4 and 8. Build the data set in signed arithmetic and refuse unless T >= 2 n_ctx + 1 (F4). Treat a non-zero exit code as "training unavailable" and keep serving the base model.
9. **Promotion.** Do not serve the file written by `llama_model_save_to_file`. Either copy trained tensors into a copy of the base GGUF with the `gguf.h` API, or extract an adapter (F11) and apply it with `llama_set_adapters_lora` or merge it with `llama-export-lora`. Always send an explicit system message.
10. **Quantisation.** Report measured bits per weight, not preset names. For the 0.5B model compare the fallback mixture (6.16) against block-32 types (`Q4_0`, `IQ4_NL`, nominal 4.5) and against overrides of `--output-tensor-type` and `--token-embedding-type`, using perplexity and KL divergence on held-out files of the demo repository. Offer the 1.5B model as the case where K-quants apply fully.
11. **Model intake.** Download with `curl -L --fail --retry 3 -C - --ssl-no-revoke`, pin the repository revision in the URL (`resolve/<commit>/<file>`; four such URLs were checked and answer 200), verify size and SHA-256 from the reference list, then run a functional smoke test before first use [97]. Prefer producing the CodeRankEmbed GGUF ourselves from `nomic-ai/CodeRankEmbed`; otherwise pin the community file by hash. Read pooling from the model **and** assert it equals the expected value (CLS, MEAN or LAST); add the query prefix in our code; cap chunks at the model's context (2048 for the 137M models); do not ship the non-commercial model.
12. **Grammar-constrained edits and honest reporting.** Use `llama_sampler_init_grammar` for edit proposals together with a hard token limit and the abort callback [100]. In the paper describe training as experimental upstream functionality, cite the maintainer's caveat, label every timing as a single-machine single-run measurement on the mimalloc fork, and reproduce the chain on `stories260K-f32.gguf` as a smoke test before each run on the 0.5B model.

## References

Numbering follows `llama_cpp_stack.sources.json`. "Tag" means llama.cpp `v0.5.0`. All items retrieved 2026-09-29.

### llama.cpp release and source files

1. ggml-org. llama.cpp release v0.5.0. GitHub release, 2026-09-23. https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0
2. G. Gerganov. dist : releases and versioning of ggml-org projects (Discussion #1579). GitHub Discussions, ggml-org/ggml, 2026-08-07. https://github.com/ggml-org/ggml/discussions/1579
3. llama.cpp contributors. include/llama.h at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/include/llama.h
4. J. Gaessler and contributors. ggml/include/ggml-opt.h at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/include/ggml-opt.h
5. llama.cpp contributors. ggml/src/ggml-opt.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/src/ggml-opt.cpp
6. llama.cpp contributors. ggml/src/ggml-cpu/ops.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/src/ggml-cpu/ops.cpp
7. llama.cpp contributors. ggml/src/ggml-cpu/ggml-cpu.c at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/src/ggml-cpu/ggml-cpu.c
8. llama.cpp contributors. ggml/include/ggml.h and ggml-cpu.h at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/include/ggml.h
9. llama.cpp contributors. examples/training at tag. https://github.com/ggml-org/llama.cpp/tree/v0.5.0/examples/training
10. llama.cpp contributors. src/llama-context.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-context.cpp
11. llama.cpp contributors. common/common.cpp, common.h, arg.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/common/common.cpp
12. llama.cpp contributors. src/llama-model-saver.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-model-saver.cpp
13. llama.cpp contributors. src/llama-adapter.cpp and llama-adapter.h at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-adapter.cpp
14. llama.cpp contributors. src/llama-graph.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-graph.cpp
15. llama.cpp contributors. src/llama-quant.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-quant.cpp
16. llama.cpp contributors. tools/quantize/README.md at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/quantize/README.md
17. llama.cpp contributors. tools/quantize/quantize.cpp and CMakeLists.txt at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/quantize/quantize.cpp
18. llama.cpp contributors. tools/imatrix/README.md at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/imatrix/README.md
19. llama.cpp contributors. tools/imatrix/imatrix.cpp and common/imatrix-loader.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/imatrix/imatrix.cpp
20. llama.cpp contributors. tools/export-lora/README.md at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/export-lora/README.md
21. llama.cpp contributors. examples/embedding/embedding.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/examples/embedding/embedding.cpp
22. llama.cpp contributors. CMake build files at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/CMakeLists.txt
23. ggml-org contributors. ggml/CMakeLists.txt at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/ggml/CMakeLists.txt
24. llama.cpp contributors. docs/build.md at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/docs/build.md
25. llama.cpp contributors. src/llama-chat.cpp and src/llama.cpp at tag. https://github.com/ggml-org/llama.cpp/blob/v0.5.0/src/llama-chat.cpp
26. llama.cpp contributors. Templates supported by llama_chat_apply_template. GitHub wiki. https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
27. llama.cpp contributors. Git tree at tag. GitHub REST API. https://api.github.com/repos/ggml-org/llama.cpp/git/trees/v0.5.0?recursive=1

### llama.cpp pull requests and issues

28. I. Kawrakow. Importance Matrix calculation. Pull request #4861, merged 2024-01-12. https://github.com/ggml-org/llama.cpp/pull/4861
29. X. S. Nguyen. examples : remove finetune and train-text-from-scratch. Pull request #8669, merged 2024-07-25. https://github.com/ggml-org/llama.cpp/pull/8669
30. J. Gaessler. llama/ggml: add LLM training support. Pull request #10544, merged 2025-05-12. https://github.com/ggml-org/llama.cpp/pull/10544
31. J. Graehl. finetune: SGD optimizer, more CLI args. Pull request #13873, merged 2025-08-14. https://github.com/ggml-org/llama.cpp/pull/13873
32. srossitto79. native QLoRA training with reward-weighted SFT and GRPO. Pull request #20453, closed without merge 2026-05-04. https://github.com/ggml-org/llama.cpp/pull/20453
33. srossitto79. Feat/qlora training. Pull request #22705, closed without merge 2026-09-24. https://github.com/ggml-org/llama.cpp/pull/22705
34. X. S. Nguyen. finetune: fix no KV cache. Pull request #27199, merged 2026-09-02. https://github.com/ggml-org/llama.cpp/pull/27199
35. daineball. qwen2 : use no-cache attention path for training graphs. Pull request #25428, open. https://github.com/ggml-org/llama.cpp/pull/25428
36. unravel1020. llama-finetune crashes with "failed to allocate buffer of size 18446744073709547520". Issue #27648, opened 2026-08-24, open. https://github.com/ggml-org/llama.cpp/issues/27648
37. unravel1020. common : fix unsigned underflow in common_opt_dataset_init. Pull request #27690, open. https://github.com/ggml-org/llama.cpp/pull/27690
38. yousefabdel1727; comments by J. Gaessler. llama-finetune won't work even with 17M parameters arch:llama. Issue #18499, opened 2025-12-30, open. https://github.com/ggml-org/llama.cpp/issues/18499
39. mmpataki. performance: MSVC compilation doesn't detect AVX-VNNI. Issue #28295, open. https://github.com/ggml-org/llama.cpp/issues/28295
40. Manus-Vindicte. [MSVC] \uHHHH escapes in BPE pre-tokenizer patterns abort model load. Issue #28788, open. https://github.com/ggml-org/llama.cpp/issues/28788
41. D. Bevenius. cmake : skip PCH for llama-server PCH when using MSVC. Pull request #28763, merged 2026-09-11. https://github.com/ggml-org/llama.cpp/pull/28763
42. J. Gaessler. LoRA training example. Issue #13485, opened 2025-05-12, open. https://github.com/ggml-org/llama.cpp/issues/13485

### Model repositories (Hugging Face)

43. Qwen team. Qwen2.5-Coder-0.5B-Instruct-GGUF. 2024. https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF
44. bartowski. Qwen2.5-Coder-0.5B-Instruct-GGUF (conversion). 2024. https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF
45. Qwen team. Qwen2.5-Coder-1.5B-Instruct-GGUF. 2024. https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
46. bartowski. Qwen2.5-Coder-1.5B-Instruct-GGUF (conversion). 2024. https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF
47. Qwen team. Qwen3-0.6B-GGUF. 2025. https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
48. Unsloth. Qwen3-0.6B-GGUF (conversion). 2025. https://huggingface.co/unsloth/Qwen3-0.6B-GGUF
49. ggml-org. gemma-3-270m-it-GGUF (conversion). 2025. https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF
50. Unsloth. gemma-3-270m-it-GGUF (conversion). 2025. https://huggingface.co/unsloth/gemma-3-270m-it-GGUF
51. Google. gemma-3-270m-it (gated). 2025. https://huggingface.co/google/gemma-3-270m-it
52. ggml-org. test-model-stories260K. 2025. https://huggingface.co/ggml-org/test-model-stories260K
53. Qwen team. Qwen3-Embedding-0.6B-GGUF. 2025. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF
54. Qwen team. Qwen3-Embedding-0.6B (model card). 2025. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
55. Nomic AI. nomic-embed-text-v1.5-GGUF. 2024. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
56. Nomic AI and UIUC. CodeRankEmbed (model card). 2024. https://huggingface.co/nomic-ai/CodeRankEmbed
57. handwoven8588. CodeRankEmbed-GGUF (conversion). 2026. https://huggingface.co/handwoven8588/CodeRankEmbed-GGUF
58. brandtcormorant. CodeRankEmbed-Q8_0-GGUF (conversion). 2025. https://huggingface.co/brandtcormorant/CodeRankEmbed-Q8_0-GGUF
59. Jina AI. jina-code-embeddings-0.5b-GGUF. 2025. https://huggingface.co/jinaai/jina-code-embeddings-0.5b-GGUF
60. ggml-org. embeddinggemma-300M-GGUF (conversion). 2025. https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF
61. Unsloth. embeddinggemma-300m-GGUF (conversion). 2025. https://huggingface.co/unsloth/embeddinggemma-300m-GGUF
62. Google. embeddinggemma-300m (gated). 2025. https://huggingface.co/google/embeddinggemma-300m

### Other dependencies and documentation

63. M. Brunsfeld and contributors. tree-sitter v0.27.0. 2026-08-30. https://github.com/tree-sitter/tree-sitter/tree/v0.27.0
64. tree-sitter contributors. tree-sitter-c v0.24.2. 2026-04-22. https://github.com/tree-sitter/tree-sitter-c/tree/v0.24.2
65. tree-sitter contributors. tree-sitter-cpp v0.23.4. 2024-11-11. https://github.com/tree-sitter/tree-sitter-cpp/tree/v0.23.4
66. Y. Hirose and contributors. cpp-httplib v0.58.0. 2026-09-26. https://github.com/yhirose/cpp-httplib/tree/v0.58.0
67. N. Lohmann and contributors. JSON for Modern C++ v3.12.0. 2025-04-11. https://github.com/nlohmann/json/tree/v3.12.0
68. P. Williams and contributors. Tectonic 0.17.0 release. 2026-07-27. https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0
69. Tectonic project. How To: Install Tectonic. https://tectonic-typesetting.github.io/book/latest/installation/index.html
70. Tectonic project. Build Your First Document. https://tectonic-typesetting.github.io/book/latest/getting-started/first-document.html
71. Kitware. CMake documentation: FetchContent module. https://cmake.org/cmake/help/latest/module/FetchContent.html
72. Kitware. CMake documentation: ExternalProject module. https://cmake.org/cmake/help/latest/module/ExternalProject.html
73. Kitware. CMake 4.0 Release Notes. 2025. https://cmake.org/cmake/help/latest/release/4.0.html
74. Git for Windows project. Documentation/config/core.adoc (core.longpaths), tag v2.53.0.windows.1. https://github.com/git-for-windows/git/blob/v2.53.0.windows.1/Documentation/config/core.adoc

### Papers and reports

75. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101. https://arxiv.org/abs/1711.05101
76. E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
77. C. Eckart, G. Young. The Approximation of One Matrix by Another of Lower Rank. Psychometrika 1(3), 211-218, 1936. https://doi.org/10.1007/BF02288367
78. L. Mirsky. Symmetric Gauge Functions and Unitarily Invariant Norms. The Quarterly Journal of Mathematics 11(1), 50-59, 1960. https://doi.org/10.1093/qmath/11.1.50
79. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. arXiv:2210.17323. https://arxiv.org/abs/2210.17323
80. J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, S. Han. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024. arXiv:2306.00978. https://arxiv.org/abs/2306.00978
81. T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314, 2023. https://arxiv.org/abs/2305.14314
82. R.-G. Dumitru, V. Yadav, R. Maheshwary, P.-I. Clotan, S. T. Madhusudhan, M. Surdeanu. Layer-Wise Quantization: A Pragmatic and Effective Method for Quantizing LLMs Beyond Integer Bit-Levels. arXiv:2406.17415, 2024. https://arxiv.org/abs/2406.17415
83. B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Lu, K. Dang, Y. Fan, Y. Zhang, A. Yang, R. Men, F. Huang, B. Zheng, Y. Miao, S. Quan, Y. Feng, X. Ren, X. Ren, J. Zhou, J. Lin. Qwen2.5-Coder Technical Report. arXiv:2409.12186, 2024. https://arxiv.org/abs/2409.12186
84. Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, J. Zhou. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176, 2025. https://arxiv.org/abs/2506.05176
85. T. Suresh, R. G. Reddy, Y. Xu, Z. Nussbaum, A. Mulyar, B. Duderstadt, H. Ji. CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking. ICLR 2025. arXiv:2412.01007. https://arxiv.org/abs/2412.01007
86. Z. Nussbaum, J. X. Morris, B. Duderstadt, A. Mulyar. Nomic Embed: Training a Reproducible Long Context Text Embedder. TMLR, 2025. arXiv:2402.01613. https://arxiv.org/abs/2402.01613
87. A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S. Kakade, P. Jain, A. Farhadi. Matryoshka Representation Learning. NeurIPS 2022. arXiv:2205.13147. https://arxiv.org/abs/2205.13147
88. D. Kryvosheieva, S. Sturua, M. Günther, H. Xiao. Efficient Code Embeddings from Code Generation Models. NeurIPS 2025 Workshop on Deep Learning for Code. arXiv:2508.21290. https://arxiv.org/abs/2508.21290
89. H. Schechter Vera, S. Dua, B. Zhang, D. Salz, R. Mullins, et al. (89 authors). EmbeddingGemma: Powerful and Lightweight Text Representations. arXiv:2509.20354, 2025. https://arxiv.org/abs/2509.20354
90. Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, C. Finn. Surgical Fine-Tuning Improves Adaptation to Distribution Shifts. ICLR 2023. arXiv:2210.11466. https://arxiv.org/abs/2210.11466
91. R. Pan, X. Liu, S. Diao, R. Pi, J. Zhang, C. Han, T. Zhang. LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning. NeurIPS 2024. arXiv:2403.17919. https://arxiv.org/abs/2403.17919
92. R. Shuttleworth, J. Andreas, A. Torralba, P. Sharma. LoRA vs Full Fine-tuning: An Illusion of Equivalence. arXiv:2410.21228, 2024-2025. https://arxiv.org/abs/2410.21228
93. I. Tenison, S. Ahn, M. Kim, E. Alshehri, L. Kagal. Parameter Efficiency Is Not Memory Efficiency: Rethinking Fine-Tuning for On-Device LLM Adaptation. arXiv:2604.22783, 2026. https://arxiv.org/abs/2604.22783
94. J. Geng, L. Zhao, Y. Lu, B. Luo. MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications. arXiv:2512.08211, 2025. https://arxiv.org/abs/2512.08211
95. Subash SN, A. Nambiar, P. Lambert, M. Gritta, G. Cordella, A. Nurman, et al. An Edge-First Generalized LLM LoRA Fine-Tuning Framework for Heterogeneous GPUs. Hugging Face blog (QVAC, Tether Data), 2025-12-01. https://huggingface.co/blog/qvac/fabric-llm-finetune
96. K. Egashira, R. Staab, M. Vero, J. He, M. Vechev. Mind the Gap: A Practical Attack on GGUF Quantization. ICML 2025. arXiv:2505.23786. https://arxiv.org/abs/2505.23786
97. A. Patodiya. Broken on Arrival: Silently Defective LLM Artifacts in Public Model Registries and How to Catch Them. arXiv:2609.05881, 2026 (preprint). https://arxiv.org/abs/2609.05881
98. U. Kurt. Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct. arXiv:2601.14277, 2026. https://arxiv.org/abs/2601.14277
99. S. Afrin, M. Z. Haque, A. Mastropaolo. Quantize with Confidence? An Empirical Study of Quantization for Code Generation. arXiv:2607.14181, 2026. https://arxiv.org/abs/2607.14181
100. L. Tang, Y. Zheng. Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation. REALM workshop at EMNLP 2026. arXiv:2609.26693. https://arxiv.org/abs/2609.26693
101. X. Qiu, C. Xu, B. Su, Z. Chen, R. Xu, S. Dai. GGUF-Metadata Prediction of Single-Sequence llama.cpp Throughput Across Three Systems. arXiv:2609.14864, 2026 (preprint). https://arxiv.org/abs/2609.14864
102. W. Zhang, T. Gu, M. Canini, C. Xu, J. Weng. Cache-Resident LLM Inference in GB-Scale Last-Level Caches. arXiv:2606.25353, 2026 (preprint). https://arxiv.org/abs/2606.25353
103. Y. Si, Y. Lin, D. Li, J. Zhang. The Ingestion Tax: Adopting File-Backed Weights in Tensor Frameworks. arXiv:2608.12114, 2026 (preprint). https://arxiv.org/abs/2608.12114

### First-party measurements

104. programmersRAG research session. Scratch experiments on the target machine, 2026-09-29. Unpublished; single runs; logs audited by the checker without re-running. Cite as own measurement, not as literature.
