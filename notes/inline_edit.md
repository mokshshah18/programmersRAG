# Applying LLM-proposed inline code edits reliably

Research note for the programmersRAG C++20 edit applier (feature 11).
Verified on 2026-09-29 by an adversarial citation and mathematics check of an earlier dossier, and re-checked on 2026-09-30 by a second independent pass (corrections C17-C21 below; one source added).
Evaluation corpus: the demo repository `Demo repo/mimalloc`, a fork/clone of microsoft/mimalloc (origin `https://github.com/microsoft/mimalloc.git`, HEAD `31d034d`, 16 September 2026). All local measurements below were made read-only on that corpus.

Evidence grades used throughout: **[P]** peer-reviewed publication, **[A]** arXiv preprint or workshop paper, **[V]** vendor or practitioner report (not peer reviewed), **[D]** official documentation, standard or source code, **[L]** local measurement made for this project, **[PROPOSAL]** our own design or derivation, not an established result.

## Summary

1. **Format.** The sources examined agree on one point: edits addressed by a bare line number, with no check of the content at that line, are the least reliable representation. Content-anchored edits (search/replace, `str_replace`, V4A context hunks) are what independent tools converged on [27, 37, 42, 44]. Which content-anchored format is best is *not* settled and depends on model size: search/replace is best for strong models [16, 23], while for fine-tuned 7B models a rewrite of one syntactic unit (function or block) matches full-file accuracy and beats line-fragment diffs [31], and for sub-1B models whole-file output beats iterative search/replace by about 40 points [33].
2. **Matching.** Whether a content-anchored edit fails *cleanly* or *silently* is decided by the applier's ambiguity policy, not only by the format. A first-match policy turns a 0.000 silent-misapplication rate into about 0.02 [34], and first-occurrence fallback is associated with most failures of small-model search/replace trajectories [33]. Two of the three reference appliers we read (OpenAI Codex, Aider) are first-match; Anthropic's text editor tool is executed by the client application, and its documentation tells implementers to refuse zero or multiple matches [40, 43, 44].
3. **Safety.** POSIX `rename` gives an atomic-visibility guarantee [60, 61]; the Windows `ReplaceFileW` and `MoveFileExW` reference pages do not state one and `ReplaceFileW` documents three partial-failure states [64, 65]. A syntax gate on the edit tool measurably improves task success [13].
4. **C is under-measured.** No edit-format or edit-application benchmark we opened uses C. Issue-resolution results on C and C++ exist and are worse than on Python [20]. Every number taken from the literature must therefore be re-measured on the mimalloc fork.
5. **Local measurements change two design decisions [L].** (a) 15.9% of mimalloc function definitions share their name with another definition in the same file and 12.5% share their first line, so a unit must never be addressed by name or signature line alone. (b) The unmodified corpus already contains 1,250 ERROR and 566 MISSING tree-sitter nodes, and a "did the error count increase" gate caught 92.2% of single-token deletions, so the tree-sitter gate is a fast filter and not a proof of validity.

### Verification status and corrections to the first dossier

The first dossier listed 68 sources. 65 were opened and confirmed (several with corrected metadata), the 2 GNU manual pages could not be opened at gnu.org and were replaced by one entry for the same text read from the manual's Texinfo source, and 1 (the local measurement) was reclassified from "source" to "measurement made in this work" after being re-run independently. 11 sources were added by the checker after being opened, giving 77 verified references; the second pass added one more (Lamberti 2026, [78]), giving 78. Main corrections:

| # | First dossier | Verified |
|---|---|---|
| C1 | SWE-agent: "1185 of 2942 trajectories (51.7%)" | 1,185 of **2,294** task instances (51.7%); venue is NeurIPS 2024 [13] |
| C2 | EDIT-Bench best model pass@1 66.67% | arXiv v2 reports claude-sonnet-4 at **64.8%**; venue ICLR 2026 (Oral) confirmed from the OpenReview record, camera-ready PDF not retrievable [24] |
| C3 | SWE-Edit Tables 4 and 5; "parity with GPT-5-nano as an editor" | In v2 the tables are 3 and 4. GRPO-trained Qwen3-8B: PR-Edit grader 68.4 vs 66.4, resolve rate 69.9 vs 70.0, edit success 81.1 vs 82.0 for GPT-5-nano. The adaptive policy beats fixed find-replace by only 0.9 points of edit success (81.1 vs 80.2); the PR-Edit correlation is r = 0.98 on n = 4 models [30] |
| C4 | "3 lines of context leaves roughly 1 window in 11 ambiguous" | Conflates a 3-line window with the convention of 3 lines above *and* below. Measured: 3-line windows are ambiguous 9.3% of the time; a single changed line with 3 lines of context on each side is ambiguous for 0.8% of edit sites [L] |
| C5 | Geometric fit A(k) = 0.452 x 0.46^(k-1) | Holds only for k <= 5. Re-measured values are 0.451, 0.215, 0.093, 0.040, 0.020, 0.012, 0.0069, 0.0040 for k = 1..8; the ratio rises to about 0.58 for k >= 5 [L] |
| C6 | "A Myers O(ND) verifier can certify ed <= k by exploring only D <= 2k" | False as stated. D > 2k proves ed > k and D <= k proves ed <= k, but k < D <= 2k is inconclusive (formula F7) |
| C7 | Fuzzy criterion d(1) <= k < d(2), "corresponds to the similarity floor 0.90 validated by Robust-Apply" | Robust-Apply also requires a margin of 0.20 and does not name its similarity function, so its measured silent rate does not transfer. The criterion now carries an explicit margin and the soundness statement is weakened to what can be proved (formula F13) |
| C8 | Invariant "not committed implies the backup holds the old content" | The backup does not exist before the replace call. Correct form: the old content is readable at the target path *or* at the backup path (formula F18) |
| C9 | Mode U addresses a unit by "file path plus the unit's signature line" | Ambiguous for 12.5% of mimalloc functions [L]; units must be addressed by an identifier issued by retrieval and checked by content hash |
| C10 | cAST "improves Recall@5 by 4.3 points on RepoEval" | 4.3 is the upper end of a range (1.8-4.3 points of recall across retrievers); venue is Findings of EMNLP 2025 [22] |
| C11 | Cascaded editing "11.1% improvement" | Relative improvement; the absolute gain is 7.5 points (67.6 to 75.1 Pass@2) [29] |
| C12 | Venues given as arXiv only | CanItEdit: COLM 2024. Agentless: Proc. ACM Softw. Eng. 2(FSE), 2025. Cascaded Code Editing: Proc. ACM Softw. Eng. 3(FSE), 2026. To Diff or Not to Diff: Findings of ACL 2026. GPT-4.1 guide: Noah MacCallum and Julian Lee, 14 April 2025 |
| C13 | hnswlib "marking deleted does not free memory" | Not stated in the README; removed [75] |
| C14 | "Location-anchored addressing is the most dangerous", "evidence from four independent sources" followed by five | Reworded to bare line-number addressing; now seven sources, each with its own caveat (E3) |
| C15 | Althoubi: "the similarity function is not specified" | Still true that it is not named, but the paper cites Ratcliff and Metzener (1988) among its matching methods, which is the basis of Python's difflib; this suggests, without confirming, a difflib-style ratio [34, 59] |
| C16 | Cheng et al.: unchanged context lines "often disorient the model" quoted as a general property of diffs | The sentence is about the unified-diff-like *hunk style* inside content-addressed formats (Appendix A). The number-indexed results were produced with a custom patch tool that uses line numbers only [31] |
| C17 | EDIT-Bench: "only 1 of 40 models exceeds 60%" | That is the arXiv wording. The ICLR 2026 proceedings abstract (mlanthology record) says "only 3 models score over 60%". The two versions differ; quote the version you cite [24] |
| C18 | "Every Ollama-served model is benchmarked with the whole format" | Not every one: `ollama/codegeex4` is run with `diff` (32.3% correct, 97.0% format compliance). Format compliance is also not correctness: Qwen2.5-Coder-0.5B-Instruct is 100.0% compliant but solves 14.3% [39] |
| C19 | "Anthropic's documented behaviour refuses both zero matches and multiple matches" | The text editor tool is executed by the client application. The quoted error strings are examples the docs tell the *implementer* to return, together with the best practice "Make sure replacements match exactly one location". The refusal policy is a recommendation, not something Anthropic's service enforces [44] |
| C20 | Robust-Apply K_min = 2 read as "2 lines" | Algorithm 1 writes the test as \|A\| < K_min but the paper does not state the unit of \|A\| (characters, tokens or lines). K_min in our proposal is defined separately, in non-blank lines [34] |
| C21 | Second pass could not re-open the diff3 PDF (connection reset from upenn.edu) | Bibliographic record re-confirmed (FSTTCS 2007, pp. 485-496); the theorem-level statements in E13 rest on the first pass's reading only and E13 is used as motivation, not as evidence [4] |

## Established results

**E1. Plain text beats JSON or function-call wrapping for code [V].** On Aider's benchmark of 133 Exercism Python exercises, returning the whole file in a fenced block was "the most reliable and effective edit format across all GPT-3.5 and GPT-4 models", and "Using the new functions API for edits performed worse than the above whole file method, for all the models" [36]. A 2024 follow-up on four models found that code returned inside JSON is "more prone to syntax errors" and that OpenAI's strict JSON mode "offered no improvement"; gpt-4o-2024-05-13 was the one model whose markdown and JSON results were close [38]. Practitioner benchmarks, not peer reviewed.

**E2. A line-number-free unified diff reduced "lazy" output [V].** With gpt-4-1106-preview on 89 Python refactoring tasks, Aider's search/replace baseline scored 20% and its simplified unified diff 61%; lazy comments fell from 12 tasks to 4. The stated design principles are FAMILIAR, SIMPLE (avoid "brittle specifiers like line numbers or line counts"), HIGH LEVEL (whole functions or blocks rather than surgical lines) and FLEXIBLE. Disabling flexible patch application produced "a 9X increase in editing errors" (on the Exercism benchmark) and omitting high-level-diff prompting a 30-50% increase [35].

**E3. Bare line-number addressing is the least reliable representation in the sources examined.** (The one line-addressed scheme reported to work well, hashline in R12, attaches a content hash to every line reference.)
- Aider: "GPT is terrible at working with source code line numbers" [V] [35].
- Cursor: "models are notoriously bad at counting line numbers" [V] [45].
- OpenAI: "we do not use line numbers in this diff format, as the context is enough to uniquely identify code" [D] [42].
- Fine-tuned Qwen2.5-Coder-7B, average pass@1 over five benchmarks: minimal unified diff 14.07, unified diff 33.15, versus full code 57.07; supplying line-numbered input raised these only to 31.13 and 37.66 [P] [31].
- Apply-only benchmark: after a one-line shift, line-range edits applied correctly in 0.000 of trials and silently corrupted 0.991 of files [A] [34].
- A training corpus of verified search/replace blocks, compared with a unified-diff corpus of similar size that also differed in filtering, was associated with a valid-patch rate of 96.3% against 89.7% for a 32B model; the authors offer the format as a hypothesis for the gain, not as an isolated effect [P] [25].
- Under a 64k-token context, single-shot unified-diff generation produced "incorrect line numbers in the diff header" and patches for non-existent files; Qwen3-Coder-30B-A3B resolved 7% and GPT-5-nano 0% of 100 SWE-bench Verified samples [A, workshop] [26].

**E4. Independent tools converged on content-anchored edits [D].** Aider SEARCH/REPLACE blocks [37]; Anthropic `str_replace`, whose `old_str` "must match exactly, including whitespace and indentation" [44]; OpenAI V4A (`*** Begin Patch`, `*** Update File:`, `@@` scope lines, 3 lines of context above and below by default, `@@` class or function anchors when that is not unique) [41, 42, 43]. OpenAI states that search/replace and a pseudo-XML format "both had high success rates" and share two properties: "(1) they do not use line numbers, and (2) they provide both the exact code to be replaced, and the exact code with which to replace it, with clear delimiters between the two" [42]. A source-code study of 13 open-source agents found the `str_replace` interface in 5 of them and describes edit formats as converging on string replacement [A] [27].

**E5. Reference appliers disagree on ambiguity [D].**
- Anthropic's text editor tool is implemented by the client; its documentation tells the implementer to return an error on zero matches ("Error: No match found for replacement. Please check your text and try again.") and on multiple matches ("Error: Found 3 matches for replacement text. Please provide more context to make a unique match."), and lists "Make sure replacements match exactly one location" as a best practice [44].
- OpenAI Codex `seek_sequence` returns the *first* index at or after a cursor, trying four passes of decreasing strictness: exact, trailing whitespace ignored, leading and trailing whitespace ignored, Unicode dash/quote/space folding [43].
- Aider's `perfect_replace` also returns the first matching window; it then tries a leading-whitespace-tolerant match and `...` elision; its edit-distance matcher (difflib `SequenceMatcher`, threshold 0.8) is present but unreachable behind a bare `return` [40].
- Cheng et al. treat zero or multiple matches as a patch failure and relax whitespace and blank lines only when no exact match exists [31].

**E6. A syntax guard on the edit tool improves task success [P].** On SWE-bench Lite with GPT-4 Turbo, SWE-agent resolved 18.0% with a linting edit command, 15.0% without linting and 10.3% with no edit command; "Invalid edits are discarded, and the agent is asked to try editing the file again". Of 2,294 task instances, 1,185 (51.7%) had at least one failed edit. An edit attempt has a 90.5% chance of eventually succeeding, falling to 57.2% after one failed edit; successful recoveries were preceded by about 2 failed attempts on average, unsuccessful ones by 4.2 to 5.6 [13].

**E7. Classical patch tools already encode the accuracy/safety trade-off [D].** GNU patch takes the hunk's line number plus the previous offset as a first guess, then "scans both forward and backward"; with fuzz factor f it ignores the first and last f context lines; "a larger fuzz factor increases the odds of making a faulty patch. The default fuzz factor is 2"; and "the results are guaranteed only when the patch is applied to an exact copy of the file that the patch was generated from" [55]. `git apply` "by default fails the whole patch and does not touch the working tree when some of the hunks do not apply", ignores no context by default, and discourages context-free patches [54].

**E8. diff-match-patch [D, V].** It locates a pattern with the Bitap algorithm and scores candidates by normalised error count plus normalised distance from the expected location (formula F8). Defaults: `Match_Threshold` 0.5, `Match_Distance` 1000, `Patch_DeleteThreshold` 0.5, `Patch_Margin` 4, `Match_MaxBits` 32 [57, 58]. Fraser's design note observes that GNU patch's strategy is "unrestricted by location" and argues that context length "must be tailored to each edit block" until the block is unique in the file [56].

**E9. Shortest edit scripts [P].** Myers' greedy algorithm finds a shortest insert/delete script in O((N+M)D) time, with O(N+M+D^2) expected time and a linear-space refinement; it "performs well when differences are small" [1]. See F6.

**E10. Patience and histogram diff [V, D, P].** The core step of patience diff is to "find all lines which occur exactly once on both sides, then do longest common subsequence on those lines, matching them up" [51]. Histogram diff extends it "to support low-occurrence common elements" and runs in O(N x D) when the chain-length cap is a small constant such as 64 [52]. Git's default is still Myers [53]. An empirical study of 14 Java projects found that diff algorithms disagree in 1.7% to 8.2% of commits for code-churn metrics and recommends Histogram for code changes [6].

**E11. Approximate string matching [P].** Edit distance is symmetric and satisfies the triangle inequality, so strings form a metric space; searching a pattern of length m in a text of length n with at most k errors is the edit-distance dynamic program with the first row set to zero, in O(mn) time and O(m) space; the error level is alpha = k/m; the probability that a random pattern matches is exponentially decreasing in m below a critical error level; Ukkonen's cut-off heuristic gives O(kn) expected time [3]. The Wu-Manber bit-parallel algorithm handles k errors with "two shifts, one AND, and three ORs" per error level per text character [2]. See F1-F5.

**E12. Pigeonhole filtering [P].** "If a pattern is cut in k+1 pieces, then at least one of the pieces must appear unchanged in an approximate occurrence", so candidates can be found by exact search of the pieces and verified in a neighbourhood of length m + 2k [2, 3].

**E13. Unique elements and merging (analogy only) [P].** For diff3, if A changes only the prefix O1 and B only the suffix O3 of O = O1 O2 O3, and the untouched region O2 "contains an element x that occurs exactly once in each of O, A, and B", the merge is unique and conflict-free (Theorem 4.1.1). A long stable region is *not* sufficient: the paper gives a conflict that occurs "even when the stable region O2 is arbitrarily large". Idempotence, stability and near-success do not hold in general [4]. This is a theorem about three-way merge; we use it only as motivation for unique anchors.

**E14. Source code is repetitive at short anchor lengths [V].** In the Linux kernel source "a fifteen-character pattern is likely to occur 250 times in the same file", mostly because of decorative comment lines, against under 2% repeat probability in English prose [56].

**E15. POSIX rename [D].** If the link named by `new` exists, "it shall be removed and old renamed to new. In this case, a link named new shall remain visible to other threads throughout the renaming operation and refer either to the file referred to by new or old before the operation began" [60]; Issue 8 (2024) keeps the requirement [61]. Linux: "there is no point at which another process attempting to access newpath will find it missing"; `EXDEV` across mount points [62]. Durability is separate: `fsync` of a file "does not necessarily ensure that the entry in the directory containing the file has also reached disk. For that an explicit fsync() on a file descriptor for the directory is also needed" [63].

**E16. Windows file replacement [D].** Microsoft recommends, for document-like files, "to write the document to a new file, then replace the original file with the new one. One method to do this is with the ReplaceFile API" [66]. `ReplaceFileW` takes an optional backup name; preserves creation time, short name, object identifier, DACLs, encryption, compression and named streams; requires all three files on one volume; opens the replacement file with no sharing mode; and is described as combining several rename and delete steps. `REPLACEFILE_WRITE_THROUGH` is "not supported". It documents three partial-failure codes: 1175 (both files keep their names), 1176 (with a backup name both keep their names, without one "the replaced file no longer exists"), 1177 (the replaced file "still exists with a different name", the backup name if one was given) [64]. `MoveFileExW` offers `MOVEFILE_REPLACE_EXISTING` and `MOVEFILE_WRITE_THROUGH` [65]. Neither page states an atomicity guarantee. An ordinary rename fails if the target has open handles; with `FILE_RENAME_POSIX_SEMANTICS` (Windows 10 RS1 and later) a rename may replace a file "even if there are existing handles to it" [67]; the Win32 route is `SetFileInformationByHandle` with `FileRenameInfoEx` [68]. `FlushFileBuffers` requires `GENERIC_WRITE` [69].

**E17. Crash consistency depends on the file system [P, D].** Across six Linux file systems "persistence properties vary widely"; rename and link "are seemingly atomic on all file systems that use techniques like journaling or copy-on-write"; append-then-rename is specially recognised by many file systems; 60 vulnerabilities were found in 11 applications [5]. The study covers Linux only. SQLite's rollback journal writes the original content first, and deleting the journal (or the super-journal for multi-file transactions) is the commit point, with hot-journal replay on recovery [70].

**E18. Syntax validation tools [D].** Tree-sitter aims to be "fast enough to parse on every keystroke" and "robust enough to provide useful results even in the presence of syntax errors"; its C API exposes `ts_node_has_error`, `ts_node_is_error`, `ts_node_is_missing`, `ts_tree_edit` and `ts_tree_get_changed_ranges`; a re-parse "internally shares structure with the old tree" [71, 72, 73]. A C grammar exists [74]. Compilers offer a front-end-only check: clang `-fsyntax-only` ("Run the preprocessor, parser and semantic analysis stages") and MSVC `/Zs` ("check only the syntax", "no output files are created") [76, 77].

**E19. Speculative decoding is exact [P, A].** Speculative sampling provably preserves the target distribution and gave 2X-3X on T5-XXL [8]. Copying spans from a reference text and verifying them in one decoding step gives "identical generation results as greedy decoding" with over 2x speed-up [9]. See F16.

**E20. Grammar-constrained decoding [P, D].** GCD guarantees "that the output follows a given structure" without fine-tuning [10]; llama.cpp implements it as GBNF [50]. It has two documented costs: it "can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM" [15], and format restrictions produced "a significant decline in LLMs' reasoning abilities" with stricter constraints causing greater degradation [A] [17].

**E21. Benchmarks are Python-centric [P, A].** SWE-bench: 2,294 problems from 12 Python repositories [11]. CanItEdit: 105 hand-written Python problems, models reproduce the entire "after" code [12]. EDIT-Bench: 540 problems in Python and JavaScript; of 40 models, only 1 scores above 60% in the arXiv version and only 3 in the ICLR 2026 abstract (C17) [24]. Diff-XYZ: 1,000 edits in Python, JavaScript, Java, Kotlin and Rust [23]. CodeEditorBench: 19 LLMs on debugging, translating, polishing and requirement switching [14]. An audit found CanItEdit and EDIT-Bench "concentrate over 90% of evaluation on Python" [28]. Multi-SWE-bench adds C and C++ (1,632 issue-resolution instances over seven languages) and reports that strong models "struggle significantly with languages like C and C++" [A] [20]; it measures issue resolution, not edit formats.

## Recent work (2024-2026)

**R1. Diff ability scales steeply with model size [A].** On Diff-XYZ with the *unified diff* format and a format-describing system prompt, the Qwen2.5-Coder family scored on the Apply task (the model itself applies a given diff) EM 0.00 (0.5B), 0.11 (1.5B), 0.36 (3B), 0.59 (7B), 0.82 (14B), 0.85 (32B). On Diff Generation the 7B model had EM 0.03, apply rate 0.19 and parsing rate 0.70; the 0.5B model had parsing rate 0.23. Search-replace "performs best for larger models", while "smaller open models benefit little from any formatting choice" [23].

**R2. Structure-aware unit rewrites [P].** BlockDiff and FuncDiff express a change as a rewrite of a tree-sitter-derived block or function, anchored by content that is expanded until unique. Fine-tuned Qwen2.5-Coder-7B average pass@1: FullCode 57.07, ContentDiff 54.43, BlockDiff 55.98, FuncDiff 57.32; with the AdaEdit adaptive rule BlockDiff 57.61 and FuncDiff 57.95. AdaEdit reduces "both latency and cost by over 30% on long-code editing tasks". The authors attribute the failure of conventional diffs to "fragile offsets" and "fragmented hunks that break the syntactic integrity of code", and note that the search/replace style "can fail to patch if the source code contains characters identical to its special delimiters". Training and evaluation are Python-centred [31].

**R3. Edit-format competence can be trained into an 8B model [A].** GRPO training of Qwen3-8B raised format success from 76.8% to 90.4% and downstream edit success from 68.6% to 81.1% (C3 gives the comparison with GPT-5-nano and the small size of the adaptive-policy effect) [30].

**R4. The "context tax" [P].** To localise an edit unambiguously, "both unified diff and search and replace must generate [Old Code] + [New Code] pairs. Since [Old Code] serves as a search anchor, models generate extensive surrounding context to disambiguate similar or repeated code patterns", which "negates theoretical efficiency gains". Whole-file and search/replace were comparable and unified diff worst. A cascade of DeepSeek-R1 (sketch) with a fine-tuned Qwen2.5-Coder-14B (apply) reached Pass@2 75.1 against 67.6 for DeepSeek-R1 alone on Aider Polyglot [29].

**R5. Very small models: whole file beats iterative search/replace [A].** Static-analysis pass rate, about 1,790 held-out Flutter/Dart tasks per model: Rainbow-Pony-100M 0.802 direct vs 0.347 stepwise; fine-tuned Qwen2.5-Coder-0.5B 0.900 vs 0.501. Among normally completed stepwise trajectories, those needing the first-occurrence fallback passed at 0.123 and 0.175 against 0.570 and 0.800 without it, and accounted for roughly 65% and 71% of failures. This split is observational: trajectories that emit ambiguous spans may be harder tasks. Diff-based editing was competitive only on short, localised edits. Single-author preprint, Dart only [33].

**R6. Content-anchored formats fail cleanly, location-anchored ones silently [A].** Apply-only benchmark, 640 edits over 224 Python files, 23,040 trials, target file fixed before any applier runs:

| Format | Clean success | Stress success | Stress silent |
|---|---|---|---|
| Search/replace, exact | 1.000 | 0.176 | 0.000 |
| Search/replace, whitespace-normalised | 1.000 | 0.571 | 0.000 |
| Search/replace, fuzzy | 1.000 | 0.977 | 0.023 |
| Unified diff (GNU patch, fuzz 0 and 2) | 1.000 | 0.375-0.536 | 0.000 |
| Line range | 1.000 | 0.000 | 0.991 |
| Whole function by name | 0.873 | 0.000 | 0.127 (occurs unperturbed, from duplicate names) |

A first-match policy raised silent misapplication from 0.000 to 0.020 (exact) and 0.023 (whitespace-normalised). Unguarded fuzzy matching reached 0.393 silent on multi-hunk edits. The refuse-when-unsure applier Robust-Apply (F10) recorded 1 silent misapplication in 8,320 trials; the price is applicability, 0.51 against 0.98 on perturbed single-hunk edits. Single-author preprint, Python only, synthetic perturbations, one Linux host [34].

**R7. Verified search/replace blocks as training data [P].** Clean-PR converts pull-request diffs into "minimal unique Search/Replace blocks, verified through round-trip patch application": blocks are applied back to the pre-change snapshot and any example that does not match the post-change state bit-for-bit is discarded. Mid-training Qwen2.5-Coder-32B on this corpus improved SWE-bench Lite by 13.6 and Verified by 12.3 absolute points over the instruction-tuned baseline [25].

**R8. Fine-tuning for editing [P].** EditCoder (fine-tuned DeepSeekCoder) reached 55.90% pass@1 on descriptive CanItEdit instructions against 49.78% for the best open baseline of the same size [12]. NextCoder adapts Qwen2.5-Coder with SeleKT, a dense gradient step followed by a sparse projection onto the base model, and reports gains on five editing benchmarks while retaining generation ability [19].

**R9. Fast apply by speculation on unchanged text [V, P, A, D].** Cursor's fine-tuned Llama-3-70b apply model reached about 1000 tokens/s, "a ~13x speedup over vanilla inference", using "speculative edits" that "speculate on future tokens using a deterministic algorithm rather than a draft model"; evaluation used about 450 full-file edits of files under 400 lines [45]. The serving side validates the speculation against greedy generation at temperature 0 [46]. OpenAI exposes the same idea as Predicted Outputs [47]. EfficientEdit reports up to 10.38x and 13.09x speed-ups on CanItEdit and CodeIF-Bench by reusing original code as the draft and using edit-oriented draft models [P] [21]. PLD+ is tuning-free and targets input-guided tasks including code editing [A] [18]. llama.cpp's server implements draft-model-free n-gram speculation (`ngram-simple`, `ngram-map-k`, `ngram-mod`) and names "the rewriting of source code by a LLM" as the example use [D] [49].

**R10. Incremental re-indexing [A, P, D].** A 2026 coding-agent paper describes an index "updated incrementally on subsequent runs via Merkle-tree diffs over the working copy, so a source edit invalidates and re-indexes only the affected chunks", built from tree-sitter parses into vector, call-graph and BM25 indices, but states the incremental path was "not load-bearing" in its evaluation [32]. AST-based chunking improved RepoEval recall by up to 4.3 points [22]. hnswlib supports incremental insertion, in-place update of an existing label, and `mark_deleted` with optional slot reuse [75]; FreshDiskANN shows graph indices can absorb real-time inserts and deletes while retaining more than 95% 5-recall@5 [7].

**R11. Line-ending preservation [D].** The Codex applier gained an opt-in `PreserveLineEndings` mode in August 2026 because updating a file "historically normalized its contents to LF, which can rewrite line endings outside the requested change"; LF normalisation remains its default (commit message of 21aa552) [43].

**R12. Hash-anchored line addressing (contested) [V].** A blog study over 16 models, 3 runs of 180 mutation-repair tasks on React files, reports that a V4A-style patch tool was the worst format for nearly every model not trained on it (patch failure 50.7% for Grok 4, 46.2% for GLM-4.7) and that "hashline", where each displayed line carries a short content hash that edits must cite, matched or beat `str_replace` for most models [48]. Treat as a hypothesis.

**R13. Local models and the whole format (observational) [V].** On Aider's code-editing leaderboard nearly all Ollama-served models are run with the `whole` format (the exception is `ollama/codegeex4`, run with `diff`: 32.3% correct, 97.0% compliant) and most reach 97-100% format compliance (qwen2.5-coder:32b 100.0%, qwen2.5-coder:14b 98.5%, qwen2.5-coder:7b-instruct-q8_0 100.0%, Qwen2.5-Coder-0.5B-Instruct 100.0%; granite3-dense:8b 78.9%, opencodeinterpreter:6.7b 91.0%), while open-weight models run with the search/replace `diff` format show lower compliance (Qwen2.5-Coder-32B-Instruct 94.7%, openrouter qwen-2.5-coder-32b-instruct 84.2%, llama-3.3-70b-instruct 88.7%, llama3-70b-8192 73.5%). Compliance is not correctness: Qwen2.5-Coder-0.5B-Instruct is 100.0% compliant but solves 14.3%. The format is chosen per model by the maintainers, so this is not a controlled comparison; the only matched pair on the page is llama-3.1-405b-instruct (whole 66.2% correct / 100.0% compliant; diff 63.9% / 92.5%) [39].

**R14. Host-side function replacement helps local models on larger files [A] (added in the second pass).** In a study of structured, line-anchored review feedback, five Ollama-served local models (llama3.1 8B, qwen2.5-coder 14B, gemma4 26B, gemma3 27B, qwen3-coder 30B) were given 36 base-R tasks in six file-size tiers (T1 under 10 lines to T6 of 1,000 lines or more). The coder-tuned local models applied "nearly all of their T4 edits (qwen3-coder 1.00, qwen2.5-coder 0.87) and none at T5-T6" (500 lines and above). Under an exploratory contract in which the model returns only corrected function definitions and the harness "replaces the same-named functions in the seed using base R's parser", correctness on files of 100+ lines rose from 0.115 to 0.342 (llama3.1), 0.099 to 0.339 (qwen2.5-coder) and 0.112 to 0.365 (qwen3-coder). Single-author preprint, R only, 36 tasks, and replacement by name, which M3 shows is unsafe for C with conditional compilation [78]. It is the only source opened that measures a unit-rewrite contract with local models, and it supports Mode U of P1.

## Formulas

Notation: x, y, S, W are strings; |x| is length in characters unless stated; ed is Levenshtein distance (insert, delete, replace, unit cost); d_LCS is the insert/delete-only distance.

**F1. Edit distance recurrence.** Source: [3], Section 5.1.1. Checked against the PDF. The simplified match case is valid because neighbouring cells differ by at most one, so C_{i-1,j-1} <= 1 + min(C_{i-1,j}, C_{i,j-1}).

$$
C_{i,0}=i,\qquad C_{0,j}=j,\qquad
C_{i,j}=\begin{cases}
C_{i-1,j-1} & \text{if } x_i=y_j\\
1+\min\{C_{i-1,j},\;C_{i,j-1},\;C_{i-1,j-1}\} & \text{otherwise}
\end{cases},
\qquad ed(x,y)=C_{|x|,|y|}
$$

Time O(|x||y|), space O(min(|x|,|y|)). ed is symmetric, ed(x,x) = 0 and ed(x,z) <= ed(x,y) + ed(y,z).

**F2. Approximate search with at most k errors.** Source: [3], Sections 3.1 and 5.1.2. P has length m, T has length n.

$$
C_{0,j}=0\ \ (0\le j\le n),\qquad C_{i,0}=i\ \ (0\le i\le m),\qquad
\text{report end position } j \iff C_{m,j}\le k \iff \exists\, i:\ ed(P,\,T_{i..j})\le k
$$

Time O(mn), space O(m).

**F3. Error level.** Source: [3], Section 3.1.

$$
\alpha=\frac{k}{m},\qquad 0<\alpha<1
$$

**F4. Pigeonhole filter and its generalisation.** Source: [3], Section 8; partition approach in [2].

$$
P=P_1P_2\cdots P_{k+1}\ \wedge\ ed(P,W)\le k\ \Longrightarrow\ \exists\, j\in\{1,\dots,k+1\}:\ P_j \text{ occurs unchanged in } W
$$

$$
P=P_1\cdots P_{j}\ \wedge\ ed(P,W)\le k\ \Longrightarrow\ \exists\, i:\ P_i \text{ occurs in } W \text{ with at most } \lfloor k/j\rfloor \text{ errors}
$$

**F5. Wu-Manber bit-parallel transition.** Source: [2], Eq. (2.1). The scan is OCR-damaged, so the recurrence was re-derived from the four cases (match, substitute, insert, delete) and agrees with the paper's operation count.

$$
R^{0}_{j+1}=\mathrm{Rshift}[R^{0}_{j}]\ \mathrm{AND}\ S_c,\qquad
R^{d}_{j+1}=\big(\mathrm{Rshift}[R^{d}_{j}]\ \mathrm{AND}\ S_c\big)\ \mathrm{OR}\ \mathrm{Rshift}\big[R^{d-1}_{j}\ \mathrm{OR}\ R^{d-1}_{j+1}\big]\ \mathrm{OR}\ R^{d-1}_{j},
\qquad c=t_{j+1},\quad R^{d}_{0}=1^{d}0^{m-d}
$$

Bit i of R^d_j is 1 iff the first i pattern characters match a suffix of t_1..t_j with at most d errors; S_c marks pattern positions holding c. Rshift "fills the first position with a 1". The grouped form uses two shifts, one AND and three ORs per error level. Time O(nk ceil(m/w)) for word size w.

**F6. Myers: duality, diagonal lemma, complexity.** Source: [1], Sections 2-4.

$$
D=N+M-2L,\qquad
\text{a } D\text{-path ends on diagonal } k=x-y\in\{-D,\,-D+2,\,\dots,\,D-2,\,D\},\qquad
T_{\text{worst}}=O((N+M)D),\quad T_{\text{expected}}=O(N+M+D^{2})
$$

N = |A|, M = |B|, L the LCS length, D the number of inserted plus deleted symbols in a shortest script.

**F7. Levenshtein versus insert/delete distance [PROPOSAL: own derivation, corrected].**

$$
ed(x,y)\ \le\ d_{LCS}(x,y)=|x|+|y|-2L\ \le\ 2\,ed(x,y)
$$

Left: every insert/delete script is a Levenshtein script. Right: replace each substitution by one deletion plus one insertion. Consequence for a verifier that runs Myers' algorithm with a cap:

$$
D>2k\ \Rightarrow\ ed>k\ \ (\text{reject}),\qquad
D\le k\ \Rightarrow\ ed\le k\ \ (\text{accept}),\qquad
k<D\le 2k\ \Rightarrow\ \lceil D/2\rceil\le ed\le D\ \ (\text{inconclusive})
$$

In the inconclusive band a banded Levenshtein computation with Ukkonen's cut-off [3] is required. The simpler alternative is to state the acceptance threshold directly in d_LCS, which is also a metric and is exactly what Myers' algorithm computes.

**F8. diff-match-patch match score.** Source: [58] (`match_bitapScore`, `match_bitap`), [57].

$$
s(e,x)=\frac{e}{|P|}+\frac{|x-\mathrm{loc}|}{\mathrm{Match\_Distance}},\qquad \text{candidate kept iff } s\le \text{current threshold},\ \text{initially } \mathrm{Match\_Threshold}
$$

e is the Bitap error count at position x, loc the expected location. If `Match_Distance` is 0 the code returns e/|P| when x = loc and 1.0 otherwise. The library keeps the best-scoring match and has no uniqueness test.

**F9. Fraser's fuzzy-patch score.** Source: [56].

$$
s=\frac{e/p}{c}+\frac{d/t}{1-c}
$$

e errors (Levenshtein distance), p pattern length, d offset in characters from the expected location, t text length, 0 < c < 1 a weighting constant. Lower is better.

**F10. Robust-Apply acceptance rule.** Source: [34], Algorithm 1.

$$
\text{apply}\iff |A|\ge K_{\min}\ \wedge\ \Big(\exists!\,m\ \big(\text{exact or whitespace-normalised match of } A \text{ in } F\big)\ \vee\ \big(s_1\ge\tau_{hi}\ \wedge\ s_1-s_2\ge\delta\big)\Big),
\qquad (K_{\min},\tau_{hi},\delta)=(2,\,0.90,\,0.20)
$$

(m_1, s_1), (m_2, s_2) are the two best fuzzy matches by similarity; otherwise the result is Clean-Fail. The similarity function is not named in the paper (C15), and the unit in which |A| and K_min are measured is not stated (C20).

**F11. difflib ratio and its relation to edit distance.** Definition from [59]; the inequalities are [PROPOSAL: own derivation].

$$
\rho(a,b)=\frac{2M}{|a|+|b|}
$$

M is the number of matched elements. The matched blocks form a common subsequence, so M <= L, and therefore

$$
\rho\ \le\ 1-\frac{d_{LCS}(a,b)}{|a|+|b|},\qquad
\rho\ge\tau\ \Longrightarrow\ ed(a,b)\ \le\ d_{LCS}(a,b)\ \le\ (1-\tau)\,(|a|+|b|)
$$

For |a| = |b| = m a ratio floor of 0.90 admits up to 0.20 m insert/delete errors but only 0.10 m substitutions. A difflib threshold is therefore up to twice as permissive as the same number read as a normalised Levenshtein similarity (F14). difflib also applies a junk heuristic by default (items accounting for more than 1% of a sequence of at least 200 items are ignored), which changes ratios on long inputs.

**F12. Tiered uniqueness criterion for the exact tiers [PROPOSAL].** F = (f_1..f_n) are the lines of the search scope, S = (s_1..s_m) the anchor lines, and nu_0..nu_3 the normalisers identity, strip trailing whitespace, strip leading and trailing whitespace, and Unicode dash/quote/space folding after stripping (the four passes of [43]).

$$
\mathcal{M}_t=\{\,i\in\{1,\dots,n-m+1\}:\ \nu_t(f_{i+j-1})=\nu_t(s_j)\ \ \forall j=1..m\,\},\qquad
\mathcal{M}_0\subseteq\mathcal{M}_1\subseteq\mathcal{M}_2\subseteq\mathcal{M}_3
$$

$$
t^{*}=\min\{t:\mathcal{M}_t\neq\emptyset\},\qquad
\mathrm{ACCEPT}_{\text{strict-first}}\iff m_{\text{non-blank}}\ge K_{\min}\ \wedge\ |\mathcal{M}_{t^{*}}|=1,\qquad
\mathrm{ACCEPT}_{\text{conservative}}\iff m_{\text{non-blank}}\ge K_{\min}\ \wedge\ |\mathcal{M}_{3}|=1
$$

The nesting holds because each normaliser factors through the previous one. The strict-first rule accepts a byte-exact match even when a whitespace variant exists elsewhere; the conservative rule refuses in that case. On mimalloc the two differ for about 0.9% of 3-line windows (8.4% ambiguous byte-exact, 9.3% after trimming) [L]. When nu_2 or nu_3 is the accepting tier, the replacement is re-indented by the common indentation delta, as Aider does [40].

**F13. Bounded-error criterion with margin, and what it guarantees [PROPOSAL: own derivation, corrected].** Let S be the anchor after nu_2 normalisation with lines joined by a newline, |S| its length in characters, and k = floor(alpha |S|). Let the candidate set be all line-aligned windows W of the scope (runs of whole lines), d(W) = ed(S, nu_2(W)). Let W_(1) be a minimiser of d, d_(1) = d(W_(1)), and

$$
d_{(2)}=\min\{\,d(W):\ W\cap W_{(1)}=\emptyset\,\},\qquad
\mathrm{ACCEPT}_{\text{fuzzy}}\iff d_{(1)}\le k\ \wedge\ d_{(2)}>d_{(1)}+\Delta,\qquad \Delta\ge 0
$$

Let W* be the window the model intended.

$$
\textbf{(a) Localisation.}\quad d(W^{*})\le k\ \wedge\ \mathrm{ACCEPT}_{\text{fuzzy}}\ \wedge\ d_{(1)}+\Delta\ge k\ \Longrightarrow\ W_{(1)}\cap W^{*}\neq\emptyset
$$

$$
\textbf{(b) No false ambiguity.}\quad d(W^{*})\le k\ \wedge\ \big[\forall W:\ W\cap W^{*}=\emptyset\Rightarrow ed(\nu_2(W^{*}),\nu_2(W))>2k+\Delta\big]\ \Longrightarrow\ d(W)>k+\Delta\ \ \text{for every such } W
$$

$$
\textbf{(c) Cost of a silent error.}\quad \mathrm{ACCEPT}_{\text{fuzzy}}\ \wedge\ W_{(1)}\cap W^{*}=\emptyset\ \Longrightarrow\ d(W^{*})\ \ge\ d_{(2)}\ >\ d(W_{(1)})+\Delta
$$

Proofs. (a) Any window disjoint from W_(1) has distance at least d_(2) > d_(1) + Delta >= k, so W*, which is within k, cannot be disjoint from W_(1). (b) Triangle inequality: d(W) >= ed(W*, W) - d(W*) > 2k + Delta - k. (c) W* is disjoint from W_(1), so d(W*) >= d_(2) by definition. Reading: (b) rules out refusals caused by windows disjoint from the target; a window that overlaps the target but not the winner can still cause a refusal, which is the safe direction. (a) proves only that the accepted window *overlaps* the intended one; equality of the spans is not provable from distances, because an anchor with errors at its ends fits several overlapping windows equally well. (c) says a silent misapplication requires the model's quotation to be closer to the wrong site than to the intended one by more than Delta edits, which is why the margin matters. With sigma = 1 - d/|S|, Robust-Apply's defaults tau_hi = 0.90 and delta = 0.20 would correspond to alpha = 0.10 and Delta = 2k; that correspondence is exact only if its similarity were normalised Levenshtein, which is unconfirmed.

**F14. Similarity floor as an error level [PROPOSAL: own definition].**

$$
\sigma(S,W)=1-\frac{ed(S,W)}{\max(|S|,|W|)}\ \ge\ \tau\iff ed(S,W)\le(1-\tau)\max(|S|,|W|)
$$

Since |W| <= |S| + ed(S,W), this admits at most (1 - tau)|S| / tau errors, slightly more than alpha |S| with alpha = 1 - tau. Thresholds are not transferable between this measure, difflib's ratio (F11) and diff-match-patch's score (F8).

**F15. AdaEdit format-selection rule.** Source: [31], Eq. (3).

$$
E_j=\arg\min_{S\in\{C'_j,\ \mathrm{Diff}(C_j,C'_j)\}}|S|
$$

|.| counts tokens under the model's tokenizer; C_j is the source and C'_j the target code. The rule defines the training target; the model learns the switch during fine-tuning.

**F16. Speculative sampling and the copy-draft special case.** Rule and Eq. (1) from [8]; the special case is [PROPOSAL: own derivation].

$$
x\sim q;\quad \text{accept with probability } \min\!\Big(1,\frac{p(x)}{q(x)}\Big);\quad \text{otherwise resample } x\sim \mathrm{norm}\big(\max(0,\,p-q)\big);
\qquad
\mathbb{E}[\#\text{tokens per target pass}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

alpha is the expected acceptance rate and gamma the number of drafted tokens. If the draft is a copied token c, q is a point mass at c: c is accepted with probability p(c), and on rejection the resampling distribution is p with c removed and renormalised, so the output has probability p(c) for c and (1 - p(c)) p(y)/(1 - p(c)) = p(y) for y != c. Copy-based speculation is therefore exact; under greedy decoding it reduces to "accept the copied token iff it is the argmax". As alpha tends to 1 the expected yield tends to gamma + 1 tokens per pass.

**F17. Anchor ambiguity on the mimalloc fork [L].** A(k) is the fraction of k-line windows, lines trimmed of leading and trailing whitespace, that occur more than once in their own file; 59 files, 31,175 lines.

$$
A(k)=\Pr[\text{a } k\text{-line window is not unique within its file}]
$$

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| A(k), file scope, all windows | 0.4509 | 0.2152 | 0.0932 | 0.0405 | 0.0201 | 0.0118 | 0.0069 | 0.0040 |
| 95% interval half-width | 0.0055 | 0.0046 | 0.0032 | 0.0022 | 0.0016 | 0.0012 | 0.0009 | 0.0007 |
| A(k), windows with no blank line | 0.3660 | 0.1485 | 0.0704 | 0.0406 | 0.0267 | 0.0169 | 0.0089 | 0.0037 |
| Inside functions, file scope | 0.4219 | 0.1863 | 0.0834 | 0.0444 | 0.0276 | 0.0176 | - | - |
| Inside functions, function scope | 0.1845 | 0.0606 | 0.0246 | 0.0123 | 0.0065 | 0.0040 | - | - |

The interval half-widths are nominal score intervals (F19); overlapping windows are not independent samples, so they understate the true uncertainty. Successive ratios A(k+1)/A(k) are 0.477, 0.433, 0.434, 0.498, 0.585, 0.586, 0.583, so a single geometric law does not fit: A(k) is approximately 0.451 x 0.46^(k-1) for k <= 5 and decays more slowly afterwards. For a single changed line with c lines of context on each side, the smallest c giving a unique window is 0 for 57.8% of non-blank lines inside functions, at most 1 for 89.9%, at most 2 for 97.6% and at most 3 for 99.2%.

**F18. Atomic-replace and rollback invariants [PROPOSAL: own formalisation, corrected].** pi is the target path, beta the backup path, h_0 the hash recorded when the file was read.

$$
\textbf{I1 (POSIX visibility)}\quad \forall t:\ \mathrm{content}_t(\pi)\in\{C_{old},\,C_{new}\}
$$

$$
\textbf{I1' (Windows, weaker)}\quad \forall t:\ \pi \text{ absent}\ \vee\ \mathrm{content}_t(\pi)\in\{C_{old},\,C_{new}\}
$$

$$
\textbf{I2 (recoverability)}\quad \mathrm{state}_t\neq\mathrm{COMMITTED}\ \Longrightarrow\ \mathrm{content}_t(\pi)=C_{old}\ \vee\ \mathrm{content}_t(\beta)=C_{old}
$$

$$
\textbf{I3 (staleness guard)}\quad \mathrm{COMMIT}\ \text{only if}\ H(\mathrm{content}(\pi))=h_0
$$

I1 is what POSIX requires of `rename` for concurrent observers [60]; it says nothing about durability after power loss. On Windows only I1' can be claimed from the documentation: with error 1177, and with error 1176 when no backup name was given, no file exists at pi [64]. I2 holds for `ReplaceFileW` only if a backup name is always passed. I3 narrows but does not close the race between the re-hash and the replace.

**F19. Score interval for reported rates [PROPOSAL: standard result, re-derived].** For a successes in n trials, p-hat = a/n and normal quantile z, solving (p-hat - p)^2 = z^2 p(1-p)/n for p gives

$$
p\in\frac{\hat p+\dfrac{z^{2}}{2n}\ \pm\ z\sqrt{\dfrac{\hat p(1-\hat p)}{n}+\dfrac{z^{2}}{4n^{2}}}}{1+\dfrac{z^{2}}{n}}
$$

This is the interval usually attributed to Wilson (1927, J. Amer. Statist. Assoc. 22(158):209-212). Only the bibliographic record of that paper was checked, not its text, so it is not in the reference list; open it before citing.

## What is novel in our proposal

Everything in this section is a proposal or a local measurement. None of it is an established result.

**P1. Two-mode, content-anchored, line-number-free format [PROPOSAL].**
- *Mode U (unit rewrite, default for the local quantized model).* Retrieval hands the model each function with a short unit identifier. The model returns the identifier and the complete new text of that one unit. The applier looks up the byte range and content hash recorded for the identifier, checks that the file still hashes to the same unit text, and replaces that range. The unit is never located by name or by signature line (M3).
- *Mode S (search/replace).* A SEARCH/REPLACE block with the file path and an optional unit identifier as scope, used when the block is shorter in tokens than the unit (F15).
- Only the envelope (markers, path, mode tag, unit identifier) is grammar-constrained; code is plain text, never inside JSON.
- What is new relative to the literature: BlockDiff/FuncDiff anchor units by content that is expanded until unique [31]; we anchor by a retrieval-issued identifier verified by hash, which removes the anchor tokens from the output and turns a stale reference into a clean failure. This borrows the stale-reference idea of hashline [48] at unit rather than line granularity. No source measures this combination.
- Support for the ingredients: unit rewrites match full-code accuracy on a fine-tuned 7B model [31]; a host-side function-replacement contract roughly tripled local-model correctness on 100+ line files [78]; small models comply with whole-output formats [33, 39]; plain text beats JSON [36, 38]; mimalloc functions are short (M4).

**P2. Acceptance criterion [PROPOSAL].** Refuse unless (1) the anchor has at least K_min non-blank lines; (2) the exact tiers give exactly one match in scope (F12); or, only if the fuzzy tier is enabled, (3) the bounded-error test with margin passes (F13) *and* the first and last non-blank anchor lines match the first and last lines of the accepted window under nu_2. Never take the first of several matches. The refusal message lists the candidate locations. What is new: the margin form with statements (a)-(c), and the boundary-line condition that addresses the span ambiguity which (a) leaves open.

**P3. Transaction protocol [PROPOSAL].** READ (record h_0) -> MATCH (all hunks of a file against one snapshot, or none) -> BUILD in memory, preserving line endings and encoding -> VALIDATE (tree-sitter error and missing-node count inside the edited unit must not increase; optional compiler front-end check) -> STAGE (temporary file in the same directory, flushed and closed) -> GUARD (re-hash the target, abort if it differs from h_0) -> COMMIT (`ReplaceFileW` with a backup name on Windows; `rename` plus directory `fsync` on POSIX) -> JOURNAL (for multi-file edits, deletion of the journal is the commit point) -> REINDEX (only chunks whose content hash changed, strictly after commit, in commit-sequence order). Each step is taken from a cited practice [13, 32, 54, 60, 63, 64, 66, 70]; the composition and its invariants (F18) are ours and are untested.

**Local measurements on the mimalloc fork [L].** Read-only; tree-sitter 0.26.0 Python binding with tree-sitter-c 0.24.2 [74]; function counts are approximate because the grammar does not expand macros.

- **M1. Corpus.** 59 files (41 `.c`, 18 `.h`), 1,273,275 bytes, 31,175 lines, LF line endings only, all valid UTF-8; largest file `src/arena.c` with 2,777 lines and 117,283 bytes; 31 files under 400 lines.
- **M2. Anchor ambiguity.** Table in F17. Scoping the search to the enclosing function cuts ambiguity by a factor of 3.4 to 4.3 for k = 3..6.
- **M3. Duplicate units.** Of 1,533 outermost function definitions, 243 (15.9%) share their name with another definition in the same file, 192 (12.5%) have a first line identical to another definition's first line in the same file, and 20 (1.3%) are textually identical to another definition; 25 of 59 files are affected. The cause is conditional compilation. This is the C analogue of the 12.7% wrong-function rate reported for name-addressed edits on Python [34].
- **M4. Unit size.** Function definitions: median 231 bytes and 5 lines; 90th percentile 1,049 bytes and 26 lines; 99th percentile 3,543 bytes and 82 lines; maximum 18,662 bytes and 589 lines. 99.2% are at most 4 KB and 99.9% at most 16 KB.
- **M5. Tree-sitter baseline.** The unmodified corpus parses with 1,250 ERROR nodes and 566 MISSING nodes; 46 of 59 files have at least one. `include/mimalloc.h` alone has 303 ERROR nodes. Parsing all files took about 0.17 s in total through the Python binding on the target machine.
- **M6. Gate sensitivity.** Deleting one real `{`, `}`, `(`, `)` or `;` token inside a function (1,452 trials, up to 30 per file) raised the file's ERROR+MISSING count in 92.2% of trials (95% interval 90.7% to 93.4%); 95.7% (201 of 210) in files with a clean baseline and 91.5% (1,137 of 1,242) in files with baseline errors. Some undetected deletions may leave valid code, so 7.8% is an upper bound on misses for this mutation class.
- **M7. Delimiter collisions.** No line in the corpus matches Aider's SEARCH, divider or REPLACE markers or the V4A `***` and `@@` markers. 58 lines in one file begin with a Markdown code fence, and 744 lines in 54 files are dash rulers.

## Risks and failure modes

1. **Transfer to C is unmeasured.** Every edit-format number cited comes from Python, JavaScript, Dart or mixed Java/Kotlin/Rust corpora, mostly with cloud models. Macro-heavy C with conditional compilation may behave differently [20, 28].
2. **Duplicate definitions under `#if`.** A rewrite of "the" function `mi_lock_acquire` is ambiguous in mimalloc (M3). Name- or signature-addressed unit edits will hit the wrong variant silently.
3. **First-match appliers.** Copying the Codex or Aider matching ladder without a uniqueness test imports a silent-misapplication rate of about 2% in the one controlled measurement available [34].
4. **Fuzzy matching.** Safe on single hunks only with a margin; 39% silent on multi-hunk edits when unguarded [34]. Distances localise a region but do not fix its boundaries (F13a). Thresholds quoted from other tools are in different units (F8, F11, F14).
5. **Refusal cost.** Refusing on ambiguity halves applicability under perturbation (0.51 vs 0.98) [34], and a failed edit lowers the chance of eventual success from 90.5% to 57.2% [13]. Refusals must carry enough information for one good retry.
6. **Weak syntax gate.** The tree-sitter count gate misses up to about 8% of single-token deletions and cannot see type errors, undeclared identifiers or macro misuse (M5, M6). A compiler check needs the project's include paths and defines, and mimalloc compiles different code per platform, so one configuration validates only one branch of each `#if`.
7. **Windows replace is not documented as atomic.** Observers can find the target missing in the 1176-without-backup and 1177 states; antivirus or indexer handles can make the replace fail; `REPLACEFILE_WRITE_THROUGH` is unsupported, so durability after power loss is unspecified [64]. The only crash-consistency study cited covers Linux [5].
8. **Staleness race.** The hash guard (I3) leaves a window between check and replace.
9. **Delimiter collision.** Search/replace fails if the code contains the delimiters [31]; fenced payloads fail if the code contains a fence, which one mimalloc file does (M7).
10. **Grammar constraints.** Constrained decoding distorts the model's distribution [15] and format restrictions reduced reasoning quality in one study [17]; whether a light envelope-only grammar harms a small code model is not measured in any source opened.
11. **Line endings and encoding.** Normalising to LF rewrites bytes outside the edit; the Codex applier does this by default and gained an opt-in preservation mode only in August 2026 [43].
12. **Long contexts.** Patch generation degrades sharply at 64k tokens even with perfect retrieval [26]; the edit prompt must stay short, which is an argument for the pruned retrieval set and for unit-level edits.
13. **Quantization.** No source opened reports edit-format compliance or anchor-reproduction fidelity as a function of quantization level.
14. **Evidence quality.** Three of the most relevant 2026 papers are single-author preprints [27, 33, 34]; Aider, Cursor, Fireworks and hashline figures are not peer reviewed.
15. **Fine-tuning on own edits.** Published gains used corpora of 10^5 to 10^6 examples [19, 25, 29]; whether a few hundred locally logged edits help or overfit is unknown.

## Design recommendations

1. **Address units by identifier and hash, never by name or line number.** Retrieval assigns each function a short identifier bound to (path, byte range, content hash). Mode U output is the identifier plus the full new unit text; the applier verifies the hash before replacing. Reject any edit expressed with line numbers, line ranges or standard unified-diff hunk headers.
2. **Default to Mode U for the local model; use Mode S only when shorter.** mimalloc functions have a median of 5 lines and a 90th percentile of 26, so re-emitting a unit is cheap. Apply the AdaEdit token-length rule per unit and log which mode was chosen.
3. **Constrain only the envelope.** Use a GBNF grammar for markers, path, mode tag and identifier; leave the payload free. Delimit the payload with a marker line that is checked for absence in the target unit before prompting, not with a fixed Markdown fence. Measure pass rate with and without the grammar.
4. **Exact-tier ladder with uniqueness.** Byte-exact, then trailing whitespace ignored, then leading and trailing whitespace ignored with re-indentation, then Unicode punctuation folding. Search the named unit first and the file second. Require exactly one match; ship the conservative variant (unique at the loosest tier) as default and measure the strict-first variant against it.
5. **Minimum anchor.** K_min = 3 non-blank lines when the search is scoped to a unit (2.5% of such windows are ambiguous on mimalloc) and 4 when scoped to the file (about 4%). Ambiguity produces a refusal that lists the candidates with their enclosing function names.
6. **Fuzzy tier off by default.** When enabled: single-hunk edits only; character-level distance after whitespace normalisation; alpha = 0.10 and margin Delta = 2k as starting values; exact match required on the first and last anchor lines; candidates from the k+1-pieces filter over a per-file line-hash index; verification by banded Levenshtein with cut-off (or state the threshold in d_LCS and verify with Myers capped at that threshold). Calibrate on the corpus before trusting any value.
7. **All-or-nothing per file, bottom-up.** Resolve every hunk against the same snapshot, reject the edit if any hunk fails or two hunks overlap, and apply from the end of the file backwards.
8. **Two-level validation.** Level 1, always on: parse the new content with tree-sitter-c and reject if the ERROR+MISSING count inside the edited unit's byte range increases. Level 2, on by default for commits to disk: `cl /Zs` (or `clang -fsyntax-only`) on the affected translation unit with the project's include paths and defines. Return the diagnostic with before and after snippets.
9. **Retry budget.** At most 2 to 3 automatic retries per edit, each with the refusal reason and candidate list; then surface the failure to the user.
10. **Commit protocol on Windows.** Write the new content to a temporary file in the same directory, `FlushFileBuffers`, close the handle, re-hash the target and abort on mismatch, then `ReplaceFileW` with a backup name. Handle 1175 and 1176 as clean failures and 1177 by renaming the backup back. For new files use `MoveFileExW` with `MOVEFILE_WRITE_THROUGH` and without `MOVEFILE_REPLACE_EXISTING`. Evaluate `SetFileInformationByHandle` with `FileRenameInfoEx` and `FILE_RENAME_FLAG_POSIX_SEMANTICS | FILE_RENAME_FLAG_REPLACE_IF_EXISTS` as an alternative when other processes hold the target open. On POSIX: write, `fsync`, `rename`, `fsync` the directory. For multi-file edits keep a journal of (path, backup path, pre-edit hash), flush it before the first replace, delete it after the last, and replay it at start-up. Preserve each file's line endings and encoding.
11. **Re-index after commit, in order.** Key chunks by content hash; re-parse with `ts_tree_edit` and use `ts_tree_get_changed_ranges` to find affected units; re-embed only changed chunks; update the vector index by label; refresh density and cluster statistics for those units. Give every commit a sequence number and apply re-index jobs in that order, so that the index never describes uncommitted content and the project's block-ordering guarantee extends to the edit path.
12. **Evaluation and training data.** Use the known-target protocol: construct the intended post-edit file first, render the edit in each format, perturb only the location surface (re-indentation, line shifts, and for C, duplicated `#if` variants), and report clean success, stress success, clean failure and silent misapplication separately with score intervals (F19). When the fine-tuning toggle is on, log (instruction, retrieved context, emitted edit, match outcome, validation outcome), keep only examples that pass a round-trip check in the deployed format, and enable llama.cpp n-gram speculation for unit rewrites after measuring the speed-up on the target hardware.

## References

Numbering follows the order of `inline_edit.sources.json`. The letter after each entry is the evidence grade defined at the top of this note. Every entry was opened by the checker on 2026-09-29; entries marked *added* were not in the first dossier.

[1] Eugene W. Myers. *An O(ND) Difference Algorithm and Its Variations*. Algorithmica 1(1-4):251-266. 1986. <http://www.xmailserver.org/diff2.pdf>. Key: `myers1986ond`. [P]

[2] Sun Wu, Udi Manber. *Fast Text Searching Allowing Errors*. Communications of the ACM 35(10):83-91. 1992. <https://www.cin.ufpe.br/~paguso/courses/if767/bib/Wu_1992.pdf>. Key: `wu1992agrep`. [P]

[3] Gonzalo Navarro. *A Guided Tour to Approximate String Matching*. ACM Computing Surveys 33(1):31-88. 2001. <https://users.dcc.uchile.cl/~gnavarro/ps/acmcs01.1.pdf>. Key: `navarro2001guided`. [P]

[4] Sanjeev Khanna, Keshav Kunal, Benjamin C. Pierce. *A Formal Investigation of Diff3*. FSTTCS 2007, Lecture Notes in Computer Science 4855, pp. 485-496 (Springer). 2007. <https://www.cis.upenn.edu/~sanjeev/papers/fsttcs07_diff3.pdf>. Key: `khanna2007diff3`. [P]

[5] Thanumalayan Sankaranarayana Pillai, Vijay Chidambaram, Ramnatthan Alagappan, Samer Al-Kiswany, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau. *All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications*. 11th USENIX Symposium on Operating Systems Design and Implementation (OSDI 14), pp. 433-448. 2014. <https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai>. Key: `pillai2014crash`. [P]

[6] Yusuf Sulistyo Nugroho, Hideaki Hata, Kenichi Matsumoto. *How Different Are Different diff Algorithms in Git? Use --histogram for Code Changes*. Empirical Software Engineering 25(1):790-823; arXiv:1902.02467. 2020 (published online 11 September 2019). <https://arxiv.org/abs/1902.02467>. Key: `nugroho2019diff`. [P]

[7] Aditi Singh, Suhas Jayaram Subramanya, Ravishankar Krishnaswamy, Harsha Vardhan Simhadri. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search*. arXiv:2105.09613 (preprint). 2021. <https://arxiv.org/abs/2105.09613>. Key: `singh2021freshdiskann`. [A]

[8] Yaniv Leviathan, Matan Kalman, Yossi Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023 (Oral, per arXiv comments field); arXiv:2211.17192. 2023. <https://arxiv.org/abs/2211.17192>. Key: `leviathan2023speculative`. [P]

[9] Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei. *Inference with Reference: Lossless Acceleration of Large Language Models*. arXiv:2304.04487 (preprint). 2023. <https://arxiv.org/abs/2304.04487>. Key: `yang2023llma`. [A] *added*

[10] Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West. *Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning*. EMNLP 2023 (main conference), pp. 10932-10952; arXiv:2305.13971. 2023. <https://arxiv.org/abs/2305.13971>. Key: `geng2023gcd`. [P]

[11] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*. ICLR 2024; arXiv:2310.06770. 2024. <https://arxiv.org/abs/2310.06770>. Key: `jimenez2023swebench`. [P]

[12] Federico Cassano, Luisa Li, Akul Sethi, Noah Shinn, Abby Brennan-Jones, Jacob Ginesin, Edward Berman, George Chakhnashvili, Anton Lozhkov, Carolyn Jane Anderson, Arjun Guha. *Can It Edit? Evaluating the Ability of Large Language Models to Follow Code Editing Instructions*. Conference on Language Modeling (COLM) 2024; arXiv:2312.12450. 2024. <https://arxiv.org/abs/2312.12450>. Key: `cassano2023canitedit`. [P]

[13] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, Ofir Press. *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. NeurIPS 2024; arXiv:2405.15793. 2024. <https://arxiv.org/abs/2405.15793>. Key: `yang2024sweagent`. [P]

[14] Jiawei Guo, Ziming Li, Xueling Liu, Kaijing Ma, Tianyu Zheng, Zhouliang Yu, Ding Pan, Yizhi Li, Ruibo Liu, Yue Wang, Shuyue Guo, Xingwei Qu, Xiang Yue, Ge Zhang, Wenhu Chen, Jie Fu. *CodeEditorBench: Evaluating Code Editing Capability of Large Language Models*. arXiv:2404.03543 (preprint, v3 April 2025). 2024. <https://arxiv.org/abs/2404.03543>. Key: `guo2024codeeditorbench`. [A]

[15] Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, Loris D'Antoni. *Grammar-Aligned Decoding*. NeurIPS 2024; arXiv:2405.21047. 2024. <https://arxiv.org/abs/2405.21047>. Key: `park2024gad`. [P] *added*

[16] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, Lingming Zhang. *Agentless: Demystifying LLM-based Software Engineering Agents*. Proceedings of the ACM on Software Engineering 2(FSE):801-824 (published as 'Demystifying LLM-Based Software Engineering Agents'); arXiv:2407.01489. 2025 (arXiv 2024). <https://arxiv.org/abs/2407.01489>. Key: `xia2024agentless`. [P]

[17] Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, Yun-Nung Chen. *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models*. arXiv:2408.02442 (preprint, v3 October 2024). 2024. <https://arxiv.org/abs/2408.02442>. Key: `tam2024speakfreely`. [A]

[18] Shwetha Somasundaram, Anirudh Phukan, Apoorv Saxena. *PLD+: Accelerating LLM inference by leveraging Language Model Artifacts*. arXiv:2412.01447 (preprint). 2024. <https://arxiv.org/abs/2412.01447>. Key: `somasundaram2024pldplus`. [A]

[19] Tushar Aggarwal, Swayam Singh, Abhijeet Awasthi, Aditya Kanade, Nagarajan Natarajan. *Robust Learning of Diverse Code Edits*. ICML 2025 (to appear as 'NextCoder: Robust Adaptation of Code LMs to Diverse Code Edits', per arXiv comments); arXiv:2503.03656. 2025. <https://arxiv.org/abs/2503.03656>. Key: `aggarwal2025nextcoder`. [P]

[20] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, Liang Xiang. *Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving*. arXiv:2504.02605 (preprint). 2025. <https://arxiv.org/abs/2504.02605>. Key: `zan2025multiswebench`. [A] *added*

[21] Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, An Fu. *EfficientEdit: Accelerating Code Editing via Edit-Oriented Speculative Decoding*. ASE 2025 (per arXiv comments field); arXiv:2506.02780. 2025. <https://arxiv.org/abs/2506.02780>. Key: `wang2025efficientedit`. [P] *added*

[22] Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu. *cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree*. Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 8106-8116; arXiv:2506.15655. 2025. <https://arxiv.org/abs/2506.15655>. Key: `zhang2025cast`. [P]

[23] Evgeniy Glukhov, Michele Conti, Egor Bogomolov, Yaroslav Golubev, Alexander Bezzubov. *Diff-XYZ: A Benchmark for Evaluating Diff Understanding*. arXiv:2510.12487 (preprint, v2 November 2025). 2025. <https://arxiv.org/abs/2510.12487>. Key: `glukhov2025diffxyz`. [A]

[24] Wayne Chi, Valerie Chen, Ryan Shar, Aditya Mittal, Jenny Liang, Wei-Lin Chiang, Anastasios Nikolas Angelopoulos, Ion Stoica, Graham Neubig, Ameet Talwalkar, Chris Donahue. *EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits*. ICLR 2026 (Oral, per OpenReview record FtL9eEmU6v); arXiv:2511.04486. 2026 (arXiv 2025). <https://arxiv.org/abs/2511.04486>. Key: `chi2025editbench`. [P]

[25] Qinglin Zhu, Tianyu Chen, Shuai Lu, Lei Ji, Runcong Zhao, Murong Ma, Xiangxiang Dai, Yulan He, Lin Gui, Peng Cheng, Yeyun Gong. *Pull Requests as a Training Signal for Repo-Level Code Editing*. ICML 2026 (per arXiv comments field); arXiv:2602.07457. 2026. <https://arxiv.org/abs/2602.07457>. Key: `zhu2026cleanpr`. [P] *added*

[26] Ravi Raju, Mengmeng Ji, Shubhangi Upasani, Bo Li, Urmish Thakker. *The Limits of Long-Context Reasoning in Automated Bug Fixing*. I Can't Believe It's Not Better (ICBINB) Workshop at ICLR 2026; arXiv:2602.16069. 2026. <https://arxiv.org/abs/2602.16069>. Key: `raju2026limits`. [A] *added*

[27] Benjamin Rombaut. *Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures*. arXiv:2604.03515 (single-author preprint, v2). 2026. <https://arxiv.org/abs/2604.03515>. Key: `rombaut2026scaffold`. [A] *added*

[28] Amir M. Ebrahimi, Gopi Krishnan Rajbahadur. *Edit, But Verify: An Empirical Audit of Instructed Code-Editing Benchmarks*. arXiv:2604.05100 (preprint). 2026. <https://arxiv.org/abs/2604.05100>. Key: `ebrahimi2026editverify`. [A]

[29] Chaozheng Wang, Zezhou Yang, Shuzheng Gao, Cuiyun Gao, Zongjie Li, Yichen Li, Ting Peng, Hailiang Huang, Yuetang Deng, Michael R. Lyu. *Cascaded Code Editing: Large-Small Model Collaboration for Effective and Efficient Code Editing*. Proceedings of the ACM on Software Engineering 3(FSE):2094-2116; arXiv:2604.19201. 2026. <https://arxiv.org/abs/2604.19201>. Key: `wang2026cascaded`. [P]

[30] Yikai Zhang, Jiaxin Pei, Kenan Li, Qirui Jin, Maoquan Wang, Jin Pan, Yu Kang, Shengyu Fu, Elsie Nallipogu, Junjie Hu, Yufan Huang, Zijian Jin. *SWE-Edit: Rethinking Code Editing for Efficient SWE-Agent*. arXiv:2604.26102 (preprint, v2 26 May 2026). 2026. <https://arxiv.org/abs/2604.26102>. Key: `zhang2026sweedit`. [A]

[31] Wei Cheng, Yongchang Cao, Chen Shen, Binhua Li, Jue Chen, Yongbin Li, Wei Hu. *To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing*. Findings of the Association for Computational Linguistics: ACL 2026, pp. 29677-29694; arXiv:2604.27296. 2026. <https://arxiv.org/abs/2604.27296>. Key: `cheng2026todiff`. [P]

[32] Ishaan Bhola, Adithyan Krishnan, Sravanth Kurmala, Mukunda NS. *Code Isn't Memory: A Structural Codebase Index Inside a Coding Agent*. arXiv:2606.22417 (preprint). 2026. <https://arxiv.org/abs/2606.22417>. Key: `bhola2026codeisntmemory`. [A]

[33] Andrej Andrejev. *Diffs vs. Whole Files: An Empirical Comparison of Iterative Edit-Based and Direct Generation for Flutter/Dart Code Models*. arXiv:2609.05779 (single-author preprint). 2026. <https://arxiv.org/abs/2609.05779>. Key: `andrejev2026diffswhole`. [A]

[34] Asaad Althoubi. *Look Before You Leap: Pre-Action Verification for LLM Agents*. arXiv:2609.11957 (single-author preprint; arXiv lists v1 as submitted 9 August 2026). 2026. <https://arxiv.org/abs/2609.11957>. Key: `althoubi2026lookbefore`. [A]

[35] Paul Gauthier (Aider project); the page itself carries no byline. *Unified diffs make GPT-4 Turbo 3X less lazy*. aider.chat documentation (practitioner benchmark, not peer reviewed). n.d. (benchmarks gpt-4-1106-preview). <https://aider.chat/docs/unified-diffs.html>. Key: `aider_udiff`. [V]

[36] Paul Gauthier (Aider project); the page itself carries no byline. *GPT code editing benchmarks*. aider.chat documentation (practitioner benchmark, not peer reviewed). n.d. (benchmarks GPT-3.5 and GPT-4 0613 models). <https://aider.chat/docs/benchmarks.html>. Key: `aider_benchmarks`. [V]

[37] Aider project. *Edit formats*. aider.chat documentation. n.d. <https://aider.chat/docs/more/edit-formats.html>. Key: `aider_editformats`. [D]

[38] Aider project (no byline on the page). *LLMs are bad at returning code in JSON*. aider.chat blog, 14 August 2024 (practitioner benchmark, not peer reviewed). 2024. <https://aider.chat/2024/08/14/code-in-json.html>. Key: `aider_codejson`. [V]

[39] Paul Gauthier (Aider project). *Code editing leaderboard*. aider.chat leaderboards (practitioner benchmark, not peer reviewed). 2025 (page states 'last updated April 12, 2025'). <https://aider.chat/docs/leaderboards/edit.html>. Key: `aider_leaderboard`. [V]

[40] Aider-AI contributors. *aider/coders/editblock_coder.py (source code)*. GitHub repository Aider-AI/aider. 2025 (main branch; last commit to this file 24 June 2025, 7c9cff2). <https://raw.githubusercontent.com/Aider-AI/aider/main/aider/coders/editblock_coder.py>. Key: `aider_editblock_src`. [D]

[41] OpenAI. *Apply Patch*. OpenAI API documentation. n.d. <https://developers.openai.com/api/docs/guides/tools-apply-patch>. Key: `openai_applypatch`. [D]

[42] Noah MacCallum, Julian Lee. *GPT-4.1 Prompting Guide*. OpenAI Cookbook, 14 April 2025. 2025. <https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide>. Key: `openai_gpt41guide`. [D]

[43] OpenAI Codex contributors. *openai/codex: codex-rs/apply-patch/src/seek_sequence.rs and parser.rs (source code)*. GitHub repository openai/codex. 2026 (main branch; last commit to seek_sequence.rs 10 August 2026, 21aa552). <https://raw.githubusercontent.com/openai/codex/main/codex-rs/apply-patch/src/seek_sequence.rs>. Key: `codex_applypatch_src`. [D]

[44] Anthropic. *Text editor tool*. Claude Platform Docs. n.d. (documents tool versions up to text_editor_20250728). <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>. Key: `anthropic_texteditor`. [D]

[45] Aman Sanger. *Editing Files at 1000 Tokens per Second*. Cursor blog (research), 14 May 2024; vendor report, not peer reviewed. 2024. <https://cursor.com/blog/instant-apply>. Key: `sanger2024instantapply`. [V]

[46] Fireworks AI (no byline). *How Cursor built Fast Apply using the Speculative Decoding API*. Fireworks AI blog, 23 June 2024; vendor report, not peer reviewed. 2024. <https://fireworks.ai/blog/cursor>. Key: `fireworks2024cursor`. [V]

[47] OpenAI. *Predicted Outputs*. OpenAI API documentation. n.d. <https://developers.openai.com/api/docs/guides/predicted-outputs>. Key: `openai_predicted`. [D]

[48] Can Boluk (Can Bölük). *We improved 15 LLMs at coding in one afternoon. Only the harness changed.*. Stencil blog, 12 February 2026; practitioner report, not peer reviewed. 2026. <https://stencil.so/blog/the-harness-problem>. Key: `boluk2026harness`. [V]

[49] llama.cpp contributors (ggml-org). *Speculative Decoding (docs/speculative.md)*. GitHub repository ggml-org/llama.cpp. 2026 (master branch; last commit to this file 27 August 2026, 2bb9bdd). <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md>. Key: `llamacpp_speculative`. [D]

[50] llama.cpp contributors (ggml-org). *GBNF Guide (grammars/README.md)*. GitHub repository ggml-org/llama.cpp. n.d. (master branch). <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/README.md>. Key: `llamacpp_gbnf`. [D]

[51] Bram Cohen. *Patience Diff Advantages*. LiveJournal blog post, 30 March 2010 (primary description by the algorithm's author). 2010. <https://bramcohen.livejournal.com/73318.html>. Key: `cohen2010patience`. [V]

[52] Eclipse JGit contributors (copyright Google Inc. and others). *org.eclipse.jgit.diff.HistogramDiff (source code and Javadoc)*. GitHub repository eclipse-jgit/jgit. 2010. <https://raw.githubusercontent.com/eclipse-jgit/jgit/master/org.eclipse.jgit/src/org/eclipse/jgit/diff/HistogramDiff.java>. Key: `jgit_histogram`. [D]

[53] Git project. *git-diff Documentation*. git-scm.com reference manual. n.d. (page states 'last updated in 2.56.0'). <https://git-scm.com/docs/git-diff>. Key: `git_diff_docs`. [D]

[54] Git project. *git-apply Documentation*. git-scm.com reference manual. n.d. (page states 'last updated in 2.55.0'). <https://git-scm.com/docs/git-apply>. Key: `git_apply_docs`. [D]

[55] David MacKenzie, Paul Eggert, Richard Stallman. *Comparing and Merging Files (GNU Diffutils manual): 'Helping patch Find Inexact Matches' and 'Detailed Description of Unified Format'*. GNU Diffutils manual, Free Software Foundation; doc/diffutils.texi read from a Git mirror. 2017 (Texinfo source snapshot of 5 May 2017). <https://raw.githubusercontent.com/gitGNU/gnu_diffutils/master/doc/diffutils.texi>. Key: `gnu_diffutils_manual`. [D]

[56] Neil Fraser. *Fuzzy Patch*. Author's technical note (neil.fraser.name), May 2006; design document for diff-match-patch. 2006. <https://neil.fraser.name/writing/patch/>. Key: `fraser2006fuzzypatch`. [V]

[57] Google (Neil Fraser) and contributors. *diff-match-patch API wiki*. GitHub wiki google/diff-match-patch. n.d. <https://github.com/google/diff-match-patch/wiki/API>. Key: `dmp_api`. [D]

[58] Google (Neil Fraser) and contributors. *diff-match-patch C++ source (cpp/diff_match_patch.cpp)*. GitHub repository google/diff-match-patch. n.d. (master branch). <https://raw.githubusercontent.com/google/diff-match-patch/master/cpp/diff_match_patch.cpp>. Key: `dmp_src`. [D]

[59] Python Software Foundation. *difflib - Helpers for computing deltas*. Python standard library documentation. n.d. (Python 3.14.7 documentation). <https://docs.python.org/3/library/difflib.html>. Key: `py_difflib`. [D] *added*

[60] The Open Group; IEEE. *rename - rename file (The Open Group Base Specifications Issue 7, 2018 edition, IEEE Std 1003.1-2017)*. POSIX.1-2017 specification. 2018. <https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html>. Key: `posix_rename`. [D]

[61] The Open Group; IEEE. *rename - rename file (The Open Group Base Specifications Issue 8, IEEE Std 1003.1-2024)*. POSIX.1-2024 specification. 2024. <https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html>. Key: `posix_rename_2024`. [D] *added*

[62] Linux man-pages project. *rename(2) - Linux manual page*. man7.org. 2026 (man-pages 6.19, 8 February 2026). <https://man7.org/linux/man-pages/man2/rename.2.html>. Key: `linux_rename_man`. [D]

[63] Linux man-pages project. *fsync(2) - Linux manual page*. man7.org. 2026 (man-pages 6.19, 8 February 2026). <https://man7.org/linux/man-pages/man2/fsync.2.html>. Key: `linux_fsync_man`. [D]

[64] Microsoft. *ReplaceFileW function (winbase.h)*. Microsoft Learn, Win32 API reference. 2025 (page metadata: updated 1 July 2025). <https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-replacefilew>. Key: `ms_replacefilew`. [D]

[65] Microsoft. *MoveFileExW function (winbase.h)*. Microsoft Learn, Win32 API reference. 2025 (page metadata: updated 1 July 2025). <https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw>. Key: `ms_movefileexw`. [D]

[66] Microsoft. *Alternatives to using Transactional NTFS*. Microsoft Learn, Win32 File I/O documentation. 2025 (page metadata: updated 12 March 2025). <https://learn.microsoft.com/en-us/windows/win32/fileio/deprecation-of-txf>. Key: `ms_txf_alternatives`. [D]

[67] Microsoft. *FILE_RENAME_INFORMATION structure (ntifs.h)*. Microsoft Learn, Windows driver reference. 2024 (page metadata: updated 22 February 2024). <https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information>. Key: `ms_file_rename_info`. [D]

[68] Microsoft. *FILE_RENAME_INFO structure (winbase.h)*. Microsoft Learn, Win32 API reference. 2026 (page metadata: updated 15 May 2026). <https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_rename_info>. Key: `ms_file_rename_info_win32`. [D] *added*

[69] Microsoft. *FlushFileBuffers function (fileapi.h)*. Microsoft Learn, Win32 API reference. 2025 (page metadata: updated 1 July 2025). <https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers>. Key: `ms_flushfilebuffers`. [D]

[70] SQLite project. *Atomic Commit In SQLite*. sqlite.org documentation. 2026 (page states 'last updated on 2026-04-21'). <https://www.sqlite.org/atomiccommit.html>. Key: `sqlite_atomiccommit`. [D]

[71] Tree-sitter project. *Tree-sitter: Introduction*. tree-sitter.github.io documentation. n.d. <https://tree-sitter.github.io/tree-sitter/>. Key: `treesitter_intro`. [D]

[72] Tree-sitter project. *Tree-sitter: Advanced Parsing*. tree-sitter.github.io documentation. n.d. <https://tree-sitter.github.io/tree-sitter/using-parsers/3-advanced-parsing.html>. Key: `treesitter_advanced`. [D]

[73] Tree-sitter contributors. *tree_sitter/api.h (public C API header)*. GitHub repository tree-sitter/tree-sitter. n.d. (master branch). <https://raw.githubusercontent.com/tree-sitter/tree-sitter/master/lib/include/tree_sitter/api.h>. Key: `treesitter_api`. [D]

[74] Tree-sitter contributors (package metadata names Max Brunsfeld and Amaan Qureshi). *tree-sitter-c: C grammar for tree-sitter*. GitHub repository tree-sitter/tree-sitter-c. n.d. (master branch; PyPI release 0.24.2 used for the local measurement). <https://raw.githubusercontent.com/tree-sitter/tree-sitter-c/master/README.md>. Key: `treesitter_c`. [D] *added*

[75] hnswlib contributors (nmslib). *Hnswlib - fast approximate nearest neighbor search (README)*. GitHub repository nmslib/hnswlib. n.d. (master branch). <https://raw.githubusercontent.com/nmslib/hnswlib/master/README.md>. Key: `hnswlib_readme`. [D]

[76] LLVM Project. *Clang command line argument reference*. clang.llvm.org documentation. n.d. <https://clang.llvm.org/docs/ClangCommandLineReference.html>. Key: `clang_cli`. [D]

[77] Microsoft. */Zs (Syntax Check Only)*. Microsoft Learn, MSVC compiler reference. 2021 (page metadata: updated 3 August 2021). <https://learn.microsoft.com/en-us/cpp/build/reference/zs-syntax-check-only>. Key: `msvc_zs`. [D]

[78] William Franz Lamberti. *Line-Anchored Feedback Cuts Token Costs and Improves Correctness in AI Code Editing*. arXiv:2607.12713 (single-author preprint, v1 14 July 2026). 2026. <https://arxiv.org/abs/2607.12713>. Key: `lamberti2026lineanchored`. [A] *added in second pass*
