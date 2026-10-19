# Cache-conscious parallelism and deterministic ordering

Verified research note for programmersRAG (features 7 and 9: multithreading with about 16 KB of work per thread so that the working set can stay in the L1 data cache, and a strict guarantee that block A is handled before block B). It also covers the int8 SIMD scoring kernel used by the retrieval stage, because its speed depends on cache residency.

- Verification date: 2026-09-29.
- Status: second-pass, adversarial check of a first-pass dossier. Every source in the reference list was opened by the checker and its title, authors, year and venue were compared with the document itself and, where a DOI exists, with the Crossref record. Quotations were re-read in the source text. Every formula was re-derived or checked numerically.
- Labels used throughout: **[E]** established result, **[R]** recent result (2024-2026), **[C]** contested, **[P]** our proposal or our own derivation (not a published result), **[M]** our own measurement on the target machine.
- Target machine (confirmed by query, see Appendix B): Windows 11, Intel Core i9-12950HX, 8 P-cores with Hyper-Threading plus 8 E-cores, 24 logical processors.
- Evaluation context: all experiments planned for the paper run on a demo repository that is a clone of `microsoft/mimalloc` (59 C source and header files, 1,273,275 bytes). This corpus is small, which limits what can be concluded about speed-up (see Section 6).

## 1. Summary

What the literature supports:

1. Cache blocking is an established technique, and Intel's current guidance gives a concrete size rule: target one half of the first-level cache, or one quarter to one half of a cache shared by two logical processors [1, 2]. For the target CPU this gives 8-16 KB on E-cores (32 KB L1d) and 12-24 KB on P-cores (48 KB L1d), so 16 KB is the largest block that satisfies the rule on both core types. **[E]** for the rule, **[P]** for the application to this CPU.
2. The benefit of L1-sized blocks is conditional. It exists when each block is touched several times. For a single streaming pass the literature says block size is not critical [9, 14], and classical analysis shows that simply "filling the cache" is wrong when accesses conflict [6]. **[C]**
3. User-space code cannot place data in L1. It can only bound the working set, keep accesses sequential, align data, avoid false sharing, and then measure. The phrase "force the computer to use L1 cache" must be written as "size the per-thread working set so that it can reside in L1d". **[E]**
4. Strict ordering in a parallel pipeline is a solved pattern: sequence numbers at dispatch, out-of-order execution, in-order commit. The same idea appears in processor reorder buffers [29], ticket locks [30], the Disruptor [31], oneTBB `serial_in_order` filters [37, 39], Cilk-P pipelines [32] and DuckDB batch indexes [17]. **[E]**
5. "Block A reaches the CPU before block B" can only be guaranteed for events that the program itself defines and observes (claim, optional start gate, commit). Physical execution order of arbitrary instructions cannot be guaranteed on a pre-emptive operating system [28, 45]. **[E]** for the principle, **[P]** for the wording we propose.
6. On a hybrid CPU, work must be handed out dynamically or in proportion to measured core speed. Equal static partitioning makes fast cores wait for slow ones [23, 24]. Intel and Microsoft both advise against hard affinity [4, 44, 46]. **[E]/[R]**
7. The relative speed of E-cores is workload dependent and must be measured. The value 0.6 used in the first dossier is a nominal clock-frequency ratio taken from one preprint [25], not a measured throughput. **Correction.**

What had to be corrected in the first dossier is listed in Appendix A (20 items). No dossier source failed verification; two unofficial mirror pages were replaced by the primary Intel manual, and ten sources were added (references 2, 5, 10, 16, 23, 24, 32, 36, 42 and 47).

## 2. Established results

### 2.1 Cache blocking and block size

**E1. Blocking reuses data while it is in a fast cache level.** Lam, Rothberg and Wolf: "blocked algorithms operate on submatrices or blocks, so that data loaded into the faster levels of the memory hierarchy are reused" [6, abstract]. Intel describes strip-mining (one dimension) and loop blocking (two or more dimensions), and states that "Each chunk should be small enough to fit all the data for a given computation into the cache, thereby maximizing data reuse" [1, Section 5.5.3].

**E2. Using the whole cache is not optimal when accesses conflict.** "The conventional wisdom of trying to use the entire cache, or even a fixed fraction of the cache, is incorrect. If a fixed block size is used for a given cache size, the block size that minimizes the expected number of cache misses is very small" [6, abstract]. In their direct-mapped example, choosing the block to use the entire cache gives a miss rate "more than 10 times the optimal", and the optimal fraction falls "from 15% for a 0.25K word cache to 3% for a 16K word cache" [6, Section 3.3]. With 4-way associativity "the average miss rates remain relatively flat as the block size increases" [6, Section 5.1]. Scope: the analysis is for two-dimensional matrix blocks accessed with non-unit stride in caches with one-word lines. It does not apply directly to a contiguous one-dimensional buffer in an 8-way or 12-way cache. What carries over is the warning that self-interference, not capacity alone, decides the miss rate.

**E3. Intel's sizing rules.** User/Source Coding Rule 8: "Optimize data structures either to fit in one-half of the first-level cache or in the second-level cache". The text continues: "Optimizing for one-half of the first-level cache will bring the greatest performance benefit in terms of cycle-cost per data access. If one-half of the first-level cache is too small to be practical, optimize for the second-level cache. Optimizing for a point in between (for example, for the entire first-level cache) will likely not bring a substantial improvement over optimizing for the second-level cache" [1, Section 3.6.10]. User/Source Coding Rule 25: "Use cache blocking to improve locality of data access. Target one quarter to one half of the cache size when targeting Intel processors supporting HT Technology or target a block size that allow all the logical processors serviced by a cache to share that cache simultaneously" [1, Section 11.6.1]. The upper limit is obtained "by placing an upper limit of the block size as the total size of the cache divided by the number of logical processors serviced by that cache", and the quarter-to-half window exists because "some cache lines are needed to access data that are not part of the source or destination buffers used in cache blocking" [1, Section 11.6.1]. The same three rules appear unchanged in revision 050 [2].

**E4. Cache geometry of the target CPU family.** "P Cores 1st level cache ... size is 48KB for data and 32KB for instructions. The 1st level cache is an 12-way associative cache." "E Cores 1st level cache ... size is 32KB for data and 64KB for instructions. The 1st level cache is an 8-way associative cache." The P-core second-level cache is 1.25 MB, and four E-cores share a 2 MB second-level cache. First-level caches are not shared between physical cores [3, Sections 2.4.1 and 2.4.2]. The datasheet covers the HX processor line. Its revision history records that revision 010 (February 2023) "Corrected L1 Data and Instruction Cache size", so earlier revisions must not be cited for these numbers. For Gracemont (E-core), Intel states that "The load-to-use latency for loads is typically four cycles" and "The L2 cache delivers 64 bytes of data per cycle at a latency of 17 cycles, and that bandwidth is shared among four cores" [1, Section 4.1.7].

**E5. SMT siblings share the L1 data cache.** "Most resources in a physical processor are fully shared to improve the dynamic utilization of the resource, including caches and all the execution units" [1, Section 2.7.1.3]. Drepper explains a two-hyper-thread measurement on a Pentium 4 by noting that "Since the hyper-threads share all the resources except the registers each thread has only half the cache and bandwidth available" [9, Section 3.5.1, Figure 3.25]. Sharing is competitive, not a fixed partition, so "half" is a planning figure. For the last-level cache Drepper gives the practical rule of dividing the cache size by the number of CPUs that share it, which is "a safe lower limit" [9, Section 6.2.3]; we apply the same rule to L1d.

**E6. Measured working-set steps.** In Drepper's sequential list walk on a Pentium 4 with "a 16kB L1d and 1MB L2" there are "three distinct levels": up to 2^14 bytes, 2^15 to 2^20 bytes, and 2^21 bytes and above. L1d hits take "around 4 cycles"; once the working set exceeds L1d "only about 9 cycles are required" instead of the 14-cycle L2 access time, because of prefetching; and "Only with effective prefetching is it possible for the processor to keep the access times as low as 9 cycles" beyond L2 [9, Section 3.3.2]. Edges are not sharp "because the caches are used by other parts of the system as well and so the cache is not exclusively available for the program data". Scope: 2007 hardware, cycles per 8-byte list element. The numbers are illustrative for the target CPU, not predictive.

**E7. Sequential access is the best case.** Scanning N contiguous elements costs at most ceil(N/B)+1 block transfers in the ideal-cache model [8, Theorem 1]. Intel: "Try to arrange data structures so they permit sequential access", because "the automatic hardware prefetcher can prefetch data", and "The prefetcher can recognize up to eight concurrent streams" [1, Section 3.6.5]. Gracemont has "three independent L1 prefetchers" (next-line on L1 load misses, instruction-pointer stride, next-page), and the stride prefetcher "works in the linear address space to cross page boundaries" [1, Section 4.1.7]. The statement that L2 prefetchers "prefetch only inside page boundary (4 KBytes)" is made in the manual for the older Intel Core microarchitecture [1, Section 3.7.3] and must not be quoted as a property of the target cores.

**E8. Cache-oblivious algorithms avoid tuning.** "no variables dependent on hardware parameters, such as cache size and cache-line length, need to be tuned to achieve optimality", analysed in an "ideal-cache" model under the tall-cache assumption Z = Omega(L^2); "an optimal cache-oblivious algorithm designed for two levels of memory is also optimal for multiple levels" [7, abstract]. The ideal cache "assumes ... page replacement is optimal, and the cache is fully associative", and LRU on a cache of size M makes at most 2T transfers if optimal replacement on size M/2 makes T [8, Section 2.2 and Lemma 1]. A fixed 16 KB block is a cache-aware choice. It is justified on this machine and must be re-derived on any other.

**E9. What user space cannot do.** "PREFETCH merely provides a hint to the hardware" and "This data access type and the cache level are specified as a hint" [1, Section 9.3.1]. "As for the cache replacement there is not much a programmer can do", and for physically addressed caches conflict avoidance "is impossible for user-level processes" [9, Section 3.3.5]. Windows assigns time slices "in a round-robin fashion to all threads with the highest priority", and "If a higher-priority thread becomes available to run, the system ceases to execute the lower-priority thread (without allowing it to finish using its time slice)" [45]. Affinity only restricts where a thread may run, and "In most cases, it is better to let the system select an available processor" [46]. CPU Sets "declare application affinity in a 'soft' manner" [44]. What user space can do is read the real cache parameters (`CACHE_RELATIONSHIP`: Level, Associativity, LineSize, CacheSize, and the group mask of the processors that share the cache) [43].

**E10. Cost of losing the cache.** For a cache of C blocks with a "simple" replacement policy (the paper names LRU and direct-mapped caches as examples), an execution that starts from one cache state incurs at most C more misses than the same execution started from another state [10, Lemma 2; paraphrased, because the symbols are damaged in text extraction]. In work stealing of a series-parallel computation, the number of drifted nodes is at most twice the number of steals [10, Lemma 11]. Consequence for us: one pre-emption or migration costs at most C/L line refills of L1d, which is 512 lines on an E-core and 768 on a P-core.

**E11. False sharing.** "False sharing occurs when processors in a shared-memory parallel system make references to different data objects within the same coherence block (cache line or page), thereby inducing 'unnecessary' coherence operations" [11]. Intel: "Beware of false sharing within a cache line (64 bytes)" [1, Rule 7] and "Place each synchronization variable alone, separated by 128 bytes or in a separate cache line" [1, Section 11.3.1]. C++17 provides `std::hardware_destructive_interference_size`, the "Minimum offset between two objects to avoid false sharing" [40], proposed in P0154R1 as "a number that's suitable as an offset between two objects to likely avoid false-sharing" [12]. The Microsoft STL defines both interference constants as 64 on x86, x64 and ARM64 [47]; the installed MSVC 14.51.36231 header has the same definition **[M]**.

### 2.2 Cache-sized work units in database engines

**E12. Vectorised execution.** MonetDB/X100 passes "small, cache-resident, vertical data fragments called vectors". "X100 uses a default vector size of 1024 ... Preferably, all vectors together should comfortably fit the CPU cache size". In TPC-H Query 1 "the optimal vector size seems to be 1000, but all values between 128 and 8K actually work well. Performance starts to deteriorate when intermediate results do not fit in the cache anymore", which happens when the vectors exceed "the 320KB combined L1 and L2 cache of the AthlonMP" [13, Section 5.1.1]. Kersten et al. confirm: "small (<64) and large vector sizes (>64 K) decrease performance significantly ... Large vectors do not fit into the CPU caches and therefore cause cache misses" [15, Section 4.3]. Note that the X100 criterion is fit in L1 plus L2, not L1 alone.

**E13. Morsel-driven parallelism separates the scheduling unit from the cache unit.** "a morsel size of about 100,000 tuples yields good tradeoff"; "the dispatcher is implemented as a lock-free data structure only. The dispatcher's code is then executed by the work-requesting query evaluation thread itself" [14, Section 3]. "there is no performance penalty if a morsel does not fit into cache ... the morsel size is not very critical for performance, it only needs to be large enough to amortize scheduling overhead while providing good response times" [14, Section 3.3]. With an unrelated process occupying one core, "query performance dropped by 36.8% with static approach, but only 4.7% with dynamic morsel assignment" [14, Section 5.4]. The abstract reports "an average speedup of over 30 with 32 cores". Two caveats for our use: the morsel system "permanently bind[s]" each worker to a hardware thread, which is contrary to hybrid-CPU guidance (E27), and the interference experiment is about a busy core, not about heterogeneous cores.

**E14. Task granularity by time, not by size.** Wagner, Kohn and Neumann replace fixed morsel sizes by a target task duration: "When a worker picks a task, the task tries to schedule morsels that exhaust this target duration", and "we chose 2ms to balance responsiveness and scheduling overhead" [16, Sections 3.1 and 4]. By our arithmetic, a 16 KB text block processed at hundreds of megabytes per second lasts tens of microseconds, which is far below this target **[P]**.

**E15. SIMD gains depend on cache residency.** "With a growing working set, gains from SIMD diminish and the execution costs are dominated by memory latency. SIMD is only beneficial when all data fits into the cache" [15, Section 5.2].

### 2.3 Ordering, commit and determinism

**E16. Observable events.** "if a system is to meet a specification correctly, then that specification must be given in terms of events observable within the system. If the specification is in terms of physical time, then the system must contain real clocks" and "The relation 'happened before' is therefore only a partial ordering of the events in the system" [28].

**E17. Reorder buffer.** "Instructions are allowed to finish out of order, but a special buffer called the reorder buffer is used to reorder them before they modify the process state." It "is a circular buffer with head and tail pointers"; an entry is allocated at issue, and "When the entry at the head of the reorder buffer contains valid results (its instruction has finished), then its exceptions are checked. If there are none, the results are written into the registers." Forcing in-order completion costs "at least 16 percent"; three of the other methods "result in as little as a 3 percent performance loss" [29].

**E18. Ticket lock.** "A ticket lock consists of two counters, one containing the number of requests to acquire the lock, and the other the number of times the lock has been released. A processor acquires the lock by performing a fetch-and-increment operation on the request counter and waiting until the result (its ticket) is equal to the value of the release counter." This "ensure[s] FIFO service" and "eliminates the possibility of starvation" [30, Section 2.2]. Caveat from the same paper: the FIFO guarantee "is likely to waste CPU resources if spinning processes may be preempted" [30, Section 4.3].

**E19. Sequence numbers as the only coordination mechanism.** In the Disruptor, "Producers claim the next slot in sequence", using "a simple counter in the case of only one producer or an atomic counter updated using CAS operations in the case of multiple producers", and commit by waiting until the cursor equals `claimedSequence - 1`. "In the case of having only one producer ... no locks or CAS operations are required" [31, Section 4.3]. The paper reports mean latency "3 orders of magnitude lower" and "approximately 8 times more throughput" than an equivalent queue-based design. These figures are vendor measurements in Java against `ArrayBlockingQueue` and are not peer reviewed.

**E20. Ordered pipelines in oneTBB.** "A serial_in_order filter processes items one at a time. The order in which items are processed is implicitly set by the first serial_in_order filter and respected by all other such filters in the pipeline" [39]. "if an item arrives at MyOutputFunc out of the order established by MyInputFunc, the pipeline automatically delays invoking MyOutputFunc::operator() on the item until its predecessors are processed", and "Once this limit is reached, the pipeline never creates a new token at the input filter until another token is destroyed at the output filter" [37]. "the throughput of a pipeline is limited by the throughput of the slowest sequential filter"; the text-processing example is "unlikely to see a speedup much more than 2"; "A good guideline is to try for a large window size that still fits in cache" [38].

**E21. Provably efficient ordered pipelines with a bounded window.** Cilk-P "inhibits the start of iteration i + K until iteration i has completed, where K = Theta(P) is the throttling limit", runs in expected time T_P <= T_1/P + O(T_inf + lg P), and uses bounded space. "a throttling limit of 4P, where P is the number of cores, seems to work well in general". For uniform pipelines a window K = aP gives T_P <= (1 + c/a) T_1/P + c T_inf (Theorem 12). For non-uniform pipelines a throttled scheduler can lose speed-up (Theorem 13) [32]. For a three-stage serial-parallel-serial pipeline with unit serial stages and parallel stage cost r the paper gives work n(r+2) and span n+r, hence parallelism n(r+2)/(n+r).

**E22. Deterministic multithreading.** Kendo "enforces a deterministic interleaving of lock acquisitions" with "a geometric mean overhead of only 16% when running on 4 processors" [33]. CoreDet: "As long as transactions commit according to the serial scheduling order, the resulting parallel execution is exactly equivalent to the (deterministic) serial execution" [34, Section 2.2]. Dthreads: "waiting for the token guarantees a global order for all operations in the serial phase", and it "eliminates cache-line false sharing" [35, Section 4.2.1]. Blelloch et al. define internal determinism and list "a strict functional style (no shared state among concurrent operations)" as one way to obtain it; their deterministic algorithms "achieve good speedup and good performance even relative to prior nondeterministic solutions" on 32 cores [36].

**E23. Order-preserving parallelism in a production engine.** DuckDB pull request 3700: "pipeline sources emit a 'batch index'"; the original order is restored "without an additional sorting step - we can scan all the data in the result ordered by batch index". The reported example (100 million rows) takes 0.53 s single-threaded and 0.07 s with 8 threads [17]. This is an engineering source, not a peer-reviewed one.

### 2.4 Speed-up laws and asymmetric cores

**E24. Amdahl and Gustafson.** Amdahl's 1967 paper contains no equation (the reprint editors note that "it has no equations and only a single figure"); it argues that sequential housekeeping "would then place an upper limit on throughput of five to seven times the sequential processing rate" [26]. Gustafson states the law as Speedup = 1/(s + p/N) and the scaled speed-up N + (1 - N)s' [27].

**E25. Asymmetric multicore.** Hill and Marty: "Asymmetric multicore chips can offer potential speedups that are much greater than symmetric multicore chips (and never worse)", and research must deal with "the scheduling and overhead challenges that Amdahl's model doesn't capture" [22].

**E26. Loop scheduling on asymmetric processors.** "conventional loop-scheduling OpenMP approaches are unable to efficiently cope with the load imbalance that naturally stems from the different performance delivered by big and small cores". Uneven iteration distribution improves on `static` by up to 56% and on `dynamic` by up to 16.8%. The dynamic schedule takes chunks with a fetch-and-add on a shared counter, and "the overhead of assigning iterations dynamically can be substantial" [23]. Platforms: ARM big.LITTLE and an emulated x86 asymmetric system, not Alder Lake.

**E27. Vendor guidance for hybrid CPUs.** "In general, software should avoid setting affinity"; if affinity must be used, "leverage CPUSets API"; "Thread creation based on processor count may not always be optimal. Some cores may be less performant than others, or even idled/parked"; "software can choose to replace spin-waits with lighter weight spins containing UMWAIT, TPAUSE, or PAUSE" [4, Sections 6.2, 6.4, 6.5].

### 2.5 Int8 SIMD kernels

**E28. VPDPBUSD.** The instruction "Multiplies the individual unsigned bytes of the first source operand by the corresponding signed bytes of the second source operand, producing intermediate signed word results. The word results are then summed and accumulated in the destination dword element size operand." The VEX-encoded 128-bit and 256-bit forms carry the CPUID feature flag AVX_VNNI [5, Vol. 2C, VPDPBUSD]. It "replaces an Intel AVX-512 three instruction sequence that accomplishes the same functionality with higher accuracy since Intel AVX-512 saturates the 16-bit intermediate results: VPMADDUBSW + VPMADDWD + VPADDD" [1, Section 8.2.1]. For PMADDUBSW "Each adjacent pair of signed words is added and the saturated result is packed to the destination operand" [5, Vol. 2B, PMADDUBSW]. The Alder Lake datasheet lists "Intel AVX2 Vector Neural Network Instructions (Intel AVX2 VNNI)" [3], and the Gracemont feature list includes "VEX-based VNNI ISA extension" [1, Section 4.1].

**E29. Signed inputs.** "VPMADDUBSW and VPDPBUSD only support the combination of an unsigned value in the first parameter and a signed value in the second parameter ... signed activations (in the first parameter) require some manipulations"; Intel adds 128 to the activations and compensates "by subtracting 128*(sum of all the weights ...)" [1, Chapter 8]. Signed-by-signed forms (VPDPBSSD and relatives) exist under the separate flag AVX_VNNI_INT8 [5], which Intel documents as new instruction support of the later Crestmont core [2, Section 4.1.8.3]. It is not in the Alder Lake datasheet feature list [3].

**E30. E-core vector width.** On Gracemont "most 256-bit uops are further subdivided into two independent 128-bit uops at allocation" [1, Section 4.1.8]. Gracemont offers 256-bit AVX and AVX2, not AVX-512 [1, Section 4.1].

## 3. Recent work (2024-2026)

**R1. Dynamic work partitioning for LLM inference on hybrid CPUs (2024).** Luo, Liu and Shen (Intel) observe that llama.cpp-style fixed equal partitions make "high-performance cores [wait] for low-performance cores". They keep a per-core, per-instruction-set performance ratio, update it after each kernel with a low-pass filter, and give each thread a share proportional to its ratio. On a Core i9-12900K (same core types as the target CPU) an int8 GEMM kernel using AVX-VNNI improves by 85% over balanced OpenMP dispatch, and by 65% on a Core Ultra 125H. End-to-end prefill latency improves by 20-30% and decode by 9-22%. The measured P-core ratio for the VNNI kernel settles "between 3 and 3.5" in the paper's relative units and changes between prefill and decode [24]. Preprint; the PDF carries a NeurIPS 2024 footer but no acceptance statement. Their thread pool binds each thread to a physical core.

**R2. Hybrid cores and caches for parallel applications (2026).** On an Alder Lake system (8 P-cores with Hyper-Threading, 8 E-cores), with OpenMP thread binding enabled, miniVite scales up to 8 threads and gets worse from 10 to 16 threads; NAS-FT shows a similar effect up to 12 threads. NAS BT, LU and SP are unaffected. With binding disabled the degradation disappears for the two affected workloads. The abstract concludes that "parallel applications with work imbalance ... scale better across hybrid cores when thread affinity is disabled" and that "hybrid cache architecture has very little impact on parallel shared data applications except for some workloads with locks" [25]. Limits: single-institution preprint, Linux, OpenMP, five workloads. The statement that E-cores are "only 60% the speed of P-cores" rests on a ratio of maximum clock frequencies (3900 MHz to 6500 MHz as printed in the paper), not on measured throughput.

**R3. Vector search is frequently memory-bound (2025).** PDX: "it processes each float in 0.1 CPU cycles when the data fits in L1d" and "When data size (X-axis) exceeds L2, all kernels become data access-bound"; "A block size of 64 achieves the highest speedups across all architectures" [19]. Bang for the Buck: vector search "is memory-bound in many scenarios"; "when the collection fits in the L2 cache, SPR achieves 10% more performance than Zen4 ... as soon as data spills to L3, the performance of SPR degrades"; "the only available instruction to do so (i.e., VPDPBUSD) expects one vector to be signed and the other to be unsigned" [20]. VSAG "reduces L3 cache misses with pre-fetching and cache-friendly vector organization" and uses scalar quantisation [21]. All three are peer reviewed (SIGMOD 2025, DaMoN 2025, VLDB 2025).

**R4. Morsel-driven parallelism remains the reference design (2025).** "First is the state of the art morsel-driven parallelism", and different choices of morsel granularity form "a design space of morsel dispatching policies" [18] (PVLDB 18(11)).

**R5. Current vendor documents.** The Intel optimization manual revision 050 (April 2024) is the latest revision served by Intel on the verification date; it keeps Rules 7, 8 and 25 verbatim but replaces the Gracemont section by Crestmont [2]. The Gracemont statements therefore must be cited from revision 049 [1]. The Alder Lake datasheet is at revision 011 (May 2025) [3]. The Intel SDM Volume 2 is at order number 325383-093US (September 2026) [5].

**Coverage limit.** The web-search tool budget for the session was exhausted, and the arXiv API began rate limiting during the scan. The 2024-2026 scan therefore relied on Crossref queries, a small number of arXiv API queries and direct retrieval of known papers. No peer-reviewed 2024-2026 paper that measures L1-sized task granularity for text or code indexing was found. This is a gap in the search, not evidence that no such paper exists.

## 4. Formulas

Each formula lists its source and the check that was performed. Symbols: C cache size in bytes, A associativity, L line size in bytes, S number of sets, h number of logical processors sharing the cache, B block size in bytes, N number of blocks (or elements, where stated), P number of workers.

### F1. Amdahl's law (fixed problem size)

$$
S(N) = \frac{s + p}{s + p/N} = \frac{1}{s + p/N}, \qquad s + p = 1
$$

Source: [27] states the law in this form; [26] contains the argument but no equation. s is the serial time fraction, p the parallel fraction, N the processor count. Maximum speed-up is 1/s. Check: quoted form compared with the source text.

### F2. Gustafson-Barsis scaled speed-up

$$
S_{\text{scaled}}(N) = \frac{s' + p' N}{s' + p'} = N + (1 - N)\, s', \qquad s' + p' = 1
$$

Source: [27]. s' and p' are measured on the parallel system. Check: s' + (1 - s')N = N + (1 - N)s'; numerical check at N = 1024, s' = 0.005 gives 1018.885 on both sides.

### F3. Hill-Marty speed-up for symmetric and asymmetric multicore chips

$$
S_{\text{sym}}(f,n,r) = \frac{1}{\dfrac{1-f}{\mathrm{perf}(r)} + \dfrac{f\, r}{\mathrm{perf}(r)\, n}}, \qquad
S_{\text{asym}}(f,n,r) = \frac{1}{\dfrac{1-f}{\mathrm{perf}(r)} + \dfrac{f}{\mathrm{perf}(r) + n - r}}
$$

Source: [22]. f is the parallelisable fraction, n the chip budget in base-core equivalents, r the budget of the large core, perf(r) its sequential performance (the paper's graphs use the square root of r). Speed-up is relative to one base core. Check: reproduces the paper's values 6.7 (f = 0.9, n = 16, r = 2), 51.2 (f = 0.975, n = 256, r = 7.1) and 125.0 (best asymmetric, f = 0.975, n = 256, at r = 66).

### F4. Heterogeneous-core Amdahl bound **[P]**

$$
S = \frac{1}{\dfrac{s}{\rho_{\text{ser}}} + \dfrac{1-s}{P_{\text{eff}}}}, \qquad P_{\text{eff}} = \sum_{i=1}^{P} \rho_i
$$

Source: our generalisation of F3. rho_i is the throughput of core i relative to a reference core, rho_ser the relative throughput of the core that runs the serial part, and the speed-up is relative to the reference core running everything. Setting rho = perf(r) for one core and 1 for n - r cores recovers S_asym. Assumptions: perfectly divisible parallel work and ideal load balancing. Illustration only: with 8 cores at 1.0 and 8 at 0.6, P_eff = 12.8, and s = 0.05 gives S = 8.05. The value 0.6 is not a measurement (Appendix A, item 2). SMT is ignored.

### F5. Optimal partition of divisible work over cores of different speed

$$
\theta_i^{*} = \frac{\rho_i}{\sum_j \rho_j}, \qquad
T^{*} = \frac{K}{\sum_j \rho_j}, \qquad
T_{\text{equal}} = \frac{K}{P\, \rho_{\min}}, \qquad
\frac{T_{\text{equal}}}{T^{*}} = \frac{\sum_j \rho_j}{P\, \rho_{\min}}
$$

Source: [24], Equations 1 and 3, for the optimal share; the equal-partition time and the ratio are our derivation **[P]**. K is the problem size and theta_i the share given to core i. Check: minimising max_i(theta_i K / rho_i) subject to sum theta_i = 1 equalises all terms, which gives theta_i proportional to rho_i. For 8 cores at 1.0 and 8 at 0.6 the ratio is 12.8 / 9.6 = 1.33. Dynamic per-block dispatch approaches T* without knowing rho_i.

### F6. Work, span and speed-up of a pipeline with ordered dispatch and ordered commit **[P]**

$$
T_1 = N(d + w + c), \qquad
T_\infty = N \max(d, c) + w + \min(d, c)
$$

$$
S \le \min\!\left(P_{\text{eff}},\; \frac{T_1}{T_\infty}\right), \qquad
\lim_{N \to \infty} \frac{T_1}{T_\infty} = \frac{d + w + c}{\max(d, c)}
$$

Source: our derivation, using the work/span method of [32]. d is the serial dispatch time per block, w the parallel work per block, c the serial commit time per block. The dependency graph has edges dispatch(i) to dispatch(i+1), dispatch(i) to work(i), work(i) to commit(i), and commit(i) to commit(i+1). The longest path goes through x+1 dispatches, one work node and N-x commits, which is maximised at x = N-1 when d >= c and at x = 0 otherwise. With d = c = 1 and w = r this agrees with the parallelism n(r+2)/(n+r) reported in [32] up to an additive 1 in the span. The limit formalises the oneTBB statement that throughput "is limited by the throughput of the slowest sequential filter" [38]. Numerical check with d = c = 1 microsecond and w = 200 microseconds: the limit is 202, but T_1/T_inf is 56.5 for N = 78 and 73.1 for N = 114. The first dossier quoted only the limit.

For reference, the Amdahl value with serial fraction s = (d + c)/(d + w + c) is 1/(s + (1 - s)/P_eff), which is 11.46 for the example at P_eff = 12.8. It is a reference point, not a guaranteed lower bound: an ideal pipeline does better because serial stages overlap with parallel work, and a real one can do worse because of contention, waiting at the commit gate and cache-line transfers.

### F7. Scanning bound

$$
Q_{\text{scan}}(N) \le \left\lceil \frac{N}{B} \right\rceil + 1, \qquad
\max_{\text{alignment}} Q_{\text{scan}}(N) = \left\lceil \frac{N - 1}{B} \right\rceil + 1
$$

Source: first inequality [8, Theorem 1]; the exact maximum is our derivation **[P]**. N contiguous elements, block (line) size B. The ceiling brackets are lost in text extraction of the PDF and were confirmed from the proof, which counts at most floor(N/B) full blocks plus two partial ones. Check: brute force over all alignments for B in {4, 8, 64} and N up to 399. For N = 16384 bytes and B = 64 both expressions give 257 lines; a 64-byte-aligned buffer needs exactly 256.

### F8. Cache-oblivious bounds in the ideal-cache model

$$
Z = \Omega(L^2); \quad
Q_{\text{transpose}}(m,n) = \Theta\!\left(1 + \frac{mn}{L}\right); \quad
Q_{\text{sort}}(n) = \Theta\!\left(1 + \frac{n}{L}\left(1 + \log_Z n\right)\right); \quad
Q_{\text{matmul}} = \Theta\!\left(1 + \frac{mn + np + mp}{L} + \frac{mnp}{L\sqrt{Z}}\right)
$$

Source: [7], abstract. Z is the cache size and L the line length. Check: compared with the author-hosted abstract; the full text was opened to confirm the ideal-cache model and the tall-cache assumption.

### F9. Intel block-size rule, formalised

$$
B_{\max} = \frac{C}{h}, \qquad \frac{C}{4} \le B \le \frac{C}{2}
$$

Source: content from [1, Section 11.6.1 and Rule 25]; symbols are ours. For L1d: E-core C = 32 KB gives 8 to 16 KB; P-core C = 48 KB gives 12 to 24 KB; the intersection is 12 to 16 KB. Caveat: Rule 25 is phrased for processors with Hyper-Threading. E-cores have none, so applying the window to them is a conservative extrapolation; Rule 8 (one half of L1) gives the same upper value of 16 KB.

### F10. L1d set geometry and ways occupied by a contiguous buffer **[P]**

$$
S = \frac{C}{A\, L}, \qquad
\text{lines}(B) \le \left\lceil \frac{B - 1}{L} \right\rceil + 1, \qquad
\text{ways}(B) = \left\lceil \frac{\text{lines}(B)}{S} \right\rceil
$$

Source: our derivation from datasheet numbers [3], confirmed on the machine (Appendix B). E-core: 32768/(8 x 64) = 64 sets. P-core: 49152/(12 x 64) = 64 sets. Because L x S = 4096 bytes equals the page size, the set index lies inside the page offset, so consecutive lines of a virtually contiguous buffer map to consecutive sets regardless of physical page placement. A 64-byte-aligned 16 KB buffer has 256 lines and occupies exactly 4 ways in every set: 4 of 8 on an E-core, 4 of 12 on a P-core. An unaligned 16 KB buffer has 257 lines and occupies 5 ways in one set. The first dossier gave ways(B) = ceil(B/(L S)), which is correct only for aligned buffers.

### F11. Working-set condition for L1d residency **[P]**

$$
W = B_{\text{in}} + B_{\text{out}} + B_{\text{scratch}} + B_{\text{stack}} \le \frac{C_{\text{L1d}}}{h}
$$

Source: our formulation; the right-hand side is Intel's upper limit [1] and Drepper's "safe lower limit" of the available cache [9, Section 6.2.3]. h = 2 on a P-core whose SMT sibling is busy and h = 1 on an E-core or on a P-core whose sibling is idle. With B_in = 16 KB this leaves 16 KB on an E-core and 8 KB on a P-core logical processor under SMT. It is a necessary condition for residency, not a guarantee.

### F12. Multi-pass block cost and in-L1 advantage **[P]**

$$
T_{\text{block}}(k) = \frac{B}{L}\Big(k\, t_1 + M\, \Delta\Big), \quad
M = \begin{cases} 1 & W \le C_{\text{eff}} \\ k & W > C_{\text{eff}} \end{cases}; \qquad
R(k) = \frac{t_1 + \Delta}{t_1 + \Delta / k}, \quad
\lim_{k \to \infty} R(k) = 1 + \frac{\Delta}{t_1}
$$

Source: our model. k is the number of passes over the block, t_1 the cost per line when the line is in L1d, Delta the effective extra cost when it must come from L2, M the number of passes that miss. R(k) is the slowdown of a block that does not fit relative to one that does. R(1) = 1: a single pass gains nothing from L1 sizing. Check: R(2) = 1.62, R(4) = 2.34, R(8) = 3.02 for t_1 = 4 and Delta = 13. Anchors: Gracemont latencies of 4 and 17 cycles [1] give a ceiling of 4.25; Drepper's Pentium 4 measurement of about 4 and about 9 cycles per element [9] gives 2.25. Both are upper bounds for memory-bound code with dependent loads. Sequential scans overlap their loads and are prefetched, and compute-bound code hides the difference, so measured gains will be smaller.

### F13. X100 cache-fit condition

$$
n_{\text{vec}} \cdot \sum_j w_j \le C
$$

Source: our notation for the condition described in [13, Section 5.1.1]. n_vec is the number of tuples per vector and w_j the width in bytes of each vector. Check: 8192 x 40 bytes = 320 KB, the combined L1 and L2 size quoted in the paper.

### F14. Int8 dot product: semantics, signed correction, saturation and overflow bounds

$$
d_i \leftarrow d_i + \sum_{j=0}^{3} \mathrm{zext}\big(a_{4i+j}\big) \cdot \mathrm{sext}\big(b_{4i+j}\big), \qquad i = 0,\dots,7 \text{ for 256-bit}
$$

$$
\sum_j x_j y_j = \sum_j (x_j + 128)\, y_j \;-\; 128 \sum_j y_j, \qquad x_j, y_j \in [-128, 127]
$$

$$
\big|a_{2j} b_{2j} + a_{2j+1} b_{2j+1}\big| \le 2\, a_{\max} b_{\max} \le 32767
\quad\Longleftarrow\quad a \in [0,127],\; b \in [-128,127] \;\; (2 \cdot 127 \cdot 128 = 32512)
$$

Source: first line [5] (operation section of VPDPBUSD; a is the unsigned first source, b the signed second source, d the 32-bit accumulators); second line is the offset correction described in [1, Chapter 8]; third line is our bound **[P]** for the AVX2 fallback VPMADDUBSW, which saturates pair sums to 16 bits [5]. Checks: with full-range operands the pair sum can reach 2 x 255 x 127 = 64770, which saturates. The offset correction produces an unsigned operand up to 255 and is therefore safe only with VPDPBUSD, not with the VPMADDUBSW fallback. For the fallback use the sign transfer |x| and sign(x) y with values quantised to [-127, 127], where the pair sum is at most 2 x 127 x 127 = 32258. The non-saturating VPDPBUSD wraps at 32 bits; the total of a D-dimensional dot product is below 2^31 for D <= 65793.

### F15. Sequence guards for ordered claim and ordered commit **[P]**

$$
\textsf{claim}: \; i \leftarrow \textsf{next};\;\; \text{if } i < N \text{ and } i - \textsf{committed} < K \text{ then } \mathrm{CAS}(\textsf{next}, i, i+1)
$$

$$
\textsf{commit}(i) \text{ enabled} \iff \textsf{committed} = i; \qquad
\textsf{commit}(i): \; \text{append } r_i;\; \textsf{committed} \leftarrow i + 1
$$

Source: our protocol, built on [30, 31, 29, 32]. Invariants: (I1) every i in 0..N-1 is claimed exactly once and claims are totally ordered by the modification order of `next` [41]; (I2) `committed` = j implies that results 0..j-1 have been appended exactly once in increasing order; (I3) 0 <= next - committed <= K. Check of I3: `committed` never decreases, so a stale read of it only makes the guard stricter; if i - c < K at the read and the CAS succeeds, then next - committed <= K afterwards. The window condition is the same as the Cilk-P throttling rule [32]. See Section 5 for the proof sketch and assumptions.

### F16. Cost of one pre-emption or migration **[P]**

$$
\Delta Q_{\text{L1d}} \le \frac{C}{L} = \begin{cases} 512 \text{ lines} & \text{E-core} \\ 768 \text{ lines} & \text{P-core} \end{cases}
$$

Source: our application of [10, Lemma 2], which holds for caches with simple replacement policies such as LRU. It equals two to three blocks' worth of first-touch misses.

### F17. Blocks per dispatch ticket **[P]**

$$
m = \max\!\left(1,\; \operatorname{round}\!\left(\frac{\tau}{\hat{t}_{\text{block}}}\right)\right)
$$

Source: our adaptation of the target-duration idea of [16]. tau is the target task duration and t_block the measured mean time per 16 KB block. The cache unit stays 16 KB; the scheduling unit becomes a run of m consecutive blocks. Output does not depend on m because commits remain in block order.

## 5. What is novel in our proposal

Everything in this section is **[P]**: our design or our analysis. None of it is an established result, and the paper must say so.

**P1. The 16 KB choice for this CPU.** 16 KB is one half of the E-core L1d, one third of the P-core L1d, and two thirds of a P-core logical processor's equal share when its sibling is busy. It is the largest size inside Intel's quarter-to-half window for both core types (F9). A 64-byte-aligned 16 KB buffer uses 4 ways in each of the 64 sets and cannot evict itself (F10). An input block plus an equally large output buffer would use all 8 ways of an E-core L1d, so the rule must be applied to the whole working set (F11).

**P2. Reconciliation of the contested evidence.** Intel Rule 8 favours half of L1 [1]; the morsel and X100 results say that L1 fit is not needed [13, 14]; Drepper says matching the working set to the cache matters when "the data set is needed more than once" [9]. Our reading is that these agree once the number of passes is considered (F12): L1 sizing pays only for multi-pass per-block work. This reading is ours and is untested on the target workload.

**P3. The ordered ticket protocol.** Shared state: atomic counters `next` and `committed`, each on its own cache line, and a ring of K result slots. A worker claims block i by compare-and-swap, computes the result as a pure function of block i into thread-local memory, and the result is committed only when `committed` equals i (F15).

- Safety 1 (ordered, unique start). Modifications of one atomic object occur in a single total order [41], so each ticket value is obtained by exactly one worker and the claim of i precedes the claim of i+1.
- Safety 2 (ordered, unique commit). The only transition of `committed` from i to i+1 is commit(i); it is enabled only when `committed` equals i and only the holder of ticket i executes it. By induction commits happen once each in the order 0, 1, 2, and so on. A release store and acquire load of `committed` make the append of result i happen before the append of result i+1, so the shared output is free of data races.
- Liveness. The holder of the smallest uncommitted ticket is never blocked by the commit guard or by the window. Every wait is on a smaller sequence number, so the wait-for relation is acyclic. If every block computation terminates and threads are eventually scheduled, all N blocks commit.
- Bounded memory. next - committed <= K (I3).
- Determinism. If block boundaries are fixed independently of thread count and timing, and the per-block function is pure, the committed output equals the sequential output for every schedule. This is external determinism in the sense of [36] and follows the commit-in-serial-order principle of [34].

Two variants exist. In the waiting variant each worker waits at the gate and commits its own result; the ring is not needed, but workers that finish early stand idle. In the draining variant a worker stores its result in slot i mod K, marks it done, and any worker that finds the head slot done commits all consecutive done slots under a try-lock, re-checking after release. The draining variant is a software reorder buffer [29] and avoids idle workers. For it, Safety 2 holds with "the holder of the commit lock" in place of "the holder of ticket i", and the guard additionally requires slot i to be marked done. The hand-off between a worker that marks a slot done and then fails the try-lock, and the lock holder that releases and then re-checks, is a store-then-load handshake on two different objects. It is correct only if these four operations are sequentially consistent (`memory_order_seq_cst`, which establishes "a single total modification order of all atomic operations that are so tagged" [41]); release and acquire ordering is not sufficient there.

**P4. Precise reading of requirement 9.** Following [28], the guarantee is stated on events: claim(i) precedes claim(i+1), and commit(i) precedes commit(i+1). User-space code cannot guarantee that the first payload instruction of block A executes before that of block B, because the operating system may pre-empt the worker of A directly after its claim [45]. A stronger, still observable property is available with a start gate: worker i waits for `started` = i, performs a prologue such as the first touch of the block, then stores `started` = i+1. This adds the prologue to the serial fraction and inherits the pre-emption sensitivity of FIFO locks [30]. Only full serialisation orders all payload instructions.

**P5. Speed-up model.** F4, F5 and F6 together: the speed-up is limited by P_eff, by the finite-N parallelism of the pipeline, and by whatever serial work (file reading, model inference on a single GPU) is outside the pipeline.

**P6. Scale check for the evaluation corpus [M].** The corpus has 59 files and 1,273,275 bytes. Concatenated, it splits into 78 blocks of 16 KB; blocked per file, it gives 114 blocks, and 33 of the 59 files fit in a single block. That is 3 to 5 blocks per logical processor on a 24-thread machine, and the whole corpus is about the size of one P-core L2 (1.25 MB). Start-up, I/O and the serial stages will dominate. Speed-up must be measured and reported with absolute times.

## 6. Risks and failure modes

1. **No L1 benefit.** If each block is processed in one streaming pass, or if the time is dominated by computation or by model inference, 16 KB blocks will not be faster than L2-sized blocks (F12, E13, E14). Embedding-model weights are hundreds of megabytes and can never be L1-resident.
2. **Working set larger than assumed.** Tokeniser tables, hash maps, parser state, output buffers and the stack count towards W (F11). On a P-core logical processor with a busy sibling only 8 KB remains beside a 16 KB input block.
3. **Cache not exclusive.** The operating system, interrupts and the SMT sibling use the same L1d (E5, E6). Each pre-emption or migration can cost up to 512 or 768 line refills (F16).
4. **Coordination overhead exceeds the work.** A 16 KB block may take only tens of microseconds. Database engines aim for tasks of about 2 ms [16]. Contended compare-and-swap and cache-line transfers on `next` and `committed` can then dominate.
5. **Head-of-line blocking.** One slow block (for example on an E-core, or pre-empted) stalls the commit cursor, fills the window and idles the other workers. Throttled pipelines with non-uniform iterations can lose speed-up [32, Theorem 13].
6. **Spinning under pre-emption.** Strict FIFO gates waste CPU when the thread whose turn it is has been descheduled [30]; spin-waits also disturb other threads on a hybrid CPU [4].
7. **Wrong P_eff.** The E-core to P-core ratio depends on the kernel and instruction set [24]. For 256-bit SIMD the E-cores split most operations into two 128-bit halves (E30). Using 0.6 without measurement will mis-predict speed-up.
8. **Static partitioning or hard affinity.** Equal shares make P-cores wait for E-cores (F5, E26, R1). Hard affinity is discouraged by Intel and Microsoft (E27, E9), and thread binding made scaling worse for two of five workloads in [25].
9. **False sharing.** Counters or per-worker state on the same 64-byte line (E11).
10. **Int8 kernel errors.** Saturation in the VPMADDUBSW fallback, the unsigned-by-signed operand order, and the missing signed-by-signed instruction on this CPU (F14, E29). No AVX-512.
11. **Determinism broken by impure blocks.** Shared mutable state, random numbers, iteration over unordered containers, or floating-point reductions merged in arrival order instead of sequence order.
12. **Block boundaries.** Cutting at function boundaries gives variable block sizes. The largest file in the corpus is 117,283 bytes; any single function above 16 KB forms an oversized block.
13. **Corpus too small.** 78 to 114 blocks (P6). Timing noise and fixed costs can hide or fake a speed-up.
14. **Measurement limits.** Hardware cache-miss counters need a profiler and usually administrative rights. Without them only timing curves versus block size are available, which is indirect evidence.
15. **Overclaiming.** "Forces L1" and "A reaches the CPU first" are not defensible statements (E9, E16).
16. **Evidence quality.** [25] and [24] are preprints; [31] and [17] are engineering documents; [9] measures 2007 hardware; [6] analyses direct-mapped caches with matrix strides.

## 7. Design recommendations

1. **Block size.** Make it a run-time parameter with a default of 16 KB. At start-up read every L1 data cache record with `GetLogicalProcessorInformationEx` (`RelationCache`) [43] and compute, over all core types, the window [max C/4, min C/2] and the working-set ceiling min C/h, where h is the number of logical processors in the record's mask. On the target machine this gives a window of 12 to 16 KB and a ceiling of 24 KB. Do not use C/(2h): it yields 12 KB on P-cores and contradicts the default.
2. **Budget the whole working set.** Input block, per-block output, scratch and hot stack together must stay below 24 KB, the smallest per-logical-processor share on this machine. Up to 32 KB is available on an E-core, and on a P-core only while its sibling is idle, which the program cannot guarantee without affinity. Keep per-block outputs compact (offsets and int8 vectors), and reuse one thread-local scratch arena. Measure the real W with a counting allocator.
3. **Make the block multi-pass.** Run all per-block stages (split into functions, tokenise, hash, quantise, score) on one block before moving to the next. This is the only case in which L1 sizing can pay (F12).
4. **Layout.** Contiguous buffers aligned to 64 bytes, sequential access, at most a few concurrent streams. Cut blocks at the function boundary nearest to the target size and let an oversized function form its own block.
5. **False sharing.** Put `next`, `committed` and each worker's state on separate lines with `alignas(std::hardware_destructive_interference_size)` (64 with MSVC [47]); consider 128-byte separation for the two counters [1]. Workers write only thread-local memory until commit.
6. **Dispatch.** Hand out work dynamically with one atomic ticket. Do not partition statically. Do not call `SetThreadAffinityMask`; if placement must be steered use CPU Sets [44]. Make the worker count configurable and test 8, 16 and 24.
7. **Separate the scheduling unit from the cache unit.** Let one ticket cover a run of m consecutive 16 KB blocks, with m chosen so that a task lasts roughly 1 to 2 ms (F17, [16]). Commit per block, in block order.
8. **Ordering.** Implement the draining variant of P3 with a ring of K slots. Start with K = 4P, the value reported to work well in [32], and sweep K from 2P to 10P. Keep dispatch and commit tiny: the dispatcher hands out (pointer, length, sequence); commit appends a small result.
9. **Waiting.** Spin briefly with `_mm_pause`, then block with `std::atomic::wait` and wake with `notify_one` or `notify_all` [42]. No unbounded spinning [4, 30].
10. **Int8 kernel.** Select at run time by CPUID: AVX-VNNI `VPDPBUSD` (256-bit) with the +128 offset correction; otherwise AVX2 `VPMADDUBSW` + `VPMADDWD` + `VPADDD` with sign transfer and values quantised to [-127, 127]; otherwise scalar. Treat AVX-VNNI-INT8 as an optional path for newer CPUs. Add a unit test that compares every path with the scalar result on extreme inputs (-128, -127, 127).
11. **Tests.** Determinism test: for each block size, index the corpus with 1, 2, 8, 16 and 24 workers and several window sizes and require byte-identical committed output. Across block sizes, require identical per-function results, which holds only if the per-function output is defined independently of the blocking. Stress test: inject random delays and yields into workers to force out-of-order completion and head-of-line blocking.
12. **Evaluation and wording.** Sweep block size (4 KB to 1 MB) and worker count; report time per byte, absolute times and speed-up; measure d, w, c and per-core-type throughput, and compare with F4 and F6. Report a flat or negative result if 16 KB is not better than L2-sized blocks. In the paper write "the per-thread working set is sized to fit the L1 data cache" and "block i is claimed before block i+1 and results are committed in sequence order".

## 8. References

Web pages without a publication date are cited with the access date 2026-09-29.

1. Intel Corporation. *Intel 64 and IA-32 Architectures Optimization Reference Manual, Volume 1.* Order Number 248966-049, January 2024. https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf
2. Intel Corporation. *Intel 64 and IA-32 Architectures Optimization Reference Manual, Volume 1.* Document Number 248966-050US, April 2024. https://cdrdv2-public.intel.com/821612/248966-Optimization-Reference-Manual-V1-050.pdf
3. Intel Corporation. *12th Generation Intel Core Processors Datasheet, Volume 1 of 2.* Doc. No. 655258, Rev. 011, May 2025. https://cdrdv2-public.intel.com/655258/655258-011.pdf
4. Intel Corporation. *Optimizing Software for Intel performance hybrid architecture.* White paper, Document Number 348851-001US, Revision 1.0, October 2021. https://cdrdv2-public.intel.com/685869/optimizing%20_software_for_performance_hybrid_architecture.pdf
5. Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2 (2A, 2B, 2C and 2D): Instruction Set Reference, A-Z.* Order Number 325383-093US, September 2026. https://cdrdv2-public.intel.com/929352/325383-093-sdm-vol-2abcd.pdf
6. Monica S. Lam, Edward E. Rothberg, Michael E. Wolf. The Cache Performance and Optimizations of Blocked Algorithms. *Proc. 4th Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS IV)*, Palo Alto, April 1991, pp. 63-74. DOI 10.1145/106972.106981. Full text read at https://www.cse.iitd.ac.in/~nagaraju/TA/2004-2005/SDS-04/project/loop_tiling.pdf
7. Matteo Frigo, Charles E. Leiserson, Harald Prokop, Sridhar Ramachandran. Cache-Oblivious Algorithms. *40th Annual Symposium on Foundations of Computer Science (FOCS '99)*, New York, October 1999, pp. 285-297. DOI 10.1109/SFFCS.1999.814600. http://www.fftw.org/~athena/abstracts/abstract12.html
8. Erik D. Demaine. Cache-Oblivious Algorithms and Data Structures. *Lecture Notes from the EEF Summer School on Massive Data Sets*, BRICS, University of Aarhus, June 27 to July 1, 2002. https://erikdemaine.org/papers/BRICS2002/paper.pdf
9. Ulrich Drepper. *What Every Programmer Should Know About Memory.* Red Hat, Inc., version 1.0, November 21, 2007. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
10. Umut A. Acar, Guy E. Blelloch, Robert D. Blumofe. The Data Locality of Work Stealing. *Proc. 12th ACM Symposium on Parallel Algorithms and Architectures (SPAA 2000)*, Bar Harbor, July 2000, pp. 1-12. DOI 10.1145/341800.341801. https://www.cs.cmu.edu/~guyb/papers/locality2000.pdf
11. William J. Bolosky, Michael L. Scott. False Sharing and its Effect on Shared Memory Performance. *USENIX Symposium on Experiences with Distributed and Multiprocessor Systems (SEDMS IV)*, San Diego, September 1993. https://www.usenix.org/conference/sedms-iv/false-sharing-and-its-effect-shared-memory-performance
12. JF Bastien, Olivier Giroux. *P0154R1: constexpr std::hardware_{constructive,destructive}_interference_size.* ISO/IEC JTC1/SC22/WG21, 2016-03-03. https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
13. Peter Boncz, Marcin Zukowski, Niels Nes. MonetDB/X100: Hyper-Pipelining Query Execution. *Proceedings of the 2005 CIDR Conference*, 2005. https://www.cidrdb.org/cidr2005/papers/P19.pdf (same text at https://ir.cwi.nl/pub/16497/16497B.pdf)
14. Viktor Leis, Peter Boncz, Alfons Kemper, Thomas Neumann. Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age. *Proc. 2014 ACM SIGMOD Int. Conf. on Management of Data*, Snowbird, June 2014, pp. 743-754. DOI 10.1145/2588555.2610507. https://db.in.tum.de/~leis/papers/morsels.pdf
15. Timo Kersten, Viktor Leis, Alfons Kemper, Thomas Neumann, Andrew Pavlo, Peter Boncz. Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask. *Proceedings of the VLDB Endowment* 11(13): 2209-2222, 2018. DOI 10.14778/3275366.3275370. http://www.vldb.org/pvldb/vol11/p2209-kersten.pdf
16. Benjamin Wagner, André Kohn, Thomas Neumann. Self-Tuning Query Scheduling for Analytical Workloads. *Proc. 2021 Int. Conf. on Management of Data (SIGMOD '21)*, June 2021, pp. 1879-1891. DOI 10.1145/3448016.3457260. https://db.in.tum.de/~kohn/papers/query-scheduling-sigmod21.pdf
17. Mark Raasveldt (GitHub user Mytherin). *Support Parallel Order-Preserving Result Set Materialization.* DuckDB pull request 3700, merged 2022-05-25. https://github.com/duckdb/duckdb/pull/3700
18. Anurag Chakraborty, Semih Salihoğlu. Robust Recursive Query Parallelism in Graph Database Management Systems. *Proceedings of the VLDB Endowment* 18(11): 4465-4477, 2025. DOI 10.14778/3749646.3749706. arXiv:2508.19379. https://arxiv.org/abs/2508.19379
19. Leonardo Kuffo, Elena Krippner, Peter Boncz. PDX: A Data Layout for Vector Similarity Search. *Proceedings of the ACM on Management of Data* 3(3) (SIGMOD), pp. 1-26, June 2025. DOI 10.1145/3725333. arXiv:2503.04422. https://arxiv.org/abs/2503.04422
20. Leonardo Kuffo, Peter Boncz. Bang for the Buck: Vector Search on Cloud CPUs. *Proc. 21st Int. Workshop on Data Management on New Hardware (DaMoN '25)*, Berlin, June 2025, pp. 1-8. DOI 10.1145/3736227.3736233. arXiv:2505.07621. https://arxiv.org/abs/2505.07621
21. Xiaoyao Zhong, Haotian Li, Jiabao Jin, Mingyu Yang, Deming Chu, Xiangyu Wang, Zhitao Shen, Wei Jia, George Gu, Yi Xie, Xuemin Lin, Heng Tao Shen, Jingkuan Song, Peng Cheng. VSAG: An Optimized Search Framework for Graph-Based Approximate Nearest Neighbor Search. *Proceedings of the VLDB Endowment* 18(12): 5017-5030, 2025. DOI 10.14778/3750601.3750624. arXiv:2503.17911. https://arxiv.org/abs/2503.17911
22. Mark D. Hill, Michael R. Marty. Amdahl's Law in the Multicore Era. *IEEE Computer* 41(7): 33-38, July 2008. DOI 10.1109/MC.2008.209. https://research.cs.wisc.edu/multifacet/papers/ieeecomputer08_amdahl_multicore.pdf
23. Juan Carlos Saez, Fernando Castro, Manuel Prieto-Matias. Enabling performance portability of data-parallel OpenMP applications on asymmetric multicore processors. *49th Int. Conf. on Parallel Processing (ICPP '20)*, Edmonton, August 2020, Article 51, 11 pages. DOI 10.1145/3404397.3404441. arXiv:2402.07664 (posted 2024). https://arxiv.org/abs/2402.07664
24. Yu Luo, Yucheng Liu, Haihao Shen (listed on arXiv as Luo Yu, Liu Yucheng, Shen Haihao). A dynamic parallel method for performance optimization on hybrid CPUs. arXiv:2411.19542 [cs.DC], 29 November 2024. Preprint. https://arxiv.org/abs/2411.19542
25. Nanda Velugoti, Joseph Manzano, Nathan Tallent, Kyle Hale. Effects of Hybrid CPU and Cache Architectures on Parallel HPC and Cloud Applications. arXiv:2608.23165 [cs.DC], 24 August 2026. Preprint. https://arxiv.org/abs/2608.23165
26. Gene M. Amdahl. Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities. *AFIPS Conference Proceedings, Vol. 30 (Spring Joint Computer Conference)*, Atlantic City, April 18-20, 1967, pp. 483-485. DOI 10.1145/1465482.1465560. Read in the IEEE SSCS reprint at https://www3.cs.stonybrook.edu/~rezaul/Spring-2012/CSE613/reading/Amdahl-1967.pdf
27. John L. Gustafson. Reevaluating Amdahl's Law. *Communications of the ACM* 31(5): 532-533, May 1988. DOI 10.1145/42411.42415. http://www.johngustafson.net/pubs/pub13/amdahl.pdf
28. Leslie Lamport. Time, Clocks, and the Ordering of Events in a Distributed System. *Communications of the ACM* 21(7): 558-565, July 1978. DOI 10.1145/359545.359563. https://lamport.azurewebsites.net/pubs/time-clocks.pdf
29. James E. Smith, Andrew R. Pleszkun. Implementing Precise Interrupts in Pipelined Processors. *IEEE Transactions on Computers* 37(5): 562-573, May 1988. DOI 10.1109/12.4607. https://www.cs.virginia.edu/~evans/greatworks/smith.pdf
30. John M. Mellor-Crummey, Michael L. Scott. Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors. *ACM Transactions on Computer Systems* 9(1): 21-65, February 1991. DOI 10.1145/103727.103729. https://www.cs.rochester.edu/u/scott/papers/1991_TOCS_synch.pdf
31. Martin Thompson, Dave Farley, Michael Barker, Patricia Gee, Andrew Stewart. *Disruptor: High performance alternative to bounded queues for exchanging data between concurrent threads.* LMAX technical paper, May 2011. https://lmax-exchange.github.io/disruptor/files/Disruptor-1.0.pdf
32. I-Ting Angelina Lee, Charles E. Leiserson, Tao B. Schardl, Jim Sukha, Zhunping Zhang. On-the-Fly Pipeline Parallelism. *Proc. 25th ACM Symposium on Parallelism in Algorithms and Architectures (SPAA '13)*, Montreal, June 2013, pp. 140-151. DOI 10.1145/2486159.2486174. Record: https://dspace.mit.edu/handle/1721.1/90258
33. Marek Olszewski, Jason Ansel, Saman Amarasinghe. Kendo: Efficient Deterministic Multithreading in Software. *Proc. 14th Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS '09)*, Washington DC, March 2009, pp. 97-108. DOI 10.1145/1508244.1508256. https://commit.csail.mit.edu/papers/2009/asplos073-olszewski.pdf
34. Tom Bergan, Owen Anderson, Joseph Devietti, Luis Ceze, Dan Grossman. CoreDet: A Compiler and Runtime System for Deterministic Multithreaded Execution. *Proc. 15th Int. Conf. on Architectural Support for Programming Languages and Operating Systems (ASPLOS '10)*, Pittsburgh, March 2010, pp. 53-64. DOI 10.1145/1736020.1736029. https://homes.cs.washington.edu/~djg/papers/asplos10-coredet.pdf
35. Tongping Liu, Charlie Curtsinger, Emery D. Berger. Dthreads: Efficient Deterministic Multithreading. *Proc. 23rd ACM Symposium on Operating Systems Principles (SOSP '11)*, October 2011, pp. 327-336. DOI 10.1145/2043556.2043587. Author draft read at https://people.cs.umass.edu/~emery/pubs/dthreads-draft.pdf
36. Guy E. Blelloch, Jeremy T. Fineman, Phillip B. Gibbons, Julian Shun. Internally Deterministic Parallel Algorithms Can Be Fast. *Proc. 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '12)*, New Orleans, February 2012, pp. 181-192. DOI 10.1145/2145816.2145840. https://www.cs.cmu.edu/~guyb/papers/BFGS12.pdf
37. UXL Foundation (oneTBB project). *oneTBB Developer Guide: Working on the Assembly Line: parallel_pipeline*, and *Using Circular Buffers*. Undated; source file last changed 2025-03-05. https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Working_on_the_Assembly_Line_pipeline.html
38. UXL Foundation (oneTBB project). *oneTBB Developer Guide: Throughput of pipeline.* Undated; source file last changed 2021-06-28. https://raw.githubusercontent.com/uxlfoundation/oneTBB/master/doc/main/tbb_userguide/Throughput_of_pipeline.rst
39. Intel Corporation and UXL Foundation. *oneAPI Specification, oneTBB: filter_mode [algorithms.parallel_pipeline.filter_mode].* Copyright 2019-2021. https://raw.githubusercontent.com/uxlfoundation/oneAPI-spec/main/source/elements/oneTBB/source/algorithms/functions/parallel_pipeline_func/filter_mode_enum.rst
40. cppreference.com contributors. *std::hardware_destructive_interference_size, std::hardware_constructive_interference_size.* Undated. https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
41. cppreference.com contributors. *std::memory_order.* Undated. https://en.cppreference.com/w/cpp/atomic/memory_order
42. cppreference.com contributors. *std::atomic<T>::wait.* Undated. https://en.cppreference.com/w/cpp/atomic/atomic/wait
43. Microsoft. *CACHE_RELATIONSHIP structure (winnt.h).* Microsoft Learn, page updated 2024-02-22. https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-cache_relationship
44. Microsoft. *CPU Sets.* Microsoft Learn, page dated 2025-07-14. https://learn.microsoft.com/en-us/windows/win32/procthread/cpu-sets
45. Microsoft. *Scheduling Priorities.* Microsoft Learn, page dated 2025-07-14. https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
46. Microsoft. *SetThreadAffinityMask function (winbase.h).* Microsoft Learn, page updated 2024-02-22. https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask
47. Microsoft. *Microsoft STL, header `<new>`* (definition of the hardware interference constants). GitHub repository microsoft/STL, main branch. https://raw.githubusercontent.com/microsoft/STL/main/stl/inc/new

## Appendix A. Corrections to the first dossier

| # | First-pass statement | Correction |
|---|---|---|
| 1 | Recommendation: B_max = min over core types of CacheSize / (2 x logical processors sharing); the worked example was cut off | That formula gives 48/(2 x 2) = 12 KB on P-cores and contradicts the 16 KB default. Replaced by the window [C/4, C/2] (12 to 16 KB on this machine) and the working-set ceiling C/h (24 KB). |
| 2 | E-cores are "only 60% the speed of P-cores" [25], used as rho = 0.6 | The quotation is accurate, but the paper derives it from a ratio of maximum clock frequencies (3900/6500 MHz), not from measured throughput. P_eff = 12.8 is an illustration. [24] shows that the ratio depends on the kernel and instruction set. |
| 3 | With binding, scaling "improves up to 8 threads and worsens from 10 to 16" | True for miniVite; NAS-FT worsens only up to 12 threads; NAS BT, LU and SP are unaffected. Linux and OpenMP, not Windows. |
| 4 | FIFO guarantee "is likely to waste CPU resources if spinning processes may be preempted" cited to Section 5 of [30] | The sentence is in Section 4.3. |
| 5 | [18] cited as an arXiv preprint; author spelled Salihoglu | Published in PVLDB 18(11): 4465-4477 (2025). The author's name is Salihoğlu. |
| 6 | [19], [20], [21] cited mainly by arXiv number | Published versions and DOIs added: Proc. ACM Manag. Data 3(3); DaMoN '25; PVLDB 18(12): 5017-5030. |
| 7 | One Intel manual revision (049) cited for everything | Revision 050 (April 2024) is the latest served by Intel. It keeps Rules 7, 8 and 25 but replaces the Gracemont section by Crestmont. Gracemont statements must cite revision 049. |
| 8 | L2 prefetchers "prefetch only inside page boundary (4 KBytes)" presented as current behaviour | The sentence describes the legacy Intel Core microarchitecture [1, Section 3.7.3]. For Gracemont the manual says the L1 stride prefetcher crosses page boundaries. |
| 9 | "Drepper measured that ... each thread has only half the cache" | It is Drepper's explanation of a Pentium 4 result, not a measurement of cache share. L1d sharing between SMT siblings is competitive; 24 KB is a planning figure. |
| 10 | Drepper's 4 and 9 cycles used as per-line costs | They are cycles per 8-byte list element on a Pentium 4. Only the ratio is usable, and only as an illustration. |
| 11 | Drepper's advice to match the working set to the cache cited generally under Section 6 | The sentence is in Section 6.2.3, "Optimizing Level 2 and Higher Cache Access". It supports L2-sized blocks at least as much as L1-sized ones. |
| 12 | ways(B) = ceil(B/(L S)) | Correct only for 64-byte-aligned buffers. An unaligned 16 KB buffer spans 257 lines and needs 5 ways in one set (F10). |
| 13 | Scan bound given only as ceil(N/B) + 1 | Confirmed from the proof. The exact worst case is ceil((N-1)/B) + 1; both give 257 for 16 KB and 64-byte lines (F7). |
| 14 | Pipeline bound (d + w + c)/max(d, c) = 202 used for the corpus | This is the limit for N to infinity. For finite N the parallelism is N(d+w+c)/(N max(d,c) + w + min(d,c)), which is 56.5 for N = 78 (F6). |
| 15 | "S >= about 1/(s + (1-s)/P_eff)" | Not a guaranteed lower bound. Real pipelines can fall below it because of contention and head-of-line blocking. Relabelled as a reference value. |
| 16 | AVX2 fallback "with inputs restricted ... (for example 7-bit magnitudes)" together with the +128 offset | The two are incompatible: the offset makes the unsigned operand as large as 255, which can saturate VPMADDUBSW. The fallback must use sign transfer with values in [-127, 127] (F14). |
| 17 | VPDPBUSD and PMADDUBSW cited from an unofficial mirror | Replaced by the primary Intel SDM Volume 2, order number 325383-093US [5]. Mirror content matched the primary text. |
| 18 | Intel's quarter-to-half window applied to E-cores without comment | Rule 25 is written for Hyper-Threading processors. For E-cores it is an extrapolation; Rule 8 gives the same 16 KB (F9). |
| 19 | Corpus "about 1.3 MB ... roughly 80 blocks" | Measured: 1,273,275 bytes in 59 files. 78 blocks if concatenated, 114 if blocked per file (P6). |
| 20 | Undated web pages given the year 2024 | cppreference and oneTBB guide pages are undated. They are cited with the access date, and with the repository file date where available. X100 is now cited from the official CIDR site. |

Minor: the Bolosky and Scott abstract puts "unnecessary" in quotation marks; the Intel and Microsoft texts write 'soft' in quotation marks; Crossref lists the first author of [6] as "Monica D. Lam", while the paper itself says Monica S. Lam.

## Appendix B. Measurements taken on the target machine (2026-09-29) **[M]**

All queries were read-only.

| Item | Result |
|---|---|
| CPU (WMI `Win32_Processor`) | 12th Gen Intel Core i9-12950HX, 16 cores, 24 logical processors |
| L1d, P-cores (`GetLogicalProcessorInformationEx`, `RelationCache`) | 49,152 bytes, 12-way, 64-byte lines, 64 sets, shared by 2 logical processors (masks 0x3, 0xC, ... 0xC000) |
| L1d, E-cores | 32,768 bytes, 8-way, 64-byte lines, 64 sets, 1 logical processor each (masks 0x10000 ... 0x800000) |
| L2, P-cores | 1,310,720 bytes, 10-way, per core |
| L2, E-cores | 2,097,152 bytes, 16-way, shared by 4 E-cores (masks 0xF0000 and 0xF00000) |
| L3 | 30 MB |
| MSVC 14.51.36231 `<new>` | `hardware_destructive_interference_size = 64` |
| Demo corpus (`Demo repo/mimalloc`, origin `https://github.com/microsoft/mimalloc.git`, HEAD `31d034d94cdb8e22f7d7ed55967f581a2d6e831d` dated 2026-09-16) | 41 `.c` and 18 `.h` files, 1,273,275 bytes; file size minimum 236, median 13,112, mean 21,581, maximum 117,283 bytes; 33 files are 16 KB or smaller; 78 blocks of 16 KB if concatenated, 114 if blocked per file |

Not measured: CPUID feature flags (AVX-VNNI is taken from the datasheet [3]), hardware cache-miss counters, and the relative throughput of E-cores.

Note for the paper: the local checkout's `origin` remote is the upstream `microsoft/mimalloc` repository. The paper should name the upstream commit above when it describes the corpus.

## Appendix C. Verification method and unverified leads

- Documents were downloaded with curl and converted with pdftotext; web pages were read as HTML; DOIs were checked against the Crossref REST API; arXiv records were read from the abs pages. Certificate revocation look-ups could not be completed from this machine, so curl was run with `--ssl-revoke-best-effort`; certificate chains were still validated. One host whose chain failed validation was not used.
- Leads that were not added because the full text could not be opened: Saez and Prieto-Matias, "Evaluation of the Intel thread director technology on an Alder Lake processor" (APSys 2022, DOI 10.1145/3546591.3547532; metadata and abstract only); Boncz, Manegold and Kersten, "Database Architecture Optimized for the New Bottleneck: Memory Access" (VLDB 1999; connection reset). They should be opened before they are cited.
- The journal version of [32] (ACM TOPC 2(3), 2015, DOI 10.1145/2809808) exists according to Crossref but was not opened; the note cites the SPAA 2013 paper.
