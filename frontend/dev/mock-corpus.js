// Synthetic corpus for the development mock server.
//
// The real server indexes a checkout of mimalloc. The mock does not read any
// file: it generates a corpus of the same shape (59 files, 1510 units, 37
// density groups) from the name lists below and a seeded random generator, so
// that every run serves the same data. The code texts are made up; only a few
// central functions are written by hand to make the demo readable.

// ---- deterministic random numbers -------------------------------------------

// mulberry32: small, fast, and good enough for mock data.
export function makeRandom(seed) {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    int: (n) => Math.floor(next() * n),
    pick: (list) => list[Math.floor(next() * list.length)],
    // Standard normal deviate (Box-Muller).
    normal: () => Math.sqrt(-2 * Math.log(1 - next())) * Math.cos(2 * Math.PI * next()),
  };
}

// ---- topics: one per density group ------------------------------------------

// sig: [return type, parameters]; the first parameter name is used in the body.
const TOPICS = [
  { key: 'alloc', files: ['src/alloc.c'], sig: ['void*', 'mi_heap_t* heap, size_t size'],
    words: 'allocate allocation malloc zalloc calloc small block size bytes heap fast path zero',
    doc: 'Allocation: takes a block of the requested size from a page of the heap.',
    names: 'mi_malloc mi_heap_malloc mi_zalloc mi_heap_zalloc mi_calloc mi_heap_calloc mi_malloc_small ' +
      'mi_heap_malloc_small mi_zalloc_small _mi_heap_malloc_zero _mi_heap_malloc_zero_ex _mi_page_malloc ' +
      '_mi_page_malloc_zero mi_mallocn mi_heap_mallocn mi_heap_malloc_small_zero mi_page_malloc_zeroed' },
  { key: 'realloc', files: ['src/alloc.c'], sig: ['void*', 'mi_heap_t* heap, void* p, size_t newsize'],
    words: 'realloc reallocate resize grow shrink expand copy rezalloc recalloc',
    doc: 'Reallocation: resizes a block in place when it fits, otherwise allocates, copies and frees.',
    names: 'mi_realloc mi_heap_realloc mi_reallocn mi_heap_reallocn mi_reallocf mi_heap_reallocf mi_rezalloc ' +
      'mi_heap_rezalloc mi_recalloc mi_heap_recalloc _mi_heap_realloc_zero mi_expand' },
  { key: 'free', files: ['src/free.c'], sig: ['void', 'mi_page_t* page, mi_block_t* block'],
    words: 'free release deallocate block thread local delayed double padding usable size',
    doc: 'Freeing: returns a block to the free list of its page, locally or from another thread.',
    names: 'mi_free mi_free_generic_local mi_free_generic_mt mi_free_block_local mi_free_block_mt ' +
      '_mi_free_delayed_block mi_free_size mi_free_aligned mi_free_size_aligned mi_free_block_delayed_mt ' +
      'mi_check_is_double_free mi_check_is_double_freex mi_list_contains mi_check_padding mi_verify_padding ' +
      'mi_usable_size _mi_usable_size _mi_padding_shrink mi_page_usable_aligned_size_of mi_stat_free ' +
      'mi_validate_ptr_page mi_page_decode_padding mi_page_usable_size_of' },
  { key: 'aligned', files: ['src/alloc-aligned.c'], sig: ['void*', 'mi_heap_t* heap, size_t size, size_t alignment'],
    words: 'aligned alignment align offset over-allocate power of two',
    doc: 'Aligned allocation: returns a block whose address is a multiple of the alignment.',
    names: 'mi_malloc_aligned mi_heap_malloc_aligned mi_malloc_aligned_at mi_heap_malloc_aligned_at ' +
      'mi_zalloc_aligned mi_zalloc_aligned_at mi_calloc_aligned mi_calloc_aligned_at mi_realloc_aligned ' +
      'mi_realloc_aligned_at mi_heap_realloc_aligned mi_heap_realloc_aligned_at mi_heap_malloc_zero_aligned_at ' +
      'mi_heap_malloc_zero_aligned_at_generic mi_heap_malloc_zero_aligned_at_overalloc mi_heap_realloc_zero_aligned_at ' +
      'mi_heap_rezalloc_aligned mi_heap_recalloc_aligned mi_malloc_is_naturally_aligned mi_heap_zalloc_aligned' },
  { key: 'posix', files: ['src/alloc-posix.c'], sig: ['int', 'void** p, size_t alignment, size_t size'],
    words: 'posix memalign valloc pvalloc aligned_alloc reallocarray usable size errno compatibility',
    doc: 'POSIX and platform compatibility entry points built on the core allocation functions.',
    names: 'mi_posix_memalign mi_memalign mi_valloc mi_pvalloc mi_aligned_alloc mi_reallocarray mi_reallocarr ' +
      'mi_malloc_size mi_malloc_usable_size mi_malloc_good_size mi_cfree mi__expand mi_wcsdup mi_mbsdup ' +
      'mi_dupenv_s mi_wdupenv_s mi_aligned_offset_recalloc mi_aligned_recalloc' },
  { key: 'strings', files: ['src/alloc.c'], sig: ['char*', 'mi_heap_t* heap, const char* s'],
    words: 'string duplicate strdup strndup realpath copy characters',
    doc: 'String helpers: duplicate a string into memory owned by the heap.',
    names: 'mi_strdup mi_heap_strdup mi_strndup mi_heap_strndup mi_realpath mi_heap_realpath' },
  { key: 'new', files: ['src/alloc.c', 'include/mimalloc-new-delete.h'], sig: ['void*', 'size_t size'],
    words: 'new delete operator c++ nothrow bad_alloc handler exception',
    doc: 'C++ operator new support: retries through the new handler and reports failure as bad_alloc.',
    names: 'mi_new mi_new_nothrow mi_new_aligned mi_new_aligned_nothrow mi_new_n mi_new_realloc mi_new_reallocn ' +
      'mi_try_new mi_try_new_handler mi_heap_alloc_new mi_heap_alloc_new_n mi_heap_try_new mi_get_new_handler ' +
      'mi_free_size_nothrow' },
  { key: 'heap', files: ['src/heap.c'], sig: ['mi_heap_t*', 'mi_heap_t* heap'],
    words: 'heap create new delete destroy collect default backing thread absorb reset',
    doc: 'Heap management: creation, collection and destruction of first-class heaps.',
    names: 'mi_heap_new mi_heap_new_in_arena mi_heap_new_ex mi_heap_delete mi_heap_destroy mi_heap_collect ' +
      'mi_heap_collect_ex mi_collect mi_heap_set_default mi_heap_get_default mi_heap_get_backing _mi_heap_init ' +
      'mi_heap_reset_pages mi_heap_free mi_heap_absorb _mi_heap_destroy_pages _mi_heap_unsafe_destroy_all ' +
      'mi_heap_is_valid _mi_heap_memid_is_suitable _mi_heap_collect_abandon _mi_heap_collect_retired ' +
      '_mi_heap_by_tag mi_heap_page_collect mi_heap_page_never_delayed_free mi_heap_page_is_valid' },
  { key: 'visit', files: ['src/heap.c'], sig: ['bool', 'const mi_heap_t* heap, mi_block_visit_fun* visitor, void* arg'],
    words: 'visit visitor blocks areas iterate enumerate owned contains check',
    doc: 'Heap inspection: visits the pages, areas and blocks of a heap.',
    names: 'mi_heap_visit_blocks mi_heap_visit_areas mi_heap_visit_pages mi_heap_area_visitor mi_heap_visit_areas_page ' +
      'mi_heap_area_visit_blocks _mi_heap_area_visit_blocks _mi_heap_area_init mi_heap_contains_block ' +
      'mi_heap_check_owned mi_check_owned mi_heap_page_check_owned mi_heap_of_block mi_heap_page_owned' },
  { key: 'page', files: ['src/page.c'], sig: ['void', 'mi_page_t* page, mi_page_queue_t* pq'],
    words: 'page fresh retire abandon full free list extend collect generic slow path find huge delayed',
    doc: 'Page life cycle: initialisation, free list extension, retirement and the generic allocation path.',
    names: '_mi_page_retire _mi_page_free _mi_page_abandon _mi_page_unfull mi_page_to_full mi_page_fresh ' +
      'mi_page_fresh_alloc mi_page_init mi_page_extend_free mi_page_free_list_extend mi_page_free_list_extend_secure ' +
      '_mi_page_free_collect _mi_page_thread_free_collect mi_page_local_free_collect mi_find_page mi_find_free_page ' +
      'mi_page_queue_find_free_ex _mi_malloc_generic _mi_page_use_delayed_free _mi_page_try_use_delayed_free ' +
      '_mi_heap_delayed_free_all _mi_heap_delayed_free_partial mi_large_huge_page_alloc mi_huge_page_alloc ' +
      '_mi_page_reclaim _mi_heap_collect_retired mi_page_is_valid_init _mi_page_is_valid mi_page_list_count ' +
      'mi_page_list_is_valid mi_page_block_at mi_register_deferred_free _mi_deferred_free' },
  { key: 'queue', files: ['src/page-queue.c'], sig: ['void', 'mi_heap_t* heap, mi_page_queue_t* queue, mi_page_t* page'],
    words: 'queue bin size class push remove enqueue append first update good size',
    doc: 'Page queues: one queue per size class, with the mapping from sizes to bins.',
    names: 'mi_page_queue_push mi_page_queue_remove mi_page_queue_enqueue_from mi_page_queue_enqueue_from_ex ' +
      '_mi_page_queue_append mi_page_queue_of mi_heap_page_queue_of mi_page_queue_is_huge mi_page_queue_is_full ' +
      'mi_page_queue_is_special _mi_bin mi_bin _mi_bin_size mi_good_size mi_heap_queue_first_update ' +
      'mi_page_queue_contains mi_heap_contains_queue mi_page_queue_move_to_front mi_page_queue_is_valid ' +
      'mi_page_queue_count' },
  { key: 'segment', files: ['src/segment.c'], sig: ['mi_segment_t*', 'mi_segment_t* segment, mi_segments_tld_t* tld'],
    words: 'segment slices span allocate os commit purge huge page coalesce split',
    doc: 'Segments: large regions obtained from the arenas and divided into spans of slices.',
    names: 'mi_segment_alloc mi_segment_free _mi_segment_page_alloc _mi_segment_page_free mi_segment_init ' +
      'mi_segment_os_alloc mi_segment_calculate_slices mi_segments_page_alloc mi_segment_page_clear ' +
      'mi_segment_span_allocate mi_segment_span_free mi_segment_span_free_coalesce mi_segment_slice_split ' +
      'mi_segments_page_find_and_allocate mi_segment_try_purge mi_segment_purge mi_segment_commit ' +
      'mi_segment_ensure_committed mi_segment_schedule_purge mi_segment_huge_page_alloc _mi_segment_huge_page_reset ' +
      '_mi_segment_huge_page_free mi_segment_is_valid mi_segment_info_size mi_segments_track_size ' +
      '_mi_segment_collect mi_segment_force_abandon' },
  { key: 'reclaim', files: ['src/segment.c'], sig: ['bool', 'mi_segment_t* segment, mi_heap_t* heap'],
    words: 'abandon abandoned reclaim thread terminated adopt segment pages',
    doc: 'Abandoned segments: pages of a terminated thread are reclaimed by another thread.',
    names: 'mi_segment_abandon _mi_segment_page_abandon mi_segment_reclaim _mi_abandoned_reclaim_all ' +
      'mi_segment_try_reclaim mi_segment_check_free _mi_abandoned_collect _mi_segment_attempt_reclaim ' +
      'mi_segment_reclaim_or_alloc mi_segment_get_reclaim_tries mi_segment_visit_page mi_segment_visit_pages' },
  { key: 'segmap', files: ['src/segment-map.c'], sig: ['bool', 'const mi_segment_t* segment'],
    words: 'segment map valid pointer region lookup heap address',
    doc: 'Segment map: records which addresses belong to segments, to validate pointers.',
    names: '_mi_segment_map_allocated_at _mi_segment_map_freed_at _mi_segment_of mi_is_valid_pointer ' +
      'mi_is_in_heap_region mi_segment_map_index_of _mi_segment_map_unsafe_destroy mi_is_valid_pointer_ex' },
  { key: 'span', files: ['src/segment.c'], sig: ['void', 'mi_span_queue_t* sq, mi_slice_t* slice'],
    words: 'span queue slice bin index first start end',
    doc: 'Span queues: free spans of slices kept per size bin inside a thread.',
    names: 'mi_span_queue_push mi_span_queue_delete mi_span_queue_for mi_slice_bin mi_slice_bin8 ' +
      '_mi_segment_page_start _mi_segment_page_start_from_slice mi_slice_first mi_slice_index ' +
      'mi_segment_slices_end mi_slice_start mi_span_queue_contains mi_slice_is_used' },
  { key: 'arena', files: ['src/arena.c'], sig: ['void*', 'size_t size, size_t alignment, mi_memid_t* memid'],
    words: 'arena reserve os memory manage huge pages numa suitable meta static',
    doc: 'Arenas: large reserved areas of memory from which segments are allocated.',
    names: '_mi_arena_alloc _mi_arena_alloc_aligned mi_arena_try_alloc mi_arena_try_alloc_at mi_arena_try_alloc_at_id ' +
      'mi_arenas_try_alloc mi_arena_reserve _mi_arena_free mi_reserve_os_memory mi_reserve_os_memory_ex ' +
      'mi_manage_os_memory mi_manage_os_memory_ex mi_arena_add mi_reserve_huge_os_pages_at ' +
      'mi_reserve_huge_os_pages_at_ex mi_reserve_huge_os_pages_interleave mi_reserve_huge_os_pages ' +
      'mi_arena_id_create _mi_arena_id_none _mi_arena_memid_is_suitable mi_arena_static_zalloc ' +
      '_mi_arena_meta_zalloc _mi_arena_meta_free mi_arena_area mi_arena_block_size mi_arena_size mi_arena_id_index ' +
      'mi_memid_create_arena mi_arena_memid_indices _mi_arena_contains mi_debug_show_arenas' },
  { key: 'purge', files: ['src/arena.c'], sig: ['void', 'mi_arena_t* arena, size_t bitmap_idx, size_t blocks'],
    words: 'purge decommit reset schedule delay expire collect destroy',
    doc: 'Arena purging: memory that stays unused is decommitted after a delay.',
    names: 'mi_arena_purge mi_arena_schedule_purge mi_arena_purge_range mi_arena_try_purge mi_arenas_try_purge ' +
      '_mi_arenas_collect mi_arena_purge_delay mi_arenas_unsafe_destroy _mi_arena_unsafe_destroy_all' },
  { key: 'abandoned', files: ['src/arena-abandon.c'], sig: ['bool', 'mi_segment_t* segment'],
    words: 'arena abandoned segment mark clear cursor count visit',
    doc: 'Abandoned segments inside arenas: marked in a bitmap and found again through a cursor.',
    names: '_mi_arena_segment_mark_abandoned _mi_arena_segment_clear_abandoned _mi_arena_field_cursor_init ' +
      '_mi_arena_field_cursor_done _mi_arena_segment_clear_abandoned_next mi_arena_segment_os_mark_abandoned ' +
      'mi_arena_segment_os_clear_abandoned _mi_arena_abandoned_count mi_abandoned_visit_blocks ' +
      '_mi_arena_segment_clear_abandoned_next_field _mi_arena_segment_clear_abandoned_next_list' },
  { key: 'bitmap', files: ['src/bitmap.c', 'src/bitmap.h'], sig: ['bool', 'mi_bitmap_t bitmap, size_t count, mi_bitmap_index_t* idx'],
    words: 'bitmap bits claim unclaim find field mask across atomic',
    doc: 'Concurrent bitmaps: sequences of bits are claimed and released atomically.',
    names: '_mi_bitmap_try_find_claim_field _mi_bitmap_try_find_from_claim _mi_bitmap_try_find_from_claim_pred ' +
      '_mi_bitmap_unclaim _mi_bitmap_try_claim _mi_bitmap_claim _mi_bitmap_is_claimed _mi_bitmap_is_any_claimed ' +
      '_mi_bitmap_try_find_from_claim_across _mi_bitmap_unclaim_across _mi_bitmap_claim_across ' +
      '_mi_bitmap_is_claimed_across _mi_bitmap_is_any_claimed_across mi_bitmap_mask_across mi_bitmap_mask_ ' +
      'mi_bitmap_index_create mi_bitmap_index_create_ex mi_bitmap_index_field mi_bitmap_index_bit_in_field ' +
      'mi_bitmap_index_bit mi_bitmap_is_claimedx mi_bitmap_try_find_claim_field_across mi_bitmap_is_claimedx_across' },
  { key: 'os', files: ['src/os.c'], sig: ['void*', 'size_t size, mi_memid_t* memid'],
    words: 'os operating system virtual memory commit decommit reset protect page size overcommit hint',
    doc: 'Operating system memory: allocation, commit and protection of virtual memory.',
    names: '_mi_os_alloc _mi_os_free _mi_os_free_ex _mi_os_alloc_aligned _mi_os_alloc_aligned_at_offset ' +
      '_mi_os_commit _mi_os_decommit _mi_os_reset _mi_os_purge _mi_os_purge_ex _mi_os_protect _mi_os_unprotect ' +
      'mi_os_prim_alloc mi_os_prim_free mi_os_prim_alloc_aligned mi_os_page_align_area_conservative ' +
      'mi_os_page_align_areax _mi_os_page_size _mi_os_large_page_size _mi_os_good_alloc_size ' +
      '_mi_os_has_overcommit _mi_os_has_virtual_reserve _mi_os_use_large_page mi_os_get_aligned_hint ' +
      '_mi_os_get_aligned_hint _mi_os_init mi_os_decommit_ex mi_os_protectx _mi_os_secure_guard_page_size' },
  { key: 'huge', files: ['src/os.c'], sig: ['void*', 'size_t pages, int numa_node, mi_msecs_t max_msecs'],
    words: 'huge os pages numa node gigabyte claim count',
    doc: 'Huge OS pages and NUMA: reserves gigabyte pages, optionally on a given node.',
    names: '_mi_os_alloc_huge_os_pages mi_os_claim_huge_pages _mi_os_numa_node _mi_os_numa_node_count ' +
      '_mi_os_numa_node_get _mi_os_numa_node_count_get mi_os_free_huge_os_pages' },
  { key: 'unix', files: ['src/prim/unix/prim.c'], sig: ['int', 'void* addr, size_t size'],
    words: 'unix linux mmap munmap madvise mprotect overcommit pthread getenv clock',
    doc: 'Primitives for Unix-like systems, implemented with mmap, madvise and mprotect.',
    names: 'unix_mmap unix_mmap_prim unix_mmap_prim_aligned unix_madvise unix_detect_overcommit unix_mmap_fd ' +
      'unix_mprotect_hint mi_prim_open mi_prim_read mi_prim_close mi_prim_access mi_prim_mbind mi_pthread_done ' +
      'mi_prim_munmap mi_get_clock_now mi_prim_thread_key_init' },
  { key: 'windows', files: ['src/prim/windows/prim.c', 'src/prim/windows/etw.h'], sig: ['void*', 'void* addr, size_t size, DWORD flags'],
    words: 'windows virtualalloc virtualfree large pages fiber fls dll process attach etw',
    doc: 'Primitives for Windows, implemented with VirtualAlloc and fiber local storage.',
    names: 'win_virtual_alloc win_virtual_alloc_prim win_virtual_alloc_prim_once win_enable_large_os_pages ' +
      'win_is_out_of_memory_error mi_win_main mi_fls_done mi_process_attach mi_win_tls_init mi_win_tls_done ' +
      'filetime_msecs mi_to_msecs mi_win_get_numa_node win_query_performance_clock' },
  { key: 'prim', files: ['src/prim/prim.c', 'src/prim/osx/prim.c', 'src/prim/wasi/prim.c', 'src/prim/emscripten/prim.c',
                         'include/mimalloc/prim.h'],
    sig: ['int', 'void* start, size_t size'],
    words: 'primitive portable interface platform wasi emscripten macos alloc free commit reset clock random',
    doc: 'The portable primitive interface that every platform layer implements.',
    names: '_mi_prim_alloc _mi_prim_free _mi_prim_commit _mi_prim_decommit _mi_prim_reset _mi_prim_protect ' +
      '_mi_prim_mem_init _mi_prim_alloc_huge_os_pages _mi_prim_numa_node _mi_prim_numa_node_count ' +
      '_mi_prim_clock_now _mi_prim_process_info _mi_prim_out_stderr _mi_prim_getenv _mi_prim_random_buf ' +
      '_mi_prim_thread_init_auto_done _mi_prim_thread_done_auto_done _mi_prim_thread_associate_default_heap ' +
      'mi_prim_get_default_heap _mi_prim_thread_id mi_prim_tls_slot mi_prim_tls_slot_set' },
  { key: 'options', files: ['src/options.c'], sig: ['long', 'mi_option_t option'],
    words: 'option options setting environment variable enable disable default clamp',
    doc: 'Run-time options: read from the environment on first use and cached.',
    names: 'mi_option_get mi_option_get_clamp mi_option_get_size mi_option_set mi_option_set_default ' +
      'mi_option_is_enabled mi_option_set_enabled mi_option_set_enabled_default mi_option_enable mi_option_disable ' +
      'mi_option_init _mi_options_init mi_option_has_size_in_kib mi_getenv mi_version' },
  { key: 'messages', files: ['src/options.c'], sig: ['void', 'const char* fmt, va_list args'],
    words: 'message output error warning verbose trace print stderr buffer register assert',
    doc: 'Messages: warnings, errors and verbose output, buffered until the output is registered.',
    names: '_mi_fputs _mi_fprintf mi_vfprintf mi_vfprintf_thread _mi_trace_message _mi_verbose_message ' +
      '_mi_warning_message _mi_error_message mi_show_error_message mi_register_output mi_register_error ' +
      'mi_out_buf mi_out_buf_flush mi_out_buf_stderr mi_out_stderr mi_error_default _mi_assert_fail ' +
      'mi_add_stderr_output mi_recurse_enter mi_recurse_exit mi_recurse_enter_prim mi_recurse_exit_prim' },
  { key: 'stats', files: ['src/stats.c'], sig: ['void', 'mi_stat_count_t* stat, int64_t amount'],
    words: 'statistics stats counter increase decrease merge print amount peak process info clock elapsed',
    doc: 'Statistics: per-thread counters that are merged into the process totals and printed.',
    names: 'mi_stat_update _mi_stat_increase _mi_stat_decrease _mi_stat_counter_increase mi_stat_add ' +
      'mi_stat_counter_add mi_stats_add mi_stats_reset mi_stats_merge _mi_stats_done mi_stats_print ' +
      'mi_stats_print_out mi_stat_print mi_stat_print_ex mi_stat_counter_print mi_stat_counter_print_avg ' +
      'mi_print_amount mi_print_count mi_printf_amount mi_print_header mi_stats_merge_from ' +
      'mi_thread_stats_print_out mi_process_info _mi_clock_now _mi_clock_start _mi_clock_end mi_stats_get_default ' +
      'mi_is_in_main mi_stat_peak_print mi_buffered_flush mi_buffered_out mi_print_line' },
  { key: 'init', files: ['src/init.c'], sig: ['void', 'void'],
    words: 'init initialise process thread startup shutdown done tld main heap load cpu features',
    doc: 'Initialisation: set-up and tear-down of the process, of threads and of their heaps.',
    names: 'mi_process_init mi_process_done _mi_process_load mi_process_setup_auto_thread_done mi_thread_init ' +
      'mi_thread_done _mi_thread_done _mi_thread_heap_init _mi_thread_heap_done mi_heap_main_init _mi_heap_main_get ' +
      'mi_tld_init mi_thread_data_zalloc mi_thread_data_free _mi_thread_data_collect _mi_is_main_thread ' +
      '_mi_current_thread_count _mi_heap_set_default_direct mi_detect_cpu_features mi_allocator_init ' +
      'mi_allocator_done mi_is_redirected _mi_preloading mi_process_load mi_init_done' },
  { key: 'random', files: ['src/random.c'], sig: ['void', 'mi_random_ctx_t* ctx'],
    words: 'random chacha20 entropy seed split next weak cookie secure',
    doc: 'Random numbers: a ChaCha20 generator used for the secure free list encoding.',
    names: '_mi_random_init _mi_random_init_weak _mi_random_reinit_if_weak _mi_random_split _mi_random_next ' +
      '_mi_heap_random_next _mi_os_random_weak chacha_block chacha_init chacha_split chacha_next32 qround rotl ' +
      'read32 mi_random_is_initialized mi_random_init_ex' },
  { key: 'libc', files: ['src/libc.c'], sig: ['size_t', 'char* dest, const char* src, size_t dest_size'],
    words: 'libc string copy length compare format printf memcpy memzero popcount without dependency',
    doc: 'Small replacements for C library functions that are safe to use during initialisation.',
    names: '_mi_toupper _mi_strnicmp _mi_strlcpy _mi_strlcat _mi_strlen _mi_strnlen _mi_vsnprintf _mi_snprintf ' +
      'mi_outc mi_outs mi_out_fill mi_out_alignright mi_out_num _mi_memcpy _mi_memzero _mi_memcpy_aligned ' +
      '_mi_memzero_aligned mi_byte_sum64 mi_popcount_generic64 mi_popcount_generic32 _mi_popcount_generic' },
  { key: 'atomic', files: ['include/mimalloc/atomic.h'], sig: ['uintptr_t', '_Atomic(uintptr_t)* p, uintptr_t x'],
    words: 'atomic load store compare exchange cas increment yield once lock guard relaxed acquire release',
    doc: 'Atomic operations and light-weight locks with a uniform interface over C11, C++ and MSVC.',
    names: 'mi_atomic_load_relaxed mi_atomic_load_acquire mi_atomic_store_release mi_atomic_store_relaxed ' +
      'mi_atomic_cas_weak mi_atomic_cas_strong mi_atomic_cas_weak_release mi_atomic_cas_strong_acq_rel ' +
      'mi_atomic_exchange mi_atomic_exchange_acq_rel mi_atomic_add mi_atomic_sub mi_atomic_increment ' +
      'mi_atomic_decrement mi_atomic_addi mi_atomic_subi mi_atomic_maxi64_relaxed mi_atomic_yield mi_atomic_once ' +
      'mi_atomic_guard mi_lock_acquire mi_lock_release mi_lock_try_acquire mi_lock_init mi_lock_done' },
  { key: 'internal', files: ['include/mimalloc/internal.h'], sig: ['bool', 'const mi_page_t* page'],
    words: 'inline helper pointer segment page of block next encode decode align up down overflow thread id',
    doc: 'Inline helpers shared by the implementation: pointer arithmetic, page and block accessors.',
    names: '_mi_ptr_segment _mi_ptr_page _mi_segment_page_of mi_page_block_size mi_page_is_huge mi_page_all_free ' +
      'mi_page_has_any_available mi_page_immediate_available mi_page_mostly_used mi_page_heap mi_page_set_heap ' +
      'mi_block_next mi_block_set_next mi_block_nextx mi_block_set_nextx mi_ptr_encode mi_ptr_decode ' +
      'mi_is_in_same_page mi_is_in_same_segment _mi_align_up _mi_align_down _mi_is_power_of_two _mi_is_aligned ' +
      'mi_mul_overflow mi_count_size_overflow _mi_wsize_from_size _mi_divide_up mi_clz mi_ctz mi_bsr _mi_thread_id ' +
      'mi_heap_is_initialized mi_heap_is_backing mi_heap_is_default _mi_heap_get_free_small_page mi_tf_make ' +
      'mi_tf_delayed mi_tf_block mi_tf_set_delayed mi_page_thread_free mi_page_thread_free_flag mi_page_is_in_full ' +
      'mi_page_set_in_full mi_page_has_aligned mi_page_set_has_aligned mi_mem_is_zero _mi_memid_create ' +
      '_mi_memid_none _mi_memid_create_os mi_slice_to_page mi_page_to_slice mi_page_start mi_page_area' },
  { key: 'types', files: ['include/mimalloc/types.h'], kind: 'type', sig: ['', ''],
    words: 'type struct definition page heap segment block thread local data statistics memid constants',
    doc: 'Data structures of the allocator.',
    names: 'mi_page_t mi_heap_t mi_segment_t mi_block_t mi_tld_t mi_stats_t mi_memid_t mi_page_queue_t mi_slice_t ' +
      'mi_delayed_t mi_page_kind_t mi_segment_kind_t mi_memkind_t mi_stat_count_t mi_stat_counter_t mi_os_tld_t ' +
      'mi_segments_tld_t mi_page_flags_t mi_thread_free_t mi_padding_t mi_span_queue_t mi_commit_mask_t ' +
      'mi_arena_id_t mi_msecs_t mi_encoded_t mi_threadid_t mi_random_ctx_t mi_subproc_t mi_memid_os_info_t ' +
      'mi_memid_arena_info_t mi_arena_t mi_bitmap_field_t mi_bitmap_index_t mi_option_desc_t mi_init_t' },
  { key: 'macros', files: ['include/mimalloc/types.h', 'include/mimalloc.h', 'include/mimalloc/track.h'], kind: 'macro',
    sig: ['', ''],
    words: 'macro constant define size limit shift debug secure padding attribute export track',
    doc: 'Compile-time constants and attribute macros.',
    names: 'MI_SEGMENT_SIZE MI_SEGMENT_SHIFT MI_SEGMENT_SLICE_SIZE MI_SEGMENT_SLICE_SHIFT MI_SLICES_PER_SEGMENT ' +
      'MI_SMALL_OBJ_SIZE_MAX MI_MEDIUM_OBJ_SIZE_MAX MI_LARGE_OBJ_SIZE_MAX MI_SMALL_SIZE_MAX MI_SMALL_WSIZE_MAX ' +
      'MI_BIN_HUGE MI_BIN_FULL MI_INTPTR_SIZE MI_INTPTR_SHIFT MI_SIZE_SIZE MI_PADDING MI_PADDING_SIZE MI_DEBUG ' +
      'MI_SECURE MI_STAT MI_MAX_ALIGN_SIZE MI_BLOCK_ALIGNMENT_MAX MI_MAX_ARENAS MI_ARENA_BLOCK_SIZE ' +
      'MI_COMMIT_MASK_BITS MI_MINIMAL_COMMIT_SIZE MI_KiB MI_MiB MI_GiB mi_decl_nodiscard mi_decl_export ' +
      'mi_decl_noinline mi_decl_thread mi_decl_cache_align mi_attr_malloc mi_attr_alloc_size mi_attr_alloc_size2 ' +
      'mi_attr_alloc_align mi_attr_noexcept mi_likely mi_unlikely mi_assert mi_assert_internal ' +
      'mi_assert_expensive mi_track_malloc mi_track_free mi_track_resize mi_track_mem_defined ' +
      'mi_track_mem_undefined mi_track_mem_noaccess mi_malloc_tp mi_zalloc_tp mi_calloc_tp mi_heap_malloc_tp' },
  { key: 'mask', files: ['src/segment.c'], sig: ['bool', 'const mi_commit_mask_t* commit, const mi_commit_mask_t* cm'],
    words: 'commit mask bits set clear intersect run committed size',
    doc: 'Commit masks: which parts of a segment are backed by committed memory.',
    names: 'mi_commit_mask_create mi_commit_mask_create_empty mi_commit_mask_create_full mi_commit_mask_clear ' +
      'mi_commit_mask_set mi_commit_mask_all_set mi_commit_mask_any_set mi_commit_mask_create_intersect ' +
      '_mi_commit_mask_committed_size _mi_commit_mask_next_run mi_commit_mask_is_empty mi_commit_mask_is_full ' +
      'mi_segment_commit_mask' },
  { key: 'override', files: ['src/alloc-override.c', 'src/prim/osx/alloc-override-zone.c', 'include/mimalloc-override.h',
                             'src/static.c'],
    sig: ['void*', 'malloc_zone_t* zone, size_t size'],
    words: 'override interpose malloc free zone macos forward symbol replace standard',
    doc: 'Overriding the standard allocator: forwarding of malloc and free, and the macOS malloc zone.',
    names: 'mi_forward mi_forward0 zone_malloc zone_free zone_size zone_calloc zone_realloc zone_valloc ' +
      'zone_destroy zone_batch_malloc zone_batch_free zone_memalign zone_free_definite_size zone_pressure_relief ' +
      'intro_enumerator intro_good_size intro_check intro_print intro_log intro_force_lock intro_force_unlock ' +
      'intro_statistics intro_zone_locked _mi_macos_override_malloc mi_get_default_zone' },
  { key: 'tests', files: ['test/test-api.c', 'test/test-api-fill.c', 'test/test-stress.c', 'test/test-wrong.c',
                          'test/main-override.c', 'test/main-override-static.c', 'test/main-override.cpp',
                          'test/main-override-dep.cpp', 'test/main-override-dep.h', 'test/main.c', 'test/testhelper.h',
                          'test/test-leak.c', 'test/test-wasm.c', 'bin/minject-test.c', 'contrib/vcpkg/usage-test.c'],
    sig: ['bool', 'void'],
    words: 'test check stress thread items overflow double free corrupt zero init expect result',
    doc: 'Tests of the public interface and stress tests with concurrent allocation and freeing.',
    names: 'test_heap1 test_heap2 test_large stress alloc_items free_items test_stress run_os_threads thread_entry ' +
      'check_result check_zero_init check_nonzero_init check_debug_fill_uninit check_debug_fill_freed ' +
      'test_process_info test_reserved test_aligned1 test_aligned2 tsan_numa_test various_tests padding_shrink ' +
      'double_free1 double_free2 corrupt_free block_overflow1 block_overflow2 heap_late_free heap_thread_free_large ' +
      'test_mt_shutdown fail_aslr strdup_test mem_is_zero test_stl_allocator1 test_stl_allocator2 ' +
      'heap_no_delete heap_thread_free_huge test_nomem1 test_manage_os_memory test_std_string test_large_pages ' +
      'test_heap_walk test_canary_leak test_leak pick chance transfer atomic_exchange_ptr' },
];

// Files that exist in the corpus besides those named by the topics. They hold
// residual regions only.
const EXTRA_FILES = [
  'include/mimalloc-stats.h', 'src/prim/readme.md.h', 'src/prim/osx/alloc-override-zone.h', 'src/alloc-override-osx.c',
  'src/page-map.c', 'cmake/mimalloc-config-version.h', 'doc/mimalloc-doc.h',
];

const SUFFIXES = ['_ex', '_at', '_checked', '_slow', '_fast', '_locked', '_init', '_done', '_size', '_count', '_is_valid',
                  '_debug', '_try', '_retry', '_internal', '_aligned', '_noinline', '_mt', '_local', '_unsafe', '_generic',
                  '_prim', '_with', '_first', '_next', '_all', '_any', '_range', '_once', '_stat'];

// Functions that many others call; they receive a large number of callers,
// which is what the hub damping of the tracer is there for.
const HUBS = ['mi_assert_internal', '_mi_memzero', 'mi_atomic_load_relaxed', '_mi_stat_increase', '_mi_warning_message',
              '_mi_align_up', 'mi_page_block_size', '_mi_ptr_segment', 'mi_option_is_enabled', '_mi_error_message'];

export const TOTAL_UNITS = 1510;

// ---- hand-written units -------------------------------------------------------

const WRITTEN = {
  mi_malloc: {
    callees: ['mi_heap_malloc', 'mi_prim_get_default_heap'],
    text: [
      '// Allocate `size` bytes from the default heap of the calling thread.',
      '// Returns NULL when out of memory; the block is not initialised.',
      'mi_decl_nodiscard mi_decl_restrict void* mi_malloc(size_t size) mi_attr_noexcept {',
      '  return mi_heap_malloc(mi_prim_get_default_heap(), size);',
      '}',
    ],
  },
  mi_heap_malloc: {
    callees: ['_mi_heap_malloc_zero'],
    text: [
      '// Allocate `size` bytes from `heap`. The heap must belong to the calling',
      '// thread; blocks may be freed from any thread.',
      'mi_decl_nodiscard extern inline mi_decl_restrict void* mi_heap_malloc(mi_heap_t* heap, size_t size) mi_attr_noexcept {',
      '  return _mi_heap_malloc_zero(heap, size, false);',
      '}',
    ],
  },
  _mi_heap_malloc_zero: {
    callees: ['_mi_heap_malloc_zero_ex'],
    text: [
      '// Allocate with optional zero initialisation and no huge alignment.',
      'extern inline void* _mi_heap_malloc_zero(mi_heap_t* heap, size_t size, bool zero) mi_attr_noexcept {',
      '  return _mi_heap_malloc_zero_ex(heap, size, zero, 0);',
      '}',
    ],
  },
  _mi_heap_malloc_zero_ex: {
    callees: ['mi_heap_malloc_small_zero', '_mi_malloc_generic', 'mi_assert_internal'],
    text: [
      '// The main allocation function: small sizes take the fast path through the',
      '// direct page table of the heap, everything else goes to the generic path.',
      'extern inline void* _mi_heap_malloc_zero_ex(mi_heap_t* heap, size_t size, bool zero, size_t huge_alignment) mi_attr_noexcept {',
      '  if mi_likely(size <= MI_SMALL_SIZE_MAX) {',
      '    mi_assert_internal(huge_alignment == 0);',
      '    return mi_heap_malloc_small_zero(heap, size, zero);',
      '  }',
      '  else {',
      '    mi_assert(heap != NULL);',
      '    mi_assert(heap->thread_id == 0 || heap->thread_id == _mi_thread_id());  // heaps are thread local',
      '    void* const p = _mi_malloc_generic(heap, size + MI_PADDING_SIZE, zero, huge_alignment);',
      '    mi_track_malloc(p, size, zero);',
      '    return p;',
      '  }',
      '}',
    ],
  },
  mi_heap_malloc_small_zero: {
    callees: ['_mi_heap_get_free_small_page', '_mi_page_malloc_zero', 'mi_assert'],
    text: [
      '// Fast path for small sizes: the page is found by indexing `pages_free_direct`.',
      'static inline mi_decl_restrict void* mi_heap_malloc_small_zero(mi_heap_t* heap, size_t size, bool zero) mi_attr_noexcept {',
      '  mi_assert(heap != NULL);',
      '  mi_assert(size <= MI_SMALL_SIZE_MAX);',
      '  mi_page_t* page = _mi_heap_get_free_small_page(heap, size + MI_PADDING_SIZE);',
      '  void* const p = _mi_page_malloc_zero(heap, page, size + MI_PADDING_SIZE, zero);',
      '  mi_track_malloc(p, size, zero);',
      '  return p;',
      '}',
    ],
  },
  _mi_page_malloc_zero: {
    callees: ['_mi_malloc_generic', 'mi_block_next', 'mi_page_block_size', '_mi_memzero_aligned', 'mi_assert_internal'],
    text: [
      '// Pop a block from the free list of `page`. When the list is empty the',
      '// generic path collects freed blocks or finds another page.',
      'extern inline void* _mi_page_malloc_zero(mi_heap_t* heap, mi_page_t* page, size_t size, bool zero) mi_attr_noexcept {',
      '  mi_assert_internal(page->block_size == 0 || mi_page_block_size(page) >= size);',
      '  mi_block_t* const block = page->free;',
      '  if mi_unlikely(block == NULL) {',
      '    return _mi_malloc_generic(heap, size, zero, 0);',
      '  }',
      '  mi_assert_internal(block != NULL && _mi_ptr_page(block) == page);',
      '  // pop from the free list',
      '  page->free = mi_block_next(page, block);',
      '  page->used++;',
      '  mi_assert_internal(page->free == NULL || _mi_ptr_page(page->free) == page);',
      '  if mi_unlikely(zero) {',
      '    mi_assert_internal(page->block_size != 0);',
      '    _mi_memzero_aligned(block, page->block_size - MI_PADDING_SIZE);',
      '  }',
      '  return block;',
      '}',
    ],
  },
  _mi_malloc_generic: {
    callees: ['mi_thread_init', '_mi_deferred_free', '_mi_heap_delayed_free_partial', 'mi_find_page',
              '_mi_page_malloc_zero', '_mi_error_message', 'mi_heap_collect', 'mi_heap_is_initialized'],
    text: [
      '// Generic allocation routine for when the fast path fails. It is also the',
      '// place where deferred work is done: delayed frees of other threads are',
      '// collected here, so that the fast path needs no synchronisation.',
      'void* _mi_malloc_generic(mi_heap_t* heap, size_t size, bool zero, size_t huge_alignment) mi_attr_noexcept {',
      '  mi_assert_internal(heap != NULL);',
      '  // initialise if necessary',
      '  if mi_unlikely(!mi_heap_is_initialized(heap)) {',
      '    heap = mi_heap_get_default();  // calls mi_thread_init',
      '    if mi_unlikely(!mi_heap_is_initialized(heap)) { return NULL; }',
      '  }',
      '  mi_assert_internal(mi_heap_is_initialized(heap));',
      '',
      '  // call potential deferred free routines, and free delayed blocks from other threads',
      '  _mi_deferred_free(heap, false);',
      '  _mi_heap_delayed_free_partial(heap);',
      '',
      '  // find (or allocate) a page of the right size',
      '  mi_page_t* page = mi_find_page(heap, size, huge_alignment);',
      '  if mi_unlikely(page == NULL) {  // first time out of memory, try to collect and retry once',
      '    mi_heap_collect(heap, true /* force */);',
      '    page = mi_find_page(heap, size, huge_alignment);',
      '  }',
      '  if mi_unlikely(page == NULL) {  // out of memory',
      '    const size_t req_size = size - MI_PADDING_SIZE;',
      '    _mi_error_message(ENOMEM, "unable to allocate memory (%zu bytes)\\n", req_size);',
      '    return NULL;',
      '  }',
      '  mi_assert_internal(mi_page_immediate_available(page));',
      '  mi_assert_internal(mi_page_block_size(page) >= size);',
      '',
      '  // and try again, this time succeeding (the page has free blocks now)',
      '  return _mi_page_malloc_zero(heap, page, size, zero);',
      '}',
    ],
  },
  mi_free: {
    callees: ['_mi_ptr_segment', '_mi_segment_page_of', 'mi_free_block_local', 'mi_free_generic_local',
              'mi_free_generic_mt', '_mi_thread_id', 'mi_page_has_aligned'],
    text: [
      '// Free a block. The common case is a block that the calling thread',
      '// allocated itself in a page without special flags: it is pushed on the',
      '// local free list of the page, without any atomic operation.',
      'void mi_free(void* p) mi_attr_noexcept {',
      '  mi_segment_t* const segment = mi_checked_ptr_segment(p, "mi_free");',
      '  if mi_unlikely(segment == NULL) return;',
      '',
      '  const bool is_local = (_mi_prim_thread_id() == mi_atomic_load_relaxed(&segment->thread_id));',
      '  mi_page_t* const page = _mi_segment_page_of(segment, p);',
      '  if mi_likely(is_local) {                      // thread-local free?',
      '    if mi_likely(page->flags.full_aligned == 0) {',
      '      // thread-local, aligned, and not a full page',
      '      mi_block_t* const block = (mi_block_t*)p;',
      '      mi_free_block_local(page, block, true /* track stats */, false /* no need to check if the page is full */);',
      '    }',
      '    else {',
      '      // page is full or contains (inner) aligned blocks; use generic path',
      '      mi_free_generic_local(page, segment, p);',
      '    }',
      '  }',
      '  else {',
      '    // not thread-local; use generic path',
      '    mi_free_generic_mt(page, segment, p);',
      '  }',
      '}',
    ],
  },
  mi_find_page: {
    callees: ['mi_large_huge_page_alloc', 'mi_find_free_page', '_mi_error_message', 'mi_assert_internal'],
    text: [
      '// Find a page with free blocks for an allocation of `size` bytes. Large',
      '// and huge sizes get a page of their own; other sizes use the page queue of',
      '// their size class.',
      'static mi_page_t* mi_find_page(mi_heap_t* heap, size_t size, size_t huge_alignment) mi_attr_noexcept {',
      '  // huge allocation?',
      '  const size_t req_size = size - MI_PADDING_SIZE;  // correct for padding_size in case of an overflow on `size`',
      '  if mi_unlikely(req_size > (MI_MEDIUM_OBJ_SIZE_MAX - MI_PADDING_SIZE) || huge_alignment > 0) {',
      '    if mi_unlikely(req_size > MI_MAX_ALLOC_SIZE) {',
      '      _mi_error_message(EOVERFLOW, "allocation request is too large (%zu bytes)\\n", req_size);',
      '      return NULL;',
      '    }',
      '    else {',
      '      return mi_large_huge_page_alloc(heap, size, huge_alignment);',
      '    }',
      '  }',
      '  else {',
      '    // otherwise find a page with free blocks in our size segregated queues',
      '    mi_assert_internal(size >= MI_PADDING_SIZE);',
      '    return mi_find_free_page(heap, size);',
      '  }',
      '}',
    ],
  },
};

// ---- generation ---------------------------------------------------------------

function subwords(name) {
  return name.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w !== '' && w !== 'mi');
}

function firstParam(params) {
  const first = params.split(',')[0].trim();
  const match = /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(first);
  return match && first !== 'void' ? match[1] : '';
}

function argumentsOf(params) {
  if (params.trim() === 'void' || params.trim() === '') return '';
  return params.split(',').map((p) => /([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(p.trim())[1]).join(', ');
}

function functionText(unit, topic, calleeNames, random) {
  const [ret, params] = topic.sig;
  const lines = [];
  const purpose = subwords(unit.name).join(' ');
  lines.push(`// ${topic.doc}`);
  lines.push(`// ${unit.name}: ${purpose}.`);
  const storage = unit.name.startsWith('_mi_') ? '' : random.next() < 0.4 ? 'static ' : '';
  const inline = random.next() < 0.3 ? 'inline ' : '';
  lines.push(`${storage}${inline}${ret} ${unit.name}(${params}) {`);
  const subject = firstParam(params);
  if (subject !== '' && /\*/.test(params.split(',')[0])) lines.push(`  mi_assert_internal(${subject} != NULL);`);
  const args = argumentsOf(params);
  calleeNames.forEach((callee, index) => {
    if (callee.startsWith('mi_assert')) return;
    const form = random.int(4);
    if (form === 0) lines.push(`  if (!${callee}(${args})) {`, `    return${ret === 'void' ? '' : ret === 'bool' ? ' false' : ret.endsWith('*') ? ' NULL' : ' 0'};`, '  }');
    else if (form === 1) lines.push(`  const size_t n${index} = (size_t)${callee}(${args});`, `  MI_UNUSED(n${index});`);
    else lines.push(`  ${callee}(${args});`);
  });
  if (ret === 'void') lines.push('}');
  else if (ret === 'bool') lines.push('  return true;', '}');
  else if (ret.endsWith('*')) lines.push(`  return ${subject !== '' && /\*/.test(params.split(',')[0]) ? `(${ret})${subject}` : 'NULL'};`, '}');
  else lines.push('  return 0;', '}');
  return lines;
}

function typeText(unit, topic, random) {
  const base = unit.name.replace(/_t$/, '');
  const lines = [`// ${topic.doc}`, `// ${unit.name}: ${subwords(base).join(' ')}.`, `typedef struct ${base}_s {`];
  const fields = ['size_t size;', 'uint32_t count;', 'mi_block_t* free;', 'uintptr_t thread_id;', 'bool is_zero;',
                  'uint8_t flags;', 'struct ' + base + '_s* next;', 'struct ' + base + '_s* prev;', 'size_t capacity;',
                  'mi_memid_t memid;', '_Atomic(size_t) used;'];
  const n = 3 + random.int(6);
  for (let i = 0; i < n; ++i) lines.push('  ' + fields[(i * 3 + random.int(3)) % fields.length]);
  lines.push(`} ${unit.name};`);
  return lines;
}

function macroText(unit, topic, random) {
  const lines = [`// ${topic.doc}`];
  if (/^[A-Z0-9_]+$/.test(unit.name)) {
    lines.push(`#define ${unit.name}  (${1 << random.int(12)}${random.next() < 0.5 ? '*MI_KiB' : 'UL'})`);
  } else {
    lines.push(`#if defined(__GNUC__) || defined(__clang__)`, `#define ${unit.name}(x)  __builtin_expect(!!(x), 1)`,
               '#else', `#define ${unit.name}(x)  (x)`, '#endif');
  }
  return lines;
}

function regionText(path, index) {
  if (index === 0) {
    return ['/* ----------------------------------------------------------------------------',
            'Copyright (c) 2018-2024, Microsoft Research, Daan Leijen',
            'This is free software; you can redistribute it and/or modify it under the',
            'terms of the MIT license. A copy of the license can be found in the file',
            '"LICENSE" at the root of this distribution.',
            '-----------------------------------------------------------------------------*/',
            '#include "mimalloc.h"', '#include "mimalloc/internal.h"', '#include "mimalloc/atomic.h"', '',
            '#include <string.h>  // memset, strlen'];
  }
  return ['', '// ------------------------------------------------------', `// ${path}: section ${index + 1}`,
          '// ------------------------------------------------------', ''];
}

// Builds the corpus. Returns { units, groups, files, byName, callers }.
export function buildCorpus(seed = 20260929) {
  const random = makeRandom(seed);
  const files = [];
  const fileSet = new Set();
  const addFile = (path) => {
    if (!fileSet.has(path)) {
      fileSet.add(path);
      files.push(path);
    }
  };
  for (const topic of TOPICS) topic.files.forEach(addFile);
  EXTRA_FILES.forEach(addFile);

  // Names per topic: the listed ones, then generated variants up to the target.
  const regions = files.length * 2;
  const used = new Set();
  const listed = TOPICS.map((t) => {
    const names = t.names.split(' ').filter((name) => !used.has(name));  // a name belongs to the first topic listing it
    names.forEach((name) => used.add(name));
    return names;
  });
  const listedTotal = listed.reduce((sum, names) => sum + names.length, 0);
  const remaining = TOTAL_UNITS - regions - listedTotal;
  const drafts = [];  // { name, topic }
  let assigned = 0;
  TOPICS.forEach((topic, t) => {
    const names = listed[t].slice();
    const last = t + 1 === TOPICS.length;
    const extra = last ? remaining - assigned : Math.round((remaining * listed[t].length) / listedTotal);
    assigned += extra;
    let made = 0;
    let guard = 0;
    while (made < extra && guard++ < 100000) {
      const base = random.pick(listed[t]);
      const upper = /^[A-Z0-9_]+$/.test(base);
      const suffix = random.pick(SUFFIXES);
      let name;
      if (topic.kind === 'type') name = base.replace(/_t$/, '') + suffix + '_t';
      else name = base + (upper ? suffix.toUpperCase() : suffix);
      if (used.has(name)) name += upper ? '_' + (2 + random.int(7)) : String(2 + random.int(7));
      if (used.has(name)) continue;
      used.add(name);
      names.push(name);
      ++made;
    }
    names.forEach((name) => drafts.push({ name, topic: t }));
  });

  // Group centres on a jittered grid, so that the groups are spread out.
  const columns = 7;
  const rows = Math.ceil(TOPICS.length / columns);
  const centres = TOPICS.map((topic, t) => {
    const col = t % columns;
    const row = Math.floor(t / columns);
    return {
      x: -0.84 + (1.68 * (col + 0.5)) / columns + 0.06 * random.normal(),
      y: -0.8 + (1.6 * (row + 0.5)) / rows + 0.05 * random.normal(),
      spread: 0.035 + 0.03 * random.next(),
    };
  });

  // Units in file order. Every file starts with a region; functions follow.
  const perFile = new Map(files.map((path) => [path, []]));
  for (const draft of drafts) {
    const topic = TOPICS[draft.topic];
    const path = topic.files[random.int(topic.files.length)];
    perFile.get(path).push(draft);
  }
  const units = [];
  const byName = new Map();
  for (const path of files) {
    const list = perFile.get(path);
    const entries = [{ region: 0 }];
    list.forEach((draft, index) => {
      entries.push(draft);
      if (index === Math.floor(list.length / 2)) entries.push({ region: 1 });
    });
    if (list.length === 0) entries.push({ region: 1 });
    for (const entry of entries) {
      const id = units.length;
      if (entry.region !== undefined) {
        // A region belongs to the group of the units around it.
        const neighbour = list.length > 0 ? list[Math.min(list.length - 1, entry.region * (list.length - 1))] : null;
        const topic = neighbour ? neighbour.topic : TOPICS.length - 1;
        units.push({ id, path, name: '', kind: 'region', group: topic, region: entry.region });
      } else {
        const topic = TOPICS[entry.topic];
        const kind = topic.kind || 'function';
        const unit = { id, path, name: entry.name, kind, group: entry.topic };
        units.push(unit);
        if (!byName.has(entry.name)) byName.set(entry.name, unit);
      }
    }
  }

  // Call graph.
  const functionsOf = TOPICS.map(() => []);
  for (const unit of units) if (unit.kind === 'function') functionsOf[unit.group].push(unit);
  const allFunctions = units.filter((u) => u.kind === 'function');
  const hubs = HUBS.map((name) => byName.get(name)).filter((u) => u !== undefined);
  for (const unit of units) {
    unit.callees = [];
    unit.callers = [];
  }
  for (const unit of allFunctions) {
    const written = WRITTEN[unit.name];
    const chosen = new Set();
    if (written) {
      for (const name of written.callees) {
        const callee = byName.get(name);
        if (callee && callee !== unit) chosen.add(callee.id);
      }
    } else {
      const k = 1 + random.int(4);
      for (let i = 0; i < k; ++i) {
        const r = random.next();
        const pool = r < 0.12 && hubs.length > 0 ? hubs : r < 0.8 ? functionsOf[unit.group] : allFunctions;
        const callee = random.pick(pool);
        if (callee.id !== unit.id) chosen.add(callee.id);
      }
    }
    unit.callees = Array.from(chosen);
  }
  for (const unit of units) for (const callee of unit.callees) units[callee].callers.push(unit.id);

  // Text, lines, tokens, coordinates.
  const lineOf = new Map(files.map((path) => [path, 1]));
  for (const unit of units) {
    const topic = TOPICS[unit.group];
    let lines;
    const written = WRITTEN[unit.name];
    if (written) lines = written.text;
    else if (unit.kind === 'function') lines = functionText(unit, topic, unit.callees.map((id) => units[id].name), random);
    else if (unit.kind === 'type') lines = typeText(unit, topic, random);
    else if (unit.kind === 'macro') lines = macroText(unit, topic, random);
    else lines = regionText(unit.path, unit.region);
    unit.text = lines.join('\n') + '\n';
    unit.start_line = lineOf.get(unit.path);
    unit.end_line = unit.start_line + lines.length - 1;
    lineOf.set(unit.path, unit.end_line + 2);
    unit.tokens = Math.max(4, Math.ceil(unit.text.length / 3.6));
    const comment = lines.filter((l) => l.startsWith('//')).map((l) => l.replace(/^\/\/ ?/, ''));
    unit.doc = unit.kind === 'region' ? '' : comment.join(' ');
    const open = lines.findIndex((l) => l.endsWith('{') && !l.startsWith('//'));
    unit.signature = unit.kind === 'function' && open >= 0 ? lines[open].replace(/\s*\{$/, '') : '';
    unit.words = new Set([...subwords(unit.name), ...topic.words.split(' '), ...subwords(unit.path)]);
    unit.nameWords = subwords(unit.name);
    const c = centres[unit.group];
    unit.x = Math.max(-1, Math.min(1, c.x + c.spread * random.normal()));
    unit.y = Math.max(-1, Math.min(1, c.y + c.spread * random.normal()));
    const d2 = ((unit.x - c.x) ** 2 + (unit.y - c.y) ** 2) / (c.spread * c.spread);
    unit.log_density = Number((2.6 - 0.5 * d2 + 0.1 * random.normal()).toFixed(3));
    delete unit.region;
  }

  const groups = TOPICS.map((topic, t) => {
    const members = units.filter((u) => u.group === t);
    const named = members.filter((u) => u.name !== '').sort((a, b) => b.log_density - a.log_density);
    const density = members.reduce((sum, u) => sum + u.log_density, 0) / Math.max(1, members.length);
    return {
      id: t,
      size: members.length,
      log_density: Number(density.toFixed(3)),
      x: Number(centres[t].x.toFixed(4)),
      y: Number(centres[t].y.toFixed(4)),
      label: named.slice(0, 3).map((u) => u.name).join(', '),
      key: topic.key,
    };
  });

  return { units, groups, files, byName };
}

export { subwords };
