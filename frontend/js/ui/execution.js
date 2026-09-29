// Inspector, Execution tab: how the scan of the index ran.
//
// The scan goes through the ordered executor in blocks sized to the L1 data
// cache. Two things are shown prominently: that no block was admitted or
// committed out of order (the counters must be zero), and how the block size
// relates to the cache sizes the operating system reports.

import { formatCount, formatBytes, formatMicros, formatRatio, plural } from '../lib/format.js';
import { h, byId, stat, banner, emptyState } from './dom.js';

function sizeBar(label, bytes, max, kind, note) {
  const width = max > 0 ? Math.min(100, (bytes / max) * 100).toFixed(2) + '%' : '0%';
  return h('div', { class: 'size-row' },
    h('span', { class: 'size-label' }, label, note ? h('span', { class: 'note', text: ' ' + note }) : null),
    h('div', { class: 'meter meter-tall' }, h('span', { class: `meter-fill meter-${kind}`, style: { width } })),
    h('span', { class: 'size-value', text: formatBytes(bytes) }));
}

export function createExecutionPanel({ store }) {
  const panel = byId('panel-execution');

  function cacheSection(scan, cache) {
    if (!cache) return h('p', { class: 'note', text: 'The cache topology is not known yet.' });
    const block = scan ? scan.block_bytes : cache.block_bytes;
    const l1 = (Array.isArray(cache.classes) ? cache.classes : [])
      .filter((c) => c.level === 1 && c.type === 'data')
      .sort((a, b) => a.size_bytes - b.size_bytes);
    const max = Math.max(block, cache.l1d_min_bytes || 0, ...l1.map((c) => c.size_bytes), 1);
    const fits = block <= cache.l1d_per_thread_min_bytes;
    return h('section', null,
      h('h3', { class: 'section-title', text: 'Block size and L1 data cache' }),
      cache.detected === false
        ? banner('warn', 'Cache sizes not detected.', 'The operating system query failed; conservative defaults are shown.') : null,
      h('div', { class: 'sizes', role: 'group', 'aria-label': 'Block size compared with the L1 data cache sizes' },
        sizeBar('Block', block, max, 'after'),
        sizeBar('L1d per thread', cache.l1d_per_thread_min_bytes, max, 'before', '(smallest share of one logical CPU)'),
        sizeBar('L1d smallest', cache.l1d_min_bytes, max, 'before'),
        l1.map((c) => sizeBar('L1d', c.size_bytes, max, 'before',
          `(${plural(c.instances, 'cache')}, ${c.sharing_threads === 1 ? 'not shared' : `shared by ${c.sharing_threads} threads`})`))),
      h('p', { class: 'note', text: fits
        ? `A block of ${formatBytes(block)} fits the L1 data cache share of a thread (${formatBytes(cache.l1d_per_thread_min_bytes)}), ` +
          'with room for the query and the scores. The program sizes the working set; it cannot choose the cache level that holds it.'
        : `A block of ${formatBytes(block)} is larger than the L1 data cache share of a thread ` +
          `(${formatBytes(cache.l1d_per_thread_min_bytes)}): the scan will miss the L1 cache.` }),
      h('dl', { class: 'facts' },
        h('div', { class: 'fact' }, h('dt', { text: 'Logical CPUs' }), h('dd', { text: formatCount(cache.logical_cpus) })),
        h('div', { class: 'fact' }, h('dt', { text: 'Cache line' }), h('dd', { text: formatBytes(cache.line_bytes) }))));
  }

  function render() {
    const retrieval = store.get('retrieval');
    const status = store.get('status');
    const scan = retrieval ? retrieval.scan : null;
    const cache = status ? status.cache : null;
    if (!scan && !cache) {
      panel.replaceChildren(emptyState('No scan yet', 'After a question this tab shows how the index was scanned.'));
      return;
    }
    const nodes = [];
    if (scan) {
      const admission = scan.admission_violations || 0;
      const commit = scan.commit_violations || 0;
      nodes.push(admission === 0 && commit === 0
        ? banner('good', 'Order preserved.',
                 `All ${plural(scan.blocks, 'block')} were admitted and committed in sequence: 0 admission and 0 commit violations.`)
        : banner('bad', 'Ordering violated.',
                 `${plural(admission, 'admission violation')} and ${plural(commit, 'commit violation')}; both must be 0. ` +
                 'Results of this scan may depend on the thread schedule.'));
      const speedup = scan.wall_us > 0 ? scan.work_us / scan.wall_us : NaN;
      nodes.push(h('div', { class: 'stat-grid' },
        stat('Admission violations', formatCount(admission), 'expected 0', admission === 0 ? null : 'bad'),
        stat('Commit violations', formatCount(commit), 'expected 0', commit === 0 ? null : 'bad'),
        stat('Blocks', formatCount(scan.blocks), `of at most ${formatBytes(scan.block_bytes)}`),
        stat('Threads', formatCount(scan.threads)),
        stat('Wall time', formatMicros(scan.wall_us), 'start to end of the scan'),
        stat('Work time', formatMicros(scan.work_us), `summed over threads, ${formatRatio(speedup)} the wall time`),
        stat('Gate wait', formatMicros(scan.gate_wait_us), 'waiting for admission in order'),
        stat('Rescored', formatCount(scan.rescored), 'units scored again exactly'),
        stat('Codes', String(scan.codes || ''), 'vector encoding that was scanned'),
        stat('SIMD kernel', String(scan.simd || ''))));
    } else {
      nodes.push(h('p', { class: 'note', text: 'No question has been asked yet; the cache topology of the machine is shown.' }));
    }
    nodes.push(cacheSection(scan, cache));
    panel.replaceChildren(...nodes);
  }

  store.on('retrieval', render);
  store.on('status', render);
  render();
  return { render };
}
