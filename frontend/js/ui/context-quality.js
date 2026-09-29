// Inspector, Context quality tab: what the recursive context quality pruning did.
//
// The context rot index is CRI = 1 - Q, the expected fraction of prompt tokens
// that are irrelevant, redundant or overhead; Q is the quality of the context
// (context/rot.hpp). The tab compares the candidate context before the
// pruning with the prompt that was sent to the model.

import { formatProbability, formatCount, formatPercent, formatSigned, clamp01, plural, shortenPath } from '../lib/format.js';
import { h, byId, stat, emptyState, replace } from './dom.js';

const ARROW = '\u{2192}';
const THETA = '\u{3b8}';
const RHO = '\u{3c1}';

function percent(x) {
  return (clamp01(x) * 100).toFixed(2) + '%';
}

// A pair of bars on a common scale: before above, after below.
function comparison(title, help, before, after, max, format, lowerIsBetter) {
  const row = (label, value, kind) => h('div', { class: 'compare-row' },
    h('span', { class: 'compare-label', text: label }),
    h('div', { class: 'meter' }, h('span', { class: `meter-fill meter-${kind}`, style: { width: percent(max > 0 ? value / max : 0) } })),
    h('span', { class: 'compare-value', text: format(value) }));
  const change = after - before;
  const better = lowerIsBetter ? change < 0 : change > 0;
  const verdict = change === 0 ? 'unchanged' : better ? 'better' : 'worse';
  return h('section', { class: 'compare', 'aria-label': `${title}: ${format(before)} before, ${format(after)} after` },
    h('div', { class: 'compare-head' },
      h('h3', { class: 'section-title', text: title }),
      h('span', { class: 'compare-change', 'data-verdict': verdict,
                  text: `${format(before)} ${ARROW} ${format(after)}` })),
    row('before', before, 'before'),
    row('after', after, 'after'),
    h('p', { class: 'note', text: help }));
}

export function createContextPanel({ store }) {
  const panel = byId('panel-context');

  function render() {
    const retrieval = store.get('retrieval');
    const c = retrieval ? retrieval.context : null;
    if (!c) {
      panel.replaceChildren(emptyState('No context yet',
        'After a question this tab shows how much of the retrieved material reached the prompt, and how much of the prompt is relevant.'));
      return;
    }
    const dropped = (c.dropped_low_quality || 0) + (c.dropped_redundant || 0) + (c.dropped_budget || 0);
    const reasons = [
      ['Low quality', c.dropped_low_quality, `dropped: discounted quality (1 - ${RHO}) p below ${THETA}`],
      ['Redundant', c.dropped_redundant, 'dropped: they repeat units placed before them'],
      ['Over budget', c.dropped_budget, 'dropped: the prompt exceeded the token budget'],
      ['Skeletonised', c.skeletonised, 'reduced to signature and documentation to fit the budget'],
      ['Support added', c.support_added, 'signatures of units that a kept unit calls (dependency closure)'],
    ];

    const groups = Array.isArray(c.groups) ? c.groups : [];
    const files = h('table', { class: 'table files' },
      h('caption', { class: 'visually-hidden', text: 'Quality of the prompt per file' }),
      h('thead', null, h('tr', null,
        h('th', { scope: 'col', text: 'File' }), h('th', { scope: 'col', text: 'Quality' }),
        h('th', { scope: 'col', class: 'num', text: 'Units' }), h('th', { scope: 'col', class: 'num', text: 'Tokens' }))),
      h('tbody', null, groups.map((g) => h('tr', null,
        h('th', { scope: 'row', class: 'path', title: g.path, text: shortenPath(String(g.path || ''), 34) }),
        h('td', null, h('div', { class: 'bar-cell' },
          h('div', { class: 'meter' }, h('span', { class: 'meter-fill meter-after', style: { width: percent(g.quality) } })),
          h('span', { class: 'bar-value', text: formatProbability(g.quality) }))),
        h('td', { class: 'num', text: formatCount(g.units) }),
        h('td', { class: 'num', text: formatCount(g.tokens) })))));

    replace(panel,
      h('div', { class: 'stat-grid' },
        stat('Context rot index', formatProbability(c.rot_after),
             `${formatSigned(c.rot_after - c.rot_before)} from ${formatProbability(c.rot_before)}`),
        stat('Prompt size', `${formatCount(c.tokens_after)} tokens`, `${formatPercent(c.compression)} of the candidates`),
        stat('Passes', formatCount(c.passes), 'prune and rescore, to the fixed point'),
        stat('History turns kept', `${formatCount(c.history_kept)} of ${formatCount(c.history_total)}`,
             'gated by relevance to the question')),
      comparison('Context rot index', 'Share of the tokens that are irrelevant, redundant or overhead. Lower is better.',
                 c.rot_before, c.rot_after, 1, formatProbability, true),
      comparison('Quality', 'Relevant mass per token, Q = 1 - rot. Higher is better.',
                 c.quality_before, c.quality_after, 1, formatProbability, false),
      comparison('Tokens', `Every candidate in full before; the pruned prompt after (${formatPercent(c.compression)}).`,
                 c.tokens_before, c.tokens_after, Math.max(c.tokens_before, c.tokens_after, 1), formatCount, true),
      h('section', null,
        h('h3', { class: 'section-title', text: 'What was changed and why' }),
        h('p', { class: 'note', text: dropped === 0 ? 'No unit was dropped.' : `${plural(dropped, 'unit')} dropped.` }),
        h('table', { class: 'table reasons' },
          h('caption', { class: 'visually-hidden', text: 'Changes made by the pruning' }),
          h('tbody', null, reasons.map(([label, count, why]) => h('tr', { 'data-zero': String(!(count > 0)) },
            h('th', { scope: 'row', text: label }),
            h('td', { class: 'num', text: formatCount(count) }),
            h('td', { class: 'why', text: why })))))),
      h('section', null,
        h('h3', { class: 'section-title', text: 'Quality per file' }),
        groups.length === 0 ? h('p', { class: 'note', text: 'The prompt holds no file.' })
          : h('div', { class: 'table-scroll' }, files)));
  }

  store.on('retrieval', render);
  render();
  return { render };
}
