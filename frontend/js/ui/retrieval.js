// Inspector, Retrieval tab: the units accepted by the probabilistic trace,
// ordered by probability.
//
// The bar of a unit shows its probability in two parts: the posterior from
// the similarity score alone, and what the evidence of the graph (callers,
// callees, units of the same density group) added to it. A tick marks tau,
// the threshold a unit has to reach to be accepted.

import { formatProbability, formatCount, formatFixed, clamp01, plural } from '../lib/format.js';
import { languageOfPath } from '../lib/highlight.js';
import { h, byId, badge, fact, banner, emptyState, replace } from './dom.js';
import { codeView } from './markdown-dom.js';

const TAU = '\u{3c4}';
const RHO = '\u{3c1}';

function percent(x) {
  return (clamp01(x) * 100).toFixed(2) + '%';
}

export function unitTitle(unit) {
  return unit.name !== '' && unit.name !== undefined ? unit.name : `${unit.kind || 'region'} (unnamed)`;
}

export function unitLocation(unit) {
  return `${unit.path}:${unit.start_line}-${unit.end_line}`;
}

// How the tracer reached the unit, in words.
function relation(unit, parent) {
  const name = parent ? unitTitle(parent) : unit.parent !== null && unit.parent !== undefined ? `unit ${unit.parent}` : '';
  switch (unit.via) {
    case 'seed': return 'seed: accepted on its own score';
    case 'callee': return `callee: called by ${name}`;
    case 'caller': return `caller: calls ${name}`;
    case 'sibling': return `sibling: same group as ${name}`;
    default: return String(unit.via || '');
  }
}

export function probabilityBar(unit, tau) {
  const p = clamp01(unit.p);
  const score = Math.min(clamp01(unit.p_score), p);
  const label = `Probability ${formatProbability(p)}, from the score alone ${formatProbability(unit.p_score)}, ` +
                `threshold ${formatProbability(tau)}`;
  return h('div', { class: 'pbar', role: 'img', 'aria-label': label, title: label },
    h('span', { class: 'pbar-score', style: { width: percent(score) } }),
    h('span', { class: 'pbar-lift', style: { left: percent(score), width: percent(p - score) } }),
    h('span', { class: 'pbar-tau', style: { left: percent(tau) } }));
}

export function createRetrievalPanel({ api, store }) {
  const panel = byId('panel-retrieval');
  const texts = new Map();   // unit id -> GET /api/unit, kept for the session
  let onlyPrompt = false;

  async function loadCode(unit, target) {
    target.replaceChildren(h('p', { class: 'note', text: 'Loading the unit' }));
    try {
      let full = texts.get(unit.id);
      if (!full) {
        full = await api.unit(unit.id);
        texts.set(unit.id, full);
      }
      const known = new Map((store.get('retrieval') ? store.get('retrieval').units : []).map((u) => [u.id, u]));
      const mapUnits = store.get('groups') ? store.get('groups').units : [];
      const nameOf = (id) => {
        const u = known.get(id) || mapUnits.find((m) => m.id === id);
        return u && u.name ? u.name : `unit ${id}`;
      };
      const related = (label, ids) => {
        if (!Array.isArray(ids) || ids.length === 0) return null;
        const shown = ids.slice(0, 12).map(nameOf).join(', ');
        return fact(label, ids.length > 12 ? `${shown} and ${ids.length - 12} more` : shown);
      };
      replace(target,
        codeView(String(full.text || ''), {
          firstLine: full.start_line, language: languageOfPath(full.path),
          label: `${unitTitle(full)}, ${unitLocation(full)}`,
        }),
        h('dl', { class: 'facts' }, related('Calls', full.callees), related('Called by', full.callers)));
    } catch (error) {
      const retry = h('button', { type: 'button', class: 'btn btn-small', text: 'Try again',
                                  on: { click: () => loadCode(unit, target) } });
      target.replaceChildren(banner('bad', 'The unit could not be loaded.', error.message), retry);
    }
  }

  function unitCard(unit, retrieval, byIdMap) {
    const parent = unit.parent !== null && unit.parent !== undefined ? byIdMap.get(unit.parent) : null;
    const form = unit.form === 'none' ? badge('not in prompt', 'neutral')
      : unit.form === 'skeleton' ? badge('skeleton', 'accent') : badge('full', 'good');

    const code = h('div', { class: 'unit-code' });
    const details = h('details', { class: 'unit-details' },
      h('summary', null, h('span', { text: 'Code' }),
        h('span', { class: 'note', text: ` ${plural(unit.end_line - unit.start_line + 1, 'line')}, ${formatCount(unit.tokens)} tokens` })),
      code);
    details.addEventListener('toggle', () => {
      if (details.open && code.childElementCount === 0) loadCode(unit, code);
    });

    const lift = unit.p - unit.p_score;
    return h('li', { class: 'unit', id: `unit-${unit.id}`, 'data-in-prompt': String(unit.in_context === true) },
      h('div', { class: 'unit-head' },
        h('div', { class: 'unit-title' },
          h('span', { class: 'unit-name', text: unitTitle(unit) }),
          h('span', { class: 'unit-kind', text: unit.kind || '' })),
        h('div', { class: 'unit-p', title: 'Probability of being relevant' }, formatProbability(unit.p))),
      h('div', { class: 'unit-path', text: unitLocation(unit), title: unit.path }),
      probabilityBar(unit, retrieval.tau),
      h('dl', { class: 'facts' },
        fact('Score only', formatProbability(unit.p_score), 'Posterior from the similarity score alone'),
        lift > 0.0005 ? fact('Graph lift', '+' + formatProbability(lift).replace(/^[<>]/, ''), 'Added by the evidence of accepted neighbours') : null,
        fact('Reached as', relation(unit, parent)),
        fact('Hops', formatCount(unit.hops), 'Distance from a seed'),
        fact('In prompt', h('span', null, form, unit.support ? badge('support', 'warn') : null),
             unit.support ? 'Present only because a kept unit requires it'
               : unit.form === 'skeleton' ? 'Signature and documentation only'
               : unit.form === 'none' ? 'Accepted by the trace, removed by the context pruning' : 'Complete text'),
        fact('Group', formatCount(unit.group), 'Density group'),
        fact('Cosine', `${formatFixed(unit.score, 2)} (z ${formatFixed(unit.z, 1)})`, 'Similarity to the question and its standard score'),
        unit.in_context ? fact(`Redundancy ${RHO}`, formatProbability(unit.redundancy), 'Similarity to the units placed before it') : null,
        unit.in_context ? fact('Quality', formatProbability(unit.quality), `(1 - ${RHO}) p`) : null),
      unit.preview ? h('pre', { class: 'unit-preview', 'aria-label': 'First lines', text: String(unit.preview).trimEnd() }) : null,
      details);
  }

  function render() {
    const retrieval = store.get('retrieval');
    if (!retrieval || !Array.isArray(retrieval.units)) {
      panel.replaceChildren(emptyState('No retrieval yet',
        'Ask a question. The units that the trace accepts appear here with their probabilities.'));
      return;
    }
    const units = retrieval.units.slice().sort((a, b) => b.p - a.p || a.id - b.id);
    const byIdMap = new Map(units.map((u) => [u.id, u]));
    const inPrompt = units.filter((u) => u.in_context).length;
    const trace = retrieval.trace || {};

    const toggle = h('input', { type: 'checkbox', id: 'only-prompt', checked: onlyPrompt });
    toggle.addEventListener('change', () => {
      onlyPrompt = toggle.checked;
      render();
    });

    const shown = onlyPrompt ? units.filter((u) => u.in_context) : units;
    replace(panel,
      retrieval.low_confidence
        ? banner('warn', 'Low confidence.', `No unit reached ${TAU}; the single best unit is shown.`) : null,
      retrieval.truncated
        ? banner('warn', 'Truncated.', 'The trace stopped at the maximum number of units (retrieve.max_units).') : null,
      h('div', { class: 'summary' },
        h('dl', { class: 'facts facts-summary' },
          fact('Units', `${formatCount(units.length)} traced, ${formatCount(inPrompt)} in the prompt`),
          fact(`Threshold ${TAU}`, formatProbability(retrieval.tau)),
          fact('Score threshold', `z ${formatFixed(retrieval.z_threshold, 2)}`, `The score posterior reaches ${TAU} at this standard score`),
          fact('Seeds', formatCount(trace.seeds)),
          fact('Evaluated', formatCount(trace.evaluated), 'Candidates whose probability was computed'),
          fact('Expansions', formatCount(trace.expansions), 'Accepted units whose neighbours were visited'))),
      h('div', { class: 'legend', 'aria-hidden': 'true' },
        h('span', { class: 'legend-item' }, h('i', { class: 'swatch swatch-score' }), 'from the score'),
        h('span', { class: 'legend-item' }, h('i', { class: 'swatch swatch-lift' }), 'added by the graph'),
        h('span', { class: 'legend-item' }, h('i', { class: 'swatch swatch-tau' }), `${TAU} = ${formatProbability(retrieval.tau)}`)),
      h('div', { class: 'list-tools' },
        h('label', { class: 'check', htmlFor: 'only-prompt' }, toggle, 'Only units in the prompt')),
      shown.length === 0 ? h('p', { class: 'note', text: 'No unit is in the prompt.' }) : null,
      h('ol', { class: 'unit-list' }, shown.map((u) => unitCard(u, retrieval, byIdMap))));
  }

  // Scrolls to a unit and opens its code; used by citations in the answer.
  function reveal(id) {
    if (id === null || id === undefined) return;
    let card = document.getElementById(`unit-${id}`);
    if (!card && onlyPrompt) {
      onlyPrompt = false;
      render();
      card = document.getElementById(`unit-${id}`);
    }
    if (!card) return;
    const details = card.querySelector('details');
    if (details) details.open = true;
    card.scrollIntoView({ block: 'start', behavior: 'auto' });
    card.classList.add('unit-focus');
    setTimeout(() => card.classList.remove('unit-focus'), 1600);
    const summary = card.querySelector('summary');
    if (summary) summary.focus({ preventScroll: true });
  }

  store.on('retrieval', render);
  store.on('focusUnit', (id) => {
    reveal(id);
  });
  render();
  return { render };
}
