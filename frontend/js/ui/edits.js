// Inspector, Edits tab: the proposed change, file by file, with Apply and
// Revert. Nothing is written to disk before Apply; Revert restores the
// content from before the edit.

import { parseUnifiedDiff, hunkHeader } from '../lib/diff.js';
import { formatCount, plural } from '../lib/format.js';
import { h, byId, badge, banner, emptyState, fact, replace } from './dom.js';

const STATUS_TEXT = {
  ok: 'ready',
  not_found: 'text not found',
  ambiguous: 'ambiguous match',
  scope_not_found: 'unit not found',
  outside_roots: 'outside the corpus roots',
  file_missing: 'file missing',
  overlap: 'overlapping blocks',
  syntax_regression: 'adds syntax errors',
  io_error: 'input/output error',
  conflict: 'file changed on disk',
};

const MATCH_TEXT = {
  exact: 'exact',
  trailing_space: 'ignoring trailing blanks',
  trailingspace: 'ignoring trailing blanks',
  indentation: 'ignoring indentation',
  fuzzy: 'fuzzy',
  unit: 'whole unit',
  none: 'no match',
};

const MINUS = '\u{2212}';
const ARROW = '\u{2192}';

function statusBadge(status) {
  const text = STATUS_TEXT[status] || String(status || 'unknown');
  return badge(text, status === 'ok' ? 'good' : 'bad');
}

function diffTable(file, parsed) {
  const body = h('tbody', null);
  for (const hunk of parsed.hunks) {
    body.appendChild(h('tr', { class: 'diff-hunk' },
      h('td', { class: 'ln' }), h('td', { class: 'ln' }), h('td', { class: 'sign' }),
      h('td', { class: 'code', text: hunkHeader(hunk) })));
    for (const line of hunk.lines) {
      const sign = line.type === 'add' ? '+' : line.type === 'del' ? MINUS : '';
      const spoken = line.type === 'add' ? 'added' : line.type === 'del' ? 'removed' : '';
      body.appendChild(h('tr', { class: `diff-${line.type}` },
        h('td', { class: 'ln', text: line.oldLine === null ? '' : String(line.oldLine) }),
        h('td', { class: 'ln', text: line.newLine === null ? '' : String(line.newLine) }),
        h('td', { class: 'sign', 'aria-label': spoken || null, text: sign }),
        h('td', { class: 'code', text: line.type === 'note' ? `(${line.text})` : line.text })));
    }
  }
  return h('div', { class: 'diff-scroll', tabIndex: 0, role: 'group', 'aria-label': `Changes to ${file.display_path || file.path}` },
    h('table', { class: 'diff' },
      h('caption', { class: 'visually-hidden', text: `Unified diff of ${file.display_path || file.path}` }),
      h('thead', { class: 'visually-hidden' }, h('tr', null,
        h('th', { scope: 'col', text: 'Old line' }), h('th', { scope: 'col', text: 'New line' }),
        h('th', { scope: 'col', text: 'Change' }), h('th', { scope: 'col', text: 'Text' }))),
      body));
}

function matchTable(matches) {
  return h('table', { class: 'table matches' },
    h('caption', { class: 'visually-hidden', text: 'How each block was located' }),
    h('thead', null, h('tr', null,
      h('th', { scope: 'col', text: 'Block' }), h('th', { scope: 'col', text: 'Match' }),
      h('th', { scope: 'col', text: 'Lines' }), h('th', { scope: 'col', class: 'num', text: 'Distance' }),
      h('th', { scope: 'col', class: 'num', text: 'Candidates' }))),
    h('tbody', null, matches.map((m, i) => {
      const kind = String(m.kind || 'none').toLowerCase();
      const loose = kind !== 'exact' && kind !== 'unit';
      return h('tr', null,
        h('th', { scope: 'row', text: String(i + 1) }),
        h('td', null, badge(MATCH_TEXT[kind] || kind, kind === 'none' ? 'bad' : loose ? 'warn' : 'neutral')),
        h('td', { text: m.first_line > 0 ? `${m.first_line}-${m.last_line}` : '' }),
        h('td', { class: 'num', text: formatCount(m.distance) }),
        h('td', { class: 'num' }, m.candidates > 1 ? badge(formatCount(m.candidates), 'bad') : formatCount(m.candidates)));
    })));
}

function fileSection(file) {
  const parsed = parseUnifiedDiff(String(file.diff || ''));
  const diff = parsed.files[0];
  const before = file.syntax_errors_before;
  const after = file.syntax_errors_after;
  const regression = after > before;
  return h('section', { class: 'edit-file' },
    h('div', { class: 'edit-file-head' },
      h('h3', { class: 'edit-path', title: file.path, text: String(file.display_path || file.path || '') }),
      h('div', { class: 'edit-badges' },
        file.creates_file ? badge('new file', 'accent') : null,
        statusBadge(file.status),
        diff ? h('span', { class: 'diff-count', 'aria-label': `${plural(diff.added, 'line')} added, ${plural(diff.removed, 'line')} removed` },
          h('span', { class: 'diff-count-add', text: `+${diff.added}` }), ' ',
          h('span', { class: 'diff-count-del', text: `${MINUS}${diff.removed}` })) : null)),
    file.message ? banner(file.status === 'ok' ? 'info' : 'bad', STATUS_TEXT[file.status] ? STATUS_TEXT[file.status] + '.' : 'Note.', file.message) : null,
    h('dl', { class: 'facts' },
      fact('Syntax errors', h('span', null, `${formatCount(before)} ${ARROW} ${formatCount(after)} `,
                              regression ? badge('regression', 'bad') : badge(after < before ? 'fewer' : 'no new errors', 'good')),
           'Error nodes of the parser before and after the edit; counts are compared, zero is not required'),
      fact('Blocks', formatCount(Array.isArray(file.matches) ? file.matches.length : 0))),
    Array.isArray(file.matches) && file.matches.length > 0 ? h('div', { class: 'table-scroll' }, matchTable(file.matches)) : null,
    diff && diff.hunks.length > 0 ? diffTable(file, diff)
      : h('p', { class: 'note', text: 'No diff: the change could not be computed for this file.' }));
}

export function createEditsPanel({ api, store, toast }) {
  const panel = byId('panel-edits');
  let working = false;

  async function act(kind) {
    const tx = store.get('transaction');
    if (!tx || working) return;
    working = true;
    render();
    try {
      const result = kind === 'apply' ? await api.editApply(tx.id) : await api.editRevert(tx.id);
      if (!result || typeof result !== 'object') throw new Error('unexpected response');
      // The same object is referenced by the turn of the conversation: update it in place.
      Object.assign(tx, result);
      if (kind === 'apply') {
        if (tx.committed) toast.good(`Applied to ${plural(tx.files.length, 'file')}. The touched files are indexed again.`);
        else toast.error(`Not applied: ${STATUS_TEXT[tx.status] || tx.status}.`);
      } else if (!tx.committed) {
        toast.good('Reverted. The files have their previous content.');
      }
    } catch (error) {
      toast.error(`${kind === 'apply' ? 'Apply' : 'Revert'} failed: ${error.message}`);
    } finally {
      working = false;
      store.set('transaction', tx);
    }
  }

  function render() {
    const tx = store.get('transaction');
    if (!tx || !Array.isArray(tx.files)) {
      panel.replaceChildren(emptyState('No edit proposed',
        'Switch the question field to Edit and describe a change. The proposal appears here as a diff; ' +
        'nothing is written until you apply it.'));
      return;
    }
    const canApply = tx.status === 'ok' && !tx.committed && !working;
    const canRevert = tx.committed === true && !working;
    const apply = h('button', { type: 'button', class: 'btn btn-primary', disabled: !canApply,
                                text: working ? 'Working' : 'Apply', on: { click: () => act('apply') } });
    const revert = h('button', { type: 'button', class: 'btn', disabled: !canRevert, text: 'Revert',
                                 on: { click: () => act('revert') } });
    const state = tx.committed ? badge('applied', 'good') : tx.status === 'ok' ? badge('not applied', 'neutral') : statusBadge(tx.status);
    let explanation = '';
    if (tx.committed) explanation = 'The change is on disk. Revert restores the previous content.';
    else if (tx.status === 'ok') explanation = 'Review the diff, then apply it. All files are written together or not at all.';
    else explanation = 'The proposal cannot be applied. Ask again with more context, or name the function to change.';

    const warnings = Array.isArray(tx.warnings) ? tx.warnings : [];
    replace(panel,
      h('div', { class: 'edit-head' },
        h('div', { class: 'edit-title' },
          h('h3', { class: 'section-title', text: 'Transaction' }),
          h('code', { class: 'edit-id', text: String(tx.id || '') }), state),
        h('div', { class: 'edit-actions' }, revert, apply)),
      h('p', { class: 'note', text: explanation }),
      warnings.map((w) => banner('warn', 'Warning.', String(w))),
      tx.files.length === 0 ? h('p', { class: 'note', text: 'The transaction holds no file.' }) : null,
      tx.files.map(fileSection));
  }

  store.on('transaction', render);
  render();
  return { render };
}
