// Settings dialog: corpus roots, the relevance threshold, the token budget,
// model paths, threads and block size, and the rebuild of the index.
//
// Only the keys that were changed are sent. Changes of the corpus or of the
// embedding model make the index stale; the dialog says so before saving and
// offers the rebuild.

import * as model from '../lib/settings-model.js';
import { formatCount, formatBytes, formatFixed, formatSeconds, formatDateTime, formatProbability, plural } from '../lib/format.js';
import { h, icon, byId, banner, fact, replace } from './dom.js';

const TAU = '\u{3c4}';
const ARROW = '\u{2192}';
const BUILD_STAGES = ['discover', 'parse', 'graph', 'embed', 'density', 'codes', 'calibrate', 'save'];

export function createSettings({ api, store, toast }) {
  const dialog = byId('settings-dialog');
  const form = byId('settings-form');
  const roots = byId('roots-list');
  const addRoot = byId('add-root');
  const tau = byId('tau');
  const tauValue = byId('tau-value');
  const tauRatio = byId('tau-ratio');
  const budget = byId('token-budget');
  const embedding = byId('model-embedding');
  const chat = byId('model-chat');
  const tuning = byId('model-tuning');
  const threads = byId('exec-threads');
  const block = byId('exec-block');
  const blockHelp = byId('exec-block-help');
  const error = byId('settings-error');
  const dirty = byId('settings-dirty');
  const save = byId('settings-save');
  const indexState = byId('index-state');
  const rebuild = byId('rebuild-index');
  const calibrate = byId('calibrate');
  const progress = byId('index-progress');
  const progressLabel = byId('index-progress-label');
  const report = byId('index-report');

  const fields = {
    'retrieve.tau': tau, 'context.token_budget': budget, 'models.embedding': embedding, 'models.chat': chat,
    'models.tuning_base': tuning, 'exec.threads': threads, 'exec.block_bytes': block,
  };
  let before = model.readForm({});
  let loaded = false;
  let rootSerial = 0;

  // ---- corpus roots -------------------------------------------------------------

  function renumber() {
    Array.from(roots.children).forEach((row, i) => {
      row.querySelector('label').textContent = `Corpus root ${i + 1}`;
      row.querySelector('button').setAttribute('aria-label', `Remove corpus root ${i + 1}`);
    });
    roots.dataset.empty = String(roots.childElementCount === 0);
  }

  function rootRow(value) {
    const id = `root-${++rootSerial}`;
    const input = h('input', { type: 'text', id, class: 'input mono', value, autocomplete: 'off', spellcheck: false,
                               placeholder: 'C:/path/to/source' });
    const remove = h('button', { type: 'button', class: 'btn-icon', title: 'Remove' }, icon('close'));
    const row = h('li', { class: 'root-row' }, h('label', { class: 'visually-hidden', htmlFor: id }), input, remove);
    remove.addEventListener('click', () => {
      const next = row.nextElementSibling || row.previousElementSibling;
      row.remove();
      renumber();
      changed();
      (next ? next.querySelector('input') : addRoot).focus();
    });
    return row;
  }

  function setRoots(list) {
    roots.replaceChildren(...list.map(rootRow));
    renumber();
  }

  addRoot.addEventListener('click', () => {
    const row = rootRow('');
    roots.appendChild(row);
    renumber();
    row.querySelector('input').focus();
  });

  // ---- form ---------------------------------------------------------------------

  function integer(input) {
    const text = input.value.trim();
    return /^-?\d+$/.test(text) ? Number(text) : NaN;
  }

  function read() {
    return {
      'corpus.roots': model.normaliseRoots(Array.from(roots.querySelectorAll('input')).map((i) => i.value)),
      'retrieve.tau': Number(tau.value),
      'context.token_budget': integer(budget),
      'models.embedding': embedding.value.trim(),
      'models.chat': chat.value.trim(),
      'models.tuning_base': tuning.value.trim(),
      'exec.threads': integer(threads),
      'exec.block_bytes': integer(block),
    };
  }

  function fill(values) {
    setRoots(values['corpus.roots']);
    tau.value = String(Math.min(model.TAU_MAX, Math.max(model.TAU_MIN, values['retrieve.tau'])));
    budget.value = String(values['context.token_budget']);
    embedding.value = values['models.embedding'];
    chat.value = values['models.chat'];
    tuning.value = values['models.tuning_base'];
    threads.value = String(values['exec.threads']);
    block.value = String(values['exec.block_bytes']);
    renderTau();
  }

  function renderTau() {
    const value = Number(tau.value);
    tauValue.textContent = formatProbability(value);
    const ratio = model.costRatio(value);
    tauRatio.textContent = `With ${TAU} = ${formatProbability(value)}, missing a relevant unit is taken to cost ` +
      `${formatFixed(ratio, ratio >= 10 ? 0 : 1)} times as much as an irrelevant unit in the prompt.`;
    tau.setAttribute('aria-valuetext', `${formatProbability(value)}`);
  }

  function showErrors(errors) {
    for (const key of Object.keys(fields)) fields[key].removeAttribute('aria-invalid');
    for (const e of errors) if (fields[e.key]) fields[e.key].setAttribute('aria-invalid', 'true');
    error.textContent = errors.map((e) => e.message).join('. ') + (errors.length > 0 ? '.' : '');
    error.hidden = errors.length === 0;
  }

  function changed() {
    const keys = model.changedKeys(before, read());
    if (keys.length === 0) dirty.textContent = '';
    else if (model.affectsIndex(keys)) dirty.textContent = 'Unsaved changes. Saving them makes the index stale; rebuild it afterwards.';
    else dirty.textContent = 'Unsaved changes.';
    dirty.dataset.index = String(model.affectsIndex(keys));
    return keys;
  }

  // Saves what was changed. Returns true when the settings on the server are
  // what the form shows.
  async function submit() {
    const values = read();
    const errors = model.validate(values);
    showErrors(errors);
    if (errors.length > 0) {
      const first = fields[errors[0].key];
      if (first) first.focus();
      return false;
    }
    const keys = model.changedKeys(before, values);
    if (keys.length === 0) return true;
    save.disabled = true;
    try {
      const doc = await api.saveSettings(model.buildPatch(before, values));
      before = model.readForm(doc && typeof doc === 'object' ? doc : {});
      // Keep what the user typed for values the server stores as sent.
      fill(before);
      changed();
      toast.good(model.affectsIndex(keys) ? 'Settings saved. The index is stale until it is rebuilt.' : 'Settings saved.');
      store.set('refresh', Date.now());
      return true;
    } catch (failure) {
      showErrors([{ key: '', message: `The settings were not saved: ${failure.message}`.replace(/\.$/, '') }]);
      return false;
    } finally {
      save.disabled = !loaded;
    }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const keys = model.changedKeys(before, read());
    const ok = await submit();
    // Stay open when the index needs a rebuild: the button for it is here.
    if (ok && !model.affectsIndex(keys)) dialog.close();
  });
  form.addEventListener('input', () => {
    renderTau();
    changed();
  });
  // Enter in a text field must not submit by accident while a path is edited.
  roots.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      addRoot.click();
    }
  });

  byId('settings-close').addEventListener('click', () => dialog.close());
  byId('settings-cancel').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });

  // ---- index --------------------------------------------------------------------

  function renderIndex() {
    const status = store.get('status');
    const p = store.get('progress');
    const running = Boolean((p && p.running) || (status && status.index && status.index.building));
    rebuild.disabled = running;
    calibrate.disabled = running || !(status && status.index && status.index.ready);

    if (status && status.index) {
      const index = status.index;
      indexState.textContent = running ? 'The index is being built.'
        : index.ready ? `Built ${formatDateTime(index.built_at)}: ${plural(index.units, 'unit')} in ${plural(index.files, 'file')}, ` +
                        `${plural(index.groups, 'group')}${index.calibrated === false ? '; scores are not calibrated' : ''}.`
        : 'The index is stale or was never built. Rebuild it to ask questions.';
    } else {
      indexState.textContent = '';
    }
    if (status && status.cache) {
      blockHelp.textContent = `0 takes the size from the cache topology. On this machine the L1 data cache share of a thread is ` +
        `${formatBytes(status.cache.l1d_per_thread_min_bytes)} and the block in use is ${formatBytes(status.cache.block_bytes)}.`;
    }

    progress.hidden = !running;
    progressLabel.hidden = !running;
    if (running) {
      const stage = p ? String(p.stage || '') : '';
      const at = Math.max(0, BUILD_STAGES.indexOf(stage));
      const within = p && p.total > 0 ? Math.min(1, p.done / p.total) : 0;
      progress.value = (at + within) / BUILD_STAGES.length;
      progressLabel.textContent = p
        ? `Stage ${at + 1} of ${BUILD_STAGES.length}, ${stage}` +
          `${p.total > 0 ? `: ${formatCount(p.done)} of ${formatCount(p.total)}` : ''}${p.message ? ` (${p.message})` : ''}`
        : 'Starting';
    }

    const r = p && !p.running ? p.report : null;
    if (p && !p.running && p.stage === 'failed') {
      report.replaceChildren(banner('bad', 'The build failed.', String((r && r.error) || p.message || '')));
    } else if (r && !running) {
      const s = r.seconds || {};
      replace(report,
        h('h4', { class: 'form-subtitle', text: 'Last build' }),
        r.error ? banner('bad', 'Error.', String(r.error)) : null,
        h('dl', { class: 'facts' },
          fact('Files', formatCount(r.files)), fact('Units', formatCount(r.units)),
          fact('Functions', formatCount(r.functions)), fact('Call edges', formatCount(r.call_edges)),
          fact('Embedded', `${formatCount(r.embedded)} (${formatCount(r.reused)} reused)`),
          fact('Groups', formatCount(r.groups)), fact('Concentration', formatFixed(r.kappa, 1)),
          fact('Reduced dimension', formatCount(r.reduced_dim)),
          fact('Explained variance', formatProbability(r.explained_variance)),
          fact('Anisotropy', `${formatFixed(r.anisotropy_before, 2)} ${ARROW} ${formatFixed(r.anisotropy_after, 2)}`,
               'Mean pairwise cosine before and after centring and reduction'),
          fact('Blocks', `${formatCount(r.blocks)} of at most ${formatBytes(r.block_bytes)}`),
          fact('Time', formatSeconds(s.total)),
          fact('Longest stage', BUILD_STAGES.filter((k) => typeof s[k] === 'number').sort((a, b) => s[b] - s[a])
            .slice(0, 1).map((k) => `${k}, ${formatSeconds(s[k])}`).join('') || 'unknown')));
    } else if (running) {
      report.replaceChildren();
    }
  }

  rebuild.addEventListener('click', async () => {
    rebuild.disabled = true;
    if (!(await submit())) {
      rebuild.disabled = false;
      return;
    }
    try {
      await api.startIndex();
      store.set('progress', { running: true, stage: 'discover', done: 0, total: 0, message: 'starting', report: null });
      store.set('building', true);
      toast.info('The index is being rebuilt.');
    } catch (failure) {
      toast.error(`The rebuild was not started: ${failure.message}`);
      renderIndex();
    }
  });

  calibrate.addEventListener('click', async () => {
    calibrate.disabled = true;
    try {
      await api.calibrate();
      toast.info('The calibration is being estimated again.');
      store.set('refresh', Date.now());
    } catch (failure) {
      toast.error(`The calibration was not started: ${failure.message}`);
    }
    renderIndex();
  });

  store.on('status', renderIndex);
  store.on('progress', renderIndex);

  // ---- open ---------------------------------------------------------------------

  async function open() {
    showErrors([]);
    dirty.textContent = '';
    loaded = false;
    save.disabled = true;
    renderIndex();
    if (!dialog.open) dialog.showModal();
    try {
      const doc = await api.settings();
      before = model.readForm(doc && typeof doc === 'object' ? doc : {});
      fill(before);
      loaded = true;
      save.disabled = false;
    } catch (failure) {
      fill(before);
      showErrors([{ key: '', message: `The settings could not be loaded: ${failure.message}`.replace(/\.$/, '') }]);
    }
    // The current progress, in case a build was started elsewhere.
    api.indexProgress().then((p) => {
      if (p && typeof p === 'object') store.set('progress', p);
    }).catch(() => {});
  }

  return { open };
}
