// The conversation: history, the question field, and the answer as it is
// streamed.
//
// A turn is one question with its answer. Turns of this session keep the
// retrieval and the edit transaction they produced, so that the inspector can
// go back to them; turns loaded from the history only know their unit ids.

import { isAbort } from '../api.js';
import { formatCount, formatMillis, formatDateTime, plural } from '../lib/format.js';
import { h, icon, byId, badge, copyText, nextFrame } from './dom.js';
import { createMarkdownView } from './markdown-dom.js';

const EXAMPLES = [
  'How does mi_malloc allocate small blocks?',
  'What happens when a block is freed from another thread?',
  'How are abandoned segments reclaimed?',
];

const STOP_REASONS = { eos: 'complete', length: 'token limit reached', cancelled: 'cancelled', stop: 'complete' };

export function createConversation({ api, store, toast, confirm, showInspector }) {
  const list = byId('messages');
  const form = byId('composer');
  const input = byId('question');
  const send = byId('send');
  const sendLabel = send.querySelector('.btn-text');
  const hint = byId('composer-hint');
  const clearButton = byId('clear-history');
  const turns = [];
  let controller = null;  // AbortController of the running query
  let follow = true;      // the view follows the newest content until the user scrolls away

  // ---- rendering --------------------------------------------------------------

  function nearBottom() {
    return list.scrollHeight - list.scrollTop - list.clientHeight < 96;
  }

  function scrollToBottom() {
    list.scrollTop = list.scrollHeight;
  }

  list.addEventListener('scroll', () => {
    follow = nearBottom();
  }, { passive: true });

  function renderEmpty() {
    if (turns.length > 0) {
      const old = list.querySelector('.empty');
      if (old) old.remove();
      return;
    }
    if (list.querySelector('.empty')) return;
    const examples = h('ul', { class: 'examples' }, EXAMPLES.map((text) =>
      h('li', null, h('button', { type: 'button', class: 'example', text, on: {
        click: () => {
          input.value = text;
          resize();
          input.focus();
        },
      } }))));
    list.appendChild(h('div', { class: 'empty' },
      h('p', { class: 'empty-title', text: 'Ask about the indexed code' }),
      h('p', { text: 'Answers are generated on this machine from the units that the retrieval selects. ' +
                     'The inspector shows which units were used and why.' }),
      examples));
  }

  // Inline code of an answer that names a retrieved unit, or cites it as path:line.
  function resolver(turn) {
    return (text) => {
      const units = turn.retrieval ? turn.retrieval.units : null;
      if (!units) return null;
      const byName = units.find((u) => u.name !== '' && u.name === text);
      if (byName) return byName.id;
      const match = /^(.+?):(\d+)(?:-\d+)?$/.exec(text);
      if (!match) return null;
      const line = Number(match[2]);
      const cited = units.find((u) => (u.path === match[1] || u.path.endsWith('/' + match[1])) &&
                                      line >= u.start_line && line <= u.end_line);
      return cited ? cited.id : null;
    };
  }

  function inspect(turn, unitId) {
    if (turn.retrieval) {
      store.set('retrieval', turn.retrieval);
      store.set('highlight', turn.retrieval.units.map((u) => u.id));
      if (turn.transaction) store.set('transaction', turn.transaction);
      showInspector('retrieval');
      if (unitId !== undefined) store.set('focusUnit', unitId);
    } else {
      store.set('highlight', turn.units.slice());
      showInspector('groups');
    }
  }

  function ratingButton(turn, value, name, label) {
    const button = h('button', { type: 'button', class: 'btn-icon rate', 'aria-label': label, title: label,
                                 'aria-pressed': String(turn.rating === value) }, icon(name));
    button.addEventListener('click', async () => {
      if (turn.rating === value || !turn.id) return;
      const before = turn.rating;
      turn.rating = value;
      updateActions(turn);
      try {
        await api.feedback(turn.id, value);
        toast.info(value > 0 ? 'Marked as helpful.' : 'Marked as not helpful.');
      } catch (error) {
        turn.rating = before;
        updateActions(turn);
        toast.error(`The rating was not saved: ${error.message}`);
      }
    });
    return button;
  }

  function updateActions(turn) {
    const actions = turn.view.actions;
    actions.replaceChildren();
    if (turn.streaming) return;
    if (turn.id && turn.answer !== '') {
      actions.appendChild(ratingButton(turn, 1, 'thumb-up', 'Helpful'));
      actions.appendChild(ratingButton(turn, -1, 'thumb-down', 'Not helpful'));
    }
    if (turn.answer !== '') {
      const copy = h('button', { type: 'button', class: 'btn-icon', 'aria-label': 'Copy the answer', title: 'Copy the answer' },
        icon('copy'));
      copy.addEventListener('click', async () => {
        if (await copyText(turn.answer)) toast.info('Answer copied.');
        else toast.warn('The clipboard is not available.');
      });
      actions.appendChild(copy);
    }
    const count = turn.retrieval ? turn.retrieval.units.length : turn.units.length;
    if (count > 0) {
      const label = turn.retrieval ? `Inspect retrieval (${plural(count, 'unit')})` : `Show ${plural(count, 'unit')} on the map`;
      actions.appendChild(h('button', { type: 'button', class: 'btn btn-quiet btn-small', text: label,
                                        on: { click: () => inspect(turn) } }));
    }
    if (turn.transaction) {
      actions.appendChild(h('button', { type: 'button', class: 'btn btn-quiet btn-small', text: 'Review edit', on: {
        click: () => {
          store.set('transaction', turn.transaction);
          showInspector('edits');
        },
      } }));
    }
  }

  function updateMeta(turn) {
    const meta = turn.view.meta;
    meta.replaceChildren();
    const parts = [];
    if (turn.time) parts.push(formatDateTime(turn.time));
    const done = turn.done;
    if (done) {
      parts.push(done.extractive ? 'extractive answer (no chat model)' : done.model || 'model');
      parts.push(`${formatCount(done.prompt_tokens)} + ${formatCount(done.completion_tokens)} tokens`);
      parts.push(formatMillis(done.total_ms));
      const reason = STOP_REASONS[done.stop_reason] || done.stop_reason;
      if (reason && reason !== 'complete') parts.push(reason);
    }
    parts.forEach((text, i) => {
      if (i > 0) meta.appendChild(h('span', { class: 'sep', 'aria-hidden': 'true', text: '\u{b7}' }));
      meta.appendChild(h('span', { text }));
    });
  }

  function renderAnswer(turn) {
    const view = turn.view;
    view.markdown.render(turn.answer);
    view.status.replaceChildren();
    if (turn.streaming) {
      const waiting = turn.answer === '';
      const text = waiting ? (turn.retrieval ? 'Reading the context' : 'Searching the index') : 'Writing';
      view.status.appendChild(h('span', { class: 'working' },
        h('span', { class: 'working-dots', 'aria-hidden': 'true' }, h('i', null), h('i', null), h('i', null)),
        h('span', { text })));
    } else if (turn.error) {
      view.status.appendChild(h('div', { class: 'banner', 'data-tone': 'bad' }, icon('error'),
        h('div', { class: 'banner-text' }, h('strong', { text: turn.answer === '' ? 'No answer.' : 'The answer is incomplete.' }),
          h('span', { text: ' ' + turn.error }))));
    } else if (turn.stopped) {
      view.status.appendChild(h('p', { class: 'note', text: 'Stopped.' }));
    }
  }

  // `before` is the turn in front of which the new one is placed; null appends.
  function createTurnView(turn, before = null) {
    const body = h('div', { class: 'md' });
    const status = h('div', { class: 'msg-status' });
    const meta = h('div', { class: 'msg-meta' });
    const actions = h('div', { class: 'msg-actions' });
    const answer = h('div', { class: 'msg msg-assistant' },
      h('div', { class: 'msg-role', text: 'Answer' }), body, status,
      h('div', { class: 'msg-footer' }, meta, actions));
    const question = h('div', { class: 'msg msg-user' },
      h('div', { class: 'msg-role' }, 'You', turn.edit ? badge('edit', 'accent') : null),
      h('div', { class: 'msg-text', text: turn.question }));
    const article = h('article', { class: 'turn' }, question, answer);
    turn.view = {
      article, body, status, meta, actions,
      markdown: createMarkdownView(body, { resolve: resolver(turn), open: (unitId) => inspect(turn, unitId) }),
    };
    list.insertBefore(article, before ? before.view.article : null);
    renderAnswer(turn);
    updateMeta(turn);
    updateActions(turn);
  }

  function addTurn(fields, before = null) {
    const turn = { id: '', time: '', question: '', answer: '', rating: 0, units: [], retrieval: null, transaction: null,
                   done: null, error: '', stopped: false, streaming: false, edit: false, ...fields };
    const at = before ? turns.indexOf(before) : -1;
    if (at >= 0) turns.splice(at, 0, turn);
    else turns.push(turn);
    renderEmpty();
    createTurnView(turn, at >= 0 ? before : null);
    return turn;
  }

  // ---- sending ------------------------------------------------------------------

  function setBusy(busy) {
    store.set('busy', busy);
    send.classList.toggle('btn-primary', !busy);
    send.classList.toggle('btn-danger', busy);
    sendLabel.textContent = busy ? 'Stop' : 'Send';
    send.setAttribute('aria-label', busy ? 'Stop the answer' : 'Send the question');
  }

  function resize() {
    input.style.setProperty('height', 'auto');
    input.style.setProperty('height', Math.min(input.scrollHeight + 2, 220) + 'px');
  }

  async function ask() {
    const question = input.value.trim();
    if (question === '' || controller !== null) return;
    const edit = form.elements.mode.value === 'edit';
    input.value = '';
    resize();

    const turn = addTurn({ question, edit, streaming: true, time: new Date().toISOString() });
    // The answer is announced when it is complete, not piece by piece.
    turn.view.body.setAttribute('aria-live', 'polite');
    turn.view.body.setAttribute('aria-busy', 'true');
    follow = true;
    scrollToBottom();

    controller = new AbortController();
    setBusy(true);
    let cancelFrame = null;
    const schedule = () => {
      if (cancelFrame !== null) return;
      cancelFrame = nextFrame(() => {
        cancelFrame = null;
        renderAnswer(turn);
        if (follow) scrollToBottom();
      });
    };

    try {
      await api.query({ question, edit }, {
        onRetrieval: (retrieval) => {
          turn.retrieval = retrieval;
          turn.units = retrieval.units.filter((u) => u.in_context).map((u) => u.id);
          store.set('retrieval', retrieval);
          store.set('highlight', retrieval.units.map((u) => u.id));
          if (!edit) store.set('transaction', null);
          store.set('inspectorNews', 'retrieval');
          schedule();
        },
        onToken: (text) => {
          turn.answer += text;
          schedule();
        },
        onEdit: (transaction) => {
          turn.transaction = transaction;
          store.set('transaction', transaction);
          store.set('inspectorNews', 'edits');
        },
        onDone: (done) => {
          turn.done = done;
          if (typeof done.id === 'string') turn.id = done.id;
          if (typeof done.answer === 'string' && done.answer !== '') turn.answer = done.answer;
        },
      }, controller.signal);
    } catch (error) {
      if (isAbort(error)) turn.stopped = true;
      else turn.error = error.message || 'The query failed.';
    }

    if (cancelFrame !== null) cancelFrame();
    controller = null;
    turn.streaming = false;
    setBusy(false);
    const following = follow;
    // Citations can be resolved now that the retrieval is known for the whole answer.
    turn.view.markdown.reset();
    renderAnswer(turn);
    turn.view.body.setAttribute('aria-busy', 'false');
    updateMeta(turn);
    updateActions(turn);
    if (following) scrollToBottom();
    if (turn.error) toast.error(turn.error);
    store.set('answered', Date.now());
  }

  function stop() {
    if (controller !== null) controller.abort();
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (controller !== null) stop();
    else ask();
  });

  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    if (controller === null) ask();
  });
  input.addEventListener('input', resize);

  for (const radio of form.querySelectorAll('input[name="mode"]')) {
    radio.addEventListener('change', () => {
      const edit = form.elements.mode.value === 'edit';
      input.placeholder = edit ? 'Describe the change to make' : 'Ask about the code base';
      hint.textContent = edit
        ? 'The change is proposed as a diff; nothing is written until you apply it'
        : 'Enter to send, Shift+Enter for a new line';
    });
  }

  clearButton.addEventListener('click', async () => {
    if (turns.length === 0) return;
    const yes = await confirm({
      title: 'Clear the conversation?',
      text: 'The history is deleted on the server. Feedback that was already given stays part of the tuning data.',
      action: 'Clear',
      danger: true,
    });
    if (!yes) return;
    try {
      stop();
      await api.clearHistory();
      turns.length = 0;
      list.replaceChildren();
      renderEmpty();
      toast.info('History cleared.');
    } catch (error) {
      toast.error(`The history was not cleared: ${error.message}`);
    }
  });

  // ---- history ------------------------------------------------------------------

  async function load() {
    try {
      const history = await api.history();
      if (!Array.isArray(history)) throw new Error('unexpected response');
      // A question may have been asked while the history was on its way: the
      // history is older and goes in front of it, and an answer that is both
      // on the page and in the history is shown once.
      const first = turns.length > 0 ? turns[0] : null;
      const shown = new Set(turns.map((turn) => turn.id).filter((id) => id !== ''));
      for (const entry of history) {
        const id = String(entry.id || '');
        if (shown.has(id)) continue;
        addTurn({
          id, time: entry.time || '', question: String(entry.question || ''),
          answer: String(entry.answer || ''), rating: entry.rating === 1 || entry.rating === -1 ? entry.rating : 0,
          units: Array.isArray(entry.units) ? entry.units.filter(Number.isInteger) : [],
        }, first);
      }
      list.scrollTop = list.scrollHeight;
    } catch (error) {
      toast.error(`The history could not be loaded: ${error.message}`);
    }
    renderEmpty();
  }

  renderEmpty();
  resize();
  return { load, focus: () => input.focus() };
}
