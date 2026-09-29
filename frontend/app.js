// programmersRAG web UI: entry point.
//
// Plain ES modules, no build step and no network dependency besides the API of
// the server that delivered this page. Model output and file content are
// untrusted: nothing in this application assigns markup; see js/ui/dom.js.

import { createApi } from './js/api.js';
import { createStore } from './js/store.js';
import { byId } from './js/ui/dom.js';
import { createToasts } from './js/ui/toast.js';
import { createTabs } from './js/ui/tabs.js';
import { createConfirm } from './js/ui/confirm.js';
import { createHeader } from './js/ui/header.js';
import { createConversation } from './js/ui/conversation.js';
import { createRetrievalPanel } from './js/ui/retrieval.js';
import { createContextPanel } from './js/ui/context-quality.js';
import { createExecutionPanel } from './js/ui/execution.js';
import { createGroupsPanel } from './js/ui/groups.js';
import { createEditsPanel } from './js/ui/edits.js';
import { createTuningPanel } from './js/ui/tuning.js';
import { createSettings } from './js/ui/settings.js';

const POLL_IDLE = 5000;     // status poll while nothing happens
const POLL_ACTIVE = 1000;   // while the index is built or the tuning pipeline runs
const POLL_OFFLINE = 3000;  // while the server does not answer
const HIDDEN_FACTOR = 4;    // a page that is not visible polls this much slower

function start() {
  const api = createApi('');
  const store = createStore({
    status: null, tune: null, progress: null, retrieval: null, transaction: null, groups: null, highlight: [],
    focusUnit: null, busy: false, offline: false, building: false,
  });
  const toast = createToasts();
  const confirm = createConfirm();
  const workspace = byId('workspace');
  const viewButtons = Array.from(document.querySelectorAll('.view-button'));
  const wide = window.matchMedia('(min-width: 960px)');

  // ---- views: on a narrow screen one pane is shown at a time -------------------

  function showView(name) {
    workspace.dataset.view = name;
    for (const button of viewButtons) button.setAttribute('aria-pressed', String(button.dataset.view === name));
    if (name === 'inspector') {
      byId('inspector-news').hidden = true;
      tabs.refresh();
      if (tabs.selected() === 'groups') groups.show();
    }
  }
  for (const button of viewButtons) button.addEventListener('click', () => showView(button.dataset.view));

  // ---- inspector ---------------------------------------------------------------

  const tabs = createTabs(byId('inspector-tabs'), (name) => {
    tabs.mark(name, false);
    if (name === 'groups') groups.show();
  });

  function showInspector(tab) {
    tabs.select(tab);
    showView('inspector');
  }

  createRetrievalPanel({ api, store });
  createContextPanel({ store });
  createExecutionPanel({ store });
  const groups = createGroupsPanel({ api, store, toast });
  createEditsPanel({ api, store, toast });
  createTuningPanel({ api, store, toast });
  const settings = createSettings({ api, store, toast });
  createHeader({ api, store, toast, openSettings: () => settings.open() });
  const conversation = createConversation({ api, store, toast, confirm, showInspector });

  // New content in the inspector: an edit is shown at once, because the user
  // asked for it; a retrieval only marks its tab, the answer is what is read first.
  store.on('inspectorNews', (tab) => {
    if (tab === 'edits') {
      tabs.select('edits');
    } else if (tabs.selected() === 'edits' || tabs.selected() === 'tuning') {
      tabs.select('retrieval');
    }
    if (!wide.matches && workspace.dataset.view !== 'inspector') byId('inspector-news').hidden = false;
  });

  // ---- status polling ------------------------------------------------------------

  let timer = 0;
  let polling = false;
  let builtAt = null;

  function interval() {
    if (store.get('offline')) return POLL_OFFLINE;
    const status = store.get('status');
    const tune = store.get('tune');
    const building = store.get('building') || (status && status.index && status.index.building);
    const tuning = tune && tune.enabled && tune.stage !== 'disabled' && tune.stage !== 'waiting' && tune.stage !== 'failed';
    return building || tuning ? POLL_ACTIVE : POLL_IDLE;
  }

  async function poll() {
    if (polling) return;
    polling = true;
    clearTimeout(timer);
    try {
      const status = await api.status();
      if (store.get('offline')) {
        store.set('offline', false);
        toast.good('The server answers again.');
      }
      store.set('status', status);
      if (status && status.tune && typeof status.tune === 'object' && 'stage' in status.tune) store.set('tune', status.tune);

      const index = status && status.index ? status.index : {};
      if (index.building || store.get('building')) {
        const progress = await api.indexProgress();
        store.set('progress', progress);
        if (progress && progress.running === false) {
          store.set('building', false);
          if (progress.stage === 'failed') toast.error(`The index build failed: ${(progress.report && progress.report.error) || progress.message || 'unknown error'}`);
          else toast.good('The index was rebuilt.');
          store.set('status', await api.status());
        }
      }
      const built = store.get('status').index ? String(store.get('status').index.built_at || '') : '';
      if (builtAt !== null && built !== builtAt) groups.reload();
      builtAt = built;
    } catch (error) {
      if (!store.get('offline')) {
        store.set('offline', true);
        toast.error(`The server is not reachable: ${error.message}`);
      }
    } finally {
      polling = false;
      timer = setTimeout(poll, interval() * (document.hidden ? HIDDEN_FACTOR : 1));
    }
  }

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) poll();
  });
  store.on('refresh', () => poll());
  store.on('building', (on) => {
    if (on) poll();
  });
  store.on('answered', () => poll());

  // ---- keyboard ------------------------------------------------------------------

  document.addEventListener('keydown', (event) => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    const typing = target instanceof HTMLElement &&
      (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
    if (event.key === '/' && !typing && !document.querySelector('dialog[open]')) {
      event.preventDefault();
      showView('conversation');
      conversation.focus();
    }
  });

  poll();
  conversation.load();
}

start();
