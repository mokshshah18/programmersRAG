// Header: index status, model name, the fine-tuning switch and the settings
// button.

import { formatCount, formatPercent, formatDateTime, isNumber } from '../lib/format.js';
import { byId } from './dom.js';

const STAGE_LABEL = { disabled: 'off', waiting: 'waiting for idle' };

export function createHeader({ api, store, toast, openSettings }) {
  const version = byId('version');
  const indexChip = byId('index-chip');
  const indexText = byId('index-status');
  const modelChip = byId('model-chip');
  const modelText = byId('model-status');
  const tuneSwitch = byId('tune-switch');
  const tuneStage = byId('tune-stage');
  const settings = byId('settings-button');
  let switching = false;

  function setChip(chip, state) {
    chip.querySelector('.dot').dataset.state = state;
  }

  function renderIndex() {
    const status = store.get('status');
    const progress = store.get('progress');
    if (store.get('offline')) {
      setChip(indexChip, 'bad');
      indexText.textContent = 'server not reachable';
      indexChip.title = 'The server does not answer. The page keeps trying.';
      return;
    }
    if (!status || !status.index) {
      setChip(indexChip, 'off');
      indexText.textContent = 'loading';
      return;
    }
    const index = status.index;
    if (index.building || (progress && progress.running)) {
      setChip(indexChip, 'busy');
      const p = progress && progress.running ? progress : null;
      indexText.textContent = p
        ? `building: ${p.stage}${p.total > 0 ? ` ${formatCount(p.done)}/${formatCount(p.total)}` : ''}`
        : 'building';
      indexChip.title = p && p.message ? String(p.message) : 'The index is being built';
    } else if (index.ready) {
      setChip(indexChip, index.calibrated === false ? 'warn' : 'good');
      indexText.textContent = `${formatCount(index.units)} units, ${formatCount(index.files)} files, ` +
        `${formatCount(index.groups)} groups${index.calibrated === false ? ', not calibrated' : ''}`;
      indexChip.title = `Built ${formatDateTime(index.built_at)} with ${index.embedder || 'an unknown embedder'}; ` +
        `${formatCount(index.functions)} functions, ${formatCount(index.call_edges)} call edges, codes ${index.codes}`;
    } else {
      setChip(indexChip, 'warn');
      indexText.textContent = index.units > 0 ? 'stale, rebuild needed' : 'not built';
      indexChip.title = 'Open the settings to rebuild the index';
    }
  }

  function renderModel() {
    const status = store.get('status');
    if (!status || !status.models) {
      setChip(modelChip, 'off');
      modelText.textContent = store.get('offline') ? 'unknown' : 'loading';
      return;
    }
    const models = status.models;
    if (models.chat_loaded && models.chat) {
      setChip(modelChip, 'good');
      modelText.textContent = String(models.chat);
      modelChip.title = `Chat model, ${models.backend || 'local'}`;
    } else {
      setChip(modelChip, 'warn');
      modelText.textContent = 'none (extractive answers)';
      modelChip.title = models.llama === false
        ? 'This build has no llama.cpp: answers list the selected units'
        : 'No chat model is configured: answers list the selected units';
    }
    version.textContent = status.version ? String(status.version) : '';
  }

  function renderTune() {
    const tune = store.get('tune');
    if (!tune) {
      tuneSwitch.disabled = true;
      tuneSwitch.setAttribute('aria-checked', 'false');
      tuneStage.textContent = '';
      return;
    }
    const available = tune.available !== false;
    tuneSwitch.disabled = !available || switching;
    tuneSwitch.setAttribute('aria-checked', String(tune.enabled === true));
    const stage = String(tune.stage || 'disabled');
    let text = STAGE_LABEL[stage] || stage;
    if (!available) text = 'not available';
    else if (!(stage in STAGE_LABEL) && stage !== 'failed' && isNumber(tune.progress)) text = `${stage} ${formatPercent(tune.progress)}`;
    tuneStage.textContent = text;
    tuneStage.dataset.state = !available ? 'off' : stage === 'failed' ? 'bad' : tune.enabled ? 'on' : 'off';
    tuneSwitch.title = available
      ? 'Fine-tune the local model on this code base while the machine is idle'
      : tune.idle && tune.idle.reason ? String(tune.idle.reason) : 'Fine-tuning is not available';
  }

  tuneSwitch.addEventListener('click', async () => {
    const tune = store.get('tune');
    if (!tune || switching) return;
    const enable = tune.enabled !== true;
    switching = true;
    // Show the new position at once; it is taken back if the server refuses.
    tuneSwitch.setAttribute('aria-checked', String(enable));
    tuneSwitch.disabled = true;
    try {
      const result = await api.setTune(enable);
      switching = false;
      store.set('tune', result && typeof result === 'object' && 'enabled' in result ? result : { ...tune, enabled: enable });
      toast.info(enable ? 'Fine-tuning is on. It runs while the machine is idle.' : 'Fine-tuning is off.');
      store.set('refresh', Date.now());
    } catch (error) {
      switching = false;
      toast.error(`Fine-tuning was not switched: ${error.message}`);
      renderTune();
    }
  });

  settings.addEventListener('click', () => openSettings());

  store.on('status', () => {
    renderIndex();
    renderModel();
  });
  store.on('progress', renderIndex);
  store.on('offline', () => {
    renderIndex();
    renderModel();
  });
  store.on('tune', renderTune);
  renderIndex();
  renderModel();
  renderTune();
}
