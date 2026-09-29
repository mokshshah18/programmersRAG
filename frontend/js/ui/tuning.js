// Inspector, Tuning tab: the opportunistic fine-tuning pipeline.
//
// The pipeline runs only while the machine is idle. The tab shows the stage
// it is in, why it is or is not running, and every run with its losses and
// with the promotion gate that decides whether the tuned model is served.
//
// The status is polled; the tab therefore updates its elements in place and
// rebuilds the table of runs only when the runs change, so that focus and
// scroll position survive an update.

import { formatCount, formatFixed, formatPercent, formatSeconds, formatAgo, formatDateTime, formatSigned,
         secondsBetween, plural, isNumber } from '../lib/format.js';
import { h, byId, badge, banner, fact, replace } from './dom.js';

const STAGES = ['waiting', 'collecting', 'training', 'calibrating', 'quantizing', 'evaluating', 'promoting'];
const STAGE_TEXT = {
  disabled: 'Switched off',
  waiting: 'Waiting for the machine to be idle',
  collecting: 'Building training examples from the index and from feedback',
  training: 'Fine-tuning the float32 base model',
  calibrating: 'Collecting the importance matrix on repository text',
  quantizing: 'Quantising with the importance matrix',
  evaluating: 'Promotion gate: comparing the candidate with the serving model',
  promoting: 'Swapping the serving model; the previous one is kept for rollback',
  failed: 'The last run failed',
};
const ARROW = '\u{2192}';

function lossTable(run) {
  const train = Array.isArray(run.train_loss) ? run.train_loss : [];
  const valid = Array.isArray(run.validation_loss) ? run.validation_loss : [];
  const epochs = Math.max(train.length, valid.length);
  if (epochs === 0) return h('p', { class: 'note', text: 'No epoch was completed.' });
  return h('table', { class: 'table loss' },
    h('caption', { text: 'Loss per epoch (nats per token)' }),
    h('thead', null, h('tr', null,
      h('th', { scope: 'col', text: 'Epoch' }), h('th', { scope: 'col', class: 'num', text: 'Training' }),
      h('th', { scope: 'col', class: 'num', text: 'Validation' }))),
    h('tbody', null, Array.from({ length: epochs }, (_, e) => h('tr', null,
      h('th', { scope: 'row', text: String(e + 1) }),
      h('td', { class: 'num', text: formatFixed(train[e], 3) }),
      h('td', { class: 'num', text: formatFixed(valid[e], 3) })))));
}

function gateList(gate) {
  if (!gate || typeof gate !== 'object') return h('p', { class: 'note', text: 'The gate was not reached.' });
  const pair = (base, candidate) =>
    `${formatFixed(base, 3)} ${ARROW} ${formatFixed(candidate, 3)}` +
    (isNumber(base) && isNumber(candidate) ? ` (${formatSigned(candidate - base, 3)})` : '');
  return h('div', { class: 'gate' },
    h('div', { class: 'gate-head' }, h('span', { class: 'gate-title', text: 'Promotion gate' }),
      gate.pass ? badge('passed', 'good') : badge('not passed', 'bad')),
    h('dl', { class: 'facts facts-column' },
      fact('Domain NLL', pair(gate.domain_nll_base, gate.domain_nll_candidate),
           'Negative log-likelihood on held-out examples of this code base: base, then candidate. Must improve.'),
      fact('p-value', formatFixed(gate.p_value, 3), 'One-sided paired test of the improvement over held-out examples'),
      fact('General NLL', pair(gate.general_nll_base, gate.general_nll_candidate),
           'Negative log-likelihood on the general probe text. Must not get worse by more than the margin.'),
      fact('Quantisation KL', formatFixed(gate.quantization_kl, 3),
           'Mean KL divergence per token between the quantised and the unquantised candidate'),
      fact('Reason', gate.reason ? String(gate.reason) : gate.pass ? 'all conditions met' : 'not given')));
}

function runRows(run) {
  const duration = isNumber(run.seconds) ? run.seconds : secondsBetween(run.started, run.finished);
  const outcome = run.error ? badge('failed', 'bad') : run.promoted ? badge('promoted', 'good') : badge('not promoted', 'warn');
  return [
    h('tr', { class: 'run' },
      h('th', { scope: 'row' }, h('code', { text: String(run.id || '') })),
      h('td', { text: formatDateTime(run.started) }),
      h('td', { class: 'num', text: formatSeconds(duration) }),
      h('td', { text: String(run.stage || '') }),
      h('td', { class: 'num', text: formatCount(run.examples) }),
      h('td', { class: 'num', text: formatCount(run.epochs) }),
      h('td', null, outcome)),
    // The box keeps the width of the visible part of the table, so that the
    // details can be read without scrolling sideways on a narrow screen.
    h('tr', { class: 'run-detail' },
      h('td', { colSpan: 7 },
        h('div', { class: 'run-box' },
          run.error ? banner('bad', 'Error.', String(run.error)) : null,
          h('div', { class: 'run-grid' }, lossTable(run), gateList(run.gate)),
          h('dl', { class: 'facts' },
            fact('Finished', formatDateTime(run.finished)),
            fact('Model', h('code', { text: String(run.model || '') })))))),
  ];
}

export function createTuningPanel({ api, store, toast }) {
  const panel = byId('panel-tuning');

  const notice = h('div', null);
  const steps = STAGES.map((stage) => h('li', { class: 'step', 'data-stage': stage }, h('span', { class: 'step-dot', 'aria-hidden': 'true' }),
    h('span', { class: 'step-name', text: stage })));
  const stepper = h('ol', { class: 'stepper', 'aria-label': 'Stages of the pipeline' }, steps);
  const stageText = h('p', { class: 'stage-text' });
  const progress = h('progress', { class: 'progress', max: 1, value: 0, id: 'tune-progress' });
  const progressLabel = h('label', { class: 'progress-label', htmlFor: 'tune-progress' });
  const message = h('p', { class: 'note', 'aria-live': 'polite' });
  const run = h('button', { type: 'button', class: 'btn', text: 'Run now' });
  const idleFacts = h('dl', { class: 'facts' });
  const modelFacts = h('dl', { class: 'facts' });
  const runsBody = h('tbody', null);
  const runsNote = h('p', { class: 'note' });
  let runsKey = '';

  replace(panel,
    notice,
    h('section', null,
      h('div', { class: 'section-head' }, h('h3', { class: 'section-title', text: 'Pipeline' }), run),
      stepper, stageText,
      h('div', { class: 'progress-row' }, progressLabel, progress),
      message),
    h('section', null,
      h('h3', { class: 'section-title', text: 'Why it is or is not running' }),
      idleFacts),
    h('section', null,
      h('h3', { class: 'section-title', text: 'Models and data' }),
      modelFacts),
    h('section', null,
      h('h3', { class: 'section-title', text: 'Runs' }),
      runsNote,
      h('div', { class: 'table-scroll runs-scroll' },
        h('table', { class: 'table runs' },
          h('caption', { class: 'visually-hidden', text: 'Fine-tuning runs, newest first' }),
          h('thead', null, h('tr', null,
            h('th', { scope: 'col', text: 'Run' }), h('th', { scope: 'col', text: 'Started' }),
            h('th', { scope: 'col', class: 'num', text: 'Duration' }), h('th', { scope: 'col', text: 'Last stage' }),
            h('th', { scope: 'col', class: 'num', text: 'Examples' }), h('th', { scope: 'col', class: 'num', text: 'Epochs' }),
            h('th', { scope: 'col', text: 'Outcome' }))),
          runsBody))));

  run.addEventListener('click', async () => {
    run.disabled = true;
    try {
      const result = await api.runTune();
      if (result && typeof result === 'object' && 'stage' in result) store.set('tune', result);
      toast.info('The pipeline was started.');
      store.set('refresh', Date.now());
    } catch (error) {
      toast.error(`The pipeline was not started: ${error.message}`);
      render();
    }
  });

  function render() {
    const tune = store.get('tune');
    if (!tune) {
      notice.replaceChildren(h('p', { class: 'note', text: 'The tuning status is not known yet.' }));
      return;
    }
    const stage = String(tune.stage || 'disabled');
    const idle = tune.idle || {};
    const active = STAGES.indexOf(stage) > 0;

    if (tune.available === false) {
      notice.replaceChildren(banner('warn', 'Fine-tuning is not available.',
        `${idle.reason ? idle.reason + '. ' : ''}It needs a build with llama.cpp and a tuning base model (models.tuning_base).`));
    } else if (stage === 'failed') {
      notice.replaceChildren(banner('bad', 'The last run failed.', String(tune.message || '')));
    } else if (!tune.enabled) {
      notice.replaceChildren(banner('info', 'Fine-tuning is switched off.',
        'Switch it on in the header. The pipeline then runs by itself whenever the machine is idle.'));
    } else {
      notice.replaceChildren();
    }

    const at = STAGES.indexOf(stage);
    steps.forEach((step, i) => {
      const state = at < 0 ? 'todo' : i < at ? 'done' : i === at ? 'current' : 'todo';
      step.dataset.state = state;
      if (state === 'current') step.setAttribute('aria-current', 'step');
      else step.removeAttribute('aria-current');
    });
    stageText.textContent = STAGE_TEXT[stage] || stage;
    const fraction = active && isNumber(tune.progress) ? Math.min(1, Math.max(0, tune.progress)) : 0;
    progress.value = fraction;
    progress.hidden = !active;
    progressLabel.hidden = !active;
    progressLabel.textContent = active ? `${stage}: ${formatPercent(fraction)}` : '';
    message.textContent = String(tune.message || '');
    run.disabled = tune.available === false || !tune.enabled || active;
    run.title = tune.available === false ? 'Fine-tuning is not available'
      : !tune.enabled ? 'Switch fine-tuning on first' : active ? 'The pipeline is running' : 'Start without waiting for the machine to be idle';

    replace(idleFacts,
      fact('State', active ? badge('running', 'accent') : idle.idle ? badge('idle', 'good') : badge('not idle', 'neutral')),
      fact('Reason', String(idle.reason || 'not given')),
      fact('Last activity', formatAgo(idle.seconds_since_activity)),
      fact('CPU load', formatPercent(idle.cpu_load)),
      fact('Power', idle.on_ac_power === true ? 'mains' : idle.on_ac_power === false ? 'battery' : 'unknown'));
    replace(modelFacts,
      fact('Pending examples', formatCount(tune.pending_examples), 'Examples collected since the last run'),
      fact('Serving model', h('code', { text: String(tune.serving_model || 'none') })),
      fact('Base model', h('code', { text: String(tune.base_model || 'none') })));

    const runs = Array.isArray(tune.runs) ? tune.runs : [];
    const key = JSON.stringify(runs);
    if (key !== runsKey) {
      runsKey = key;
      const promoted = runs.filter((r) => r.promoted).length;
      runsNote.textContent = runs.length === 0 ? 'No run has been recorded yet.'
        : `${plural(runs.length, 'run')}, ${formatCount(promoted)} promoted. Every run is recorded, promoted or not.`;
      runsBody.replaceChildren(...runs.slice().reverse().flatMap(runRows));
    }
  }

  store.on('tune', render);
  render();
  return { render };
}
