// State and behaviour of the development mock server: everything the HTTP
// layer (mock-server.js) answers comes from here.
//
// The engine imitates the real one closely enough to exercise the user
// interface: scores are lexical instead of learned, but the probabilistic
// trace and the context pruning follow the formulas of retrieve/trace.hpp and
// context/rot.hpp, so that the numbers shown are consistent with each other.
//
// Time never advances by itself. Every request passes the current time, and the
// index build and the tuning pipeline compute where they are from the time that
// has passed. `speed` multiplies the passage of time (tests use a high value).

import { buildCorpus, makeRandom, subwords } from './mock-corpus.js';

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const VERSION = '2.0.0';
const CHAT_MODEL = 'qwen2.5-coder-0.5b-instruct-q4_k_m';
const BASE_MODEL = 'qwen2.5-coder-0.5b-instruct-fp16';
const EMBEDDER = 'Qwen3-Embedding-0.6B-Q8_0';

const DEFAULT_SETTINGS = {
  corpus: {
    roots: ['C:/Users/dev/Documents/GitHub/programmersRAG/Demo repo/mimalloc'],
    extensions: ['.c', '.h', '.cc', '.cpp', '.cxx', '.hpp', '.hh', '.py'],
    exclude: ['/.git/', '/build/', '/out/', '/.deps/', '/node_modules/'],
    max_file_bytes: 1048576,
  },
  paths: { data: 'data', frontend: 'frontend' },
  models: {
    embedding: 'models/Qwen3-Embedding-0.6B-Q8_0.gguf',
    embedding_query_prefix: 'Instruct: Given a question about a C code base, retrieve the code that answers it\nQuery: ',
    embedding_context: 2048,
    chat: 'models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf',
    chat_context: 4096,
    tuning_base: 'models/qwen2.5-coder-0.5b-instruct-fp16.gguf',
    threads: 0,
  },
  exec: { threads: 0, block_bytes: 0 },
  index: { codes: 'rabitq' },
  kde: { dim: 32, drop_top: 0, kappa: 0 },
  retrieve: { tau: 0.25, max_units: 64, max_hops: 6, lift: { callee: 4.0, caller: 2.0, sibling: 1.5 } },
  context: { theta: 0.25, token_budget: 3000, history_budget: 600 },
  generate: { max_tokens: 512, temperature: 0.2 },
  edit: { allow_fuzzy: false, validate_syntax: true },
  server: { host: '127.0.0.1', port: 8000 },
  tune: {
    enabled: false, idle_seconds: 300, max_cpu_load: 0.25, require_ac_power: true, min_examples: 64,
    distractors: 4, oracle_fraction: 0.8, epochs: 2, trainable_blocks: 2, quantization: 'Q4_K_M',
    gate_alpha: 0.05, gate_margin: 0.02, gate_kl: 0.05,
  },
};

// Stage durations in seconds at speed 1.
const INDEX_STAGES = [
  ['discover', 0.4, 'looking for source files'],
  ['parse', 0.9, 'parsing files into units'],
  ['graph', 0.4, 'building the call graph'],
  ['embed', 3.0, 'embedding units'],
  ['density', 0.9, 'estimating the density and its modes'],
  ['codes', 0.4, 'encoding vectors into cache blocks'],
  ['calibrate', 0.8, 'calibrating scores'],
  ['save', 0.3, 'writing the index'],
];
const TUNE_STAGES = [
  ['collecting', 2.0], ['training', 6.0], ['calibrating', 2.5], ['quantizing', 2.5], ['evaluating', 2.5],
  ['promoting', 1.0],
];
// The real pipeline waits tune.idle_seconds (300 by default); nobody wants to
// wait that long in front of a mock, so the wait is capped.
const MAX_IDLE_WAIT = 8;

const STOP_WORDS = new Set(('a an and are as at be by do does for from how i in is it of on or that the this to what when ' +
  'where which who why with work works can could should would into about there their then than its your my we you ' +
  'explain show tell me please function functions code get gets called call calls happen happens use used').split(' '));

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function logit(p) {
  return Math.log(p / (1 - p));
}

function round(x, digits = 4) {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}

function hash32(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; ++i) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function isObject(x) {
  return x !== null && typeof x === 'object' && !Array.isArray(x);
}

function merge(target, patch) {
  for (const key of Object.keys(patch)) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;
    if (isObject(patch[key]) && isObject(target[key])) merge(target[key], patch[key]);
    else target[key] = isObject(patch[key]) ? merge({}, patch[key]) : patch[key];
  }
  return target;
}

function flatten(object, prefix, out) {
  for (const key of Object.keys(object)) {
    const path = prefix === '' ? key : prefix + '.' + key;
    if (isObject(object[key])) flatten(object[key], path, out);
    else out.push(path);
  }
  return out;
}

function iso(ms) {
  return new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z');
}

// Similarity of two units used as the cosine of their embeddings when the
// redundancy is computed: units with nearly the same name say nearly the same.
function similarity(a, b) {
  if (a.nameWords.length === 0 || b.nameWords.length === 0) return 0;
  const other = new Set(b.nameWords);
  let common = 0;
  for (const w of a.nameWords) if (other.has(w)) ++common;
  const union = a.nameWords.length + b.nameWords.length - common;
  const jaccard = union === 0 ? 0 : common / union;
  return round((a.group === b.group ? 0.85 : 0.6) * jaccard, 3);
}

export function createEngine(options = {}) {
  const speed = options.speed > 0 ? options.speed : 1;
  const llama = options.llama !== false;
  const start = typeof options.now === 'number' ? options.now : Date.now();
  const corpus = buildCorpus(options.seed);
  const { units, groups, files, byName } = corpus;
  const settings = merge({}, DEFAULT_SETTINGS);
  const functions = units.filter((u) => u.kind === 'function').length;
  const callEdges = units.reduce((sum, u) => sum + u.callees.length, 0);
  const siblingCache = new Map();

  const seconds = (from, to) => ((to - from) * speed) / 1000;

  // ---- index ------------------------------------------------------------------

  const index = { ready: true, stale: false, builtAt: start - 3600e3, build: null, report: null, calibrated: true };

  function blockBytes() {
    return settings.exec.block_bytes > 0 ? settings.exec.block_bytes : 16384;
  }

  function threads() {
    return settings.exec.threads > 0 ? settings.exec.threads : 24;
  }

  function blockCount() {
    // RaBitQ codes: 1024 bits and two side values per unit.
    const codeBytes = settings.index.codes === 'int8' ? 1024 : 136;
    return Math.max(1, Math.ceil((units.length * codeBytes) / blockBytes()));
  }

  function makeReport(total) {
    const s = { discover: 0.01, parse: 0.41, graph: 0.05, embed: 298.6, density: 1.22, codes: 0.09, calibrate: 2.04,
                save: 0.11 };
    s.total = round(Object.values(s).reduce((a, b) => a + b, 0), 2);
    return {
      files: files.length, units: units.length, functions, call_edges: callEdges, embedded: units.length, reused: 0,
      groups: groups.length, kappa: 41.5, reduced_dim: settings.kde.dim, explained_variance: 0.71,
      anisotropy_before: 0.62, anisotropy_after: 0.01, blocks: blockCount(), block_bytes: blockBytes(),
      seconds: s, mock_seconds: round(total, 2), error: '',
    };
  }

  index.report = makeReport(7.1);

  function advanceIndex(now) {
    if (!index.build) return;
    const total = INDEX_STAGES.reduce((sum, stage) => sum + stage[1], 0);
    if (seconds(index.build.startedAt, now) >= total) {
      index.report = makeReport(total);
      index.build = null;
      index.ready = true;
      index.stale = false;
      index.calibrated = true;
      index.builtAt = now;
    }
  }

  function indexProgress(now) {
    advance(now);
    if (!index.build) {
      return { running: false, stage: index.report ? 'done' : 'done', done: 0, total: 0,
               message: index.stale ? 'the index is stale' : index.report ? 'index built' : '', report: index.report };
    }
    let elapsed = seconds(index.build.startedAt, now);
    for (const [stage, duration, message] of INDEX_STAGES) {
      if (elapsed < duration) {
        const total = stage === 'discover' || stage === 'parse' ? files.length
          : stage === 'density' ? groups.length : stage === 'codes' ? blockCount() : units.length;
        return { running: true, stage, done: Math.min(total, Math.floor((elapsed / duration) * total)), total, message,
                 report: null };
      }
      elapsed -= duration;
    }
    return { running: true, stage: 'save', done: 1, total: 1, message: 'writing the index', report: null };
  }

  // ---- tuning -----------------------------------------------------------------

  const tune = {
    enabled: false, stage: 'disabled', stageStartedAt: start, resume: 0, forced: false, message: '',
    lastActivityAt: start, pending: 180, serving: CHAT_MODEL,
    runs: [
      { id: 'run-0001', started: iso(start - 50 * 3600e3), finished: iso(start - 50 * 3600e3 + 2400e3),
        stage: 'promoting', promoted: true, examples: 180, epochs: 2, train_loss: [1.9, 1.6],
        validation_loss: [1.8, 1.7],
        gate: { pass: true, domain_nll_base: 1.92, domain_nll_candidate: 1.71, p_value: 0.003, general_nll_base: 2.4,
                general_nll_candidate: 2.41, quantization_kl: 0.012, reason: '' },
        model: 'data/tune/run-0001/model-Q4_K_M.gguf', seconds: 2400.0, error: '' },
      { id: 'run-0002', started: iso(start - 26 * 3600e3), finished: iso(start - 26 * 3600e3 + 1985e3),
        stage: 'evaluating', promoted: false, examples: 96, epochs: 2, train_loss: [1.68, 1.61],
        validation_loss: [1.72, 1.71],
        gate: { pass: false, domain_nll_base: 1.71, domain_nll_candidate: 1.69, p_value: 0.21, general_nll_base: 2.41,
                general_nll_candidate: 2.42, quantization_kl: 0.014,
                reason: 'domain negative log-likelihood did not improve significantly (p = 0.21 > 0.05)' },
        model: 'data/tune/run-0002/model-Q4_K_M.gguf', seconds: 1985.0, error: '' },
    ],
    current: null,
  };

  function tuneAvailable() {
    return llama && settings.models.tuning_base !== '';
  }

  function cpuLoad(now) {
    const t = seconds(start, now);
    return round(0.11 + 0.05 * Math.sin(t / 7) + 0.03 * Math.sin(t / 2.3), 2);
  }

  function idleWait() {
    return Math.min(settings.tune.idle_seconds, MAX_IDLE_WAIT);
  }

  function finishRun(now) {
    const number = tune.runs.length + 1;
    const id = 'run-' + String(number).padStart(4, '0');
    const examples = Math.max(tune.pending, settings.tune.min_examples);
    const base = 1.6 + 0.05 * (number % 3);
    const train = Array.from({ length: settings.tune.epochs }, (_, e) => round(base + 0.25 - 0.17 * (e + 1), 3));
    const valid = Array.from({ length: settings.tune.epochs }, (_, e) => round(base + 0.22 - 0.09 * (e + 1), 3));
    // The duration agrees with the two time stamps, as it does in a real record.
    const duration = Math.max(0, (now - tune.current.startedAt) / 1000);
    tune.runs.push({
      id, started: iso(tune.current.startedAt), finished: iso(now), stage: 'promoting', promoted: true, examples,
      epochs: settings.tune.epochs, train_loss: train, validation_loss: valid,
      gate: { pass: true, domain_nll_base: round(base + 0.09, 3), domain_nll_candidate: round(base - 0.06, 3),
              p_value: 0.008, general_nll_base: 2.41, general_nll_candidate: 2.415, quantization_kl: 0.011,
              reason: '' },
      model: `data/tune/${id}/model-${settings.tune.quantization}.gguf`, seconds: round(duration, 1), error: '',
    });
    tune.serving = `${id}/model-${settings.tune.quantization}`;
    tune.pending = 0;
    tune.current = null;
    tune.resume = 0;
    tune.forced = false;
  }

  function advanceTune(now) {
    if (!tuneAvailable() || !tune.enabled) {
      tune.stage = 'disabled';
      tune.message = '';
      return;
    }
    for (let guard = 0; guard < 32; ++guard) {
      if (tune.stage === 'disabled') {
        tune.stage = 'waiting';
        tune.stageStartedAt = now;
      }
      if (tune.stage === 'waiting') {
        const enough = tune.pending >= settings.tune.min_examples;
        const idleAt = tune.lastActivityAt + (idleWait() * 1000) / speed;
        if (tune.forced || (enough && now >= idleAt)) {
          const startedAt = tune.forced ? tune.stageStartedAt : Math.max(idleAt, tune.stageStartedAt);
          if (!tune.current) tune.current = { startedAt };
          tune.stage = TUNE_STAGES[tune.resume][0];
          tune.stageStartedAt = startedAt;
          continue;
        }
        return;
      }
      const at = TUNE_STAGES.findIndex((s) => s[0] === tune.stage);
      if (at < 0) return;
      const duration = TUNE_STAGES[at][1];
      if (seconds(tune.stageStartedAt, now) < duration) return;
      const finishedAt = tune.stageStartedAt + (duration * 1000) / speed;
      if (at + 1 === TUNE_STAGES.length) {
        finishRun(finishedAt);
        tune.stage = 'waiting';
        tune.stageStartedAt = finishedAt;
        tune.lastActivityAt = Math.max(tune.lastActivityAt, finishedAt);
        return;
      }
      tune.resume = at + 1;
      tune.stage = TUNE_STAGES[at + 1][0];
      tune.stageStartedAt = finishedAt;
    }
  }

  function tuneStatus(now) {
    advance(now);
    const since = Math.max(0, seconds(tune.lastActivityAt, now));
    const load = cpuLoad(now);
    const available = tuneAvailable();
    const enough = tune.pending >= settings.tune.min_examples;
    const running = tune.stage !== 'disabled' && tune.stage !== 'waiting';
    const idle = since >= idleWait() && load < settings.tune.max_cpu_load;

    let reason;
    if (!available) reason = llama ? 'no tuning base model is configured' : 'this build has no llama.cpp';
    else if (!tune.enabled) reason = 'fine-tuning is switched off';
    else if (running) reason = tune.forced ? 'started by request' : `idle for ${Math.floor(since)} s, on AC power`;
    else if (!enough) reason = `waiting for examples: ${tune.pending} of ${settings.tune.min_examples}`;
    else if (since < idleWait()) reason = `user active ${Math.floor(since)} s ago`;
    else if (load >= settings.tune.max_cpu_load) reason = `CPU load ${load} is above ${settings.tune.max_cpu_load}`;
    else reason = 'idle';

    let progress = 0;
    let message = tune.message;
    if (running) {
      const at = TUNE_STAGES.findIndex((s) => s[0] === tune.stage);
      progress = Math.min(1, Math.max(0, seconds(tune.stageStartedAt, now) / TUNE_STAGES[at][1]));
      const epochs = settings.tune.epochs;
      const epoch = Math.min(epochs, 1 + Math.floor(progress * epochs));
      message = {
        collecting: `building examples: ${Math.floor(progress * Math.max(tune.pending, settings.tune.min_examples))}`,
        training: `epoch ${epoch} of ${epochs}, loss ${(1.92 - 0.3 * progress).toFixed(2)}`,
        calibrating: `importance matrix: window ${Math.floor(progress * 64)} of 64`,
        quantizing: `quantising to ${settings.tune.quantization}`,
        evaluating: 'promotion gate: held-out likelihood',
        promoting: 'swapping the serving model',
      }[tune.stage];
    }
    return {
      available, enabled: tune.enabled && available, stage: available ? tune.stage : 'disabled',
      progress: round(progress, 3), message: message || '',
      idle: { idle, seconds_since_activity: round(since, 1), cpu_load: load, on_ac_power: true, reason },
      pending_examples: tune.pending, serving_model: tune.serving, base_model: BASE_MODEL,
      runs: tune.runs,
    };
  }

  function setTune(enabled, now) {
    advance(now);
    if (enabled && !tuneAvailable()) {
      throw new HttpError(409, llama ? 'Fine-tuning needs a tuning base model (models.tuning_base).'
        : 'Fine-tuning is not available: this build has no llama.cpp.');
    }
    tune.enabled = enabled;
    settings.tune.enabled = enabled;
    if (enabled) {
      tune.stage = 'waiting';
      tune.stageStartedAt = now;
      tune.lastActivityAt = now;
      tune.message = '';
    } else {
      tune.stage = 'disabled';
      tune.forced = false;
      tune.message = '';
    }
    return tuneStatus(now);
  }

  function runTune(now) {
    advance(now);
    if (!tuneAvailable()) throw new HttpError(409, 'Fine-tuning is not available.');
    if (!tune.enabled) throw new HttpError(409, 'Fine-tuning is switched off. Switch it on first.');
    if (tune.stage !== 'waiting') throw new HttpError(409, `The pipeline is already running (${tune.stage}).`);
    tune.forced = true;
    tune.stageStartedAt = now;
    tune.message = '';
    return tuneStatus(now);
  }

  // A user request: the running stage is cancelled and resumes later from
  // the last finished one.
  function notifyActivity(now) {
    advance(now);
    tune.lastActivityAt = now;
    if (tune.stage !== 'disabled' && tune.stage !== 'waiting') {
      const stage = tune.stage;
      tune.resume = Math.max(0, TUNE_STAGES.findIndex((s) => s[0] === stage));
      tune.stage = 'waiting';
      tune.stageStartedAt = now;
      tune.forced = false;
      tune.message = `paused by user activity; resumes with ${stage}`;
    }
  }

  function advance(now) {
    advanceIndex(now);
    advanceTune(now);
  }

  // ---- status, settings, history ------------------------------------------------

  function status(now) {
    advance(now);
    return {
      version: VERSION,
      index: {
        ready: index.ready && !index.stale && !index.build, building: index.build !== null, files: files.length,
        units: units.length, functions, call_edges: callEdges, groups: groups.length, dim: 1024,
        reduced_dim: settings.kde.dim, kappa: 41.5, codes: settings.index.codes, blocks: blockCount(),
        block_bytes: blockBytes(), embedder: settings.models.embedding === '' ? 'hashing-384' : EMBEDDER,
        built_at: iso(index.builtAt), calibrated: index.calibrated, stale: index.stale,
      },
      models: {
        llama, chat: settings.models.chat === '' || !llama ? '' : tune.serving,
        chat_loaded: llama && settings.models.chat !== '', embedding_loaded: llama && settings.models.embedding !== '',
        backend: llama ? 'llama.cpp b11257, AVX2' : 'none',
      },
      cache: {
        detected: true, logical_cpus: 24, l1d_min_bytes: 32768, l1d_per_thread_min_bytes: 24576, line_bytes: 64,
        block_bytes: blockBytes(), threads: threads(), simd: 'avx2',
        classes: [
          { level: 1, type: 'data', size_bytes: 49152, sharing_threads: 2, instances: 8 },
          { level: 1, type: 'data', size_bytes: 32768, sharing_threads: 1, instances: 8 },
          { level: 1, type: 'instruction', size_bytes: 32768, sharing_threads: 2, instances: 8 },
          { level: 1, type: 'instruction', size_bytes: 65536, sharing_threads: 1, instances: 8 },
          { level: 2, type: 'unified', size_bytes: 2097152, sharing_threads: 2, instances: 8 },
          { level: 2, type: 'unified', size_bytes: 4194304, sharing_threads: 4, instances: 2 },
          { level: 3, type: 'unified', size_bytes: 31457280, sharing_threads: 24, instances: 1 },
        ],
      },
      tune: tuneStatus(now),
      roots: settings.corpus.roots.slice(),
    };
  }

  function patchSettings(patch, now) {
    if (!isObject(patch)) throw new HttpError(400, 'The body must be a JSON object.');
    advance(now);
    const tau = isObject(patch.retrieve) ? patch.retrieve.tau : undefined;
    if (tau !== undefined && !(typeof tau === 'number' && tau > 0 && tau < 1)) {
      throw new HttpError(400, 'retrieve.tau must be a number between 0 and 1.');
    }
    const keys = flatten(patch, '', []);
    merge(settings, patch);
    const stale = keys.some((k) => k.startsWith('corpus.') || k.startsWith('kde.') || k.startsWith('index.') ||
                                   k === 'models.embedding');
    if (stale) index.stale = true;
    if (isObject(patch.tune) && typeof patch.tune.enabled === 'boolean') {
      try {
        setTune(patch.tune.enabled, now);
      } catch {
        settings.tune.enabled = false;
      }
    }
    return settings;
  }

  function startIndex(now) {
    advance(now);
    if (index.build) throw new HttpError(409, 'An index build is already running.');
    if (settings.corpus.roots.length === 0) throw new HttpError(400, 'No corpus roots are configured.');
    notifyActivity(now);
    index.build = { startedAt: now };
    index.report = null;
    return { ok: true, started: true };
  }

  const history = [
    { id: 'q-000001', time: iso(start - 5400e3), question: 'Where is the fast path for small allocations?',
      answer: 'The fast path is `mi_heap_malloc_small_zero` (`src/alloc.c`). It looks the page up in ' +
        '`pages_free_direct` with `_mi_heap_get_free_small_page` and pops a block with `_mi_page_malloc_zero`; ' +
        'no lock and no atomic operation is involved.',
      names: ['mi_heap_malloc_small_zero', '_mi_page_malloc_zero', 'mi_heap_malloc'], rating: 1 },
    { id: 'q-000002', time: iso(start - 4800e3), question: 'What happens when a block is freed from another thread?',
      answer: '`mi_free` compares the thread id of the segment with the id of the caller. When they differ it ' +
        'takes the generic path, `mi_free_generic_mt`:\n\n1. the block is pushed on the **thread free list** of ' +
        'the page with a compare-and-swap;\n2. a page that is full is first moved to the delayed free list of ' +
        'the owning heap, so that the owner can reuse it.\n\nThe owner collects these blocks in ' +
        '`_mi_malloc_generic`.',
      names: ['mi_free', 'mi_free_generic_mt', '_mi_malloc_generic', 'mi_free_block_mt'], rating: 0 },
  ].map((h) => ({ id: h.id, time: h.time, question: h.question, answer: h.answer, rating: h.rating,
                  units: h.names.map((n) => byName.get(n)).filter((u) => u).map((u) => u.id) }));
  let nextQuery = history.length + 1;

  function feedback(body, now) {
    advance(now);
    if (!isObject(body) || typeof body.id !== 'string') throw new HttpError(400, 'id is missing.');
    if (body.rating !== 1 && body.rating !== -1) throw new HttpError(400, 'rating must be +1 or -1.');
    const entry = history.find((h) => h.id === body.id);
    if (!entry) throw new HttpError(404, `No history entry ${body.id}.`);
    if (entry.rating !== 1 && body.rating === 1) tune.pending += 4;  // a confirmed answer yields training examples
    entry.rating = body.rating;
    return { ok: true };
  }

  // ---- units and groups -----------------------------------------------------------

  function unitPayload(unit) {
    return {
      id: unit.id, path: unit.path, name: unit.name, kind: unit.kind, start_line: unit.start_line,
      end_line: unit.end_line, signature: unit.signature, doc: unit.doc, text: unit.text,
      callees: unit.callees.slice(), callers: unit.callers.slice(), group: unit.group, tokens: unit.tokens,
    };
  }

  function unit(idText) {
    if (!/^\d{1,9}$/.test(String(idText))) throw new HttpError(400, 'id must be a unit id.');
    const found = units[Number(idText)];
    if (!found) throw new HttpError(404, `No unit ${idText}.`);
    return unitPayload(found);
  }

  function groupsPayload() {
    return {
      kappa: 41.5, reduced_dim: settings.kde.dim,
      groups: groups.map((g) => ({ id: g.id, size: g.size, log_density: g.log_density, x: g.x, y: g.y, label: g.label })),
      units: units.map((u) => ({ id: u.id, group: u.group, x: round(u.x), y: round(u.y), name: u.name, path: u.path,
                                 kind: u.kind, log_density: u.log_density })),
    };
  }

  function siblingsOf(unit, fanout) {
    let list = siblingCache.get(unit.id);
    if (!list) {
      list = units
        .filter((v) => v.group === unit.group && v.id !== unit.id)
        .map((v) => ({ id: v.id, d: (v.x - unit.x) ** 2 + (v.y - unit.y) ** 2 }))
        .sort((a, b) => a.d - b.d || a.id - b.id)
        .slice(0, fanout)
        .map((v) => v.id);
      siblingCache.set(unit.id, list);
    }
    return list;
  }

  // ---- query --------------------------------------------------------------------

  const A = 1.8;      // slope of the score posterior
  const Z0 = 3.71;    // the posterior is 0.25 at z = 3.1

  function scoreUnits(question) {
    const words = subwords(question).filter((w) => !STOP_WORDS.has(w) && w.length > 1);
    const mentioned = new Set((question.match(/[A-Za-z_][A-Za-z0-9_]{2,}/g) || []).filter((w) => byName.has(w)));
    const random = makeRandom(hash32(question));
    const scores = new Float64Array(units.length);
    let sum = 0;
    for (const u of units) {
      let hit = 0;
      for (const w of words) {
        if (u.nameWords.includes(w)) hit += 1;
        else if (u.nameWords.some((n) => n.length > 3 && (n.startsWith(w) || w.startsWith(n)))) hit += 0.6;
        else if (u.words.has(w)) hit += 0.35;
      }
      const norm = Math.sqrt(Math.max(1, words.length) * (u.nameWords.length + 1.5));
      let s = 0.18 + 0.62 * Math.min(1, hit / norm) ** 1.3 + 0.14 * random.next();
      if (mentioned.has(u.name)) s += 0.3;
      if (u.kind === 'region') s -= 0.05;
      s = Math.max(0.02, Math.min(0.97, s));
      scores[u.id] = s;
      sum += s;
    }
    const mean = sum / units.length;
    let variance = 0;
    for (const s of scores) variance += (s - mean) ** 2;
    const stddev = Math.sqrt(variance / units.length) || 1;
    return { scores, mean, stddev };
  }

  function trace(scores, mean, stddev, tau) {
    const lift = settings.retrieve.lift;
    const maxUnits = settings.retrieve.max_units;
    const maxHops = settings.retrieve.max_hops;
    const posterior = (id) => sigmoid(A * ((scores[id] - mean) / stddev - Z0));
    const candidates = new Map();  // id -> { ps, product, p, gain, parent, via, hops }
    const accepted = new Map();
    let seeds = 0;
    for (const u of units) {
      const ps = posterior(u.id);
      if (ps >= tau) {
        candidates.set(u.id, { ps, product: 1, p: ps, gain: 0, parent: null, via: 'seed', hops: 0 });
        ++seeds;
      }
    }
    let evaluated = candidates.size;
    let expansions = 0;
    let truncated = false;
    for (;;) {
      let best = -1;
      let bestP = -1;
      for (const [id, c] of candidates) {
        if (c.p >= tau && (c.p > bestP || (c.p === bestP && id < best))) {
          best = id;
          bestP = c.p;
        }
      }
      if (best < 0) break;
      if (accepted.size >= maxUnits) {
        truncated = true;
        break;
      }
      const c = candidates.get(best);
      candidates.delete(best);
      accepted.set(best, c);
      if (c.hops >= maxHops) continue;
      ++expansions;
      const u = units[best];
      const edges = [];
      for (const v of u.callees) {
        edges.push([v, 'callee', Math.log(lift.callee) / (1 + Math.log2(Math.max(1, units[v].callers.length)))]);
      }
      for (const v of u.callers) {
        edges.push([v, 'caller', Math.log(lift.caller) / (1 + Math.log2(Math.max(1, units[v].callees.length)))]);
      }
      for (const v of siblingsOf(u, 4)) edges.push([v, 'sibling', Math.log(lift.sibling)]);
      for (const [v, via, logLift] of edges) {
        if (accepted.has(v)) continue;
        let cand = candidates.get(v);
        if (!cand) {
          const ps = posterior(v);
          cand = { ps, product: 1, p: ps, gain: 0, parent: null, via: 'seed', hops: 0 };
          candidates.set(v, cand);
          ++evaluated;
        }
        const conditional = sigmoid(logit(Math.min(0.999999, Math.max(1e-9, cand.ps))) + logLift);
        const q = c.p * conditional + (1 - c.p) * cand.ps;
        const gain = (q - cand.ps) / (1 - cand.ps);
        cand.product *= 1 - gain;
        cand.p = 1 - (1 - cand.ps) * cand.product;
        // A unit that reaches tau on its own score stays a seed.
        if (gain > cand.gain && cand.ps < tau) {
          cand.gain = gain;
          cand.parent = best;
          cand.via = via;
          cand.hops = c.hops + 1;
        }
      }
    }
    let lowConfidence = false;
    if (accepted.size === 0) {
      let top = 0;
      for (const u of units) if (scores[u.id] > scores[top]) top = u.id;
      const ps = posterior(top);
      accepted.set(top, { ps, product: 1, p: ps, gain: 0, parent: null, via: 'seed', hops: 0 });
      lowConfidence = true;
    }
    return { accepted, seeds, evaluated, expansions, truncated, lowConfidence,
             zThreshold: Z0 + logit(tau) / A, posterior };
  }

  function skeletonTokens(unit) {
    if (unit.signature === '') return 0;
    return Math.ceil((unit.signature.length + unit.doc.length) / 3.6) + 4;
  }

  // Quality of an ordered list of entries grouped by file (context/rot.hpp, step 6).
  function evaluate(entries, options) {
    const perFile = new Map();
    for (const e of entries) {
      if (!perFile.has(e.unit.path)) perFile.set(e.unit.path, []);
      perFile.get(e.unit.path).push(e);
    }
    let length = 0;
    let mass = 0;
    const nodes = [];
    for (const [path, list] of perFile) {
      let l = options.group_overhead;
      let m = 0;
      for (const e of list) {
        l += e.tokens;
        m += (1 - e.redundancy) * e.p * e.tokens;
      }
      nodes.push({ path, tokens: l, quality: round(l > 0 ? m / l : 0, 3), units: list.length });
      length += l;
      mass += m;
    }
    return { tokens: length, quality: length > 0 ? mass / length : 0, groups: nodes };
  }

  function redundancyPass(list) {
    list.forEach((e, j) => {
      let rho = 0;
      for (let i = 0; i < j; ++i) rho = Math.max(rho, similarity(list[i].unit, e.unit));
      e.redundancy = rho;
      e.quality = (1 - rho) * e.p;
    });
  }

  function select(traced, posterior) {
    const options = { theta: settings.context.theta, token_budget: settings.context.token_budget, group_overhead: 12,
                      unit_overhead: 8, closure_min_p: 0.05 };
    const make = (id, p) => ({ unit: units[id], p, form: 'full', support: false, redundancy: 0, quality: p,
                               tokens: units[id].tokens + options.unit_overhead });
    const candidates = Array.from(traced, ([id, c]) => make(id, c.p)).sort((a, b) => b.p - a.p || a.unit.id - b.unit.id);

    const before = candidates.map((e) => ({ ...e }));
    redundancyPass(before);
    const qualityBefore = evaluate(before, options);

    const stats = { passes: 0, dropped_low_quality: 0, dropped_redundant: 0, dropped_budget: 0, skeletonised: 0,
                    support_added: 0 };
    let kept = candidates;
    for (;;) {
      ++stats.passes;
      redundancyPass(kept);
      const next = kept.filter((e) => e.quality >= options.theta);
      for (const e of kept) {
        if (e.quality >= options.theta) continue;
        if (e.p < options.theta) ++stats.dropped_low_quality;
        else ++stats.dropped_redundant;
      }
      const changed = next.length !== kept.length;
      kept = next;
      if (!changed || stats.passes >= 16) break;
    }
    // A low-confidence answer still needs its single unit.
    if (kept.length === 0 && candidates.length > 0) {
      kept = [candidates[0]];
      stats.dropped_low_quality = Math.max(0, stats.dropped_low_quality - 1);
    }

    const present = new Set(kept.map((e) => e.unit.id));
    for (const e of kept.slice()) {
      for (const callee of e.unit.callees) {
        if (present.has(callee) || stats.support_added >= 3) continue;
        const p = traced.has(callee) ? traced.get(callee).p : posterior(callee);
        const tokens = skeletonTokens(units[callee]);
        if (p < options.closure_min_p || tokens === 0) continue;
        const entry = make(callee, p);
        entry.form = 'skeleton';
        entry.support = true;
        entry.tokens = tokens + options.unit_overhead;
        entry.parent = e.unit.id;
        kept.push(entry);
        present.add(callee);
        ++stats.support_added;
      }
    }

    const total = () => evaluate(kept, options).tokens;
    while (total() > options.token_budget && kept.length > 1) {
      const full = kept.filter((e) => !e.support && e.form === 'full' && skeletonTokens(e.unit) > 0)
        .sort((a, b) => a.quality - b.quality);
      if (full.length > 0) {
        full[0].form = 'skeleton';
        full[0].tokens = skeletonTokens(full[0].unit) + options.unit_overhead;
        ++stats.skeletonised;
        continue;
      }
      const order = kept.slice().sort((a, b) => Number(b.support) - Number(a.support) ||
                                                a.quality * a.tokens - b.quality * b.tokens);
      kept.splice(kept.indexOf(order[0]), 1);
      ++stats.dropped_budget;
    }

    // Files by their best probability; strongest first, second strongest last.
    const filesByP = [];
    for (const e of kept) {
      let f = filesByP.find((x) => x.path === e.unit.path);
      if (!f) filesByP.push((f = { path: e.unit.path, p: 0, entries: [] }));
      f.p = Math.max(f.p, e.p);
      f.entries.push(e);
    }
    filesByP.sort((a, b) => b.p - a.p);
    const head = [];
    const tail = [];
    filesByP.forEach((f, i) => (i % 2 === 0 ? head.push(f) : tail.unshift(f)));
    const ordered = [];
    for (const f of head.concat(tail)) {
      f.entries.sort((a, b) => a.unit.start_line - b.unit.start_line);
      for (const e of f.entries) ordered.push(e);
    }
    const after = evaluate(ordered, options);

    return {
      entries: ordered,
      context: {
        tokens_before: qualityBefore.tokens, tokens_after: after.tokens,
        quality_before: round(qualityBefore.quality, 3), quality_after: round(after.quality, 3),
        rot_before: round(1 - qualityBefore.quality, 3), rot_after: round(1 - after.quality, 3),
        compression: round(after.tokens / Math.max(1, qualityBefore.tokens), 3), ...stats,
        groups: after.groups,
      },
    };
  }

  function citation(unit) {
    return `${unit.path}:${unit.start_line}`;
  }

  function label(unit) {
    return unit.name !== '' ? unit.name : `${unit.kind} at ${citation(unit)}`;
  }

  function composeAnswer(question, entries, context, lowConfidence) {
    // The prompt is in file order; the answer starts from the most probable unit.
    const full = entries.filter((e) => !e.support).sort((a, b) => b.p - a.p || a.unit.id - b.unit.id);
    const top = full.find((e) => e.unit.kind === 'function') || full[0] || entries[0];
    const lines = [];
    if (lowConfidence) {
      lines.push('The context does not contain the answer to this question. No unit reached the relevance ' +
                 `threshold; the closest one is \`${label(top.unit)}\` (\`${citation(top.unit)}\`), with a probability ` +
                 `of ${top.p.toFixed(2)}.`);
      lines.push('', 'Try naming the function or the file you are interested in.');
      return lines.join('\n');
    }
    lines.push(`**Short answer.** \`${label(top.unit)}\` (\`${citation(top.unit)}\`) is the place to look: ` +
               `${top.unit.doc.split('. ')[0].replace(/\.$/, '')}.`);
    lines.push('');
    if (full.length > 1) {
      lines.push('### How the pieces fit together', '');
      full.slice(0, 5).forEach((e, i) => {
        const callees = e.unit.callees.map((id) => units[id]).filter((u) => full.some((f) => f.unit === u));
        const calls = callees.length > 0 ? ` It calls ${callees.slice(0, 3).map((u) => `\`${u.name}\``).join(', ')}.` : '';
        lines.push(`${i + 1}. \`${label(e.unit)}\` (\`${citation(e.unit)}\`) \u{2014} relevance ${e.p.toFixed(2)}.${calls}`);
      });
      lines.push('');
    }
    const body = top.unit.text.split('\n').filter((l) => !l.startsWith('//')).slice(0, 12).join('\n').trimEnd();
    lines.push('### The code', '', '```c', body, '```', '');
    const support = entries.filter((e) => e.support);
    if (support.length > 0) {
      lines.push(`Only the signatures of ${support.map((e) => `\`${e.unit.name}\``).join(', ')} were in the context, ` +
                 'so their behaviour is described from their documentation.', '');
    }
    lines.push(`> The prompt held ${entries.length} units in ${context.tokens_after} tokens; the context rot index ` +
               `went from ${context.rot_before.toFixed(2)} to ${context.rot_after.toFixed(2)} (\u{3c4} = ` +
               `${settings.retrieve.tau}, \u{3b8} = ${settings.context.theta}).`, '');
    lines.push('*This answer comes from the development mock server; the wording is canned.*');
    return lines.join('\n');
  }

  // ---- edits ----------------------------------------------------------------------

  const transactions = new Map();

  // Unified diff of one contiguous change inside a unit, numbered by file lines.
  function diffOf(unit, oldLines, at, removed, inserted) {
    const context = 3;
    const from = Math.max(0, at - context);
    const to = Math.min(oldLines.length, at + removed + context);
    const out = [`--- a/${unit.path}`, `+++ b/${unit.path}`];
    const oldCount = to - from;
    const newCount = oldCount - removed + inserted.length;
    const name = unit.name !== '' ? ' ' + unit.name : '';
    out.push(`@@ -${unit.start_line + from},${oldCount} +${unit.start_line + from},${newCount} @@${name}`);
    for (let i = from; i < at; ++i) out.push(' ' + oldLines[i]);
    for (let i = at; i < at + removed; ++i) out.push('-' + oldLines[i]);
    for (const line of inserted) out.push('+' + line);
    for (let i = at + removed; i < to; ++i) out.push(' ' + oldLines[i]);
    return out.join('\n') + '\n';
  }

  function proposeEdit(question, entries) {
    const usable = entries.filter((e) => e.unit.kind === 'function' && !e.support)
      .sort((a, b) => b.p - a.p || a.unit.id - b.unit.id);
    if (usable.length === 0) return null;
    const first = usable[0].unit;
    const lines = first.text.replace(/\n$/, '').split('\n');
    const open = lines.findIndex((l) => l.endsWith('{') && !l.startsWith('//'));
    if (open < 0 || open + 1 >= lines.length) return null;
    const hasSize = /\bsize_t size\b/.test(lines[open]);
    const returnsPointer = /\*\s*[A-Za-z_][A-Za-z0-9_]*\(/.test(lines[open]);
    const inserted = hasSize && returnsPointer
      ? ['  if mi_unlikely(size > MI_MAX_ALLOC_SIZE) {', '    return NULL;  // refuse sizes that cannot be served', '  }']
      : ['  mi_assert_expensive(mi_heap_is_valid(mi_heap_get_default()));'];
    const search = [lines[open], lines[open + 1]];
    const replace = [lines[open], ...inserted, lines[open + 1]];

    const answer = [
      hasSize && returnsPointer
        ? `The change makes \`${first.name}\` refuse a size above \`MI_MAX_ALLOC_SIZE\` before it touches the heap.`
        : `The change adds a validity check of the heap at the start of \`${first.name}\`.`,
      '', first.path, `@@ ${first.name}`, '<<<<<<< SEARCH', ...search, '=======', ...replace, '>>>>>>> REPLACE', '',
    ];
    const tx = {
      id: 'tx-' + hash32(question + first.name + transactions.size).toString(16).padStart(8, '0').slice(0, 4),
      status: 'ok', committed: false,
      files: [{
        path: `${settings.corpus.roots[0] || 'C:/corpus'}/${first.path}`, display_path: first.path, status: 'ok',
        creates_file: false, diff: diffOf(first, lines, open + 1, 0, inserted),
        matches: [{ kind: 'exact', first_line: first.start_line + open, last_line: first.start_line + open + 1,
                    distance: 0, candidates: 1 }],
        syntax_errors_before: first.path.endsWith('.h') ? 2 : 0,
        syntax_errors_after: first.path.endsWith('.h') ? 2 : 0, message: '',
      }],
      warnings: [],
    };

    const second = usable.map((e) => e.unit).find((u) => u.path !== first.path);
    if (second) {
      const old = second.text.replace(/\n$/, '').split('\n');
      const note = `// Note: sizes above MI_MAX_ALLOC_SIZE are refused by ${first.name}.`;
      answer.push(`The documentation of \`${second.name}\` is updated to say so; the unit is rewritten as a whole.`, '',
                  second.path, `<<<<<<< UNIT ${second.name}`, note, ...old, '>>>>>>> UNIT', '');
      tx.files.push({
        path: `${settings.corpus.roots[0] || 'C:/corpus'}/${second.path}`, display_path: second.path, status: 'ok',
        creates_file: false, diff: diffOf(second, old, 0, 0, [note]),
        matches: [{ kind: 'unit', first_line: second.start_line, last_line: second.end_line, distance: 0,
                    candidates: 1 }],
        syntax_errors_before: 0, syntax_errors_after: 0, message: '',
      });
    }

    // Markers in the question select the failure cases.
    const target = tx.files[0];
    if (question.includes('#ambiguous')) {
      target.status = 'ambiguous';
      target.matches[0].candidates = 3;
      target.diff = '';
      target.message = `The SEARCH text matches 3 locations in ${first.name}; add more lines to make it unique.`;
    } else if (question.includes('#notfound')) {
      target.status = 'not_found';
      target.matches[0] = { kind: 'none', first_line: 0, last_line: 0, distance: 0, candidates: 0 };
      target.diff = '';
      target.message = 'The SEARCH text was not found on any rung (exact, trailing space, indentation).';
    } else if (question.includes('#syntax')) {
      target.status = 'syntax_regression';
      target.syntax_errors_after = target.syntax_errors_before + 2;
      target.message = 'The edit would add 2 syntax errors; it is refused.';
    } else if (question.includes('#indent')) {
      target.matches[0].kind = 'indentation';
      tx.warnings.push('Block 1 matched after ignoring indentation; the replacement was re-indented.');
    }
    if (target.status !== 'ok') tx.status = target.status;
    tx.conflict = question.includes('#conflict');
    transactions.set(tx.id, tx);
    return { answer: answer.join('\n'), transaction: tx };
  }

  function publicTransaction(tx) {
    const { conflict, ...rest } = tx;
    void conflict;
    return rest;
  }

  function editPreview(body, now) {
    notifyActivity(now);
    if (!isObject(body) || (typeof body.text !== 'string' && !Array.isArray(body.blocks))) {
      throw new HttpError(400, 'The body must hold "text" or "blocks".');
    }
    let blocks = body.blocks;
    if (!Array.isArray(blocks)) {
      blocks = [];
      const pattern = /(?:^|\n)([^\n]+)\n(?:@@ ([^\n]+)\n)?<{5,9} SEARCH\n([\s\S]*?)\n={5,9}\n([\s\S]*?)\n>{5,9} REPLACE/g;
      for (let m = pattern.exec(body.text); m !== null; m = pattern.exec(body.text)) {
        blocks.push({ path: m[1].trim(), scope: m[2] || '', search: m[3], replace: m[4] });
      }
    }
    const tx = { id: 'tx-' + hash32(JSON.stringify(blocks) + transactions.size).toString(16).padStart(8, '0').slice(0, 4),
                 status: blocks.length > 0 ? 'ok' : 'not_found', committed: false, files: [],
                 warnings: blocks.length > 0 ? [] : ['No edit block was found in the text.'] };
    for (const block of blocks) {
      const path = String(block.path || '');
      const search = String(block.search || '');
      const target = units.find((u) => (path.endsWith(u.path) || u.path.endsWith(path)) && search !== '' &&
                                       u.text.includes(search));
      const file = { path: `${settings.corpus.roots[0] || 'C:/corpus'}/${path}`, display_path: path, status: 'ok',
                     creates_file: false, diff: '', matches: [], syntax_errors_before: 0, syntax_errors_after: 0,
                     message: '' };
      if (!files.some((f) => path.endsWith(f) || f.endsWith(path))) {
        file.status = 'file_missing';
        file.message = 'The file does not exist under any corpus root.';
      } else if (!target) {
        file.status = 'not_found';
        file.message = 'The SEARCH text was not found on any rung (exact, trailing space, indentation).';
        file.matches.push({ kind: 'none', first_line: 0, last_line: 0, distance: 0, candidates: 0 });
      } else {
        const old = target.text.replace(/\n$/, '').split('\n');
        const at = target.text.slice(0, target.text.indexOf(search)).split('\n').length - 1;
        const removed = search.split('\n').length;
        file.display_path = target.path;
        file.diff = diffOf(target, old, at, removed, String(block.replace || '').split('\n'));
        file.matches.push({ kind: 'exact', first_line: target.start_line + at,
                            last_line: target.start_line + at + removed - 1, distance: 0, candidates: 1 });
      }
      if (file.status !== 'ok' && tx.status === 'ok') tx.status = file.status;
      tx.files.push(file);
    }
    transactions.set(tx.id, tx);
    return publicTransaction(tx);
  }

  function findTransaction(body) {
    if (!isObject(body) || typeof body.id !== 'string') throw new HttpError(400, 'id is missing.');
    const tx = transactions.get(body.id);
    if (!tx) throw new HttpError(404, `No transaction ${body.id}.`);
    return tx;
  }

  function editApply(body, now) {
    notifyActivity(now);
    const tx = findTransaction(body);
    if (tx.committed) throw new HttpError(409, 'The transaction is already applied.');
    if (tx.conflict) {
      tx.status = 'conflict';
      tx.files[0].status = 'conflict';
      tx.files[0].message = `${tx.files[0].display_path} changed on disk after the proposal was prepared.`;
      return publicTransaction(tx);
    }
    if (tx.status !== 'ok') throw new HttpError(409, `The transaction cannot be applied (${tx.status}).`);
    tx.committed = true;
    return publicTransaction(tx);
  }

  function editRevert(body, now) {
    notifyActivity(now);
    const tx = findTransaction(body);
    if (!tx.committed) throw new HttpError(409, 'The transaction is not applied.');
    tx.committed = false;
    return publicTransaction(tx);
  }

  // Splits an answer into pieces of a few characters, the way a tokeniser would.
  function tokenise(text) {
    const pieces = text.match(/\s*\S{1,6}|\s+$/g) || [];
    return pieces;
  }

  // Computes everything a query returns. The HTTP layer streams it.
  function query(body, now) {
    if (!isObject(body) || typeof body.question !== 'string' || body.question.trim() === '') {
      throw new HttpError(400, 'The question is empty.');
    }
    if (body.tau !== undefined && !(typeof body.tau === 'number' && body.tau > 0 && body.tau < 1)) {
      throw new HttpError(400, 'tau must be a number between 0 and 1.');
    }
    notifyActivity(now);
    if (index.build) throw new HttpError(409, 'The index is being rebuilt. Try again when the build has finished.');
    if (index.stale || !index.ready) throw new HttpError(409, 'The index is stale. Rebuild it in the settings.');

    const question = body.question.trim();
    const editMode = body.edit === true;
    const tau = body.tau !== undefined ? body.tau : settings.retrieve.tau;
    const id = 'q-' + String(nextQuery++).padStart(6, '0');
    const random = makeRandom(hash32(id + question));

    const { scores, mean, stddev } = scoreUnits(question);
    const traced = trace(scores, mean, stddev, tau);
    const selection = select(traced.accepted, traced.posterior);
    const inContext = new Map(selection.entries.map((e) => [e.unit.id, e]));

    const list = [];
    const describe = (unit, t, entry) => ({
      id: unit.id, path: unit.path, name: unit.name, kind: unit.kind, start_line: unit.start_line,
      end_line: unit.end_line, group: unit.group, tokens: unit.tokens, score: round(scores[unit.id], 3),
      z: round((scores[unit.id] - mean) / stddev, 2), p_score: round(t.ps, 3), p: round(t.p, 3), hops: t.hops,
      parent: t.parent, via: t.via, in_context: entry !== undefined, form: entry ? entry.form : 'none',
      support: entry ? entry.support : false, redundancy: entry ? round(entry.redundancy, 3) : 0,
      quality: entry ? round(entry.quality, 3) : 0, preview: unit.text.split('\n').slice(0, 3).join('\n'),
    });
    for (const [unitId, t] of traced.accepted) list.push(describe(units[unitId], t, inContext.get(unitId)));
    for (const e of selection.entries) {
      if (!e.support || traced.accepted.has(e.unit.id)) continue;
      const parent = traced.accepted.get(e.parent);
      const ps = traced.posterior(e.unit.id);
      list.push(describe(e.unit, { ps, p: e.p, hops: (parent ? parent.hops : 0) + 1, parent: e.parent, via: 'callee' }, e));
    }

    const kept = Math.min(history.length, history.length > 0 ? 1 + (hash32(question) % 2) : 0);
    const violation = question.includes('#violation');
    const wall = 250 + random.int(140);
    const retrieval = {
      id, tau, z_threshold: round(traced.zThreshold, 2), low_confidence: traced.lowConfidence,
      truncated: traced.truncated, units: list,
      context: { ...selection.context, history_kept: kept, history_total: Math.min(history.length, 6) },
      scan: { blocks: blockCount(), block_bytes: blockBytes(), threads: threads(), wall_us: wall,
              work_us: Math.round(wall * (2.4 + random.next())), gate_wait_us: 25 + random.int(40),
              admission_violations: violation ? 2 : 0, commit_violations: violation ? 1 : 0,
              rescored: 12 + random.int(30), codes: settings.index.codes, simd: 'avx2' },
      trace: { seeds: traced.seeds, evaluated: traced.evaluated, expansions: traced.expansions },
    };

    const extractive = !llama || settings.models.chat === '';
    let answer;
    let transaction = null;
    if (extractive) {
      answer = ['No chat model is loaded; these are the units selected for the question.', '']
        .concat(selection.entries.map((e) => `- \`${label(e.unit)}\` (\`${citation(e.unit)}\`): ${e.unit.doc || e.unit.kind}`))
        .join('\n');
    } else if (editMode) {
      const proposal = traced.lowConfidence ? null : proposeEdit(question, selection.entries);
      if (proposal) {
        answer = proposal.answer;
        transaction = publicTransaction(proposal.transaction);
      } else {
        answer = 'No edit is proposed: the context does not contain a function that the request applies to.';
      }
    } else {
      answer = composeAnswer(question, selection.entries, selection.context, traced.lowConfidence);
    }

    const tokens = tokenise(answer);
    const promptMs = 600 + random.int(500);
    const completionMs = tokens.length * (18 + random.int(8));
    const done = {
      id, answer, prompt_tokens: selection.context.tokens_after + 140, completion_tokens: tokens.length,
      prompt_ms: promptMs, completion_ms: completionMs, total_ms: promptMs + completionMs + wall / 1000 + 35,
      stop_reason: 'eos', model: extractive ? '' : tune.serving, extractive,
    };
    return {
      retrieval, tokens, transaction, done,
      failure: question.includes('#error') ? 'The model stopped: the context window was exceeded (mock failure).' : null,
      drop: question.includes('#drop'),
      // Called when the answer is complete. The end of a request is user
      // activity too: the idle time counts from the last thing the user saw.
      commit: (finishedAt) => {
        history.push({ id, time: iso(now), question, answer, units: list.filter((u) => u.in_context).map((u) => u.id),
                       rating: 0 });
        tune.pending += 1;
        notifyActivity(typeof finishedAt === 'number' ? finishedAt : now);
      },
    };
  }

  return {
    status,
    settings: () => settings,
    patchSettings,
    startIndex,
    indexProgress,
    calibrate: (now) => {
      advance(now);
      if (index.build) throw new HttpError(409, 'An index build is running.');
      index.calibrated = true;
      return { ok: true, started: true };
    },
    history: () => history,
    clearHistory: () => {
      history.length = 0;
      return { ok: true };
    },
    feedback,
    groups: groupsPayload,
    unit,
    tune: tuneStatus,
    setTune,
    runTune,
    query,
    editPreview,
    editApply,
    editRevert,
    corpus,
  };
}
