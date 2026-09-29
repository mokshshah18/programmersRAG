// The settings document (docs/DESIGN.md, section 8) as the dialog sees it.
//
// The server stores one JSON document addressed by dotted keys. It may hold
// only the keys that were ever set, so the defaults are repeated here for the
// keys the dialog edits. A save sends a merge patch that contains the changed
// keys and nothing else.

export const DEFAULTS = {
  'corpus.roots': [],
  'retrieve.tau': 0.25,
  'context.token_budget': 3000,
  'models.embedding': '',
  'models.chat': '',
  'models.tuning_base': '',
  'exec.threads': 0,
  'exec.block_bytes': 0,
};

export const TAU_MIN = 0.05;
export const TAU_MAX = 0.9;

// Bounds of the block size override. The rule of exec::recommended_block_bytes
// stays within 4 KiB to 64 KiB; the override is wider so that the effect of a
// block that does not fit the L1 data cache can be measured.
export const BLOCK_MIN = 1024;
export const BLOCK_MAX = 4194304;

// Prefixes of the keys whose change makes the index stale.
const INDEX_PREFIXES = ['corpus.', 'kde.', 'index.'];
const INDEX_KEYS = ['models.embedding'];

function isObject(x) {
  return x !== null && typeof x === 'object' && !Array.isArray(x);
}

// Value at a dotted key. A document that uses the dotted key itself as a
// member name is understood as well.
export function getPath(doc, key, fallback) {
  if (!isObject(doc)) return fallback;
  if (Object.prototype.hasOwnProperty.call(doc, key) && doc[key] !== null) return doc[key];
  let node = doc;
  for (const part of key.split('.')) {
    if (!isObject(node) || !Object.prototype.hasOwnProperty.call(node, part)) return fallback;
    node = node[part];
  }
  return node === null || node === undefined ? fallback : node;
}

// Sets the value at a dotted key, creating the objects on the way.
export function setPath(doc, key, value) {
  const parts = key.split('.');
  let node = doc;
  for (let i = 0; i + 1 < parts.length; ++i) {
    const part = parts[i];
    if (part === '__proto__' || part === 'constructor' || part === 'prototype') return doc;
    if (!isObject(node[part])) node[part] = {};
    node = node[part];
  }
  const last = parts[parts.length - 1];
  if (last !== '__proto__' && last !== 'constructor' && last !== 'prototype') node[last] = value;
  return doc;
}

// Values of the edited keys, with the default where the document has none or
// has a value of the wrong type.
export function readForm(doc) {
  const form = {};
  for (const key of Object.keys(DEFAULTS)) {
    const fallback = DEFAULTS[key];
    const value = getPath(doc, key, fallback);
    if (Array.isArray(fallback)) {
      form[key] = Array.isArray(value) ? value.filter((v) => typeof v === 'string') : fallback.slice();
    } else {
      form[key] = typeof value === typeof fallback ? value : fallback;
    }
  }
  return form;
}

function same(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  return a === b;
}

// Dotted keys whose value differs between two forms.
export function changedKeys(before, after) {
  return Object.keys(after).filter((key) => !same(before[key], after[key]));
}

// Merge patch (nested objects) with the changed keys only.
export function buildPatch(before, after) {
  const patch = {};
  for (const key of changedKeys(before, after)) setPath(patch, key, after[key]);
  return patch;
}

// True when a change of one of `keys` makes the index stale.
export function affectsIndex(keys) {
  return keys.some((key) => INDEX_KEYS.includes(key) || INDEX_PREFIXES.some((p) => key.startsWith(p)));
}

// Corpus roots as entered: trimmed, backslashes as forward slashes, without
// empty entries and duplicates.
export function normaliseRoots(roots) {
  const seen = new Set();
  const out = [];
  for (const raw of roots) {
    let path = String(raw).trim().replace(/\\/g, '/');
    if (path.length > 1 && path.endsWith('/') && !/^[A-Za-z]:\/$/.test(path)) path = path.slice(0, -1);
    if (path === '' || seen.has(path)) continue;
    seen.add(path);
    out.push(path);
  }
  return out;
}

// tau = c_fp / (c_fp + c_fn), hence c_fn / c_fp = (1 - tau) / tau: how many
// times more a missed relevant unit costs than an irrelevant unit in the context.
export function costRatio(tau) {
  if (!(tau > 0) || !(tau < 1)) return NaN;
  return (1 - tau) / tau;
}

// Checks the form. Returns a list of { key, message }; empty when valid.
export function validate(form) {
  const errors = [];
  const tau = form['retrieve.tau'];
  if (!(typeof tau === 'number' && tau >= TAU_MIN && tau <= TAU_MAX)) {
    errors.push({ key: 'retrieve.tau', message: `tau must be between ${TAU_MIN} and ${TAU_MAX}` });
  }
  const integer = (key, min, max, name) => {
    const v = form[key];
    if (!(Number.isInteger(v) && v >= min && v <= max)) {
      errors.push({ key, message: `${name} must be a whole number between ${min} and ${max}` });
    }
  };
  integer('context.token_budget', 256, 131072, 'The token budget');
  integer('exec.threads', 0, 1024, 'Threads');
  const block = form['exec.block_bytes'];
  if (!(Number.isInteger(block) && (block === 0 || (block >= BLOCK_MIN && block <= BLOCK_MAX)))) {
    errors.push({
      key: 'exec.block_bytes',
      message: `The block size must be 0 or between ${BLOCK_MIN} and ${BLOCK_MAX} bytes`,
    });
  }
  return errors;
}
