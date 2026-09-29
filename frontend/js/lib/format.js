// Number, size and time formatting.
//
// Pure functions without DOM or locale dependence: the same input gives the
// same string on every machine, which keeps the tests deterministic. Values
// that are missing or not finite are shown as an en dash.

// Written as code point escapes so that this file is plain ASCII.
export const MISSING = '\u{2013}';  // en dash
const MINUS = '\u{2212}';
const MICRO = '\u{b5}';
const TIMES = '\u{d7}';
const ELLIPSIS = '\u{2026}';

export function isNumber(x) {
  return typeof x === 'number' && Number.isFinite(x);
}

export function clamp(x, lo, hi) {
  if (!isNumber(x)) return lo;
  return x < lo ? lo : x > hi ? hi : x;
}

export function clamp01(x) {
  return clamp(x, 0, 1);
}

// Inserts thousands separators into the integer part of a plain decimal string.
function group(digits) {
  const negative = digits.startsWith('-');
  const body = negative ? digits.slice(1) : digits;
  const dot = body.indexOf('.');
  const whole = dot < 0 ? body : body.slice(0, dot);
  const rest = dot < 0 ? '' : body.slice(dot);
  let out = '';
  for (let i = 0; i < whole.length; ++i) {
    if (i > 0 && (whole.length - i) % 3 === 0) out += ',';
    out += whole[i];
  }
  return (negative ? MINUS : '') + out + rest;
}

// Removes trailing zeros of a fixed-point string: "1.50" -> "1.5", "2.00" -> "2".
function trimZeros(text) {
  if (text.indexOf('.') < 0) return text;
  return text.replace(/\.?0+$/, '');
}

// Integer with thousands separators: 1510 -> "1,510".
export function formatCount(n) {
  if (!isNumber(n)) return MISSING;
  return group(String(Math.round(n)));
}

// Fixed number of decimals with thousands separators: (1234.5, 2) -> "1,234.50".
export function formatFixed(x, digits = 2) {
  if (!isNumber(x)) return MISSING;
  const text = x.toFixed(digits);
  // toFixed may give "-0.00"; a negative zero is shown without the sign.
  return group(Number(text) === 0 ? text.replace('-', '') : text);
}

// Probability or quality in [0, 1] with two decimals. Values that would round
// to 0 or 1 without being 0 or 1 are shown as bounds, so that "0.00" always
// means exactly zero.
export function formatProbability(p) {
  if (!isNumber(p)) return MISSING;
  const q = clamp01(p);
  if (q > 0 && q < 0.005) return '<0.01';
  if (q < 1 && q >= 0.995) return '>0.99';
  return q.toFixed(2);
}

// Fraction as a percentage: (0.37) -> "37%", (0.375, 1) -> "37.5%".
export function formatPercent(x, digits = 0) {
  if (!isNumber(x)) return MISSING;
  return formatFixed(x * 100, digits) + '%';
}

// Signed difference with a real minus sign (U+2212): (0.31) -> "+0.31".
export function formatSigned(x, digits = 2) {
  if (!isNumber(x)) return MISSING;
  const text = Math.abs(x).toFixed(digits);
  if (Number(text) === 0) return text;
  return (x < 0 ? MINUS : '+') + group(text);
}

// Ratio such as a speed-up: 2.9 followed by a multiplication sign.
export function formatRatio(x, digits = 1) {
  if (!isNumber(x)) return MISSING;
  return formatFixed(x, digits) + TIMES;
}

// Sizes in binary units, as cache sizes are specified: 16384 -> "16 KiB".
export function formatBytes(n) {
  if (!isNumber(n) || n < 0) return MISSING;
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    ++unit;
  }
  const digits = unit === 0 || value >= 100 || Number.isInteger(value) ? 0 : value >= 10 ? 1 : 2;
  return trimZeros(value.toFixed(digits)) + ' ' + units[unit];
}

// Three significant digits at most, without trailing zeros: 1.234 -> "1.23", 12.34 -> "12.3".
function significant(value) {
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return trimZeros(value.toFixed(digits));
}

// Duration given in seconds, in the unit that fits its magnitude:
// 0.00031 -> "310 us" (with the micro sign), 0.9 -> "900 ms", 5.1 -> "5.1 s", 303.9 -> "5 min 4 s".
export function formatSeconds(seconds) {
  if (!isNumber(seconds) || seconds < 0) return MISSING;
  if (seconds === 0) return '0 s';
  if (seconds < 1e-6) return significant(seconds * 1e9) + ' ns';
  if (seconds < 1e-3) return significant(seconds * 1e6) + ' ' + MICRO + 's';
  if (seconds < 1) return significant(seconds * 1e3) + ' ms';
  if (seconds < 60) return significant(seconds) + ' s';
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return m > 0 ? `${h} h ${m} min` : `${h} h`;
  return s > 0 ? `${m} min ${s} s` : `${m} min`;
}

export function formatMillis(ms) {
  return isNumber(ms) ? formatSeconds(ms / 1e3) : MISSING;
}

export function formatMicros(us) {
  return isNumber(us) ? formatSeconds(us / 1e6) : MISSING;
}

// "12 s ago", "5 min ago"; used for the idle state of the tuning pipeline.
export function formatAgo(seconds) {
  if (!isNumber(seconds) || seconds < 0) return MISSING;
  if (seconds < 1) return 'just now';
  return formatSeconds(Math.floor(seconds)) + ' ago';
}

function pad2(n) {
  return n < 10 ? '0' + n : String(n);
}

// ISO 8601 time stamp as "2026-09-29 12:00". Local time by default; pass
// { utc: true } for a result that does not depend on the time zone.
export function formatDateTime(iso, options = {}) {
  if (typeof iso !== 'string' || iso === '') return MISSING;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return MISSING;
  const utc = options.utc === true;
  const y = utc ? date.getUTCFullYear() : date.getFullYear();
  const mo = (utc ? date.getUTCMonth() : date.getMonth()) + 1;
  const d = utc ? date.getUTCDate() : date.getDate();
  const h = utc ? date.getUTCHours() : date.getHours();
  const mi = utc ? date.getUTCMinutes() : date.getMinutes();
  return `${y}-${pad2(mo)}-${pad2(d)} ${pad2(h)}:${pad2(mi)}`;
}

// Seconds between two ISO time stamps, or NaN when either is missing.
export function secondsBetween(startIso, endIso) {
  const a = new Date(startIso).getTime();
  const b = new Date(endIso).getTime();
  return Number.isNaN(a) || Number.isNaN(b) ? NaN : (b - a) / 1000;
}

// Shortens a path in the middle, keeping the file name: the last component is
// what identifies a unit, the leading directories give the context.
export function shortenPath(path, max = 40) {
  if (typeof path !== 'string') return '';
  if (path.length <= max) return path;
  const slash = path.lastIndexOf('/');
  const name = slash < 0 ? path : path.slice(slash + 1);
  if (name.length + 2 >= max) return ELLIPSIS + name.slice(name.length - (max - 1));
  const head = path.slice(0, max - name.length - 2);
  return head + ELLIPSIS + '/' + name;
}

// "1 unit", "3 units".
export function plural(n, singular, pluralForm) {
  const word = n === 1 ? singular : pluralForm || singular + 's';
  return `${formatCount(n)} ${word}`;
}
