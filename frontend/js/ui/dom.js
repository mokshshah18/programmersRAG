// Element construction without markup strings.
//
// Everything the application shows goes through h(): text becomes text nodes,
// attributes are set one by one. There is no way to pass markup, so content
// that comes from the model or from files cannot become elements.

const SVG_NS = 'http://www.w3.org/2000/svg';

// Properties that are assigned directly; everything else becomes an attribute.
const PROPERTIES = new Set([
  'id', 'type', 'value', 'title', 'hidden', 'disabled', 'checked', 'tabIndex', 'htmlFor', 'name', 'href', 'target', 'rel',
  'min', 'max', 'step', 'placeholder', 'open', 'colSpan', 'rowSpan', 'scope', 'rows', 'maxLength', 'autocomplete',
  'spellcheck', 'required', 'readOnly', 'inputMode', 'lang',
]);

const FORBIDDEN = new Set(['innerHTML', 'outerHTML', 'srcdoc']);

function append(parent, child) {
  if (child === null || child === undefined || child === false || child === true) return;
  if (Array.isArray(child)) {
    for (const c of child) append(parent, c);
    return;
  }
  parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
}

// h('div', { class: 'unit', 'aria-label': 'x', on: { click: f } }, 'text', child, ...)
export function h(tag, props, ...children) {
  const element = document.createElement(tag);
  if (props) {
    for (const key of Object.keys(props)) {
      const value = props[key];
      if (value === undefined || value === null) continue;
      if (FORBIDDEN.has(key) || (key.startsWith('on') && key !== 'on')) {
        throw new Error(`h(): "${key}" is not allowed`);
      }
      if (key === 'class') element.className = value;
      else if (key === 'text') element.textContent = String(value);
      else if (key === 'on') {
        for (const type of Object.keys(value)) element.addEventListener(type, value[type]);
      } else if (key === 'style') {
        for (const name of Object.keys(value)) element.style.setProperty(name, String(value[name]));
      } else if (key === 'dataset') {
        for (const name of Object.keys(value)) element.dataset[name] = String(value[name]);
      } else if (PROPERTIES.has(key)) element[key] = value;
      else if (value === false) element.removeAttribute(key);
      else element.setAttribute(key, value === true ? '' : String(value));
    }
  }
  append(element, children);
  return element;
}

export function clear(element) {
  element.replaceChildren();
  return element;
}

export function replace(element, ...children) {
  element.replaceChildren();
  append(element, children);
  return element;
}

// An icon from the sprite in index.html. Decorative: the control that holds it
// carries the accessible name.
export function icon(name) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', '#i-' + name);
  svg.appendChild(use);
  return svg;
}

export function byId(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`element #${id} is missing from index.html`);
  return element;
}

// Badge with a tone: neutral, good, warn, bad, accent.
export function badge(text, tone = 'neutral') {
  return h('span', { class: 'badge', 'data-tone': tone, text });
}

// One label-value pair of a definition grid.
export function fact(label, value, title) {
  return h('div', { class: 'fact', title }, h('dt', { text: label }), h('dd', null, value));
}

// Statistic tile: label above, value below, optional note. The tone "bad"
// marks a value that must not occur.
export function stat(label, value, note, tone) {
  return h('div', { class: 'stat', 'data-tone': tone || null },
    h('div', { class: 'stat-label', text: label }),
    h('div', { class: 'stat-value' }, value),
    note ? h('div', { class: 'stat-note' }, note) : null);
}

// Message shown where a panel has nothing to show yet.
export function emptyState(title, text) {
  return h('div', { class: 'empty' }, h('p', { class: 'empty-title', text: title }), text ? h('p', { text }) : null);
}

// Banner with an icon and a label; the tone is never carried by colour alone.
export function banner(tone, title, text) {
  const names = { good: 'check', warn: 'warning', bad: 'error', info: 'info' };
  return h('div', { class: 'banner', 'data-tone': tone, role: tone === 'bad' ? 'alert' : null },
    icon(names[tone] || 'info'),
    h('div', { class: 'banner-text' }, h('strong', { text: title }), text ? h('span', { text: ' ' + text }) : null));
}

// Runs the callback before the next repaint. A page that is not visible gets
// no animation frames, but an answer that arrives meanwhile must still be put
// on the page, so a timer takes over there. Returns a function that cancels.
export function nextFrame(callback) {
  let done = false;
  let frame = 0;
  let timer = 0;
  const stop = () => {
    done = true;
    clearTimeout(timer);
    if (frame !== 0) cancelAnimationFrame(frame);
  };
  // The frame and the timer race; whichever comes first cancels the other.
  const run = () => {
    if (done) return;
    stop();
    callback();
  };
  timer = setTimeout(run, document.hidden ? 100 : 500);
  if (!document.hidden) frame = requestAnimationFrame(run);
  return stop;
}

// True when the user asked the system to reduce motion.
export function reducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

// Copies text to the clipboard. Resolves to false when that is not possible.
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
