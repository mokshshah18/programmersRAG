// Status messages. They appear in a corner and are announced by screen readers:
// information politely (role status), errors at once (role alert).

import { h, icon, byId } from './dom.js';

const LIFETIME = { info: 4000, good: 4000, warn: 7000, bad: 10000 };
const MAX_VISIBLE = 4;

export function createToasts() {
  const polite = byId('toast-polite');
  const assertive = byId('toast-assertive');

  function visible() {
    return [...polite.children, ...assertive.children];
  }

  function show(tone, text) {
    const message = String(text);
    // The same message twice in a row is shown once.
    for (const old of visible()) {
      if (old.dataset.text === message) old.remove();
    }
    const names = { good: 'check', warn: 'warning', bad: 'error', info: 'info' };
    const close = h('button', { type: 'button', class: 'btn-icon toast-close', 'aria-label': 'Dismiss message' }, icon('close'));
    const toast = h('div', { class: 'toast', 'data-tone': tone, dataset: { text: message } },
      icon(names[tone] || 'info'), h('span', { class: 'toast-text', text: message }), close);
    const remove = () => toast.remove();
    close.addEventListener('click', remove);
    const timer = setTimeout(remove, LIFETIME[tone] || LIFETIME.info);
    // Reading takes time: the message stays while the pointer or the focus is on it.
    toast.addEventListener('pointerenter', () => clearTimeout(timer));
    toast.addEventListener('focusin', () => clearTimeout(timer));
    (tone === 'bad' ? assertive : polite).appendChild(toast);
    const all = visible();
    for (let i = 0; i + MAX_VISIBLE < all.length; ++i) all[i].remove();
  }

  return {
    info: (text) => show('info', text),
    good: (text) => show('good', text),
    warn: (text) => show('warn', text),
    error: (text) => show('bad', text),
  };
}
