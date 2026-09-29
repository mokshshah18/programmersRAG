// Confirmation dialog. A <dialog> opened with showModal() keeps the focus
// inside, closes on Escape and returns the focus to the element that had it.

import { byId } from './dom.js';

export function createConfirm() {
  const dialog = byId('confirm-dialog');
  const title = byId('confirm-title');
  const text = byId('confirm-text');
  const accept = byId('confirm-accept');
  const cancel = byId('confirm-cancel');
  let settle = null;

  function finish(value) {
    if (dialog.open) dialog.close();
    if (settle) {
      const resolve = settle;
      settle = null;
      resolve(value);
    }
  }

  accept.addEventListener('click', () => finish(true));
  cancel.addEventListener('click', () => finish(false));
  dialog.addEventListener('close', () => finish(false));  // Escape
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) finish(false);           // click on the backdrop
  });

  // confirm({ title, text, action, danger }) -> Promise<boolean>
  return function confirm(options) {
    if (settle) finish(false);
    title.textContent = options.title;
    text.textContent = options.text || '';
    accept.textContent = options.action || 'OK';
    accept.classList.toggle('btn-danger', options.danger === true);
    accept.classList.toggle('btn-primary', options.danger !== true);
    return new Promise((resolve) => {
      settle = resolve;
      dialog.showModal();
      cancel.focus();  // the safe choice has the focus
    });
  };
}
