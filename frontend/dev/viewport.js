// Development helper, see viewport.html: shows the application in a frame of a
// chosen size and colour scheme and reports elements that stick out of it.
//
// The window of a desktop browser cannot be made 360 pixels wide, and a script
// cannot change what prefers-color-scheme reports for its own document. A frame
// can do both: its width is the viewport of the document inside, and the colour
// scheme of the frame element is the preferred scheme of that document.
//
// The frame and this page come from the same server, so the page may look into
// the frame. Only the mock server is expected to serve this file.

const frame = document.getElementById('frame');
const form = document.getElementById('bar');
const result = document.getElementById('result');

function clampInt(text, lo, hi, fallback) {
  const n = Number.parseInt(String(text), 10);
  return Number.isInteger(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

function apply(width, height, scheme) {
  frame.style.setProperty('width', width + 'px');
  frame.style.setProperty('height', height + 'px');
  frame.dataset.scheme = scheme === 'dark' ? 'dark' : 'light';
  form.elements.w.value = String(width);
  form.elements.h.value = String(height);
  form.elements.scheme.value = frame.dataset.scheme;
}

// Containers that are meant to scroll sideways: code, diffs, tables and the
// tab list may be wider than the screen as long as they scroll by themselves.
const SCROLLS_SIDEWAYS = '.code-text, .code-view, .diff-scroll, .table-scroll, .tablist';

function nameOf(element) {
  const classes = typeof element.className === 'string' ? element.className.trim() : '';
  return element.tagName.toLowerCase() + (element.id ? '#' + element.id : '') +
    (classes ? '.' + classes.split(/\s+/).join('.') : '');
}

// Layout faults of the framed document at its current width:
//   * the document or the application frame is wider than the viewport;
//   * a container that should only scroll vertically scrolls sideways;
//   * a rendered element reaches beyond the edge of the viewport without being
//     inside a container that scrolls sideways by design.
// The body clips what sticks out, so it does not count as such a container:
// what it clips is lost to the user.
export function findOverflow(doc) {
  const view = doc.defaultView;
  const width = doc.documentElement.clientWidth;
  const found = [];

  if (doc.documentElement.scrollWidth > width) {
    found.push(`the document is wider than the viewport: ${doc.documentElement.scrollWidth} > ${width}`);
  }
  const contained = (element) => {
    for (let e = element.parentElement; e && e !== doc.body; e = e.parentElement) {
      if (view.getComputedStyle(e).overflowX === 'visible') continue;
      const rect = e.getBoundingClientRect();
      if (rect.right <= width + 0.5 && rect.left >= -0.5) return true;
    }
    return false;
  };
  for (const element of doc.body.querySelectorAll('*')) {
    if (found.length >= 40) break;
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const style = view.getComputedStyle(element);
    if (style.visibility === 'hidden' || element.closest('.visually-hidden, .sprite')) continue;

    const scrolls = style.overflowX === 'auto' || style.overflowX === 'scroll';
    if (scrolls && element.scrollWidth > element.clientWidth + 1 && !element.matches(SCROLLS_SIDEWAYS) &&
        !element.closest('dialog:not([open])')) {
      found.push(`${nameOf(element)} scrolls sideways: ${element.scrollWidth} > ${element.clientWidth}`);
    }
    if (rect.right <= width + 0.5 && rect.left >= -0.5) continue;
    if (contained(element)) continue;
    found.push(`${nameOf(element)}: ${Math.round(rect.left)}..${Math.round(rect.right)} of ${width}`);
  }
  return found;
}

// Walks through both panes and every tab of the inspector and collects the
// faults of each. Resolves with { view: [fault, ...] }; views without faults
// are left out.
export async function walk(doc) {
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const report = {};
  const note = (view) => {
    const found = findOverflow(doc);
    if (found.length > 0) report[view] = found;
  };
  const press = (selector) => {
    const element = doc.querySelector(selector);
    if (element) element.click();
    return element !== null;
  };

  press('.view-button[data-view="conversation"]');
  await pause(100);
  note('conversation');
  press('.view-button[data-view="inspector"]');
  for (const tab of doc.querySelectorAll('[role="tab"]')) {
    tab.click();
    await pause(tab.id === 'tab-groups' ? 700 : 150);
    for (const details of doc.querySelectorAll(`#${tab.getAttribute('aria-controls')} details`)) details.open = true;
    await pause(150);
    note(tab.id.replace(/^tab-/, ''));
  }
  if (press('#settings-button')) {
    await pause(500);
    note('settings');
    press('#settings-close');
  }
  press('.view-button[data-view="conversation"]');
  return report;
}

function check() {
  const doc = frame.contentDocument;
  if (!doc || !doc.body) {
    result.textContent = 'the frame is not loaded';
    return [];
  }
  const found = findOverflow(doc);
  const dark = doc.defaultView.matchMedia('(prefers-color-scheme: dark)').matches;
  result.textContent = `${doc.documentElement.clientWidth} px, ${dark ? 'dark' : 'light'}: ` +
    (found.length === 0 ? 'nothing sticks out' : `${found.length} element(s) stick out\n${found.join('\n')}`);
  return found;
}

const params = new URLSearchParams(location.search);
apply(clampInt(params.get('w'), 240, 3840, 360), clampInt(params.get('h'), 240, 2160, 740), params.get('scheme'));

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const width = clampInt(form.elements.w.value, 240, 3840, 360);
  const height = clampInt(form.elements.h.value, 240, 2160, 740);
  const scheme = form.elements.scheme.value;
  apply(width, height, scheme);
  history.replaceState(null, '', `?w=${width}&h=${height}&scheme=${frame.dataset.scheme}`);
});
document.getElementById('check').addEventListener('click', check);

// For use from the console and from browser automation.
window.viewportCheck = { apply, check, frame, walk: () => walk(frame.contentDocument) };
