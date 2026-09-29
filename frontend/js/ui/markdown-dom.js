// Turns the tree of lib/markdown.js into DOM nodes, and shows code.
//
// Only createElement and text nodes are used. A link gets its target from
// safeUrl() in the parser, opens in a new tab and carries no referrer.

import { parseMarkdown } from '../lib/markdown.js';
import { tokenize } from '../lib/highlight.js';
import { h, icon, copyText } from './dom.js';

const MAX_HIGHLIGHT_TOKENS = 20000;

// <code> element with the tokens of `text` as spans.
export function highlighted(text, language) {
  const code = h('code', null);
  const tokens = tokenize(text, language);
  if (tokens.length > MAX_HIGHLIGHT_TOKENS) {
    code.textContent = text;
    return code;
  }
  for (const token of tokens) {
    if (token.kind === 'plain') code.appendChild(document.createTextNode(token.text));
    else code.appendChild(h('span', { class: 'tok-' + token.kind, text: token.text }));
  }
  return code;
}

// Code with line numbers. `firstLine` is the number of the first line.
export function codeView(text, options = {}) {
  const body = text.endsWith('\n') ? text.slice(0, -1) : text;
  const count = body === '' ? 1 : body.split('\n').length;
  const first = Number.isInteger(options.firstLine) && options.firstLine > 0 ? options.firstLine : 1;
  let numbers = '';
  for (let i = 0; i < count; ++i) numbers += (i > 0 ? '\n' : '') + (first + i);
  return h('div', { class: 'code-view' },
    h('pre', { class: 'code-gutter', 'aria-hidden': 'true', text: numbers }),
    h('pre', { class: 'code-text', tabIndex: 0, 'aria-label': options.label || 'Code' }, highlighted(body, options.language)));
}

function copyButton(getText) {
  const label = h('span', { class: 'btn-text', text: 'Copy' });
  const button = h('button', { type: 'button', class: 'btn btn-quiet btn-small' }, icon('copy'), label);
  button.addEventListener('click', async () => {
    const ok = await copyText(getText());
    label.textContent = ok ? 'Copied' : 'Copy failed';
    setTimeout(() => (label.textContent = 'Copy'), 1600);
  });
  return button;
}

// Code block of an answer: header with the language label, then the code.
function codeBlock(node) {
  const names = { edit: 'edit block', '': 'code' };
  const label = Object.prototype.hasOwnProperty.call(names, node.lang) ? names[node.lang] : node.lang;
  return h('div', { class: 'code-block' },
    h('div', { class: 'code-block-header' }, h('span', { class: 'code-lang', text: label }), copyButton(() => node.text)),
    h('pre', { class: 'code-text', tabIndex: 0, 'aria-label': `Code, ${label}` }, highlighted(node.text, node.lang)));
}

function inline(nodes, parent, context) {
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        parent.appendChild(document.createTextNode(node.text));
        break;
      case 'code': {
        const target = context.resolve ? context.resolve(node.text) : null;
        if (target !== null && target !== undefined) {
          // A citation of a retrieved unit: opens the unit in the inspector.
          const button = h('button', { type: 'button', class: 'cite', title: 'Show this unit in the inspector' },
            h('code', { text: node.text }));
          button.addEventListener('click', () => context.open(target));
          parent.appendChild(button);
        } else {
          parent.appendChild(h('code', { text: node.text }));
        }
        break;
      }
      case 'break':
        parent.appendChild(h('br', null));
        break;
      case 'strong':
        inline(node.children, parent.appendChild(h('strong', null)), context);
        break;
      case 'em':
        inline(node.children, parent.appendChild(h('em', null)), context);
        break;
      case 'link': {
        const a = h('a', { href: node.href, target: '_blank', rel: 'noopener noreferrer nofollow',
                           referrerpolicy: 'no-referrer', title: node.title || null });
        inline(node.children, a, context);
        parent.appendChild(a);
        break;
      }
      default:
        break;
    }
  }
}

function block(node, context, tight) {
  switch (node.type) {
    case 'heading': {
      // Answers are part of a page that has its own h1 and h2: their headings start at h3.
      const level = Math.min(6, node.level + 2);
      const element = h('h' + level, { class: 'md-h' + node.level });
      inline(node.children, element, context);
      return element;
    }
    case 'paragraph': {
      const element = tight ? document.createDocumentFragment() : h('p', null);
      inline(node.children, element, context);
      return element;
    }
    case 'code':
      return codeBlock(node);
    case 'rule':
      return h('hr', null);
    case 'quote': {
      const element = h('blockquote', null);
      for (const child of node.children) element.appendChild(block(child, context, false));
      return element;
    }
    case 'list': {
      const element = h(node.ordered ? 'ol' : 'ul', null);
      if (node.ordered && node.start !== 1) element.start = node.start;
      for (const item of node.items) {
        const li = h('li', null);
        for (const child of item) li.appendChild(block(child, context, node.tight));
        element.appendChild(li);
      }
      return element;
    }
    default:
      return document.createDocumentFragment();
  }
}

// Renders Markdown into `container` and keeps what it rendered, so that the
// next call replaces only the blocks that changed. While an answer is
// streamed that is the last block; everything before it stays in place,
// keeps its selection and is not announced again by a screen reader.
//
// context: { resolve(codeText) -> target or null, open(target) } makes inline
// code that names a retrieved unit clickable.
export function createMarkdownView(container, context = {}) {
  let rendered = [];  // [{ key, node }]

  function render(source) {
    const blocks = parseMarkdown(source);
    const next = [];
    let same = 0;
    while (same < blocks.length && same < rendered.length &&
           rendered[same].key === JSON.stringify(blocks[same])) {
      ++same;
    }
    for (let i = rendered.length - 1; i >= same; --i) rendered[i].node.remove();
    for (let i = 0; i < same; ++i) next.push(rendered[i]);
    for (let i = same; i < blocks.length; ++i) {
      let node = block(blocks[i], context, false);
      if (node.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
        const wrapper = h('div', null);
        wrapper.appendChild(node);
        node = wrapper;
      }
      container.appendChild(node);
      next.push({ key: JSON.stringify(blocks[i]), node });
    }
    rendered = next;
  }

  return {
    render,
    // Forgets what was rendered, e.g. when citations became resolvable.
    reset() {
      for (const item of rendered) item.node.remove();
      rendered = [];
    },
  };
}
