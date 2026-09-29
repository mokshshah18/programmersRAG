// A small Markdown parser for model output.
//
// Model output is untrusted. This module never produces markup that is handed
// to the browser: it turns the text into a tree of plain objects, and the DOM
// renderer (ui/markdown-dom.js) creates elements and text nodes from that tree.
// Raw HTML in the input has no meaning here; it stays text.
//
// Supported: ATX headings, paragraphs, bold, italic, inline code, fenced code
// blocks with a language label, ordered and unordered lists (nested), block
// quotes, thematic breaks, links and automatic links. Link targets are
// restricted to http, https and relative URLs (safeUrl). The two edit block
// forms of prag::edit are recognised and shown as code, so that the code inside
// them is not read as Markdown.
//
// Block nodes:
//   { type: 'heading', level, children }        { type: 'paragraph', children }
//   { type: 'code', lang, text, closed }        { type: 'rule' }
//   { type: 'list', ordered, start, tight, items: [ [block, ...], ... ] }
//   { type: 'quote', children: [block, ...] }
// Inline nodes:
//   { type: 'text', text }   { type: 'code', text }   { type: 'break' }
//   { type: 'strong', children }   { type: 'em', children }
//   { type: 'link', href, title, children }
//
// The parser is tolerant of incomplete input, because it runs on every
// animation frame while an answer is streamed: an unclosed fence extends to the
// end of the text, an unclosed emphasis mark stays literal.
//
// Every loop over the input is linear or has a fixed bound. Patterns of the
// form /x+$/ are avoided, because they take quadratic time on adversarial text.

const MAX_BLOCK_DEPTH = 8;      // nesting of lists and quotes
const MAX_INLINE_DEPTH = 8;     // nesting of emphasis and links
const MAX_URL_LENGTH = 2048;
const MAX_LABEL_LENGTH = 1000;  // bounds the search for the end of a link label

const FENCE_OPEN = /^( {0,3})(`{3,}|~{3,})(.*)$/;
const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*))?$/;
const RULE = /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const QUOTE = /^ {0,3}> ?(.*)$/;
const ITEM = /^( *)([-*+]|\d{1,9}[.)])(?:( +)(.*))?$/;
const EDIT_OPEN = /^ {0,3}<{5,9} (?:SEARCH|UNIT)\b/;
const EDIT_CLOSE = /^ {0,3}>{5,9} (?:REPLACE|UNIT)\s*$/;
const EDIT_SCOPE = /^@@ \S/;                                     // "@@ unit_name" above a SEARCH block
const EDIT_PATH = /^[^\s`*<>|"]{1,260}[./][^\s`*<>|"]{0,260}$/;  // one word that holds a dot or a slash
const LANGUAGE = /^[A-Za-z0-9_+#.-]{1,24}$/;
const ESCAPABLE = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';

// Characters that browsers drop or ignore when they parse a URL: C0 and C1
// controls, the space, zero-width and directional marks, line and paragraph
// separators, the byte order mark.
const URL_IGNORED = /[\x00-\x20\x7f-\x9f\u{200b}-\u{200f}\u{2028}\u{2029}\u{feff}]/gu;

// ---- escaping and URLs ------------------------------------------------------

const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
}

// Returns the URL when it is safe to use as a link target, otherwise null.
// Safe means: scheme http or https, or no scheme at all (a relative URL).
// Browsers remove tabs and line breaks from a URL before they look at the
// scheme, and ignore leading control characters and spaces, so the test is
// made on the text with those characters removed.
export function safeUrl(href) {
  if (typeof href !== 'string') return null;
  const url = href.replace(URL_IGNORED, '');
  if (url === '' || url.length > MAX_URL_LENGTH) return null;
  const scheme = /^([A-Za-z][A-Za-z0-9+.-]*):/.exec(url);
  if (scheme) {
    const name = scheme[1].toLowerCase();
    return name === 'http' || name === 'https' ? url : null;
  }
  return url;
}

// ---- inline -----------------------------------------------------------------

function isSpace(c) {
  return c === undefined || c === ' ' || c === '\n' || c === '\t' || c === '\xa0';
}

function isWord(c) {
  return c !== undefined && /[\p{L}\p{N}]/u.test(c);
}

function trimEndBlanks(text) {
  let end = text.length;
  while (end > 0 && (text[end - 1] === ' ' || text[end - 1] === '\t')) --end;
  return end === text.length ? text : text.slice(0, end);
}

function endsWithTwoSpaces(text) {
  return text.length >= 2 && text[text.length - 1] === ' ' && text[text.length - 2] === ' ';
}

// Text of an ATX heading without the optional closing sequence of number signs.
function headingText(raw) {
  const text = raw.trim();
  let end = text.length;
  while (end > 0 && text[end - 1] === '#') --end;
  if (end === 0) return '';
  if (end < text.length && (text[end - 1] === ' ' || text[end - 1] === '\t')) return text.slice(0, end).trim();
  return text;
}

// Index of the run of exactly `n` backticks that closes a code span, or -1.
function findBackticks(text, from, n) {
  let i = from;
  while (i < text.length) {
    if (text[i] !== '`') {
      ++i;
      continue;
    }
    let run = 1;
    while (text[i + run] === '`') ++run;
    if (run === n) return i;
    i += run;
  }
  return -1;
}

// Finds the closing delimiter (`ch` repeated `len` times) of an emphasis that
// opened before `from`. Code spans and escaped characters are skipped. Returns
// the index of the first character of the closing delimiter, or -1.
function findCloser(text, from, ch, len) {
  let i = from;
  while (i < text.length) {
    const c = text[i];
    if (c === '\\') {
      i += 2;
      continue;
    }
    if (c === '`') {
      let run = 1;
      while (text[i + run] === '`') ++run;
      const close = findBackticks(text, i + run, run);
      i = close < 0 ? i + run : close + run;
      continue;
    }
    if (c !== ch) {
      ++i;
      continue;
    }
    let run = 1;
    while (text[i + run] === ch) ++run;
    const before = text[i - 1];
    const after = text[i + run];
    const closes = i > from && !isSpace(before) && (ch !== '_' || !isWord(after));
    // A run of two inside an italic span belongs to a bold span nested in it.
    const fits = len === 1 ? run !== 2 : run >= 2;
    if (closes && fits) return i + (run - len);
    i += run;
  }
  return -1;
}

// Parses "[label](destination "title")" starting at the opening bracket.
function matchLink(text, start) {
  let depth = 0;
  let i = start;
  let labelEnd = -1;
  const limit = Math.min(text.length, start + MAX_LABEL_LENGTH);
  while (i < limit) {
    const c = text[i];
    if (c === '\\') {
      i += 2;
      continue;
    }
    if (c === '[') ++depth;
    else if (c === ']' && --depth === 0) {
      labelEnd = i;
      break;
    }
    ++i;
  }
  if (labelEnd < 0 || text[labelEnd + 1] !== '(') return null;

  i = labelEnd + 2;
  const end = Math.min(text.length, i + MAX_URL_LENGTH + MAX_LABEL_LENGTH);
  while (text[i] === ' ' || text[i] === '\n') ++i;
  let href = '';
  if (text[i] === '<') {
    const close = text.indexOf('>', i + 1);
    if (close < 0 || close > end) return null;
    href = text.slice(i + 1, close);
    if (href.indexOf('\n') >= 0) return null;
    i = close + 1;
  } else {
    let parens = 0;
    const begin = i;
    while (i < end) {
      const c = text[i];
      if (c === '\\' && i + 1 < end) {
        i += 2;
        continue;
      }
      if (c === ' ' || c === '\n') break;
      if (c === '(') ++parens;
      else if (c === ')') {
        if (parens === 0) break;
        --parens;
      }
      ++i;
    }
    href = text.slice(begin, i).replace(/\\([!-/:-@[-`{-~])/g, '$1');
  }
  while (text[i] === ' ' || text[i] === '\n') ++i;
  let title = '';
  const quote = text[i];
  if (quote === '"' || quote === "'") {
    const close = text.indexOf(quote, i + 1);
    if (close < 0 || close > end) return null;
    title = text.slice(i + 1, close);
    i = close + 1;
    while (text[i] === ' ' || text[i] === '\n') ++i;
  }
  if (text[i] !== ')') return null;
  return { label: text.slice(start + 1, labelEnd), href, title, end: i + 1 };
}

// Length of the URL that starts at `start` ("http://" or "https://"), without
// trailing punctuation that belongs to the sentence rather than to the URL.
function bareUrlLength(text, start) {
  const match = /^https?:\/\/[^\s<>`"]+/.exec(text.slice(start, start + MAX_URL_LENGTH));
  if (!match) return 0;
  const url = match[0];
  let opening = 0;
  let closing = 0;
  for (const c of url) {
    if (c === '(') ++opening;
    else if (c === ')') ++closing;
  }
  let end = url.length;
  while (end > 0) {
    const last = url[end - 1];
    if ('.,;:!?\'"*_'.indexOf(last) >= 0) {
      --end;
    } else if (last === ')' && closing > opening) {
      --closing;  // a closing parenthesis without its opening one inside the URL
      --end;
    } else {
      break;
    }
  }
  return end > 'https://'.length ? end : 0;
}

export function parseInline(text, depth = 0, inLink = false) {
  const out = [];
  let plain = '';
  const flush = () => {
    if (plain !== '') {
      out.push({ type: 'text', text: plain });
      plain = '';
    }
  };
  // Delimiters for which a search has failed: no closer exists further right either.
  const hopeless = new Set();
  const nested = depth < MAX_INLINE_DEPTH;

  let i = 0;
  while (i < text.length) {
    const c = text[i];

    if (c === '\\') {
      const next = text[i + 1];
      if (next === '\n') {
        flush();
        out.push({ type: 'break' });
        i += 2;
        while (text[i] === ' ') ++i;
        continue;
      }
      if (next !== undefined && ESCAPABLE.indexOf(next) >= 0) {
        plain += next;
        i += 2;
        continue;
      }
      plain += c;
      ++i;
      continue;
    }

    if (c === '`') {
      let run = 1;
      while (text[i + run] === '`') ++run;
      const key = '`' + run;
      const close = hopeless.has(key) ? -1 : findBackticks(text, i + run, run);
      if (close >= 0) {
        let code = text.slice(i + run, close).replace(/\n/g, ' ');
        if (code.length > 1 && code[0] === ' ' && code[code.length - 1] === ' ' && code.trim() !== '') {
          code = code.slice(1, -1);
        }
        flush();
        out.push({ type: 'code', text: code });
        i = close + run;
      } else {
        hopeless.add(key);
        plain += text.slice(i, i + run);
        i += run;
      }
      continue;
    }

    if (c === '\n') {
      const hard = endsWithTwoSpaces(plain);
      plain = trimEndBlanks(plain);
      if (hard) {
        flush();
        out.push({ type: 'break' });
      } else {
        plain += '\n';
      }
      ++i;
      while (text[i] === ' ' || text[i] === '\t') ++i;
      continue;
    }

    if ((c === '[' || (c === '!' && text[i + 1] === '[')) && nested && !inLink) {
      // An image is shown as a link to the image: loading it would be a
      // request to a server chosen by the model.
      const image = c === '!';
      const link = matchLink(text, image ? i + 1 : i);
      if (link) {
        const href = safeUrl(link.href);
        const children = parseInline(link.label, depth + 1, true);
        flush();
        if (href !== null) {
          if (children.length === 0) children.push({ type: 'text', text: href });
          out.push({ type: 'link', href, title: link.title, children });
        } else {
          for (const child of children) out.push(child);  // unsafe target: keep the words, drop the link
        }
        i = link.end;
        continue;
      }
    }

    if (c === '<' && !inLink) {
      const match = /^<(https?:\/\/[^\s<>]+)>/.exec(text.slice(i, i + MAX_URL_LENGTH));
      const href = match ? safeUrl(match[1]) : null;
      if (href !== null) {
        flush();
        out.push({ type: 'link', href, title: '', children: [{ type: 'text', text: match[1] }] });
        i += match[0].length;
        continue;
      }
    }

    if (c === 'h' && !inLink && !isWord(text[i - 1]) && text.startsWith('http', i)) {
      const length = bareUrlLength(text, i);
      const href = length > 0 ? safeUrl(text.slice(i, i + length)) : null;
      if (href !== null) {
        flush();
        out.push({ type: 'link', href, title: '', children: [{ type: 'text', text: text.slice(i, i + length) }] });
        i += length;
        continue;
      }
    }

    if (c === '*' || c === '_') {
      let run = 1;
      while (text[i + run] === c) ++run;
      const before = text[i - 1];
      const after = text[i + run];
      // Underscores inside a word are part of an identifier: mi_heap_malloc.
      const opens = !isSpace(after) && (c !== '_' || !isWord(before));
      const len = run >= 2 ? 2 : 1;
      const key = c + len;
      if (opens && nested && !hopeless.has(key)) {
        const close = findCloser(text, i + len, c, len);
        // __attribute__, __init__, __FILE__: a single word between double
        // underscores is an identifier, not bold text.
        const dunder = c === '_' && len === 2 && close > 0 && /^[A-Za-z0-9_]+$/.test(text.slice(i + len, close));
        if (close > i + len && !dunder) {
          flush();
          out.push({
            type: len === 2 ? 'strong' : 'em',
            children: parseInline(text.slice(i + len, close), depth + 1, inLink),
          });
          i = close + len;
          continue;
        }
        if (close < 0) hopeless.add(key);
      }
      plain += text.slice(i, i + run);
      i += run;
      continue;
    }

    plain += c;
    ++i;
  }
  flush();
  return out;
}

// ---- blocks -----------------------------------------------------------------

function isBlank(line) {
  return line.trim() === '';
}

function indentOf(line) {
  let n = 0;
  while (n < line.length && line[n] === ' ') ++n;
  return n;
}

function languageOf(info) {
  const word = info.trim().split(/\s+/)[0] || '';
  // "{.python}" style labels: keep what is a plain identifier.
  const name = word.replace(/^[{.]+/, '').replace(/[}]+$/, '');
  return LANGUAGE.test(name) ? name.toLowerCase() : '';
}

function isItem(line) {
  const m = ITEM.exec(line);
  return m !== null && m[1].length <= 3;
}

// True when the line starts a block other than a paragraph.
function startsBlock(line) {
  return FENCE_OPEN.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) ||
         EDIT_OPEN.test(line) || isItem(line);
}

// Bullet character, or the delimiter after the number: items of one list share it.
function flavourOf(marker) {
  return marker[marker.length - 1];
}

// True when `line` is an item that continues the list described by the arguments.
function continuesList(line, base, ordered, flavour) {
  const m = ITEM.exec(line);
  if (!m || RULE.test(line)) return false;
  if (m[1].length > base + 1 || m[1].length + 1 < base) return false;
  return /\d/.test(m[2]) === ordered && flavourOf(m[2]) === flavour;
}

// Parses a list that starts at lines[start]. Returns { node, next }.
function parseList(lines, start, depth) {
  const first = ITEM.exec(lines[start]);
  const ordered = /\d/.test(first[2]);
  const flavour = flavourOf(first[2]);
  const base = first[1].length;
  const node = {
    type: 'list',
    ordered,
    start: ordered ? parseInt(first[2], 10) : 1,
    tight: true,
    items: [],
  };

  let i = start;
  while (i < lines.length && continuesList(lines[i], base, ordered, flavour)) {
    const m = ITEM.exec(lines[i]);
    // More than four spaces after the marker start an indented text; the
    // content column is then one past the marker.
    const wide = m[3] !== undefined && m[3].length > 4;
    const spaces = m[3] === undefined || wide ? 1 : m[3].length;
    const width = m[1].length + m[2].length + spaces;  // column where the content starts
    const nestedFrom = Math.min(width, m[1].length + 2);
    const body = [m[4] === undefined ? '' : (wide ? m[3].slice(1) : '') + m[4]];
    let blanks = 0;         // blank lines seen and not yet attributed
    let inner = false;      // a blank line separates two parts of this item
    let fence = null;       // marker of the code fence that is open inside the item
    ++i;
    while (i < lines.length) {
      const line = lines[i];
      if (isBlank(line)) {
        ++blanks;
        ++i;
        continue;
      }
      const indent = indentOf(line);
      if (indent >= nestedFrom) {
        if (blanks > 0 && fence === null) inner = true;
        for (; blanks > 0; --blanks) body.push('');
        const text = line.slice(Math.min(indent, width));
        body.push(text);
        const f = FENCE_OPEN.exec(text);
        if (fence === null) {
          if (f) fence = f[2];
        } else if (f && f[2][0] === fence[0] && f[2].length >= fence.length && f[3].trim() === '') {
          fence = null;
        }
        ++i;
        continue;
      }
      if (blanks === 0 && fence === null && !startsBlock(line)) {
        body.push(line.trim());  // lazy continuation of the paragraph
        ++i;
        continue;
      }
      break;
    }
    node.items.push(parseBlocks(body, depth + 1));
    if (inner) node.tight = false;
    if (blanks > 0) {
      if (i < lines.length && continuesList(lines[i], base, ordered, flavour)) {
        node.tight = false;  // blank lines between two items make the list loose
      } else {
        i -= blanks;         // the blank lines belong to what follows the list
        break;
      }
    }
  }
  return { node, next: i };
}

function parseBlocks(lines, depth) {
  const blocks = [];
  let paragraph = [];
  const flush = () => {
    if (paragraph.length > 0) {
      blocks.push({ type: 'paragraph', children: parseInline(paragraph.join('\n')) });
      paragraph = [];
    }
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) {
      flush();
      ++i;
      continue;
    }

    let m = FENCE_OPEN.exec(line);
    if (m && !(m[2][0] === '`' && m[3].indexOf('`') >= 0)) {
      flush();
      const indent = m[1].length;
      const marker = m[2];
      const body = [];
      let closed = false;
      ++i;
      while (i < lines.length) {
        const c = FENCE_OPEN.exec(lines[i]);
        if (c && c[2][0] === marker[0] && c[2].length >= marker.length && c[3].trim() === '') {
          closed = true;
          ++i;
          break;
        }
        body.push(lines[i].slice(Math.min(indentOf(lines[i]), indent)));
        ++i;
      }
      blocks.push({ type: 'code', lang: languageOf(m[3]), text: body.join('\n'), closed });
      continue;
    }

    if (EDIT_OPEN.test(line)) {
      // The lines that name the file and the unit belong to the edit block.
      const head = [];
      if (paragraph.length > 0 && EDIT_SCOPE.test(paragraph[paragraph.length - 1])) head.unshift(paragraph.pop());
      if (paragraph.length > 0 && EDIT_PATH.test(paragraph[paragraph.length - 1])) {
        head.unshift(paragraph.pop());
      } else if (head.length > 0) {
        paragraph.push(head.pop());  // a scope line without a path is ordinary text
      }
      flush();
      const body = head;
      let closed = false;
      while (i < lines.length) {
        body.push(lines[i]);
        if (EDIT_CLOSE.test(lines[i++])) {
          closed = true;
          break;
        }
      }
      blocks.push({ type: 'code', lang: 'edit', text: body.join('\n'), closed });
      continue;
    }

    m = HEADING.exec(line);
    if (m) {
      flush();
      blocks.push({ type: 'heading', level: m[1].length, children: parseInline(headingText(m[2] || '')) });
      ++i;
      continue;
    }

    if (RULE.test(line)) {
      flush();
      blocks.push({ type: 'rule' });
      ++i;
      continue;
    }

    if (depth < MAX_BLOCK_DEPTH) {
      m = QUOTE.exec(line);
      if (m) {
        flush();
        const body = [];
        while (i < lines.length) {
          const q = QUOTE.exec(lines[i]);
          if (q) body.push(q[1]);
          else if (!isBlank(lines[i]) && !startsBlock(lines[i])) body.push(lines[i]);  // lazy continuation
          else break;
          ++i;
        }
        blocks.push({ type: 'quote', children: parseBlocks(body, depth + 1) });
        continue;
      }

      if (isItem(line)) {
        flush();
        const list = parseList(lines, i, depth);
        blocks.push(list.node);
        i = list.next;
        continue;
      }
    }

    paragraph.push(line.replace(/^[ \t]+/, ''));
    ++i;
  }
  flush();
  return blocks;
}

// Parses Markdown text into a list of block nodes.
export function parseMarkdown(source) {
  const text = String(source === undefined || source === null ? '' : source)
    .replace(/\r\n?/g, '\n')
    .replace(/\x00/g, '\u{fffd}');
  return parseBlocks(text.split('\n'), 0);
}

// ---- serialisation ----------------------------------------------------------

function inlineHtml(nodes) {
  let html = '';
  for (const node of nodes) {
    switch (node.type) {
      case 'text':
        html += escapeHtml(node.text);
        break;
      case 'code':
        html += `<code>${escapeHtml(node.text)}</code>`;
        break;
      case 'break':
        html += '<br>';
        break;
      case 'strong':
        html += `<strong>${inlineHtml(node.children)}</strong>`;
        break;
      case 'em':
        html += `<em>${inlineHtml(node.children)}</em>`;
        break;
      case 'link': {
        const title = node.title ? ` title="${escapeHtml(node.title)}"` : '';
        html += `<a href="${escapeHtml(node.href)}"${title} rel="noopener noreferrer">${inlineHtml(node.children)}</a>`;
        break;
      }
      default:
        break;
    }
  }
  return html;
}

function blockHtml(blocks, tight) {
  let html = '';
  for (const node of blocks) {
    switch (node.type) {
      case 'heading':
        html += `<h${node.level}>${inlineHtml(node.children)}</h${node.level}>\n`;
        break;
      case 'paragraph':
        html += tight ? `${inlineHtml(node.children)}\n` : `<p>${inlineHtml(node.children)}</p>\n`;
        break;
      case 'code': {
        const lang = node.lang ? ` data-lang="${escapeHtml(node.lang)}"` : '';
        html += `<pre${lang}><code>${escapeHtml(node.text)}</code></pre>\n`;
        break;
      }
      case 'rule':
        html += '<hr>\n';
        break;
      case 'quote':
        html += `<blockquote>\n${blockHtml(node.children, false)}</blockquote>\n`;
        break;
      case 'list': {
        const tag = node.ordered ? 'ol' : 'ul';
        const start = node.ordered && node.start !== 1 ? ` start="${node.start}"` : '';
        html += `<${tag}${start}>\n`;
        for (const item of node.items) html += `<li>${blockHtml(item, node.tight).trimEnd()}</li>\n`;
        html += `</${tag}>\n`;
        break;
      }
      default:
        break;
    }
  }
  return html;
}

// HTML text of a parsed document with every string escaped. The application
// does not use it (it builds DOM nodes instead); it exists so that the escaping
// rules can be tested without a DOM, and for tools that need a string.
export function toHtml(blocks) {
  return blockHtml(blocks, false);
}

export function renderMarkdownToHtml(source) {
  return toHtml(parseMarkdown(source));
}

// Text content of inline nodes without any markup.
export function inlineText(nodes) {
  let text = '';
  for (const node of nodes) {
    if (node.type === 'text' || node.type === 'code') text += node.text;
    else if (node.type === 'break') text += '\n';
    else if (node.children) text += inlineText(node.children);
  }
  return text;
}
