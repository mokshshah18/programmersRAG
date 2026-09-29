// Tests of the frontend logic. Run with:  node frontend/dev/test.js [filter]
//
// Covers the modules that import nothing from the DOM (Markdown, event stream
// parser, diff parser, formatting, highlighting, map geometry, settings model)
// and the API client, which is run against the mock server on a free port.
// No dependencies. The harness mirrors tests/prag_test.hpp: every case prints
// "[ ok ]" or "[FAIL]", and the exit status is the number of failed cases > 0.

import { parseMarkdown, parseInline, renderMarkdownToHtml, safeUrl, escapeHtml, inlineText } from '../js/lib/markdown.js';
import { SseParser, parseEventData, readEventStream } from '../js/lib/sse.js';
import { parseUnifiedDiff, hunkHeader, displayPath } from '../js/lib/diff.js';
import * as fmt from '../js/lib/format.js';
import { tokenize, languageOfPath } from '../js/lib/highlight.js';
import * as scatter from '../js/lib/scatter.js';
import * as model from '../js/lib/settings-model.js';
import { createApi, ApiError, isAbort } from '../js/api.js';
import { createMockServer } from './mock-server.js';

// ---- harness ------------------------------------------------------------------

const cases = [];
let failures = 0;

function test(name, body) {
  cases.push({ name, body });
}

function report(message) {
  ++failures;
  const where = (new Error().stack || '').split('\n').find((l) => l.includes('test.js') && !l.includes('report') &&
                                                                   !l.includes('check'));
  console.error(`  FAIL ${message}${where ? '\n      ' + where.trim() : ''}`);
}

function show(value) {
  return typeof value === 'string' ? JSON.stringify(value) : JSON.stringify(value);
}

function check(condition, message = 'check failed') {
  if (!condition) report(message);
}

function checkEq(actual, expected, message = '') {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a !== b) report(`${message ? message + ': ' : ''}expected ${show(expected)}, got ${show(actual)}`);
}

function checkNear(actual, expected, tolerance, message = '') {
  if (!(Math.abs(actual - expected) <= tolerance)) {
    report(`${message ? message + ': ' : ''}expected ${expected} +- ${tolerance}, got ${actual}`);
  }
}

async function checkThrows(body, predicate, message = 'expected an exception') {
  try {
    await body();
  } catch (error) {
    if (predicate && !predicate(error)) report(`${message}: wrong exception: ${error && error.message}`);
    return;
  }
  report(message);
}

const encoder = new TextEncoder();

function bytes(text) {
  return encoder.encode(text);
}

// Code points written as escapes keep this file plain ASCII.
const EN_DASH = '\u{2013}';
const MINUS = '\u{2212}';
const MICRO = '\u{b5}';
const TIMES = '\u{d7}';
const ELLIPSIS = '\u{2026}';
const EURO = '\u{20ac}';          // 3 bytes in UTF-8
const GRINNING = '\u{1f600}';     // 4 bytes in UTF-8, a surrogate pair in UTF-16
const E_ACUTE = '\u{e9}';         // 2 bytes in UTF-8

// ---- format -------------------------------------------------------------------

test('format_counts_and_fixed', () => {
  checkEq(fmt.formatCount(0), '0');
  checkEq(fmt.formatCount(999), '999');
  checkEq(fmt.formatCount(1510), '1,510');
  checkEq(fmt.formatCount(1234567), '1,234,567');
  checkEq(fmt.formatCount(-1234), MINUS + '1,234');
  checkEq(fmt.formatCount(1509.6), '1,510');
  checkEq(fmt.formatCount(NaN), EN_DASH);
  checkEq(fmt.formatCount(undefined), EN_DASH);
  checkEq(fmt.formatCount('12'), EN_DASH);
  checkEq(fmt.formatFixed(1234.5, 2), '1,234.50');
  checkEq(fmt.formatFixed(41.5, 1), '41.5');
  checkEq(fmt.formatFixed(-0.001, 2), '0.00');
  checkEq(fmt.formatFixed(Infinity, 2), EN_DASH);
});

test('format_probability', () => {
  checkEq(fmt.formatProbability(0.83), '0.83');
  checkEq(fmt.formatProbability(0), '0.00');
  checkEq(fmt.formatProbability(1), '1.00');
  checkEq(fmt.formatProbability(0.004), '<0.01');
  checkEq(fmt.formatProbability(0.996), '>0.99');
  checkEq(fmt.formatProbability(0.995), '>0.99');
  checkEq(fmt.formatProbability(0.25), '0.25');
  checkEq(fmt.formatProbability(1.7), '1.00');
  checkEq(fmt.formatProbability(-0.2), '0.00');
  checkEq(fmt.formatProbability(null), EN_DASH);
});

test('format_percent_signed_ratio', () => {
  checkEq(fmt.formatPercent(0.37), '37%');
  checkEq(fmt.formatPercent(0.375, 1), '37.5%');
  checkEq(fmt.formatPercent(1), '100%');
  checkEq(fmt.formatPercent(undefined), EN_DASH);
  checkEq(fmt.formatSigned(0.31), '+0.31');
  checkEq(fmt.formatSigned(-0.2), MINUS + '0.20');
  checkEq(fmt.formatSigned(0), '0.00');
  checkEq(fmt.formatSigned(-0.001), '0.00');
  checkEq(fmt.formatRatio(2.9), '2.9' + TIMES);
  checkEq(fmt.formatRatio(NaN), EN_DASH);
});

test('format_bytes', () => {
  checkEq(fmt.formatBytes(0), '0 B');
  checkEq(fmt.formatBytes(512), '512 B');
  checkEq(fmt.formatBytes(1024), '1 KiB');
  checkEq(fmt.formatBytes(16384), '16 KiB');
  checkEq(fmt.formatBytes(24576), '24 KiB');
  checkEq(fmt.formatBytes(49152), '48 KiB');
  checkEq(fmt.formatBytes(1536), '1.5 KiB');
  checkEq(fmt.formatBytes(2097152), '2 MiB');
  checkEq(fmt.formatBytes(31457280), '30 MiB');
  checkEq(fmt.formatBytes(1288490189), '1.2 GiB');
  checkEq(fmt.formatBytes(-1), EN_DASH);
  checkEq(fmt.formatBytes('16384'), EN_DASH);
});

test('format_durations', () => {
  checkEq(fmt.formatMicros(310), `310 ${MICRO}s`);
  checkEq(fmt.formatMicros(40), `40 ${MICRO}s`);
  checkEq(fmt.formatMicros(900), `900 ${MICRO}s`);
  checkEq(fmt.formatMicros(1500), '1.5 ms');
  checkEq(fmt.formatMicros(0), '0 s');
  checkEq(fmt.formatMillis(900), '900 ms');
  checkEq(fmt.formatMillis(4100), '4.1 s');
  checkEq(fmt.formatMillis(5100.4), '5.1 s');
  checkEq(fmt.formatMillis(12.345), '12.3 ms');
  checkEq(fmt.formatSeconds(0.4), '400 ms');
  checkEq(fmt.formatSeconds(303.9), '5 min 4 s');
  checkEq(fmt.formatSeconds(2400), '40 min');
  checkEq(fmt.formatSeconds(3900), '1 h 5 min');
  checkEq(fmt.formatSeconds(7200), '2 h');
  checkEq(fmt.formatSeconds(59.4), '59.4 s');
  checkEq(fmt.formatSeconds(2e-7), '200 ns');
  checkEq(fmt.formatSeconds(-1), EN_DASH);
  checkEq(fmt.formatAgo(12), '12 s ago');
  checkEq(fmt.formatAgo(12.9), '12 s ago');
  checkEq(fmt.formatAgo(0.2), 'just now');
  checkEq(fmt.formatAgo(125), '2 min 5 s ago');
});

test('format_dates_paths_plural', () => {
  checkEq(fmt.formatDateTime('2026-09-29T12:00:00Z', { utc: true }), '2026-09-29 12:00');
  checkEq(fmt.formatDateTime('2026-01-05T03:07:59Z', { utc: true }), '2026-01-05 03:07');
  checkEq(fmt.formatDateTime('not a date'), EN_DASH);
  checkEq(fmt.formatDateTime(''), EN_DASH);
  checkEq(fmt.formatDateTime(undefined), EN_DASH);
  checkEq(fmt.secondsBetween('2026-09-29T12:00:00Z', '2026-09-29T12:40:00Z'), 2400);
  check(Number.isNaN(fmt.secondsBetween('x', '2026-09-29T12:40:00Z')));
  checkEq(fmt.shortenPath('src/alloc.c', 40), 'src/alloc.c');
  const long = 'src/prim/windows/very/deep/directory/structure/prim.c';
  const short = fmt.shortenPath(long, 24);
  check(short.length <= 24, 'shortened path fits');
  check(short.endsWith('/prim.c'), 'file name kept');
  check(short.includes(ELLIPSIS), 'ellipsis present');
  checkEq(fmt.shortenPath('a-very-long-file-name-without-directory.c', 12).length, 12);
  checkEq(fmt.plural(1, 'unit'), '1 unit');
  checkEq(fmt.plural(3, 'unit'), '3 units');
  checkEq(fmt.plural(1510, 'unit'), '1,510 units');
  checkEq(fmt.plural(2, 'entry', 'entries'), '2 entries');
  checkEq(fmt.clamp01(1.2), 1);
  checkEq(fmt.clamp01(NaN), 0);
});

// ---- event stream parser --------------------------------------------------------

test('sse_basic_events', () => {
  const p = new SseParser();
  const events = p.feed(bytes('event: token\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\n'));
  checkEq(events.length, 2);
  checkEq(events[0], { event: 'token', data: '{"text":"a"}', id: '' });
  checkEq(events[1].event, 'done');
  checkEq(p.end(), []);
});

test('sse_field_rules', () => {
  const p = new SseParser();
  const events = p.feed(
    ': comment\n' +
    'data:no space\n\n' +                       // value without the optional space
    'data:  two spaces\n\n' +                   // only one space is removed
    'data: line 1\ndata: line 2\n\n' +          // joined with LF
    'event: named\n\n' +                        // no data: not dispatched, and the name is reset
    'data: after\n\n' +
    'unknown: x\ndata\n\n' +                    // a field without colon has the empty value
    'id: 7\nretry: 1500\ndata: with id\n\n' +
    'data: keeps id\n\n');
  checkEq(events.map((e) => e.data), ['no space', ' two spaces', 'line 1\nline 2', 'after', '', 'with id', 'keeps id']);
  checkEq(events.map((e) => e.event), ['message', 'message', 'message', 'message', 'message', 'message', 'message']);
  checkEq(events[5].id, '7');
  checkEq(events[6].id, '7');
  checkEq(p.retry, 1500);
});

test('sse_line_endings', () => {
  for (const eol of ['\n', '\r\n', '\r']) {
    const p = new SseParser();
    const events = p.feed(bytes(`event: a${eol}data: 1${eol}${eol}event: b${eol}data: 2${eol}${eol}`));
    checkEq(events.map((e) => [e.event, e.data]), [['a', '1'], ['b', '2']], `line ending ${JSON.stringify(eol)}`);
  }
});

test('sse_crlf_split_between_chunks', () => {
  const p = new SseParser();
  const events = [];
  // The CR ends one chunk and the LF starts the next, at every line ending.
  for (const chunk of ['event: a\r', '\ndata: 1\r', '\n\r', '\ndata: 2\r', '\n\r', '\n']) {
    events.push(...p.feed(bytes(chunk)));
  }
  checkEq(events.map((e) => [e.event, e.data]), [['a', '1'], ['message', '2']]);
});

test('sse_every_split_position', () => {
  const stream = `event: retrieval\ndata: {"id":"q-1","name":"caf${E_ACUTE}"}\n\n` +
                 `event: token\r\ndata: {"text":"${EURO} and ${GRINNING}"}\r\n\r\n` +
                 `: keep-alive\n\nevent: done\ndata: {"answer":"${EURO}${GRINNING}${E_ACUTE}"}\n\n`;
  const all = bytes(stream);
  const expected = new SseParser().feed(all);
  checkEq(expected.length, 3);
  checkEq(JSON.parse(expected[1].data).text, `${EURO} and ${GRINNING}`);
  // Two chunks, cut at every byte position: inside UTF-8 sequences, inside
  // CRLF pairs, inside field names.
  for (let cut = 0; cut <= all.length; ++cut) {
    const p = new SseParser();
    const events = [...p.feed(all.subarray(0, cut)), ...p.feed(all.subarray(cut)), ...p.end()];
    if (JSON.stringify(events) !== JSON.stringify(expected)) {
      report(`cut at byte ${cut} changes the result`);
      break;
    }
  }
  // One byte at a time.
  const p = new SseParser();
  const events = [];
  for (let i = 0; i < all.length; ++i) events.push(...p.feed(all.subarray(i, i + 1)));
  events.push(...p.end());
  checkEq(events, expected, 'byte by byte');
});

test('sse_random_chunking', () => {
  let state = 12345;
  const random = (n) => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state >>> 8) % n;
  };
  const parts = [];
  for (let i = 0; i < 200; ++i) {
    const text = `${EURO}${GRINNING}`.repeat(random(4)) + 'x'.repeat(random(30)) + E_ACUTE;
    parts.push(`event: token\ndata: ${JSON.stringify({ text, i })}\n\n`);
  }
  const all = bytes(parts.join(''));
  for (let round = 0; round < 20; ++round) {
    const p = new SseParser();
    const events = [];
    let at = 0;
    while (at < all.length) {
      const n = 1 + random(17);
      events.push(...p.feed(all.subarray(at, at + n)));
      at += n;
    }
    events.push(...p.end());
    checkEq(events.length, 200, 'number of events');
    const indices = events.map((e) => JSON.parse(e.data).i);
    check(indices.every((v, k) => v === k), 'events complete and in order');
  }
});

test('sse_incomplete_and_bom', () => {
  const p = new SseParser();
  checkEq(p.feed(bytes('\u{feff}data: first\n\n')).map((e) => e.data), ['first']);
  checkEq(p.feed(bytes('data: unfinished\n')), []);
  checkEq(p.end(), [], 'an event without its blank line is discarded');
  const q = new SseParser();
  checkEq(q.feed('data: a string chunk\n\n').map((e) => e.data), ['a string chunk']);
  // A truncated UTF-8 sequence at the end of the stream becomes U+FFFD and does not throw.
  const r = new SseParser();
  r.feed(bytes('data: x').subarray(0, 7));
  r.feed(bytes(EURO).subarray(0, 2));
  checkEq(r.end(), []);
});

test('sse_event_data_json', () => {
  checkEq(parseEventData({ event: 'token', data: '{"text":"a"}' }), { ok: true, value: { text: 'a' } });
  const bad = parseEventData({ event: 'token', data: '{"text":' });
  checkEq(bad.ok, false);
  check(bad.error.includes('token'));
});

test('sse_read_event_stream', async () => {
  const all = bytes(`event: a\ndata: 1\n\nevent: b\ndata: ${EURO}\n\n`);
  const body = new ReadableStream({
    start(controller) {
      for (let i = 0; i < all.length; i += 3) controller.enqueue(all.subarray(i, i + 3));
      controller.close();
    },
  });
  const seen = [];
  const count = await readEventStream(body, (e) => seen.push([e.event, e.data]));
  checkEq(count, 2);
  checkEq(seen, [['a', '1'], ['b', EURO]]);
});

// ---- diff -----------------------------------------------------------------------

const DIFF = [
  '--- a/src/alloc.c',
  '+++ b/src/alloc.c',
  '@@ -120,5 +120,8 @@ mi_heap_malloc',
  ' // comment',
  ' void* mi_heap_malloc(mi_heap_t* heap, size_t size) {',
  '+  if (size > MAX) {',
  '+    return NULL;',
  '+  }',
  '   return _mi_heap_malloc_zero(heap, size, false);',
  ' }',
  ' ',
  '@@ -200,3 +203,2 @@',
  ' a',
  '-b',
  ' c',
  '',
].join('\n');

test('diff_parse_hunks', () => {
  const d = parseUnifiedDiff(DIFF);
  checkEq(d.files.length, 1);
  const f = d.files[0];
  checkEq([f.oldPath, f.newPath, f.added, f.removed], ['src/alloc.c', 'src/alloc.c', 3, 1]);
  checkEq([d.added, d.removed], [3, 1]);
  checkEq(f.hunks.length, 2);
  const h = f.hunks[0];
  checkEq([h.oldStart, h.oldLines, h.newStart, h.newLines, h.section], [120, 5, 120, 8, 'mi_heap_malloc']);
  checkEq(h.lines.length, 8);
  checkEq(h.lines[0], { type: 'context', text: '// comment', oldLine: 120, newLine: 120 });
  checkEq(h.lines[2], { type: 'add', text: '  if (size > MAX) {', oldLine: null, newLine: 122 });
  checkEq(h.lines[5], { type: 'context', text: '  return _mi_heap_malloc_zero(heap, size, false);', oldLine: 122,
                        newLine: 125 });
  checkEq(h.lines[7], { type: 'context', text: '', oldLine: 124, newLine: 127 });
  const g = f.hunks[1];
  checkEq(g.lines.map((l) => l.type), ['context', 'del', 'context']);
  checkEq(g.lines[1], { type: 'del', text: 'b', oldLine: 201, newLine: null });
  checkEq(g.lines[2], { type: 'context', text: 'c', oldLine: 202, newLine: 204 });
  checkEq(hunkHeader(h), '@@ -120,5 +120,8 @@ mi_heap_malloc');
  checkEq(hunkHeader(g), '@@ -200,3 +203,2 @@');
  checkEq(displayPath(f), 'src/alloc.c');
});

test('diff_lines_that_look_like_headers', () => {
  // A removed line "-- x" reads "--- x", an added line "++ y" reads "+++ y".
  const text = ['--- a/q.sql', '+++ b/q.sql', '@@ -1,3 +1,3 @@', ' select 1;', '--- old comment', '+++ new comment',
                ' select 2;', ''].join('\n');
  const d = parseUnifiedDiff(text);
  checkEq(d.files.length, 1);
  checkEq(d.files[0].hunks[0].lines.map((l) => [l.type, l.text]),
          [['context', 'select 1;'], ['del', '-- old comment'], ['add', '++ new comment'], ['context', 'select 2;']]);
});

test('diff_several_files_and_special_cases', () => {
  const text = [
    'diff --git a/new.c b/new.c',
    'new file mode 100644',
    'index 0000000..e69de29',
    '--- /dev/null',
    '+++ b/new.c',
    '@@ -0,0 +1,2 @@',
    '+int x;',
    '+int y;',
    '\\ No newline at end of file',
    'diff --git a/old.c b/old.c',
    '--- a/old.c\t2026-09-29 12:00:00',
    '+++ /dev/null',
    '@@ -1 +0,0 @@',
    '-gone',
    '--- a/third.c',
    '+++ b/third.c',
    '@@ -5 +5 @@',
    '-a\r',
    '+b\r',
  ].join('\n');
  const d = parseUnifiedDiff(text);
  checkEq(d.files.length, 3);
  checkEq([d.files[0].oldPath, d.files[0].newPath], [null, 'new.c']);
  checkEq(d.files[0].hunks[0].lines.map((l) => l.type), ['add', 'add', 'note']);
  checkEq(d.files[0].hunks[0].lines[2].text, 'No newline at end of file');
  checkEq([d.files[1].oldPath, d.files[1].newPath], ['old.c', null]);
  checkEq(displayPath(d.files[1]), 'old.c');
  checkEq(d.files[1].hunks[0], { oldStart: 1, oldLines: 1, newStart: 0, newLines: 0, section: '',
                                 lines: [{ type: 'del', text: 'gone', oldLine: 1, newLine: null }] });
  checkEq(d.files[2].hunks[0].lines.map((l) => [l.type, l.text]), [['del', 'a'], ['add', 'b']]);
  checkEq([d.added, d.removed], [3, 2]);
});

test('diff_empty_and_garbage', () => {
  checkEq(parseUnifiedDiff(''), { files: [], added: 0, removed: 0 });
  checkEq(parseUnifiedDiff(undefined), { files: [], added: 0, removed: 0 });
  checkEq(parseUnifiedDiff('just some text\nwithout a diff').files, []);
  const d = parseUnifiedDiff('@@ -1,2 +1,2 @@\n a\n-b\n+c\n');
  checkEq(d.files.length, 1, 'a hunk without file header');
  checkEq(d.files[0].hunks[0].lines.length, 3);
});

// ---- markdown ---------------------------------------------------------------------

test('markdown_blocks', () => {
  const blocks = parseMarkdown('# Title\n\nFirst paragraph\nsecond line.\n\n## Sub ##\n\n---\n\nLast.');
  checkEq(blocks.map((b) => b.type), ['heading', 'paragraph', 'heading', 'rule', 'paragraph']);
  checkEq([blocks[0].level, inlineText(blocks[0].children)], [1, 'Title']);
  checkEq([blocks[2].level, inlineText(blocks[2].children)], [2, 'Sub']);
  checkEq(inlineText(blocks[1].children), 'First paragraph\nsecond line.');
  checkEq(parseMarkdown(''), []);
  checkEq(parseMarkdown(null), []);
  checkEq(parseMarkdown('\n\n  \n'), []);
  checkEq(parseMarkdown('#include <stdio.h>')[0].type, 'paragraph', 'no space after the number sign');
  checkEq(parseMarkdown('####### seven')[0].type, 'paragraph');
});

test('markdown_inline', () => {
  checkEq(renderMarkdownToHtml('**bold** and *italic* and __bold text__ and _italic_'),
          '<p><strong>bold</strong> and <em>italic</em> and <strong>bold text</strong> and <em>italic</em></p>\n');
  checkEq(renderMarkdownToHtml('***both***'), '<p><strong><em>both</em></strong></p>\n');
  checkEq(renderMarkdownToHtml('*a **b** c*'), '<p><em>a <strong>b</strong> c</em></p>\n');
  checkEq(renderMarkdownToHtml('**a *b* c**'), '<p><strong>a <em>b</em> c</strong></p>\n');
  checkEq(renderMarkdownToHtml('a * b * c'), '<p>a * b * c</p>\n', 'stars surrounded by spaces');
  checkEq(renderMarkdownToHtml('2 * 3 and 4*5'), '<p>2 * 3 and 4*5</p>\n');
  checkEq(renderMarkdownToHtml('\\*not italic\\*'), '<p>*not italic*</p>\n');
  checkEq(renderMarkdownToHtml('open **bold without end'), '<p>open **bold without end</p>\n');
  checkEq(renderMarkdownToHtml('line one  \nline two'), '<p>line one<br>line two</p>\n');
  checkEq(renderMarkdownToHtml('line one\\\nline two'), '<p>line one<br>line two</p>\n');
});

test('markdown_identifiers_keep_underscores', () => {
  checkEq(renderMarkdownToHtml('mi_heap_malloc calls _mi_heap_malloc_zero and _mi_page_malloc'),
          '<p>mi_heap_malloc calls _mi_heap_malloc_zero and _mi_page_malloc</p>\n');
  checkEq(renderMarkdownToHtml('MI_SMALL_SIZE_MAX and __attribute__ and size_t* p'),
          '<p>MI_SMALL_SIZE_MAX and __attribute__ and size_t* p</p>\n');
  // A single word between double underscores is an identifier; two words are bold.
  checkEq(renderMarkdownToHtml('__init__ and __FILE__ but __two words__'),
          '<p>__init__ and __FILE__ but <strong>two words</strong></p>\n');
});

test('markdown_inline_code', () => {
  checkEq(renderMarkdownToHtml('use `mi_malloc(<n>)` here'), '<p>use <code>mi_malloc(&lt;n&gt;)</code> here</p>\n');
  checkEq(renderMarkdownToHtml('``a ` b``'), '<p><code>a ` b</code></p>\n');
  checkEq(renderMarkdownToHtml('`` `x` ``'), '<p><code>`x`</code></p>\n');
  checkEq(renderMarkdownToHtml('`**not bold**`'), '<p><code>**not bold**</code></p>\n');
  checkEq(renderMarkdownToHtml('a ` b'), '<p>a ` b</p>\n', 'a single backtick stays');
  checkEq(renderMarkdownToHtml('*a `*` b*'), '<p><em>a <code>*</code> b</em></p>\n', 'a star inside code does not close');
});

test('markdown_fenced_code', () => {
  const blocks = parseMarkdown('Before\n```c\nint x = 1 < 2;\n\n# not a heading\n* not a list\n```\nAfter');
  checkEq(blocks.map((b) => b.type), ['paragraph', 'code', 'paragraph']);
  checkEq(blocks[1], { type: 'code', lang: 'c', text: 'int x = 1 < 2;\n\n# not a heading\n* not a list', closed: true });
  checkEq(parseMarkdown('~~~python\nx = 1\n~~~')[0], { type: 'code', lang: 'python', text: 'x = 1', closed: true });
  checkEq(parseMarkdown('```\nplain\n```')[0].lang, '');
  checkEq(parseMarkdown('```C++ title="x"\na\n```')[0].lang, 'c++');
  checkEq(parseMarkdown('```<script>\na\n```')[0].lang, '', 'a label that is not an identifier is dropped');
  // A longer fence is closed only by a fence that is at least as long.
  checkEq(parseMarkdown('````\n```\ninner\n```\n````')[0].text, '```\ninner\n```');
  // Streaming: the closing fence has not arrived yet.
  checkEq(parseMarkdown('```c\nint x;\nint y'), [{ type: 'code', lang: 'c', text: 'int x;\nint y', closed: false }]);
  checkEq(parseMarkdown('```'), [{ type: 'code', lang: '', text: '', closed: false }]);
  checkEq(parseMarkdown('\tint tab;\n```c\n\tkeep\ttabs\n```')[1].text, '\tkeep\ttabs');
  checkEq(renderMarkdownToHtml('```c\na < b && c > d\n```'), '<pre data-lang="c"><code>a &lt; b &amp;&amp; c &gt; d</code></pre>\n');
});

test('markdown_lists', () => {
  checkEq(renderMarkdownToHtml('- a\n- b\n- c'), '<ul>\n<li>a</li>\n<li>b</li>\n<li>c</li>\n</ul>\n');
  checkEq(renderMarkdownToHtml('1. one\n2. two'), '<ol>\n<li>one</li>\n<li>two</li>\n</ol>\n');
  checkEq(renderMarkdownToHtml('3) three\n4) four'), '<ol start="3">\n<li>three</li>\n<li>four</li>\n</ol>\n');
  checkEq(renderMarkdownToHtml('- a\n  - nested\n  - more\n- b'),
          '<ul>\n<li>a\n<ul>\n<li>nested</li>\n<li>more</li>\n</ul></li>\n<li>b</li>\n</ul>\n');
  checkEq(renderMarkdownToHtml('1. first\n   - sub\n2. second'),
          '<ol>\n<li>first\n<ul>\n<li>sub</li>\n</ul></li>\n<li>second</li>\n</ol>\n');
  checkEq(renderMarkdownToHtml('- a\n\n- b'), '<ul>\n<li><p>a</p></li>\n<li><p>b</p></li>\n</ul>\n', 'loose list');
  checkEq(renderMarkdownToHtml('- a\ncontinued\n- b'), '<ul>\n<li>a\ncontinued</li>\n<li>b</li>\n</ul>\n', 'lazy line');
  checkEq(renderMarkdownToHtml('- a\n\ntext'), '<ul>\n<li>a</li>\n</ul>\n<p>text</p>\n');
  checkEq(renderMarkdownToHtml('- a\n+ b'), '<ul>\n<li>a</li>\n</ul>\n<ul>\n<li>b</li>\n</ul>\n', 'another bullet, another list');
  checkEq(renderMarkdownToHtml('-not a list\n*neither*'), '<p>-not a list\n<em>neither</em></p>\n');
  const withCode = parseMarkdown('1. step\n   ```c\n   int x;\n   ```\n2. next');
  checkEq(withCode.length, 1);
  checkEq(withCode[0].items[0].map((b) => b.type), ['paragraph', 'code']);
  checkEq(withCode[0].items[0][1].text, 'int x;');
  checkEq(parseMarkdown('* * *')[0].type, 'rule', 'a rule, not a list');
});

test('markdown_quotes', () => {
  checkEq(renderMarkdownToHtml('> quoted **text**\n> second line\n\nafter'),
          '<blockquote>\n<p>quoted <strong>text</strong>\nsecond line</p>\n</blockquote>\n<p>after</p>\n');
  checkEq(renderMarkdownToHtml('> - a\n> - b'), '<blockquote>\n<ul>\n<li>a</li>\n<li>b</li>\n</ul>\n</blockquote>\n');
  checkEq(renderMarkdownToHtml('> > inner'), '<blockquote>\n<blockquote>\n<p>inner</p>\n</blockquote>\n</blockquote>\n');
});

test('markdown_links', () => {
  checkEq(renderMarkdownToHtml('[text](https://example.org/a?b=1&c=2)'),
          '<p><a href="https://example.org/a?b=1&amp;c=2" rel="noopener noreferrer">text</a></p>\n');
  checkEq(renderMarkdownToHtml('[t](http://a.b "the title")'),
          '<p><a href="http://a.b" title="the title" rel="noopener noreferrer">t</a></p>\n');
  checkEq(renderMarkdownToHtml('[rel](src/alloc.c#L120)'),
          '<p><a href="src/alloc.c#L120" rel="noopener noreferrer">rel</a></p>\n');
  checkEq(renderMarkdownToHtml('[p](https://a.b/x_(y))'),
          '<p><a href="https://a.b/x_(y)" rel="noopener noreferrer">p</a></p>\n');
  checkEq(renderMarkdownToHtml('<https://a.b/c>'),
          '<p><a href="https://a.b/c" rel="noopener noreferrer">https://a.b/c</a></p>\n');
  checkEq(renderMarkdownToHtml('see https://a.b/c_d.'),
          '<p>see <a href="https://a.b/c_d" rel="noopener noreferrer">https://a.b/c_d</a>.</p>\n');
  checkEq(renderMarkdownToHtml('(https://a.b/c)'),
          '<p>(<a href="https://a.b/c" rel="noopener noreferrer">https://a.b/c</a>)</p>\n');
  checkEq(renderMarkdownToHtml('[**b**](https://a.b)'),
          '<p><a href="https://a.b" rel="noopener noreferrer"><strong>b</strong></a></p>\n');
  checkEq(renderMarkdownToHtml('[not a link] (x) and [broken](x'), '<p>[not a link] (x) and [broken](x</p>\n');
  checkEq(renderMarkdownToHtml('array[0](1)'), '<p>array<a href="1" rel="noopener noreferrer">0</a></p>\n');
  // Images are never loaded; they are shown as links.
  checkEq(renderMarkdownToHtml('![alt](https://a.b/i.png)'),
          '<p><a href="https://a.b/i.png" rel="noopener noreferrer">alt</a></p>\n');
});

test('markdown_unsafe_links_are_dropped', () => {
  const unsafe = [
    'javascript:alert(1)', 'JaVaScRiPt:alert(1)', ' javascript:alert(1)', 'java\tscript:alert(1)',
    'java\nscript:alert(1)', '\x01javascript:alert(1)', 'data:text/html,<script>alert(1)</script>',
    'vbscript:msgbox(1)', 'file:///etc/passwd', 'blob:https://a.b/x', 'mailto:a@b.c', 'ftp://a.b/c',
    'javascript\u{200b}:alert(1)', 'x-custom:1', 'alloc.c:120',
  ];
  for (const url of unsafe) checkEq(safeUrl(url), null, `safeUrl(${JSON.stringify(url)})`);
  const safe = ['http://a.b', 'https://a.b/c?d#e', 'HTTPS://A.B', '/abs/path', 'rel/path.c', './x', '../x', '#frag', '?q=1',
                '//host/path', 'src/alloc.c:120'];
  for (const url of safe) check(safeUrl(url) !== null, `safeUrl(${JSON.stringify(url)}) is safe`);
  checkEq(safeUrl(''), null);
  checkEq(safeUrl(undefined), null);
  checkEq(safeUrl('https://a.b/' + 'x'.repeat(3000)), null, 'overlong');
  checkEq(safeUrl('  https://a.b/c  '), 'https://a.b/c');

  checkEq(renderMarkdownToHtml('[click](javascript:alert(1))'), '<p>click</p>\n');
  checkEq(renderMarkdownToHtml('[click](JAVASCRIPT:alert(1))'), '<p>click</p>\n');
  checkEq(renderMarkdownToHtml('[click](<javascript:alert(1)>)'), '<p>click</p>\n');
  checkEq(renderMarkdownToHtml('[click](data:text/html;base64,PHNjcmlwdD4=)'), '<p>click</p>\n');
  checkEq(renderMarkdownToHtml('<javascript:alert(1)>'), '<p>&lt;javascript:alert(1)&gt;</p>\n');
  check(!renderMarkdownToHtml('![x](javascript:alert(1))').includes('href'), 'image with script URL');
});

test('markdown_html_is_text', () => {
  const attacks = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<a href="javascript:alert(1)">x</a>',
    '<iframe src="https://evil.example"></iframe>',
    '</code></pre><script>alert(1)</script>',
    '<svg/onload=alert(1)>',
    '<!-- comment --><style>body{display:none}</style>',
    '&lt;script&gt; &amp; &#x3c;script&#x3e;',
    '"><script>alert(1)</script>',
  ];
  for (const attack of attacks) {
    for (const source of [attack, `# ${attack}`, `- ${attack}`, `> ${attack}`, `**${attack}**`, `\`\`\`\n${attack}\n\`\`\``,
                          `\`${attack}\``, `[${attack}](https://a.b)`, `[x](https://a.b "${attack.replace(/"/g, '')}")`]) {
      const html = renderMarkdownToHtml(source);
      // Remove the tags the renderer itself writes; no other angle bracket may remain.
      const rest = html.replace(/<\/?(?:p|h[1-6]|ul|ol|li|blockquote|pre|code|strong|em|br|hr)>/g, '')
        .replace(/<a href="[^"<>]*"(?: title="[^"<>]*")? rel="noopener noreferrer">/g, '').replace(/<\/a>/g, '');
      check(!/[<>]/.test(rest), `markup survives in ${JSON.stringify(source)}: ${JSON.stringify(html)}`);
    }
  }
  checkEq(renderMarkdownToHtml('a & b < c > d "e" \'f\''), '<p>a &amp; b &lt; c &gt; d &quot;e&quot; &#39;f&#39;</p>\n');
  checkEq(escapeHtml('<a href="x">&\'</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;');
  // A language label cannot carry an attribute.
  check(!renderMarkdownToHtml('```c" onmouseover="alert(1)\nx\n```').includes('onmouseover="'));
  // A title cannot break out of its attribute.
  checkEq(renderMarkdownToHtml('[x](https://a.b \'a"b\')'),
          '<p><a href="https://a.b" title="a&quot;b" rel="noopener noreferrer">x</a></p>\n');
});

test('markdown_edit_blocks', () => {
  const source = [
    'The change:', '', 'src/alloc.c', '@@ mi_heap_malloc', '<<<<<<< SEARCH', '  return *p_value * 2;  // *not italic*',
    '=======', '  return *p_value * 4;', '>>>>>>> REPLACE', '', 'Done.',
  ].join('\n');
  const blocks = parseMarkdown(source);
  checkEq(blocks.map((b) => b.type), ['paragraph', 'code', 'paragraph']);
  checkEq(blocks[1].lang, 'edit');
  checkEq(blocks[1].closed, true);
  // The lines that name the file and the unit are part of the block.
  check(blocks[1].text.startsWith('src/alloc.c\n@@ mi_heap_malloc\n<<<<<<< SEARCH\n'), 'path and scope lead the block');
  check(blocks[1].text.endsWith('\n>>>>>>> REPLACE'));
  check(blocks[1].text.includes('*p_value * 2;  // *not italic*'), 'code kept verbatim');
  checkEq(inlineText(blocks[0].children), 'The change:');
  // Prose before the block stays prose, also when it directly precedes the path.
  const prose = parseMarkdown('Here is the change\nsrc/a.c\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE');
  checkEq(prose.map((b) => b.type), ['paragraph', 'code']);
  checkEq(inlineText(prose[0].children), 'Here is the change');
  const bare = parseMarkdown('Look at this:\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE');
  checkEq(inlineText(bare[0].children), 'Look at this:');
  check(bare[1].text.startsWith('<<<<<<< SEARCH'));
  const unit = parseMarkdown('<<<<<<< UNIT mi_malloc\nvoid* mi_malloc(size_t n) { }\n>>>>>>> UNIT');
  checkEq(unit.length, 1);
  checkEq([unit[0].lang, unit[0].closed], ['edit', true]);
  checkEq(parseMarkdown('<<<<<<< SEARCH\nstill streaming')[0].closed, false);
});

test('markdown_crlf_and_nul', () => {
  checkEq(renderMarkdownToHtml('# A\r\n\r\ntext\r\n- x\r\n- y\r\n'), '<h1>A</h1>\n<p>text</p>\n<ul>\n<li>x</li>\n<li>y</li>\n</ul>\n');
  checkEq(inlineText(parseMarkdown('a\x00b')[0].children), 'a\u{fffd}b');
});

test('markdown_adversarial_input_is_fast', () => {
  const started = Date.now();
  const inputs = [
    '*a '.repeat(20000),
    '**a '.repeat(20000),
    '_a '.repeat(20000),
    '['.repeat(30000),
    '[a]('.repeat(10000),
    '`'.repeat(1) + 'a `` b ``` c '.repeat(8000),
    '*'.repeat(50000),
    '> '.repeat(20000) + 'x',
    '- '.repeat(20000) + 'x',
    ('  '.repeat(50) + '- x\n').repeat(500),
    '# ' + ' '.repeat(50000) + 'x',
    'a' + ' '.repeat(50000) + '\nb',
    '*a **b '.repeat(10000) + '** *'.repeat(10000),
    '<'.repeat(50000),
    'https://a.b/' + '('.repeat(5000),
    '\\'.repeat(50001),
    '1. '.repeat(10000),
  ];
  for (const input of inputs) {
    const blocks = parseMarkdown(input);
    check(Array.isArray(blocks), 'parses');
    renderMarkdownToHtml(input);
  }
  const elapsed = Date.now() - started;
  check(elapsed < 5000, `adversarial inputs took ${elapsed} ms`);
});

test('markdown_nesting_is_bounded', () => {
  const deep = '> '.repeat(40) + 'deep';
  let node = parseMarkdown(deep);
  let depth = 0;
  while (node.length > 0 && node[0].type === 'quote') {
    node = node[0].children;
    ++depth;
  }
  check(depth >= 1 && depth <= 9, `quote depth ${depth}`);
  const inline = parseInline('*a '.repeat(30) + 'x' + ' a*'.repeat(30));
  check(JSON.stringify(inline).length > 0);
});

test('markdown_streaming_prefixes', () => {
  // Every prefix of a document parses, and prefixes converge to the full result.
  const source = '## Title\n\nText with **bold**, `code` and a [link](https://a.b).\n\n1. one\n2. two\n\n```c\nint x;\n```\n\n> quote\n';
  for (let n = 0; n <= source.length; ++n) {
    const html = renderMarkdownToHtml(source.slice(0, n));
    check(typeof html === 'string');
  }
  checkEq(renderMarkdownToHtml(source),
          '<h2>Title</h2>\n<p>Text with <strong>bold</strong>, <code>code</code> and a ' +
          '<a href="https://a.b" rel="noopener noreferrer">link</a>.</p>\n<ol>\n<li>one</li>\n<li>two</li>\n</ol>\n' +
          '<pre data-lang="c"><code>int x;</code></pre>\n<blockquote>\n<p>quote</p>\n</blockquote>\n');
});

// ---- highlighting -----------------------------------------------------------------

test('highlight_reproduces_the_input', () => {
  const samples = [
    ['c', '#include <stdio.h>\n#define X(a) ((a) + 1)\n/* block\n comment */\nstatic inline void* f(size_t n) {\n' +
          '  // line comment\n  const char* s = "str \\" with // no comment";\n  char c = \'\\\'\';\n' +
          '  return n > 0x1F ? g(n, 1.5e-3f) : NULL;\n}\n'],
    ['cpp', 'template <class T> auto f(T&& x) -> decltype(auto) { return std::forward<T>(x); }'],
    ['python', 'def f(x):\n    """doc\n    string"""\n    return x + 1  # comment\ns = \'a\\\'b\'\n'],
    ['js', 'const s = `template ${x} string`; // c\nlet r = /re/g;'],
    ['sh', 'if [ -f x ]; then echo "yes"; fi # comment'],
    ['diff', '--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n context\n'],
    ['edit', 'path.c\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n'],
    ['', 'no language at all <b>'],
    ['unknown-language', 'x = 1'],
    ['c', '"unterminated string\nnext line /* unterminated comment'],
    ['c', ''],
  ];
  for (const [lang, code] of samples) {
    const tokens = tokenize(code, lang);
    checkEq(tokens.map((t) => t.text).join(''), code, `language ${JSON.stringify(lang)}`);
    check(tokens.every((t) => t.text !== '' && typeof t.kind === 'string'), 'no empty token');
  }
});

test('highlight_kinds', () => {
  const kinds = (code, lang) => tokenize(code, lang).filter((t) => t.kind !== 'plain').map((t) => [t.kind, t.text]);
  checkEq(kinds('static size_t n = 42; // note', 'c'),
          [['keyword', 'static'], ['type', 'size_t'], ['number', '42'], ['comment', '// note']]);
  checkEq(kinds('#include <a.h>', 'c'), [['preproc', '#include'], ['string', '<a.h>']]);
  checkEq(kinds('x = mi_malloc(n);', 'c'), [['call', 'mi_malloc']]);
  checkEq(kinds('s = "a // b";', 'c'), [['string', '"a // b"']]);
  checkEq(kinds('/* a */ b', 'c'), [['comment', '/* a */']]);
  checkEq(kinds('def f(): return None  # c', 'python'),
          [['keyword', 'def'], ['call', 'f'], ['keyword', 'return'], ['keyword', 'None'], ['comment', '# c']]);
  checkEq(kinds('+a\n-b\n c\n@@ -1 +1 @@', 'diff'), [['add', '+a\n'], ['del', '-b\n'], ['meta', '@@ -1 +1 @@']]);
  checkEq(kinds('<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE', 'edit'),
          [['meta', '<<<<<<< SEARCH\n'], ['del', 'old\n'], ['meta', '=======\n'], ['add', 'new\n'],
           ['meta', '>>>>>>> REPLACE']]);
  checkEq(tokenize('a < b', 'text'), [{ kind: 'plain', text: 'a < b' }]);
  checkEq(languageOfPath('src/alloc.c'), 'c');
  checkEq(languageOfPath('include/mimalloc.h'), 'h');
  checkEq(languageOfPath('tools/run.PY'), 'py');
  checkEq(languageOfPath('CMakeLists.txt'), 'cmake');
  checkEq(languageOfPath('README'), '');
  checkEq(languageOfPath(undefined), '');
});

// ---- map geometry -------------------------------------------------------------------

test('scatter_view_transform', () => {
  const bounds = scatter.boundsOf([{ x: -1, y: -0.5 }, { x: 1, y: 0.5 }, { x: NaN, y: 0 }]);
  checkEq(bounds, { minX: -1, maxX: 1, minY: -0.5, maxY: 0.5 });
  checkEq(scatter.boundsOf([]), { minX: -1, maxX: 1, minY: -1, maxY: 1 });
  const view = scatter.fitView(bounds, 400, 300, 20);
  checkNear(view.scale, 180, 1e-9, 'limited by the width');
  const a = scatter.toScreen(view, -1, 0);
  const b = scatter.toScreen(view, 1, 0);
  checkNear(a.x, 20, 1e-9);
  checkNear(b.x, 380, 1e-9);
  checkNear(a.y, 150, 1e-9);
  check(scatter.toScreen(view, 0, 0.5).y < scatter.toScreen(view, 0, -0.5).y, 'y points up');
  const w = scatter.toWorld(view, 123, 45);
  const s = scatter.toScreen(view, w.x, w.y);
  checkNear(s.x, 123, 1e-9);
  checkNear(s.y, 45, 1e-9);
  // A single point still gives a finite view.
  const single = scatter.fitView(scatter.boundsOf([{ x: 0.3, y: 0.3 }]), 200, 200);
  check(Number.isFinite(single.scale) && single.scale > 0);
});

test('scatter_zoom_and_pan', () => {
  const view = scatter.fitView({ minX: -1, maxX: 1, minY: -1, maxY: 1 }, 400, 400, 0);
  const before = scatter.toWorld(view, 100, 300);
  const zoomed = scatter.zoomAt(view, 2, 100, 300, view.scale);
  checkNear(zoomed.scale, view.scale * 2, 1e-9);
  const after = scatter.toWorld(zoomed, 100, 300);
  checkNear(after.x, before.x, 1e-9, 'the point under the pointer stays');
  checkNear(after.y, before.y, 1e-9);
  let v = view;
  for (let i = 0; i < 40; ++i) v = scatter.zoomAt(v, 2, 0, 0, view.scale);
  checkNear(v.scale, view.scale * scatter.MAX_ZOOM, 1e-6, 'zoom in is limited');
  for (let i = 0; i < 80; ++i) v = scatter.zoomAt(v, 0.5, 0, 0, view.scale);
  checkNear(v.scale, view.scale * scatter.MIN_ZOOM, 1e-6, 'zoom out is limited');
  const panned = scatter.panBy(view, 10, -5);
  checkEq([panned.tx - view.tx, panned.ty - view.ty, panned.scale], [10, -5, view.scale]);
});

test('scatter_nearest_point', () => {
  const points = [{ x: 0, y: 0 }, { x: 0.5, y: 0.5 }, { x: -0.5, y: 0.25 }];
  const view = scatter.fitView({ minX: -1, maxX: 1, minY: -1, maxY: 1 }, 200, 200, 0);
  checkEq(scatter.nearestPoint(points, view, 100, 100, 10), 0);
  checkEq(scatter.nearestPoint(points, view, 148, 52, 10), 1);
  checkEq(scatter.nearestPoint(points, view, 10, 190, 10), -1, 'nothing within the radius');
  checkEq(scatter.nearestPoint([], view, 0, 0, 10), -1);
  checkEq(scatter.nearestPoint([{ x: 0, y: 0 }, { x: 0, y: 0 }], view, 100, 100, 10), 0, 'ties go to the lower index');
});

test('scatter_colour_slots', () => {
  const groups = [];
  for (let i = 0; i < 37; ++i) {
    groups.push({ id: i, size: 100 - i, x: -0.85 + 1.7 * ((i % 7) / 6), y: -0.8 + 1.6 * (Math.floor(i / 7) / 5) });
  }
  const slots = scatter.assignColorSlots(groups, 8);
  checkEq(slots.size, 37);
  for (let i = 0; i < 8; ++i) checkEq(slots.get(i), i, 'the largest groups take the slots in order');
  check([...slots.values()].every((s) => Number.isInteger(s) && s >= 0 && s < 8));
  // Neighbours on the grid never share a colour.
  for (const g of groups) {
    const nearest = groups.filter((o) => o !== g).sort((a, b) =>
      Math.hypot(a.x - g.x, a.y - g.y) - Math.hypot(b.x - g.x, b.y - g.y))[0];
    check(slots.get(nearest.id) !== slots.get(g.id), `groups ${g.id} and ${nearest.id} share a colour`);
  }
  // The assignment does not depend on the order in which the groups are listed.
  const reversed = scatter.assignColorSlots(groups.slice().reverse(), 8);
  checkEq([...slots.entries()].sort((a, b) => a[0] - b[0]), [...reversed.entries()].sort((a, b) => a[0] - b[0]));
  checkEq(scatter.assignColorSlots([], 8).size, 0);
  checkEq(scatter.assignColorSlots([{ id: 5, size: 1 }], 8).get(5), 5, 'a group without coordinates');
  checkEq(scatter.niceStep(2, 4), 0.5);
  checkEq(scatter.niceStep(10, 5), 2);
  checkEq(scatter.niceStep(0, 5), 1);
});

// ---- settings model -----------------------------------------------------------------

test('settings_paths', () => {
  const doc = { retrieve: { tau: 0.3, lift: { callee: 4 } }, 'exec.threads': 8, models: { chat: null } };
  checkEq(model.getPath(doc, 'retrieve.tau', 0.25), 0.3);
  checkEq(model.getPath(doc, 'retrieve.lift.callee', 1), 4);
  checkEq(model.getPath(doc, 'retrieve.missing', 7), 7);
  checkEq(model.getPath(doc, 'exec.threads', 0), 8, 'dotted member name');
  checkEq(model.getPath(doc, 'models.chat', 'x'), 'x', 'null counts as absent');
  checkEq(model.getPath(null, 'a.b', 1), 1);
  checkEq(model.getPath(doc, 'retrieve.tau.deeper', 5), 5);
  checkEq(model.setPath({}, 'a.b.c', 1), { a: { b: { c: 1 } } });
  checkEq(model.setPath({ a: { x: 1 } }, 'a.b', 2), { a: { x: 1, b: 2 } });
  const polluted = model.setPath({}, '__proto__.polluted', true);
  checkEq(polluted, {});
  checkEq({}.polluted, undefined, 'the prototype is untouched');
});

test('settings_form_and_patch', () => {
  const form = model.readForm({ corpus: { roots: ['C:/a', 5, 'C:/b'] }, retrieve: { tau: '0.5' }, exec: { threads: 4 } });
  checkEq(form['corpus.roots'], ['C:/a', 'C:/b'], 'values of the wrong type are dropped');
  checkEq(form['retrieve.tau'], 0.25, 'a string is not a number: the default applies');
  checkEq(form['exec.threads'], 4);
  checkEq(form['context.token_budget'], 3000);
  checkEq(model.readForm({})['models.chat'], '');
  checkEq(Object.keys(model.readForm(undefined)).sort(), Object.keys(model.DEFAULTS).sort());

  const before = model.readForm({});
  const after = { ...before, 'retrieve.tau': 0.4, 'corpus.roots': ['C:/x'], 'exec.block_bytes': 8192 };
  checkEq(model.changedKeys(before, after).sort(), ['corpus.roots', 'exec.block_bytes', 'retrieve.tau']);
  checkEq(model.buildPatch(before, after),
          { corpus: { roots: ['C:/x'] }, retrieve: { tau: 0.4 }, exec: { block_bytes: 8192 } });
  checkEq(model.buildPatch(before, { ...before }), {});
  checkEq(model.changedKeys({ 'corpus.roots': ['a', 'b'] }, { 'corpus.roots': ['a', 'b'] }), []);
  checkEq(model.changedKeys({ 'corpus.roots': ['a', 'b'] }, { 'corpus.roots': ['b', 'a'] }), ['corpus.roots']);
  check(model.affectsIndex(['corpus.roots']));
  check(model.affectsIndex(['models.embedding']));
  check(model.affectsIndex(['retrieve.tau', 'kde.dim']));
  check(!model.affectsIndex(['retrieve.tau', 'models.chat', 'exec.threads', 'context.token_budget']));
});

test('settings_roots_tau_validation', () => {
  checkEq(model.normaliseRoots(['  C:\\Users\\me\\repo\\ ', '', 'C:/Users/me/repo', '/srv/code/', '/', 'C:\\']),
          ['C:/Users/me/repo', '/srv/code', '/', 'C:/']);
  checkNear(model.costRatio(0.25), 3, 1e-12, 'tau = 0.25 means c_fn = 3 c_fp');
  checkNear(model.costRatio(0.5), 1, 1e-12);
  checkNear(model.costRatio(0.05), 19, 1e-9);
  check(Number.isNaN(model.costRatio(0)));
  check(Number.isNaN(model.costRatio(1)));
  const good = model.readForm({});
  checkEq(model.validate(good), []);
  checkEq(model.validate({ ...good, 'retrieve.tau': 0.95 }).map((e) => e.key), ['retrieve.tau']);
  checkEq(model.validate({ ...good, 'retrieve.tau': 0.01 }).map((e) => e.key), ['retrieve.tau']);
  checkEq(model.validate({ ...good, 'context.token_budget': 10 }).map((e) => e.key), ['context.token_budget']);
  checkEq(model.validate({ ...good, 'exec.threads': -1 }).map((e) => e.key), ['exec.threads']);
  checkEq(model.validate({ ...good, 'exec.threads': 2.5 }).map((e) => e.key), ['exec.threads']);
  checkEq(model.validate({ ...good, 'exec.block_bytes': 100 }).map((e) => e.key), ['exec.block_bytes']);
  checkEq(model.validate({ ...good, 'exec.block_bytes': 16384 }), []);
  checkEq(model.validate({ ...good, 'retrieve.tau': model.TAU_MIN }), []);
  checkEq(model.validate({ ...good, 'retrieve.tau': model.TAU_MAX }), []);
});

// ---- API client against the mock server -------------------------------------------------

let mock = null;
let api = null;
let base = '';

async function startMock() {
  if (mock) return;
  mock = createMockServer({ speed: 40, quiet: true });
  const address = await mock.listen(0, '127.0.0.1');
  base = address.url;
  api = createApi(base);
}

function hasKeys(object, keys, what) {
  for (const key of keys) {
    if (!(object !== null && typeof object === 'object' && key in object)) report(`${what}: field "${key}" is missing`);
  }
}

test('api_static_files', async () => {
  await startMock();
  const page = await fetch(base + '/');
  checkEq(page.status, 200);
  check((page.headers.get('content-type') || '').startsWith('text/html'));
  const html = await page.text();
  check(html.includes('<title>'), 'the page has a title');
  check(!/https?:\/\/(?!www\.w3\.org)/.test(html.replace(/<!--[\s\S]*?-->/g, '')), 'the page refers to no other host');
  for (const [file, type] of [['/style.css', 'text/css'], ['/app.js', 'text/javascript'],
                              ['/js/lib/markdown.js', 'text/javascript'], ['/js/api.js', 'text/javascript']]) {
    const response = await fetch(base + file);
    checkEq(response.status, 200, file);
    check((response.headers.get('content-type') || '').startsWith(type), `${file} is served as ${type}`);
    await response.arrayBuffer();
  }
  checkEq((await fetch(base + '/missing.js')).status, 404);
  checkEq((await fetch(base + '/..%2f..%2fCMakeLists.txt')).status, 403, 'path traversal is refused');
  checkEq((await fetch(base + '/%2e%2e/%2e%2e/docs/DESIGN.md')).status !== 200, true);
  checkEq((await fetch(base + '/api/nothing')).status, 404);
  check((await fetch(base + '/api/status')).headers.get('access-control-allow-origin') === null, 'no CORS header');
});

test('api_status_and_settings', async () => {
  await startMock();
  const status = await api.status();
  hasKeys(status, ['version', 'index', 'models', 'cache', 'tune', 'roots'], 'status');
  hasKeys(status.index, ['ready', 'building', 'files', 'units', 'functions', 'call_edges', 'groups', 'dim', 'reduced_dim',
                         'kappa', 'codes', 'blocks', 'block_bytes', 'embedder', 'built_at', 'calibrated'], 'status.index');
  hasKeys(status.models, ['llama', 'chat', 'chat_loaded', 'embedding_loaded', 'backend'], 'status.models');
  hasKeys(status.cache, ['detected', 'logical_cpus', 'l1d_min_bytes', 'l1d_per_thread_min_bytes', 'line_bytes',
                         'block_bytes', 'threads', 'simd', 'classes'], 'status.cache');
  hasKeys(status.cache.classes[0], ['level', 'type', 'size_bytes', 'sharing_threads', 'instances'], 'cache class');
  checkEq([status.index.units, status.index.files, status.index.groups], [1510, 59, 37]);
  check(status.cache.block_bytes <= status.cache.l1d_per_thread_min_bytes, 'the block fits the L1 share of a thread');

  const settings = await api.settings();
  checkEq(settings.retrieve.tau, 0.25);
  const patched = await api.saveSettings({ retrieve: { tau: 0.3 }, context: { token_budget: 2500 } });
  checkEq([patched.retrieve.tau, patched.context.token_budget, patched.retrieve.max_units], [0.3, 2500, 64]);
  checkEq((await api.status()).index.ready, true, 'tau does not make the index stale');
  await checkThrows(() => api.saveSettings({ retrieve: { tau: 7 } }), (e) => e instanceof ApiError && e.status === 400);
  await api.saveSettings({ retrieve: { tau: 0.25 }, context: { token_budget: 3000 } });
});

test('api_history_and_feedback', async () => {
  await startMock();
  const history = await api.history();
  check(Array.isArray(history) && history.length >= 2);
  hasKeys(history[0], ['id', 'time', 'question', 'answer', 'units', 'rating'], 'history entry');
  check(history[0].time <= history[history.length - 1].time, 'oldest first');
  checkEq(await api.feedback(history[1].id, -1), { ok: true });
  checkEq((await api.history())[1].rating, -1);
  await api.feedback(history[1].id, 1);
  checkEq((await api.history())[1].rating, 1);
  await checkThrows(() => api.feedback('q-unknown', 1), (e) => e.status === 404 && /q-unknown/.test(e.message));
  await checkThrows(() => api.feedback(history[1].id, 5), (e) => e.status === 400);
});

test('api_groups_and_units', async () => {
  await startMock();
  const groups = await api.groups();
  hasKeys(groups, ['kappa', 'reduced_dim', 'groups', 'units'], 'groups');
  checkEq([groups.groups.length, groups.units.length], [37, 1510]);
  hasKeys(groups.groups[0], ['id', 'size', 'log_density', 'x', 'y', 'label'], 'group');
  hasKeys(groups.units[0], ['id', 'group', 'x', 'y', 'name', 'path', 'kind', 'log_density'], 'group unit');
  check(groups.units.every((u) => u.x >= -1 && u.x <= 1 && u.y >= -1 && u.y <= 1), 'coordinates in [-1, 1]');
  checkEq(groups.groups.reduce((sum, g) => sum + g.size, 0), 1510, 'every unit belongs to a group');

  const named = groups.units.find((u) => u.name === 'mi_heap_malloc');
  const unit = await api.unit(named.id);
  hasKeys(unit, ['id', 'path', 'name', 'kind', 'start_line', 'end_line', 'signature', 'doc', 'text', 'callees', 'callers',
                 'group', 'tokens'], 'unit');
  checkEq([unit.name, unit.path, unit.kind], ['mi_heap_malloc', 'src/alloc.c', 'function']);
  checkEq(unit.text.split('\n').length - 1, unit.end_line - unit.start_line + 1, 'line range matches the text');
  await checkThrows(() => api.unit(999999), (e) => e.status === 404);
  await checkThrows(() => api.unit('1;drop'), (e) => e.status === 400);
});

test('api_query_stream_ask', async () => {
  await startMock();
  const order = [];
  let retrieval = null;
  let text = '';
  const done = await api.query({ question: 'How does mi_malloc allocate small blocks?', edit: false }, {
    onRetrieval: (r) => {
      order.push('retrieval');
      retrieval = r;
    },
    onToken: (t) => {
      if (order[order.length - 1] !== 'token') order.push('token');
      text += t;
    },
    onEdit: () => order.push('edit'),
    onDone: () => order.push('done'),
  });
  checkEq(order, ['retrieval', 'token', 'done']);
  hasKeys(retrieval, ['id', 'tau', 'z_threshold', 'low_confidence', 'truncated', 'units', 'context', 'scan', 'trace'],
          'retrieval');
  hasKeys(retrieval.units[0], ['id', 'path', 'name', 'kind', 'start_line', 'end_line', 'group', 'tokens', 'score', 'z',
                               'p_score', 'p', 'hops', 'parent', 'via', 'in_context', 'form', 'support', 'redundancy',
                               'quality', 'preview'], 'retrieved unit');
  hasKeys(retrieval.context, ['tokens_before', 'tokens_after', 'quality_before', 'quality_after', 'rot_before',
                              'rot_after', 'compression', 'passes', 'dropped_low_quality', 'dropped_redundant',
                              'dropped_budget', 'skeletonised', 'support_added', 'history_kept', 'history_total',
                              'groups'], 'context');
  hasKeys(retrieval.scan, ['blocks', 'block_bytes', 'threads', 'wall_us', 'work_us', 'gate_wait_us',
                           'admission_violations', 'commit_violations', 'rescored', 'codes', 'simd'], 'scan');
  hasKeys(retrieval.trace, ['seeds', 'evaluated', 'expansions'], 'trace');
  hasKeys(done, ['id', 'answer', 'prompt_tokens', 'completion_tokens', 'prompt_ms', 'completion_ms', 'total_ms',
                 'stop_reason', 'model', 'extractive'], 'done');
  checkEq(text, done.answer, 'the tokens concatenate to the answer');
  check(done.answer.includes('\u{2014}') && done.answer.includes('\u{3c4}'), 'multi-byte characters arrive intact');
  checkEq(done.id, retrieval.id);

  // Consistency of what the mock computes.
  check(retrieval.units.length > 1);
  check(retrieval.units.some((u) => u.name === 'mi_malloc'), 'the named function is retrieved');
  for (const u of retrieval.units) {
    check(u.p >= u.p_score - 1e-9, `${u.name}: p >= p_score`);
    check(u.support || u.p >= retrieval.tau - 1e-9, `${u.name}: accepted units reach tau`);
    check(['seed', 'callee', 'caller', 'sibling'].includes(u.via), 'via');
    check(['full', 'skeleton', 'none'].includes(u.form), 'form');
    checkEq(u.in_context, u.form !== 'none', 'in_context agrees with form');
    check(u.via === 'seed' ? u.parent === null && u.hops === 0 : Number.isInteger(u.parent) && u.hops >= 1, 'parent and hops');
  }
  const c = retrieval.context;
  checkNear(c.rot_before, 1 - c.quality_before, 1e-3);
  checkNear(c.rot_after, 1 - c.quality_after, 1e-3);
  check(c.rot_after <= c.rot_before, 'pruning does not increase the rot index');
  check(c.tokens_after <= c.tokens_before);
  checkNear(c.compression, c.tokens_after / c.tokens_before, 1e-3);
  checkEq(c.groups.reduce((sum, g) => sum + g.tokens, 0), c.tokens_after, 'file tokens add up');
  checkEq([retrieval.scan.admission_violations, retrieval.scan.commit_violations], [0, 0]);

  const history = await api.history();
  checkEq(history[history.length - 1].id, done.id, 'the exchange is appended to the history');
});

test('api_query_stream_edit', async () => {
  await startMock();
  const order = [];
  let tx = null;
  const done = await api.query({ question: 'Refuse oversized requests in mi_heap_malloc', edit: true, tau: 0.3 }, {
    onRetrieval: (r) => {
      order.push('retrieval');
      checkEq(r.tau, 0.3, 'tau of the request is used');
    },
    onToken: () => {
      if (order[order.length - 1] !== 'token') order.push('token');
    },
    onEdit: (t) => {
      order.push('edit');
      tx = t;
    },
    onDone: () => order.push('done'),
  });
  checkEq(order, ['retrieval', 'token', 'edit', 'done']);
  check(/<{7} SEARCH[\s\S]*={7}[\s\S]*>{7} REPLACE/.test(done.answer), 'the answer holds an edit block');
  hasKeys(tx, ['id', 'status', 'committed', 'files', 'warnings'], 'transaction');
  hasKeys(tx.files[0], ['path', 'display_path', 'status', 'creates_file', 'diff', 'matches', 'syntax_errors_before',
                        'syntax_errors_after', 'message'], 'file proposal');
  hasKeys(tx.files[0].matches[0], ['kind', 'first_line', 'last_line', 'distance', 'candidates'], 'match');
  checkEq([tx.status, tx.committed], ['ok', false]);
  check(!('conflict' in tx), 'internal fields are not sent');

  const diff = parseUnifiedDiff(tx.files[0].diff);
  checkEq(diff.files.length, 1);
  checkEq(diff.files[0].newPath, tx.files[0].display_path);
  check(diff.added > 0);
  for (const hunk of diff.files[0].hunks) {
    checkEq(hunk.lines.filter((l) => l.type !== 'add' && l.type !== 'note').length, hunk.oldLines, 'old line count');
    checkEq(hunk.lines.filter((l) => l.type !== 'del' && l.type !== 'note').length, hunk.newLines, 'new line count');
  }

  const applied = await api.editApply(tx.id);
  checkEq([applied.id, applied.committed, applied.status], [tx.id, true, 'ok']);
  await checkThrows(() => api.editApply(tx.id), (e) => e.status === 409);
  const reverted = await api.editRevert(tx.id);
  checkEq(reverted.committed, false);
  await checkThrows(() => api.editRevert(tx.id), (e) => e.status === 409);
  await checkThrows(() => api.editApply('tx-none'), (e) => e.status === 404);
});

test('api_edit_failure_cases', async () => {
  await startMock();
  const propose = async (marker) => {
    let tx = null;
    await api.query({ question: `Refuse oversized requests in mi_heap_malloc ${marker}`, edit: true },
                    { onEdit: (t) => (tx = t) });
    return tx;
  };
  const ambiguous = await propose('#ambiguous');
  checkEq([ambiguous.status, ambiguous.files[0].status, ambiguous.files[0].matches[0].candidates], ['ambiguous', 'ambiguous', 3]);
  await checkThrows(() => api.editApply(ambiguous.id), (e) => e.status === 409);
  const syntax = await propose('#syntax');
  checkEq(syntax.status, 'syntax_regression');
  check(syntax.files[0].syntax_errors_after > syntax.files[0].syntax_errors_before);
  const conflict = await propose('#conflict');
  checkEq(conflict.status, 'ok');
  const result = await api.editApply(conflict.id);
  checkEq([result.status, result.committed], ['conflict', false]);

  const preview = await api.editPreview({
    text: 'src/alloc.c\n@@ mi_heap_malloc\n<<<<<<< SEARCH\n  return _mi_heap_malloc_zero(heap, size, false);\n=======\n' +
          '  return _mi_heap_malloc_zero(heap, size, true);\n>>>>>>> REPLACE\n',
  });
  checkEq([preview.status, preview.files.length, preview.files[0].status], ['ok', 1, 'ok']);
  const d = parseUnifiedDiff(preview.files[0].diff);
  checkEq([d.added, d.removed], [1, 1]);
  checkEq((await api.editPreview({ text: 'no block here' })).status, 'not_found');
  await checkThrows(() => api.editPreview({}), (e) => e.status === 400);
});

test('api_query_errors', async () => {
  await startMock();
  await checkThrows(() => api.query({ question: '   ' }), (e) => e instanceof ApiError && e.status === 400);
  let tokens = 0;
  await checkThrows(() => api.query({ question: 'How does mi_free work? #error' }, { onToken: () => ++tokens }),
                    (e) => e instanceof ApiError && /context window/.test(e.message), 'error event');
  check(tokens > 0, 'tokens before the error are delivered');
  await checkThrows(() => api.query({ question: 'How does mi_free work? #drop' }),
                    (e) => e instanceof ApiError && /before it was complete/.test(e.message), 'stream without done');

  const low = await new Promise((resolve, reject) => {
    api.query({ question: 'Where is the weather forecast computed?' }, { onRetrieval: resolve }).catch(reject);
  });
  checkEq([low.low_confidence, low.units.length], [true, 1]);

  let scan = null;
  await api.query({ question: 'mi_malloc #violation' }, { onRetrieval: (r) => (scan = r.scan) });
  check(scan.admission_violations > 0 && scan.commit_violations > 0, 'the violation marker works');
});

test('api_query_abort', async () => {
  await startMock();
  const controller = new AbortController();
  let tokens = 0;
  const started = Date.now();
  await checkThrows(
    () => api.query({ question: 'How does the generic allocation path find a page?' }, {
      onToken: () => {
        if (++tokens === 5) controller.abort();
      },
    }, controller.signal),
    (e) => isAbort(e), 'abort');
  check(tokens >= 5 && tokens < 50, `stopped after ${tokens} tokens`);
  check(Date.now() - started < 5000);
});

test('api_unreachable_server', async () => {
  const dead = createApi('http://127.0.0.1:9');
  await checkThrows(() => dead.status(), (e) => e instanceof ApiError && e.status === 0 && /cannot be reached/.test(e.message));
  await checkThrows(() => dead.query({ question: 'x' }), (e) => e instanceof ApiError && e.status === 0);
});

test('api_index_rebuild', async () => {
  await startMock();
  const idle = await api.indexProgress();
  hasKeys(idle, ['running', 'stage', 'done', 'total', 'message', 'report'], 'progress');
  checkEq(idle.running, false);
  hasKeys(idle.report, ['files', 'units', 'functions', 'call_edges', 'embedded', 'reused', 'groups', 'kappa', 'reduced_dim',
                        'explained_variance', 'anisotropy_before', 'anisotropy_after', 'blocks', 'block_bytes', 'seconds',
                        'error'], 'report');

  // A change of the corpus makes the index stale; queries are refused until it is rebuilt.
  const roots = (await api.settings()).corpus.roots;
  await api.saveSettings({ corpus: { roots: roots.concat(['C:/another/root']) } });
  checkEq((await api.status()).index.ready, false);
  await checkThrows(() => api.query({ question: 'mi_malloc' }), (e) => e.status === 409 && /stale/.test(e.message));

  await api.startIndex();
  await checkThrows(() => api.startIndex(), (e) => e.status === 409);
  const stages = [];
  let last = null;
  for (let i = 0; i < 400; ++i) {
    last = await api.indexProgress();
    if (stages[stages.length - 1] !== last.stage) stages.push(last.stage);
    if (!last.running) break;
    check(last.done >= 0 && last.done <= last.total, 'done within total');
    checkEq((await api.status()).index.building, true);
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  checkEq(last.running, false, 'the build finishes');
  checkEq(last.stage, 'done');
  check(last.report !== null && last.report.error === '');
  const known = ['discover', 'parse', 'graph', 'embed', 'density', 'codes', 'calibrate', 'save', 'done'];
  check(stages.every((s) => known.includes(s)), 'known stages');
  check(stages.every((s, i) => i === 0 || known.indexOf(s) > known.indexOf(stages[i - 1])), `stages in order: ${stages}`);
  check(stages.length >= 3, `several stages were observed: ${stages}`);
  const status = await api.status();
  checkEq([status.index.ready, status.index.building], [true, false]);
  await api.saveSettings({ corpus: { roots } });
  await api.startIndex();
  for (let i = 0; i < 400 && (await api.indexProgress()).running; ++i) await new Promise((r) => setTimeout(r, 5));
  checkEq(await api.calibrate(), { ok: true, started: true });
});

test('api_tuning_pipeline', async () => {
  await startMock();
  let tune = await api.tune();
  hasKeys(tune, ['available', 'enabled', 'stage', 'progress', 'message', 'idle', 'pending_examples', 'serving_model',
                 'base_model', 'runs'], 'tune');
  hasKeys(tune.idle, ['idle', 'seconds_since_activity', 'cpu_load', 'on_ac_power', 'reason'], 'tune.idle');
  hasKeys(tune.runs[0], ['id', 'started', 'finished', 'stage', 'promoted', 'examples', 'epochs', 'train_loss',
                         'validation_loss', 'gate', 'model', 'seconds', 'error'], 'run');
  hasKeys(tune.runs[0].gate, ['pass', 'domain_nll_base', 'domain_nll_candidate', 'p_value', 'general_nll_base',
                              'general_nll_candidate', 'quantization_kl', 'reason'], 'gate');
  checkEq([tune.available, tune.enabled, tune.stage], [true, false, 'disabled']);
  await checkThrows(() => api.runTune(), (e) => e.status === 409);
  const runsBefore = tune.runs.length;

  tune = await api.setTune(true);
  checkEq([tune.enabled, tune.stage], [true, 'waiting']);
  check(tune.idle.reason.length > 0, 'the idle state explains why nothing runs');
  checkEq((await api.status()).tune.enabled, true, 'the status carries the tuning state');

  tune = await api.runTune();
  const stages = [];
  for (let i = 0; i < 600; ++i) {
    tune = await api.tune();
    if (stages[stages.length - 1] !== tune.stage) stages.push(tune.stage);
    check(tune.progress >= 0 && tune.progress <= 1, 'progress in [0, 1]');
    if (tune.runs.length > runsBefore) break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  const order = ['collecting', 'training', 'calibrating', 'quantizing', 'evaluating', 'promoting', 'waiting'];
  check(stages.every((s) => order.includes(s)), `known stages: ${stages}`);
  check(stages.every((s, i) => i === 0 || order.indexOf(s) > order.indexOf(stages[i - 1])), `stages in order: ${stages}`);
  check(stages.includes('training'), `training was observed: ${stages}`);
  checkEq(tune.runs.length, runsBefore + 1, 'the run is recorded');
  const run = tune.runs[tune.runs.length - 1];
  checkEq([run.promoted, run.gate.pass, run.train_loss.length, run.validation_loss.length], [true, true, 2, 2]);
  check(tune.serving_model.includes(run.id), 'the promoted model is served');
  // The time stamps have a resolution of one second.
  checkNear(run.seconds, fmt.secondsBetween(run.started, run.finished), 1.5, 'the duration agrees with the time stamps');

  tune = await api.setTune(false);
  checkEq([tune.enabled, tune.stage], [false, 'disabled']);
  await checkThrows(() => fetch(base + '/api/tune', { method: 'POST', body: '{"enabled":"yes"}' })
    .then((r) => (r.ok ? null : Promise.reject(Object.assign(new Error('x'), { status: r.status })))),
                    (e) => e.status === 400);
});

test('api_tuning_is_interrupted_by_activity', async () => {
  await startMock();
  await api.setTune(true);
  await api.runTune();
  let tune = await api.tune();
  check(tune.stage !== 'waiting' && tune.stage !== 'disabled', `the pipeline runs: ${tune.stage}`);
  await api.query({ question: 'mi_malloc' });
  tune = await api.tune();
  checkEq(tune.stage, 'waiting', 'a query pauses the pipeline');
  check(/paused/.test(tune.message), tune.message);
  check(tune.idle.seconds_since_activity < 60);
  await api.setTune(false);
});

test('api_no_llama_build', async () => {
  const other = createMockServer({ speed: 40, quiet: true, llama: false });
  const address = await other.listen(0, '127.0.0.1');
  try {
    const client = createApi(address.url);
    const status = await client.status();
    checkEq([status.models.llama, status.models.chat_loaded, status.tune.available], [false, false, false]);
    await checkThrows(() => client.setTune(true), (e) => e.status === 409 && /llama/.test(e.message));
    const done = await client.query({ question: 'How does mi_free work?' });
    checkEq(done.extractive, true);
    check(done.answer.includes('mi_free'));
  } finally {
    await other.close();
  }
});

test('api_latency_option', async () => {
  const slow = createMockServer({ speed: 40, quiet: true, latency: 80 });
  const address = await slow.listen(0, '127.0.0.1');
  try {
    const client = createApi(address.url);
    const began = Date.now();
    const status = await client.status();
    const elapsed = Date.now() - began;
    checkEq(status.version, '2.0.0');
    // Timers may fire a little early; the delay is there when most of it has passed.
    check(elapsed >= 60, `an API response is delayed: ${elapsed} ms`);
    const page = await fetch(address.url + '/index.html');
    checkEq(page.status, 200, 'static files are served as usual');
    await page.text();
  } finally {
    await slow.close();
  }
});

test('api_history_clear', async () => {
  await startMock();
  check((await api.history()).length > 0);
  checkEq(await api.clearHistory(), { ok: true });
  checkEq(await api.history(), []);
});

test('api_malformed_requests', async () => {
  await startMock();
  const post = (path, body) => fetch(base + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
  let response = await post('/api/query', '{not json');
  checkEq(response.status, 400);
  checkEq(typeof (await response.json()).error, 'string', 'errors are {"error": "..."}');
  response = await post('/api/settings', '[1,2]');
  checkEq(response.status, 400);
  await response.json();
  response = await fetch(base + '/api/status', { method: 'DELETE' });
  checkEq(response.status, 404);
  await response.json();
  response = await fetch(base + '/style.css', { method: 'POST' });
  checkEq(response.status, 405);
  await response.json();
});

// ---- run ----------------------------------------------------------------------------

const filter = process.argv[2] || '';
let ran = 0;
let failed = 0;
const started = Date.now();
for (const c of cases) {
  if (filter !== '' && !c.name.includes(filter)) continue;
  const before = failures;
  try {
    await c.body();
  } catch (error) {
    report(`unexpected exception: ${error && error.stack ? error.stack : error}`);
  }
  ++ran;
  const ok = failures === before;
  if (!ok) ++failed;
  console.log(`[${ok ? ' ok ' : 'FAIL'}] ${c.name}`);
}
if (mock) await mock.close();
console.log(`${ran} case(s) run, ${failed} failed, ${((Date.now() - started) / 1000).toFixed(1)} s`);
process.exit(failed === 0 ? 0 : 1);
