// Tokeniser for the code views.
//
// tokenize(code, language) splits source text into [{ kind, text }]. The
// concatenation of all token texts is exactly the input, so a renderer that
// creates one text node per token reproduces the code byte for byte. Kinds:
//
//   plain  comment  string  number  keyword  type  call  preproc
//   add  del  meta  path                (diffs and edit blocks, whole lines)
//
// This is a lexer, not a parser: it knows comments, literals and word lists of
// a few language families, which is what makes code readable at a glance. An
// unknown language gives a single plain token.

const MAX_LENGTH = 200000;  // longer texts are shown without colours

const C_KEYWORDS = words(
  'alignas alignof auto break case catch class const constexpr const_cast continue decltype default ' +
  'delete do dynamic_cast else enum explicit export extern false final for friend goto if inline ' +
  'mutable namespace new noexcept nullptr operator override private protected public register ' +
  'reinterpret_cast restrict return sizeof static static_assert static_cast struct switch template ' +
  'this thread_local throw true try typedef typeid typename union using virtual volatile while NULL ' +
  'concept requires co_await co_return co_yield');
const C_TYPES = words(
  'void bool char short int long float double signed unsigned size_t ssize_t ptrdiff_t intptr_t ' +
  'uintptr_t int8_t int16_t int32_t int64_t uint8_t uint16_t uint32_t uint64_t wchar_t char8_t ' +
  'char16_t char32_t FILE va_list');
const PY_KEYWORDS = words(
  'and as assert async await break class continue def del elif else except finally for from global ' +
  'if import in is lambda nonlocal not or pass raise return try while with yield True False None ' +
  'match case self');
const PY_TYPES = words('int float str bytes bool list dict set tuple object type');
const JS_KEYWORDS = words(
  'async await break case catch class const continue debugger default delete do else export extends ' +
  'false finally for from function if import in instanceof let new null of return static super ' +
  'switch this throw true try typeof undefined var void while yield');
const SH_KEYWORDS = words(
  'if then else elif fi for in do done while until case esac function return exit local export set ' +
  'unset echo cd');

function words(list) {
  return new Set(list.split(' '));
}

const FAMILIES = {
  c: { keywords: C_KEYWORDS, types: C_TYPES, line: '//', block: true, preproc: true, hash: false },
  js: { keywords: JS_KEYWORDS, types: new Set(), line: '//', block: true, preproc: false, hash: false, template: true },
  py: { keywords: PY_KEYWORDS, types: PY_TYPES, line: '#', block: false, preproc: false, hash: true, triple: true },
  sh: { keywords: SH_KEYWORDS, types: new Set(), line: '#', block: false, preproc: false, hash: true },
};

const ALIASES = {
  c: 'c', h: 'c', cpp: 'c', 'c++': 'c', cc: 'c', cxx: 'c', hpp: 'c', hh: 'c', objc: 'c', java: 'c',
  cs: 'c', csharp: 'c', go: 'c', rust: 'c', rs: 'c',
  js: 'js', javascript: 'js', mjs: 'js', ts: 'js', typescript: 'js', json: 'js',
  py: 'py', python: 'py', python3: 'py',
  sh: 'sh', bash: 'sh', shell: 'sh', zsh: 'sh', cmake: 'sh', yaml: 'sh', yml: 'sh', toml: 'sh',
  make: 'sh', makefile: 'sh', dockerfile: 'sh',
};

// Language name for a file path, from its extension; '' when unknown.
export function languageOfPath(path) {
  if (typeof path !== 'string') return '';
  const slash = path.lastIndexOf('/');
  const name = slash < 0 ? path : path.slice(slash + 1);
  if (name === 'CMakeLists.txt') return 'cmake';
  if (name === 'Makefile' || name === 'Dockerfile') return name.toLowerCase();
  const dot = name.lastIndexOf('.');
  if (dot < 0) return '';
  const ext = name.slice(dot + 1).toLowerCase();
  return ALIASES[ext] ? ext : '';
}

function isIdentStart(c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_' || c === '$';
}

function isDigit(c) {
  return c >= '0' && c <= '9';
}

function isIdentPart(c) {
  return isIdentStart(c) || isDigit(c);
}

// Whole-line kinds for unified diffs.
function tokenizeDiff(code) {
  const tokens = [];
  const lines = code.split('\n');
  lines.forEach((line, index) => {
    const text = index + 1 < lines.length ? line + '\n' : line;
    if (text === '') return;
    let kind = 'plain';
    if (line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@') || line.startsWith('diff ')) {
      kind = 'meta';
    } else if (line.startsWith('+')) {
      kind = 'add';
    } else if (line.startsWith('-')) {
      kind = 'del';
    }
    tokens.push({ kind, text });
  });
  return tokens;
}

// Edit blocks of prag::edit: the marker lines are meta, the text to find is
// shown as removed and the replacement as added.
function tokenizeEdit(code) {
  const tokens = [];
  const lines = code.split('\n');
  let section = 'plain';
  let opened = false;  // false until the first marker: the lines naming the file and the unit
  lines.forEach((line, index) => {
    const text = index + 1 < lines.length ? line + '\n' : line;
    if (text === '') return;
    let kind = opened ? section : 'path';
    if (/^ {0,3}[<=>]{5,9}/.test(line)) opened = true;
    if (/^ {0,3}<{5,9} SEARCH\b/.test(line)) {
      kind = 'meta';
      section = 'del';
    } else if (/^ {0,3}<{5,9} UNIT\b/.test(line)) {
      kind = 'meta';
      section = 'add';
    } else if (/^ {0,3}={5,9}\s*$/.test(line) && section === 'del') {
      kind = 'meta';
      section = 'add';
    } else if (/^ {0,3}>{5,9} (?:REPLACE|UNIT)\s*$/.test(line)) {
      kind = 'meta';
      section = 'plain';
    }
    tokens.push({ kind, text });
  });
  return tokens;
}

export function tokenize(code, language) {
  const text = String(code === undefined || code === null ? '' : code);
  if (text === '') return [];
  const lang = String(language || '').toLowerCase();
  if (text.length > MAX_LENGTH) return [{ kind: 'plain', text }];
  if (lang === 'diff' || lang === 'patch') return tokenizeDiff(text);
  if (lang === 'edit') return tokenizeEdit(text);
  const family = FAMILIES[ALIASES[lang]];
  if (!family) return [{ kind: 'plain', text }];

  const tokens = [];
  let plain = '';
  const push = (kind, value) => {
    if (value === '') return;
    if (kind === 'plain') {
      plain += value;
      return;
    }
    if (plain !== '') {
      tokens.push({ kind: 'plain', text: plain });
      plain = '';
    }
    tokens.push({ kind, text: value });
  };

  const n = text.length;
  let i = 0;
  let lineStart = true;  // only blanks since the start of the line
  while (i < n) {
    const c = text[i];

    if (c === '\n') {
      push('plain', c);
      lineStart = true;
      ++i;
      continue;
    }
    if (c === ' ' || c === '\t' || c === '\r') {
      push('plain', c);
      ++i;
      continue;
    }

    // Preprocessor directive: the directive word only, so that the rest of
    // the line (a macro body, an include path) is still tokenised.
    if (family.preproc && c === '#' && lineStart) {
      let j = i + 1;
      while (j < n && (text[j] === ' ' || text[j] === '\t')) ++j;
      while (j < n && isIdentPart(text[j])) ++j;
      const directive = text.slice(i, j);
      push('preproc', directive);
      i = j;
      lineStart = false;
      if (/include/.test(directive)) {
        // #include <path> : the angle brackets delimit a string here.
        let k = i;
        while (k < n && (text[k] === ' ' || text[k] === '\t')) ++k;
        if (text[k] === '<') {
          const close = text.indexOf('>', k);
          const end = text.indexOf('\n', k);
          if (close >= 0 && (end < 0 || close < end)) {
            push('plain', text.slice(i, k));
            push('string', text.slice(k, close + 1));
            i = close + 1;
          }
        }
      }
      continue;
    }
    lineStart = false;

    if (text.startsWith(family.line, i) || (family.hash && c === '#')) {
      let j = text.indexOf('\n', i);
      if (j < 0) j = n;
      push('comment', text.slice(i, j));
      i = j;
      continue;
    }
    if (family.block && c === '/' && text[i + 1] === '*') {
      const close = text.indexOf('*/', i + 2);
      const j = close < 0 ? n : close + 2;
      push('comment', text.slice(i, j));
      i = j;
      continue;
    }

    if (family.triple && (text.startsWith('"""', i) || text.startsWith("'''", i))) {
      const close = text.indexOf(text.slice(i, i + 3), i + 3);
      const j = close < 0 ? n : close + 3;
      push('string', text.slice(i, j));
      i = j;
      continue;
    }
    if (c === '"' || c === "'" || (family.template && c === '`')) {
      let j = i + 1;
      while (j < n && text[j] !== c) {
        if (text[j] === '\\') ++j;
        else if (text[j] === '\n' && c !== '`') break;  // unterminated: stop at the end of the line
        ++j;
      }
      j = Math.min(n, j < n && text[j] === c ? j + 1 : j);
      push('string', text.slice(i, j));
      i = j;
      continue;
    }

    if (isDigit(c) || (c === '.' && isDigit(text[i + 1] || ''))) {
      let j = i + 1;
      while (j < n) {
        const d = text[j];
        if (isIdentPart(d) || d === '.' || d === "'") ++j;
        else if ((d === '+' || d === '-') && /[eEpP]/.test(text[j - 1]) && !/^0[xX]/.test(text.slice(i, i + 2))) ++j;
        else break;
      }
      push('number', text.slice(i, j));
      i = j;
      continue;
    }

    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < n && isIdentPart(text[j])) ++j;
      const word = text.slice(i, j);
      let kind = 'plain';
      if (family.keywords.has(word)) kind = 'keyword';
      else if (family.types.has(word) || (family === FAMILIES.c && /_t$/.test(word))) kind = 'type';
      else {
        let k = j;
        while (k < n && (text[k] === ' ' || text[k] === '\t')) ++k;
        if (text[k] === '(') kind = 'call';
      }
      push(kind, word);
      i = j;
      continue;
    }

    push('plain', c);
    ++i;
  }
  if (plain !== '') tokens.push({ kind: 'plain', text: plain });
  return tokens;
}
