// Parser for unified diffs as produced by prag::edit::unified_diff.
//
// The result is plain data; the caller inserts every string as text.
//
//   parseUnifiedDiff(text) -> {
//     files: [ { oldPath, newPath, added, removed,
//                hunks: [ { oldStart, oldLines, newStart, newLines, section,
//                           lines: [ { type, text, oldLine, newLine } ] } ] } ],
//     added, removed }
//
// `type` is "context", "add", "del" or "note" (the "\ No newline at end of
// file" marker). `oldLine` and `newLine` are 1-based line numbers or null.
//
// Inside a hunk the line counts of the hunk header decide where the hunk ends.
// This matters: a removed line whose content starts with "-- " reads "--- ..."
// in the diff and must not be taken for a file header.

const HUNK = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$/;

// Removes the "a/" or "b/" prefix and a trailing tab-separated time stamp.
function cleanPath(raw) {
  let path = raw;
  const tab = path.indexOf('\t');
  if (tab >= 0) path = path.slice(0, tab);
  path = path.trim();
  if (path === '/dev/null') return null;
  if (path.startsWith('a/') || path.startsWith('b/')) path = path.slice(2);
  return path;
}

export function parseUnifiedDiff(text) {
  const result = { files: [], added: 0, removed: 0 };
  if (typeof text !== 'string' || text === '') return result;

  const lines = text.split('\n');
  // A final newline yields one empty trailing element that is not a line.
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();

  let file = null;
  let hunk = null;
  let awaitingPaths = false;  // a "diff" line opened the file and "---" has not been seen yet
  let oldLeft = 0;  // lines of the old file still expected in the current hunk
  let newLeft = 0;
  let oldLine = 0;
  let newLine = 0;

  const openFile = () => {
    file = { oldPath: null, newPath: null, hunks: [], added: 0, removed: 0 };
    result.files.push(file);
    hunk = null;
  };

  for (let raw of lines) {
    if (raw.endsWith('\r')) raw = raw.slice(0, -1);

    if (hunk && (oldLeft > 0 || newLeft > 0)) {
      const mark = raw.charAt(0);
      const body = raw.slice(1);
      if (mark === '+') {
        hunk.lines.push({ type: 'add', text: body, oldLine: null, newLine: newLine++ });
        --newLeft;
        ++file.added;
        ++result.added;
        continue;
      }
      if (mark === '-') {
        hunk.lines.push({ type: 'del', text: body, oldLine: oldLine++, newLine: null });
        --oldLeft;
        ++file.removed;
        ++result.removed;
        continue;
      }
      if (mark === ' ' || raw === '') {
        // Some tools strip the single space of an empty context line.
        hunk.lines.push({ type: 'context', text: body, oldLine: oldLine++, newLine: newLine++ });
        --oldLeft;
        --newLeft;
        continue;
      }
      if (mark === '\\') {
        hunk.lines.push({ type: 'note', text: body.trim(), oldLine: null, newLine: null });
        continue;
      }
      // Anything else ends the hunk early; fall through to the header rules.
      oldLeft = 0;
      newLeft = 0;
    }

    if (raw.startsWith('\\') && hunk) {
      hunk.lines.push({ type: 'note', text: raw.slice(1).trim(), oldLine: null, newLine: null });
      continue;
    }
    if (raw.startsWith('diff ')) {
      openFile();
      awaitingPaths = true;
      continue;
    }
    if (raw.startsWith('--- ')) {
      // Starts a new file unless a "diff" line has just done so.
      if (!file || !awaitingPaths) openFile();
      awaitingPaths = false;
      file.oldPath = cleanPath(raw.slice(4));
      continue;
    }
    if (raw.startsWith('+++ ')) {
      if (!file) openFile();
      file.newPath = cleanPath(raw.slice(4));
      continue;
    }
    const match = HUNK.exec(raw);
    if (match) {
      if (!file) openFile();
      hunk = {
        oldStart: Number(match[1]),
        oldLines: match[2] === undefined ? 1 : Number(match[2]),
        newStart: Number(match[3]),
        newLines: match[4] === undefined ? 1 : Number(match[4]),
        section: match[5] || '',
        lines: [],
      };
      file.hunks.push(hunk);
      oldLeft = hunk.oldLines;
      newLeft = hunk.newLines;
      oldLine = hunk.oldStart;
      newLine = hunk.newStart;
      continue;
    }
    // Other lines ("index ...", "new file mode ...", prose) carry nothing we show.
  }

  return result;
}

// Text of a hunk header, rebuilt from the parsed numbers.
export function hunkHeader(hunk) {
  const range = (start, count) => (count === 1 ? String(start) : `${start},${count}`);
  const head = `@@ -${range(hunk.oldStart, hunk.oldLines)} +${range(hunk.newStart, hunk.newLines)} @@`;
  return hunk.section ? `${head} ${hunk.section}` : head;
}

// Path to show for a file of a diff: the new path, or the old one for a deletion.
export function displayPath(file) {
  return file.newPath || file.oldPath || '';
}
