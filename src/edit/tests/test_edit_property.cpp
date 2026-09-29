// S1, no silent misapplication: whenever prepare() reports Ok, the new content
// of the file is the intended one.
//
// Known-target protocol. A case starts from a text file and a range of lines
// in it. The intended replacement is written first, which fixes the intended
// content of the file after the edit. The edit is then rendered as a
// SEARCH/REPLACE block, the way a model would write it, and only the surface
// that is used to find the place is disturbed:
//
//   the file on disk    re-indented by a fixed prefix, trailing blanks added,
//                       line endings converted, unrelated lines inserted or
//                       deleted above the target, the target text repeated
//                       elsewhere
//   the block           indentation removed from or added to every line
//                       (SEARCH and REPLACE alike, which is how models get it
//                       wrong), trailing blanks added to SEARCH lines, CRLF
//
// The intended content of the disturbed file is known by construction: the
// lines outside of the target as they are on disk, and in place of the target
// the replacement in the frame of the file, that is with the prefix the file
// was re-indented by and with the line ending of the file. It is computed
// here, by code that shares nothing with the module under test. Every case is
// run without and with disturbance; the outcome is a clean success (Ok and the
// content is the intended one), a clean failure (NotFound or Ambiguous,
// nothing to write) or a silent misapplication (Ok and any other content).
//
// Two further experiments use the same generator and are reported separately:
// files whose indentation was converted between tabs and spaces, and SEARCH
// text with typing errors under the fuzzy rung.
#define PRAG_TEST_MAIN
#include "prag_test.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "../internal.hpp"
#include "edit_test_util.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/edit/edit.hpp"

using namespace prag;
using namespace prag::edit;
using edit_test::Random;

namespace {

// ---- the model of a text file used by the test ---------------------------------------

struct Document {
  std::vector<std::string> lines;  // without terminators
  std::string ending = "\n";
  bool final_newline = true;
};

std::string render(const Document& d) {
  std::string out;
  for (std::size_t i = 0; i < d.lines.size(); ++i) {
    out += d.lines[i];
    if (i + 1 < d.lines.size() || d.final_newline) out += d.ending;
  }
  return out;
}

// Text with LF endings and a final newline.
Document parse(std::string_view text) {
  Document d;
  std::size_t begin = 0;
  while (begin < text.size()) {
    std::size_t end = text.find('\n', begin);
    if (end == std::string_view::npos) end = text.size();
    d.lines.emplace_back(text.substr(begin, end - begin));
    begin = end + 1;
  }
  return d;
}

bool is_blank(const std::string& line) { return line.find_first_not_of(" \t") == std::string::npos; }

std::string indentation(const std::string& line) {
  const std::size_t n = line.find_first_not_of(" \t");
  return line.substr(0, n == std::string::npos ? line.size() : n);
}

std::string without_indentation(const std::string& line) { return line.substr(indentation(line).size()); }

// A payload line that the parser would read as a marker cannot be written in a block.
bool like_a_marker(const std::string& line) {
  if (line.empty() || (line[0] != '<' && line[0] != '=' && line[0] != '>')) return false;
  std::size_t run = 0;
  while (run < line.size() && line[run] == line[0]) ++run;
  return run >= 5;
}

std::size_t offset_of_line(const Document& d, std::size_t line) {
  std::size_t offset = 0;
  for (std::size_t i = 0; i < line && i < d.lines.size(); ++i) offset += d.lines[i].size() + d.ending.size();
  if (line >= d.lines.size() && !d.final_newline && !d.lines.empty()) offset -= d.ending.size();
  return offset;
}

// ---- corpus --------------------------------------------------------------------------

const char* const kSpaces = R"(/* ----------------------------------------------------------------------------
  Bookkeeping of blocks in a fixed table, in the style of the corpus.
-----------------------------------------------------------------------------*/
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SLOT_COUNT  (64)
#define SLOT_EMPTY  ((size_t)0)

typedef struct slot_s {
  size_t size;
  void*  block;
  struct slot_s* next;
} slot_t;

static slot_t slots[SLOT_COUNT];
static size_t slots_used = 0;

// Find the slot that holds `block`, or NULL.
static slot_t* slot_find(const void* block) {
  for (size_t i = 0; i < SLOT_COUNT; i++) {
    if (slots[i].block == block) {
      return &slots[i];
    }
  }
  return NULL;
}

// Claim a free slot for `block`.
static slot_t* slot_claim(void* block, size_t size) {
  if (block == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < SLOT_COUNT; i++) {
    if (slots[i].size == SLOT_EMPTY) {
      slots[i].size = size;
      slots[i].block = block;
      slots_used++;
      return &slots[i];
    }
  }
  return NULL;
}

// Release the slot of `block`; returns false if the block is unknown.
static bool slot_release(void* block) {
  slot_t* slot = slot_find(block);
  if (slot == NULL) {
    return false;
  }
  slot->size = SLOT_EMPTY;
  slot->block = NULL;
  slots_used--;
  return true;
}

size_t slots_total_size(void) {
  size_t total = 0;
  for (size_t i = 0; i < SLOT_COUNT; i++) {
    if (slots[i].size != SLOT_EMPTY) {
      total += slots[i].size;
    }
  }
  return total;
}

size_t slots_largest(void) {
  size_t largest = 0;
  for (size_t i = 0; i < SLOT_COUNT; i++) {
    if (slots[i].size > largest) {
      largest = slots[i].size;
    }
  }
  return largest;
}

#if defined(SLOT_DEBUG)
void slots_check(void) {
  size_t used = 0;
  for (size_t i = 0; i < SLOT_COUNT; i++) {
    if (slots[i].size != SLOT_EMPTY) {
      used++;
    }
  }
  if (used != slots_used) {
    abort();
  }
}
#endif
)";

const char* const kTabs = R"(/*
 * A ring buffer of bytes. Indented with tabs; continuation lines are aligned
 * with spaces after the tabs.
 */
#include <stddef.h>
#include <string.h>

struct ring {
	unsigned char *data;
	size_t capacity;
	size_t head;
	size_t tail;
	size_t used;
};

static size_t ring_free(const struct ring *r)
{
	return r->capacity - r->used;
}

size_t ring_write(struct ring *r, const unsigned char *bytes,
                  size_t count)
{
	size_t written = 0;

	if (count > ring_free(r))
		count = ring_free(r);
	while (written < count) {
		size_t chunk = r->capacity - r->tail;

		if (chunk > count - written)
			chunk = count - written;
		memcpy(r->data + r->tail, bytes + written,
		       chunk);
		r->tail = (r->tail + chunk) % r->capacity;
		written += chunk;
	}
	r->used += written;
	return written;
}

size_t ring_read(struct ring *r, unsigned char *bytes,
                 size_t count)
{
	size_t done = 0;

	if (count > r->used)
		count = r->used;
	while (done < count) {
		size_t chunk = r->capacity - r->head;

		if (chunk > count - done)
			chunk = count - done;
		memcpy(bytes + done, r->data + r->head,
		       chunk);
		r->head = (r->head + chunk) % r->capacity;
		done += chunk;
	}
	r->used -= done;
	return done;
}

void ring_clear(struct ring *r)
{
	r->head = 0;
	r->tail = 0;
	r->used = 0;
}

int ring_peek(const struct ring *r, size_t index)
{
	if (index >= r->used)
		return -1;
	switch (r->capacity) {
	case 0:
		return -1;
	default:
		return r->data[(r->head + index) % r->capacity];
	}
}
)";

const char* const kNested = R"(class Pool:
    """Blocks of one size, kept on a free list."""

    def __init__(self, block_size, count):
        self.block_size = block_size
        self.free = list(range(count))
        self.used = set()

    def allocate(self):
        if not self.free:
            return None
        index = self.free.pop()
        self.used.add(index)
        return index

    def release(self, index):
        if index not in self.used:
            raise ValueError(index)
        self.used.remove(index)
        self.free.append(index)

    def resize(self, count):
        total = len(self.free) + len(self.used)
        if count < total:
            for index in range(count, total):
                if index in self.used:
                    raise ValueError(index)
                self.free.remove(index)
        else:
            for index in range(total, count):
                self.free.append(index)


class Arena:
    def __init__(self):
        self.pools = {}

    def pool(self, block_size):
        if block_size not in self.pools:
            self.pools[block_size] = Pool(block_size, 64)
        return self.pools[block_size]

    def allocate(self, size):
        block_size = 8
        while block_size < size:
            block_size *= 2
        return block_size, self.pool(block_size).allocate()

    def release(self, block_size, index):
        if block_size not in self.pools:
            raise ValueError(block_size)
        self.pools[block_size].release(index)
)";

struct Source {
  std::string name;
  Document document;
  std::string unit;  // one level of indentation in this file
  bool from_corpus = false;
};

// The embedded texts, and files of the mimalloc checkout when it is present.
// The checkout is only read: every file is copied to the temporary directory
// and the copy is what the test works with.
std::vector<Source> read_sources() {
  std::vector<Source> sources;
  sources.push_back({"embedded/spaces.c", parse(kSpaces), "  ", false});
  sources.push_back({"embedded/tabs.c", parse(kTabs), "\t", false});
  sources.push_back({"embedded/nested.py", parse(kNested), "    ", false});

  const std::string repo = PRAG_DEMO_REPO;
  if (!fs::is_directory(repo)) {
    std::printf("  (the corpus at '%s' is missing: embedded texts only)\n", repo.c_str());
    return sources;
  }
  const std::string work = edit_test::temp_dir("corpus");
  fs::WalkOptions walk;
  walk.extensions = {".c", ".h"};
  walk.max_file_bytes = 48 * 1024;
  std::size_t copied = 0;
  for (const char* directory : {"/src", "/include"}) {
    for (const std::string& path : fs::walk(repo + directory, walk)) {
      const std::string name = path.substr(fs::normalize(repo).size() + 1);
      const std::string copy = work + "/" + name;
      const std::string original = edit_test::read_file(path);
      if (!edit_test::write_file(copy, original)) continue;
      const std::string text = edit_test::read_file(copy);
      if (text != original || text.empty() || text.back() != '\n') continue;
      if (text.find('\r') != std::string::npos || text.find('\0') != std::string::npos) continue;
      Document document = parse(text);
      if (document.lines.size() < 40) continue;
      sources.push_back({name, std::move(document), "  ", true});
      ++copied;
    }
  }
  std::printf("  (corpus: %zu files of %s, copied to %s)\n", copied, repo.c_str(), work.c_str());
  return sources;
}

// Read once and shared by the experiments.
const std::vector<Source>& load_sources() {
  static const std::vector<Source> sources = read_sources();
  return sources;
}

// ---- cases ---------------------------------------------------------------------------

enum Disturbance : unsigned {
  kFileIndent = 1u << 0,     // every non-blank line of the file starts with a further prefix
  kFileTrailing = 1u << 1,   // blanks at the end of lines of the file
  kFileCrlf = 1u << 2,       // the file has CRLF endings
  kFileInsert = 1u << 3,     // unrelated lines inserted above the target
  kFileDelete = 1u << 4,     // unrelated lines deleted above the target
  kFileRepeat = 1u << 5,     // the target text occurs a second time
  kBlockOutdent = 1u << 6,   // common indentation removed from the block
  kBlockIndent = 1u << 7,    // indentation added to the block
  kBlockTrailing = 1u << 8,  // blanks at the end of SEARCH lines
  kBlockCrlf = 1u << 9,      // the block is written with CRLF endings
  kFileConvert = 1u << 10,   // indentation converted between tabs and spaces (second experiment)
  kBlockTypos = 1u << 11,    // typing errors in the SEARCH text (third experiment)
};

const std::pair<unsigned, const char*> kDisturbanceNames[] = {
    {kFileIndent, "file: re-indented"},
    {kFileTrailing, "file: trailing blanks"},
    {kFileCrlf, "file: CRLF endings"},
    {kFileInsert, "file: lines inserted above"},
    {kFileDelete, "file: lines deleted above"},
    {kFileRepeat, "file: target text repeated"},
    {kBlockOutdent, "block: indentation removed"},
    {kBlockIndent, "block: indentation added"},
    {kBlockTrailing, "block: trailing blanks"},
    {kBlockCrlf, "block: CRLF endings"},
    {kFileConvert, "file: tabs and spaces converted"},
    {kBlockTypos, "block: typing errors"},
};

struct Instance {
  Document file;                        // what is on disk
  std::size_t first = 0;                // target lines [first, first + count) of the file
  std::size_t count = 0;
  std::vector<std::string> search;      // as the model writes them
  std::vector<std::string> replacement;
  std::vector<std::string> intended;    // the replacement in the frame of the file
  std::string unit;                     // one level of indentation
  std::string source;                   // name of the text the case was made from
  bool from_corpus = false;
  unsigned disturbances = 0;
  bool scoped = false;
  std::size_t scope_first = 0;          // lines [scope_first, scope_last) form the unit
  std::size_t scope_last = 0;
};

Document intended_file(const Instance& instance) {
  Document target = instance.file;
  const auto at = target.lines.begin() + static_cast<std::ptrdiff_t>(instance.first);
  target.lines.erase(at, at + static_cast<std::ptrdiff_t>(instance.count));
  target.lines.insert(target.lines.begin() + static_cast<std::ptrdiff_t>(instance.first), instance.intended.begin(),
                      instance.intended.end());
  return target;
}

std::string tag(std::size_t id, std::size_t k) { return std::to_string(id) + "_" + std::to_string(k); }

// The replacement: the target lines with a few changes, as an edit would make them.
std::vector<std::string> edited(const std::vector<std::string>& lines, const std::string& unit, std::size_t id,
                                Random& random) {
  if (random.chance(0.06)) return {};  // the lines are removed
  std::vector<std::string> out = lines;
  const std::size_t changes = random.between(1, 3);
  for (std::size_t k = 0; k < changes; ++k) {
    const std::size_t what = random.below(3);
    if (what == 0 || out.empty()) {
      // A new line, indented like a neighbour, or one level deeper or shallower.
      const std::size_t at = random.below(out.size() + 1);
      std::string indent;
      for (std::size_t probe = at; probe-- > 0;) {
        if (!is_blank(out[probe])) {
          indent = indentation(out[probe]);
          break;
        }
      }
      if (indent.empty() && at < out.size()) indent = indentation(out[at]);
      const std::size_t depth = random.below(10);
      if (depth < 3) indent += unit;
      if (depth == 3 && indent.size() >= unit.size() && indent.compare(indent.size() - unit.size(), unit.size(), unit) == 0) {
        indent.resize(indent.size() - unit.size());
      }
      std::string line = random.chance(0.1) ? std::string() : indent + "inserted_" + tag(id, k) + "();";
      out.insert(out.begin() + static_cast<std::ptrdiff_t>(at), std::move(line));
    } else if (what == 1 && out.size() > 1) {
      out.erase(out.begin() + static_cast<std::ptrdiff_t>(random.below(out.size())));
    } else {
      std::string& line = out[random.below(out.size())];
      if (is_blank(line)) {
        line = "changed_" + tag(id, k) + "();";
      } else if (random.chance(0.5)) {
        line += " /* changed " + tag(id, k) + " */";
      } else {
        line = indentation(line) + "changed_" + tag(id, k) + "(" + std::to_string(line.size()) + ");";
      }
    }
  }
  if (out == lines) out.push_back("appended_" + std::to_string(id) + "();");
  return out;
}

// A case without disturbance: a source, a range of its lines and the replacement.
Instance make_case(const std::vector<Source>& sources, std::size_t id, Random& random, std::size_t min_lines,
                   std::size_t max_lines) {
  for (;;) {
    const Source& source = sources[random.below(sources.size())];
    Instance instance;
    instance.file = source.document;
    instance.unit = source.unit;
    instance.source = source.name;
    instance.from_corpus = source.from_corpus;
    instance.file.final_newline = !random.chance(0.1);
    const std::size_t n = instance.file.lines.size();
    instance.count = std::min(n, random.between(min_lines, max_lines));
    instance.first = random.below(n - instance.count + 1);
    const auto begin = instance.file.lines.begin() + static_cast<std::ptrdiff_t>(instance.first);
    instance.search.assign(begin, begin + static_cast<std::ptrdiff_t>(instance.count));
    // A block needs something to hold on to, and nothing the parser takes for a marker.
    if (std::all_of(instance.search.begin(), instance.search.end(), is_blank)) continue;
    instance.replacement = edited(instance.search, source.unit, id, random);
    if (std::any_of(instance.search.begin(), instance.search.end(), like_a_marker)) continue;
    if (std::any_of(instance.replacement.begin(), instance.replacement.end(), like_a_marker)) continue;
    instance.intended = instance.replacement;
    return instance;
  }
}

void choose_scope(Instance* instance, Random& random) {
  instance->scoped = true;
  const std::size_t before = random.below(13);
  const std::size_t after = random.below(13);
  instance->scope_first = instance->first > before ? instance->first - before : 0;
  instance->scope_last = std::min(instance->file.lines.size(), instance->first + instance->count + after);
}

std::string random_blanks(Random& random) {
  static const char* const kBlanks[] = {" ", "  ", "\t", "   ", " \t"};
  return kBlanks[random.below(5)];
}

std::string common_indentation(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  bool first = true;
  std::string common;
  for (const std::vector<std::string>* lines : {&a, &b}) {
    for (const std::string& line : *lines) {
      if (is_blank(line)) continue;
      const std::string indent = indentation(line);
      if (first) {
        common = indent;
        first = false;
      } else {
        std::size_t k = 0;
        while (k < common.size() && k < indent.size() && common[k] == indent[k]) ++k;
        common.resize(k);
      }
    }
  }
  return common;
}

void shift_block(Instance* instance, std::size_t remove, const std::string& add) {
  for (std::vector<std::string>* lines : {&instance->search, &instance->replacement}) {
    for (std::string& line : *lines) {
      if (!is_blank(line)) line = add + line.substr(remove);
    }
  }
}

// Inserts and deletes lines above the target, and repeats the target text.
void disturb_structure(Instance* instance, const Document& pool, unsigned which, std::size_t id, Random& random) {
  Document& file = instance->file;
  if (which & kFileInsert) {
    for (std::size_t k = random.between(1, 5); k-- > 0;) {
      const std::size_t at = random.below(instance->first + 1);
      std::string line;
      if (random.chance(0.5)) {
        line = pool.lines[random.below(pool.lines.size())];  // may well be a line of the target
      } else {
        const std::string indent = at > 0 ? indentation(file.lines[at - 1]) : std::string();
        line = indent + "/* note " + tag(id, k) + " */";
      }
      file.lines.insert(file.lines.begin() + static_cast<std::ptrdiff_t>(at), std::move(line));
      ++instance->first;
      instance->disturbances |= kFileInsert;
    }
  }
  if (which & kFileDelete) {
    for (std::size_t k = random.between(1, 5); k-- > 0 && instance->first > 0;) {
      file.lines.erase(file.lines.begin() + static_cast<std::ptrdiff_t>(random.below(instance->first)));
      --instance->first;
      instance->disturbances |= kFileDelete;
    }
  }
  if (which & kFileRepeat) {
    // Above or below the target, never inside it.
    const std::size_t below = file.lines.size() - (instance->first + instance->count);
    std::size_t at = random.below(instance->first + 1 + below + 1);
    const bool above = at <= instance->first;
    if (!above) at = instance->first + instance->count + (at - instance->first - 1);
    const std::vector<std::string> copy(file.lines.begin() + static_cast<std::ptrdiff_t>(instance->first),
                                        file.lines.begin() + static_cast<std::ptrdiff_t>(instance->first + instance->count));
    file.lines.insert(file.lines.begin() + static_cast<std::ptrdiff_t>(at), copy.begin(), copy.end());
    if (above) instance->first += instance->count;
    instance->disturbances |= kFileRepeat;
  }
}

// The disturbances of the main experiment; `which` selects among them.
void disturb(Instance* instance, const Document& pool, unsigned which, std::size_t id, Random& random) {
  disturb_structure(instance, pool, which, id, random);
  Document& file = instance->file;

  if (which & kFileIndent) {
    static const char* const kPrefixes[] = {"  ", "    ", "\t", " "};
    // Mostly the kind of blank the file is indented with.
    std::string prefix = instance->unit;
    if (random.chance(0.5)) prefix = kPrefixes[random.below(4)];
    for (std::string& line : file.lines) {
      if (!is_blank(line)) line = prefix + line;
    }
    for (std::string& line : instance->intended) {
      if (!is_blank(line)) line = prefix + line;
    }
    instance->disturbances |= kFileIndent;
  }
  if (which & kFileTrailing) {
    for (std::size_t i = 0; i < file.lines.size(); ++i) {
      const bool target = i >= instance->first && i < instance->first + instance->count;
      if (random.chance(target ? 0.6 : 0.2)) file.lines[i] += random_blanks(random);
    }
    instance->disturbances |= kFileTrailing;
  }
  if (which & kFileCrlf) {
    file.ending = "\r\n";
    instance->disturbances |= kFileCrlf;
  }

  if (which & (kBlockOutdent | kBlockIndent)) {
    const std::string common = common_indentation(instance->search, instance->replacement);
    if ((which & kBlockOutdent) && !common.empty()) {
      const std::size_t remove = random.chance(0.7) ? common.size() : random.between(1, common.size());
      shift_block(instance, remove, "");
      instance->disturbances |= kBlockOutdent;
    } else {
      static const char* const kPrefixes[] = {"  ", "    ", " "};
      const std::string add = instance->unit == "\t" && random.chance(0.7) ? "\t" : kPrefixes[random.below(3)];
      shift_block(instance, 0, add);
      instance->disturbances |= kBlockIndent;
    }
  }
  if (which & kBlockTrailing) {
    for (std::string& line : instance->search) {
      if (random.chance(0.5)) line += random_blanks(random);
    }
    instance->search.front() += " ";
    instance->disturbances |= kBlockTrailing;
  }
  if (which & kBlockCrlf) instance->disturbances |= kBlockCrlf;
}

std::string convert_indentation(const std::string& line, bool tabs_to_spaces) {
  if (is_blank(line)) return line;
  const std::string indent = indentation(line);
  std::string converted;
  if (tabs_to_spaces) {
    for (const char c : indent) converted += c == '\t' ? std::string("    ") : std::string(1, c);
  } else {
    std::size_t i = 0;
    while (i + 1 < indent.size() && indent[i] == ' ' && indent[i + 1] == ' ') {
      converted += '\t';
      i += 2;
    }
    converted += indent.substr(i);
  }
  return converted + line.substr(indent.size());
}

// ---- running a case --------------------------------------------------------------------

struct Rendered {
  std::string text;
  std::string scope;
};

// The block as a model writes it: optional prose and fences around it, markers
// of 5 to 9 characters.
Rendered render_block(const Instance& instance, std::size_t id, Random& random) {
  const std::string newline = (instance.disturbances & kBlockCrlf) ? "\r\n" : "\n";
  const std::size_t width = random.between(5, 9);
  const std::size_t fence = random.chance(0.3) ? random.between(1, 2) : 0;  // before or after the file name
  Rendered out;
  if (instance.scoped) out.scope = "unit_" + std::to_string(id);
  std::vector<std::string> lines;
  if (random.chance(0.3)) {
    lines.push_back("The change for case " + std::to_string(id) + " follows.");
    lines.push_back("");
  }
  if (fence == 1) lines.push_back("```c");
  lines.push_back("case.c");
  if (fence == 2) lines.push_back("```c");
  if (instance.scoped) lines.push_back("@@ " + out.scope);
  lines.push_back(std::string(width, '<') + " SEARCH");
  lines.insert(lines.end(), instance.search.begin(), instance.search.end());
  lines.push_back(std::string(width, '='));
  lines.insert(lines.end(), instance.replacement.begin(), instance.replacement.end());
  lines.push_back(std::string(width, '>') + " REPLACE");
  if (fence != 0) lines.push_back("```");
  if (random.chance(0.2)) lines.push_back("That is all.");
  for (const std::string& line : lines) out.text += line + newline;
  return out;
}

enum class Outcome { Success, NotFound, Ambiguous, Refused, Silent, Deviation };

struct Tally {
  std::size_t success = 0;
  std::size_t not_found = 0;
  std::size_t ambiguous = 0;
  std::size_t refused = 0;    // any other status; none is expected
  std::size_t silent = 0;
  std::size_t deviation = 0;  // second experiment only
  std::size_t total() const { return success + not_found + ambiguous + refused + silent + deviation; }
  void add(Outcome outcome) {
    switch (outcome) {
      case Outcome::Success:
        ++success;
        break;
      case Outcome::NotFound:
        ++not_found;
        break;
      case Outcome::Ambiguous:
        ++ambiguous;
        break;
      case Outcome::Refused:
        ++refused;
        break;
      case Outcome::Silent:
        ++silent;
        break;
      case Outcome::Deviation:
        ++deviation;
        break;
    }
  }
};

struct Result {
  Outcome outcome = Outcome::Refused;
  MatchKind rung = MatchKind::None;
  Status status = Status::NotFound;
};

// One run in `kDiskEvery` goes all the way: the file is written, prepare()
// reads it from the disk, commit() writes the proposal and revert() takes it
// back. The other runs hand the content of the file to prepare() in memory
// (detail::set_snapshot_source), which is the same code from the resolution of
// the path to the diff, without a file written for each of them. The number
// is odd, so that runs without and with disturbance take turns.
constexpr std::size_t kDiskEvery = 97;

struct Runner {
  std::string root;
  std::string path;
  ApplyOptions options;
  std::size_t runs = 0;
  std::size_t disk_runs = 0;
  std::size_t parser_errors = 0;
  std::size_t disk_errors = 0;      // the file system refused an operation
  std::size_t disk_mismatches = 0;  // what is on disk is not what was proposed
  std::size_t reports = 0;
  double prepare_s = 0.0;  // time spent in prepare(), and on the disk
  double disk_s = 0.0;
  std::string snapshot;    // content handed to prepare() in memory
  bool use_snapshot = false;

  explicit Runner(const char* name) : root(edit_test::temp_dir(name)), path(root + "/case.c") {
    options.allowed_roots = {root};
    options.backup_dir = root + "/.edits";
    detail::set_snapshot_source([this](const std::string& asked, std::string* text) {
      if (!use_snapshot || asked != path) return false;
      *text = snapshot;
      return true;
    });
  }
  ~Runner() { detail::set_snapshot_source({}); }
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // Renders and parses the block and prepares the edit. With `relaxed`,
  // content that differs from the intended one only in the indentation of the
  // new lines counts as a deviation.
  Result run(const Instance& instance, std::size_t id, Random& random, bool relaxed = false) {
    const bool through_disk = runs++ % kDiskEvery == 0;
    Result result;
    const std::string on_disk = render(instance.file);
    const std::string intended = render(intended_file(instance));

    const Rendered block = render_block(instance, id, random);
    const ParseResult parsed = parse_edit_blocks(block.text);
    std::string search;
    for (const std::string& line : instance.search) search += line + "\n";
    std::string replacement;
    for (const std::string& line : instance.replacement) replacement += line + "\n";
    if (parsed.blocks.size() != 1 || !parsed.warnings.empty() || parsed.blocks[0].path != "case.c" ||
        parsed.blocks[0].scope != block.scope || parsed.blocks[0].search != search ||
        parsed.blocks[0].replace != replacement) {
      ++parser_errors;
      return result;
    }

    ApplyOptions local = options;
    if (instance.scoped) {
      const std::size_t scope_begin = offset_of_line(instance.file, instance.scope_first);
      const std::size_t scope_end = offset_of_line(instance.file, instance.scope_last);
      const std::string name = block.scope;
      local.unit_locator = [=](std::string_view, std::string_view, std::string_view unit, std::size_t* begin,
                               std::size_t* end) {
        if (unit != name) return false;
        *begin = scope_begin;
        *end = scope_end;
        return true;
      };
    }

    Stopwatch watch;
    use_snapshot = !through_disk;
    if (through_disk) {
      ++disk_runs;
      if (!edit_test::write_file(path, on_disk)) {
        std::fprintf(stderr, "  case %zu: cannot write %s\n", id, path.c_str());
        ++disk_errors;
        return result;
      }
      disk_s += watch.elapsed_s();
      watch.reset();
    } else {
      snapshot = on_disk;
    }
    Transaction tx = prepare(parsed.blocks, local);
    prepare_s += watch.elapsed_s();
    result.status = tx.status;
    if (tx.files.size() != 1 || tx.files[0].matches.size() != 1 || tx.files[0].original != on_disk) {
      std::fprintf(stderr, "  case %zu: prepare() did not see the file: %s %s\n", id, to_string(tx.status),
                   tx.message.c_str());
      return result;
    }
    const FileProposal& file = tx.files[0];
    result.rung = file.matches[0].kind;

    if (tx.status == Status::Ok) {
      if (file.updated == intended) {
        result.outcome = Outcome::Success;
      } else if (relaxed && differs_in_new_indentation_only(instance, file)) {
        result.outcome = Outcome::Deviation;
        if (reports++ < 5) report(instance, id, "other indentation of new lines", block.text, file, intended);
      } else {
        result.outcome = Outcome::Silent;
        if (reports++ < 5) report(instance, id, "silent misapplication", block.text, file, intended);
      }
    } else {
      result.outcome = tx.status == Status::NotFound    ? Outcome::NotFound
                       : tx.status == Status::Ambiguous ? Outcome::Ambiguous
                                                        : Outcome::Refused;
      if (file.updated != on_disk) result.outcome = Outcome::Silent;  // a failure proposes no change
    }
    if (through_disk) {
      watch.reset();
      round_trip(&tx, local, on_disk, id);
      disk_s += watch.elapsed_s();
    }
    return result;
  }

  // commit() writes what prepare() proposed, or nothing; revert() restores the bytes.
  void round_trip(Transaction* tx, const ApplyOptions& local, const std::string& on_disk, std::size_t id) {
    const Status proposed = tx->status;
    const std::string updated = tx->files[0].updated;
    const auto trouble = [&](const char* what, Status status) {
      std::fprintf(stderr, "  case %zu: %s gives %s %s\n", id, what, to_string(status), tx->message.c_str());
      ++(status == Status::IoError ? disk_errors : disk_mismatches);
    };
    const Status committed = commit(*tx, local);
    if (committed != proposed) return trouble("commit()", committed);
    if (edit_test::read_file(path) != (proposed == Status::Ok ? updated : on_disk)) {
      std::fprintf(stderr, "  case %zu: the file does not hold what commit() reported\n", id);
      ++disk_mismatches;
      return;
    }
    if (proposed != Status::Ok) return;
    const Status reverted = revert(*tx);
    if (reverted != Status::Ok) return trouble("revert()", reverted);
    if (edit_test::read_file(path) != on_disk) {
      std::fprintf(stderr, "  case %zu: revert() did not restore the file\n", id);
      ++disk_mismatches;
    }
  }

  static bool differs_in_new_indentation_only(const Instance& instance, const FileProposal& file) {
    Document got = parse(file.updated);
    for (std::string& line : got.lines) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
    }
    const Document want = intended_file(instance);
    if (got.lines.size() != want.lines.size()) return false;
    for (std::size_t i = 0; i < got.lines.size(); ++i) {
      const bool fresh = i >= instance.first && i < instance.first + instance.intended.size();
      if (got.lines[i] == want.lines[i]) continue;
      if (!fresh || without_indentation(got.lines[i]) != without_indentation(want.lines[i])) return false;
    }
    return true;
  }

  static void report(const Instance& instance, std::size_t id, const char* what, const std::string& block,
                     const FileProposal& file, const std::string& intended) {
    std::fprintf(stderr, "  %s in case %zu (%s, disturbances 0x%x, rung %s, target lines %zu-%zu)\n", what, id,
                 instance.source.c_str(), instance.disturbances, to_string(file.matches[0].kind), instance.first + 1,
                 instance.first + instance.count);
    std::fprintf(stderr, "--- block\n%s--- difference between the intended and the proposed content\n%s\n", block.c_str(),
                 unified_diff(intended, file.updated, "intended", "proposed", 2).c_str());
  }
};

using Table = std::map<std::string, Tally>;

void print_table(const char* title, const Table& table, bool with_deviation = false) {
  std::printf("  %-34s %8s %8s %9s %9s %8s %8s%s\n", title, "cases", "success", "not found", "ambiguous", "refused",
              "silent", with_deviation ? " deviation" : "");
  for (const auto& [name, tally] : table) {
    std::printf("  %-34s %8zu %8zu %9zu %9zu %8zu %8zu", name.c_str(), tally.total(), tally.success, tally.not_found,
                tally.ambiguous, tally.refused, tally.silent);
    if (with_deviation) std::printf(" %9zu", tally.deviation);
    std::printf("\n");
  }
}

std::string rung_name(const Result& result) {
  static const char* const kOrder[] = {"0 no rung", "1 exact", "2 trailing space", "3 indentation", "4 fuzzy", "5 unit"};
  switch (result.rung) {
    case MatchKind::None:
      return kOrder[0];
    case MatchKind::Exact:
      return kOrder[1];
    case MatchKind::TrailingSpace:
      return kOrder[2];
    case MatchKind::Indentation:
      return kOrder[3];
    case MatchKind::Fuzzy:
      return kOrder[4];
    case MatchKind::Unit:
      return kOrder[5];
  }
  return kOrder[0];
}

void add_by_disturbance(Table* table, unsigned disturbances, Outcome outcome) {
  for (const auto& [bit, name] : kDisturbanceNames) {
    if (disturbances & bit) (*table)[name].add(outcome);
  }
}

Tally sum(const Table& table) {
  Tally all;
  for (const auto& [name, tally] : table) {
    all.success += tally.success;
    all.not_found += tally.not_found;
    all.ambiguous += tally.ambiguous;
    all.refused += tally.refused;
    all.silent += tally.silent;
    all.deviation += tally.deviation;
  }
  return all;
}

}  // namespace

// ---- S1 ----------------------------------------------------------------------------------

PRAG_TEST(s1_no_silent_misapplication) {
  constexpr std::size_t kCases = 2400;
  const Stopwatch watch;
  Runner runner("s1");
  const std::vector<Source>& sources = load_sources();
  Random random(0x5EED0001);

  Table plain;         // by rung, without disturbance
  Table disturbed;     // by rung, with disturbance
  Table by_disturbance;
  std::size_t from_corpus = 0;
  std::size_t scoped = 0;
  std::size_t plain_unique = 0;
  std::size_t plain_unique_kept = 0;

  for (std::size_t id = 0; id < kCases; ++id) {
    Instance base = make_case(sources, id, random, 1, 8);
    const Document pool = base.file;
    const bool use_scope = random.chance(0.25);

    Instance undisturbed = base;
    if (use_scope) choose_scope(&undisturbed, random);
    const Result first = runner.run(undisturbed, id, random);
    plain[rung_name(first)].add(first.outcome);
    // Without disturbance the text is in the file: found, or found several times.
    CHECK(first.outcome == Outcome::Success || first.outcome == Outcome::Ambiguous);
    if (first.outcome == Outcome::Success) CHECK(first.rung == MatchKind::Exact);

    // One to four disturbances, at least one.
    unsigned which = 0;
    while (which == 0) {
      for (const unsigned bit : {kFileIndent, kFileTrailing, kFileCrlf, kFileInsert, kFileDelete, kBlockTrailing, kBlockCrlf}) {
        if (random.chance(0.22)) which |= bit;
      }
      if (random.chance(0.08)) which |= kFileRepeat;
      if (random.chance(0.30)) which |= random.chance(0.7) ? kBlockOutdent : kBlockIndent;
    }
    Instance instance = base;
    disturb(&instance, pool, which, id, random);
    if (use_scope) choose_scope(&instance, random);
    const Result second = runner.run(instance, id, random);
    disturbed[rung_name(second)].add(second.outcome);
    add_by_disturbance(&by_disturbance, instance.disturbances, second.outcome);

    if (base.from_corpus) ++from_corpus;
    if (use_scope) ++scoped;
    if (first.outcome == Outcome::Success && !(instance.disturbances & (kFileRepeat | kFileInsert))) {
      ++plain_unique;
      if (second.outcome == Outcome::Success) ++plain_unique_kept;
    }
  }

  const Tally all_plain = sum(plain);
  const Tally all_disturbed = sum(disturbed);
  std::printf("\n  S1: %zu cases, each run without and with disturbance: %zu runs of prepare(), %zu of them on a\n"
              "  file on disk and followed by commit() and revert(), the others on the same content in memory\n",
              kCases, runner.runs, runner.disk_runs);
  std::printf("  %zu cases on corpus files, %zu on embedded texts; %zu with a scope\n", from_corpus,
              kCases - from_corpus, scoped);
  std::printf("  %.1f s in total: %.1f s in prepare(), %.1f s on the disk\n\n", watch.elapsed_s(), runner.prepare_s,
              runner.disk_s);
  print_table("without disturbance, by rung", plain);
  std::printf("\n");
  print_table("with disturbance, by rung", disturbed);
  std::printf("\n");
  print_table("with disturbance, by disturbance", by_disturbance);
  std::printf("\n  totals without disturbance: %zu success, %zu clean failure, %zu silent\n", all_plain.success,
              all_plain.not_found + all_plain.ambiguous, all_plain.silent);
  std::printf("  totals with disturbance:    %zu success, %zu clean failure, %zu silent\n", all_disturbed.success,
              all_disturbed.not_found + all_disturbed.ambiguous, all_disturbed.silent);
  std::printf("  of %zu cases that are unique without disturbance and get no inserted or repeated lines, %zu succeed with disturbance\n\n",
              plain_unique, plain_unique_kept);

  // The property.
  CHECK_EQ(all_plain.silent, std::size_t{0});
  CHECK_EQ(all_disturbed.silent, std::size_t{0});
  // The experiment is what it claims to be.
  CHECK(kCases >= 2000);
  CHECK_EQ(all_plain.total(), kCases);
  CHECK_EQ(all_disturbed.total(), kCases);
  CHECK_EQ(runner.parser_errors, std::size_t{0});
  CHECK_EQ(all_plain.refused + all_disturbed.refused, std::size_t{0});
  CHECK_EQ(all_plain.not_found, std::size_t{0});
  // What commit() wrote is what prepare() proposed, and revert() took it back.
  CHECK(runner.disk_runs >= 40u);
  CHECK_EQ(runner.disk_mismatches, std::size_t{0});
  CHECK_EQ(runner.disk_errors, std::size_t{0});
  // Every rung is reached and succeeds, and failures occur as well.
  CHECK(disturbed["1 exact"].success > 100u);
  CHECK(disturbed["2 trailing space"].success > 100u);
  CHECK(disturbed["3 indentation"].success > 100u);
  CHECK(all_disturbed.ambiguous > 50u);
  CHECK(all_plain.ambiguous > 50u);
  for (const auto& [bit, name] : kDisturbanceNames) {
    if (bit == kFileConvert || bit == kBlockTypos) continue;
    CHECK(by_disturbance[name].total() > 100u);
    CHECK(by_disturbance[name].success > 0u || bit == kFileRepeat);
  }
  // Disturbance alone does not make the module give up.
  CHECK(plain_unique_kept * 10 >= plain_unique * 8);
}

// ---- files with converted indentation ----------------------------------------------------
//
// The file on disk has its indentation converted (every tab to four spaces, or
// every two spaces to a tab), the block has not: the model wrote it with the
// other kind of blank. The intended content carries the converted indentation
// on the new lines as well. No rule can derive the width of a tab, so the
// module must either have seen the indentation of a new line in the SEARCH
// text, where the file shows what it becomes, or refuse the block; the option
// ApplyOptions::keep_indent_style covers the lines that match without any
// indentation and give nothing to learn from. Content that differs from the
// intended one in the indentation of new lines only is counted apart from a
// misplaced edit, and neither may occur.
//
// A file without any indented line is the same before and after the
// conversion and does not show which blank it would be indented with. The new
// lines of such a case are intended as they are written.
PRAG_TEST(s1_converted_indentation) {
  constexpr std::size_t kCases = 800;
  const Stopwatch watch;
  Runner runner("s1convert");
  const std::vector<Source>& sources = load_sources();
  Random random(0x5EED0002);
  Table by_rung;
  std::size_t converted_targets = 0;
  std::size_t converted_files = 0;

  for (std::size_t id = 0; id < kCases; ++id) {
    Instance instance = make_case(sources, id, random, 1, 8);
    const Document pool = instance.file;
    unsigned which = 0;
    for (const unsigned bit : {kFileTrailing, kFileCrlf, kFileInsert, kFileDelete, kBlockTrailing}) {
      if (random.chance(0.15)) which |= bit;
    }
    if (random.chance(0.25)) which |= kBlockOutdent;
    disturb_structure(&instance, pool, which, id, random);
    const bool tabs_to_spaces = instance.unit == "\t";
    bool changed = false;  // a line of the target
    bool visible = false;  // a line of the file
    for (std::size_t i = 0; i < instance.file.lines.size(); ++i) {
      const std::string converted = convert_indentation(instance.file.lines[i], tabs_to_spaces);
      const bool target = i >= instance.first && i < instance.first + instance.count;
      if (converted != instance.file.lines[i]) {
        visible = true;
        if (target) changed = true;
      }
      instance.file.lines[i] = converted;
    }
    if (visible) {
      for (std::string& line : instance.intended) line = convert_indentation(line, tabs_to_spaces);
      ++converted_files;
    }
    instance.disturbances |= kFileConvert;
    if (changed) ++converted_targets;
    disturb(&instance, pool, which & ~(kFileInsert | kFileDelete | kFileRepeat), id, random);
    if (random.chance(0.25)) choose_scope(&instance, random);

    const Result result = runner.run(instance, id, random, true);
    by_rung[rung_name(result)].add(result.outcome);
  }

  const Tally all = sum(by_rung);
  std::printf("\n  converted indentation: %zu cases, %zu in files that show the conversion, %zu with converted target lines\n",
              kCases, converted_files, converted_targets);
  std::printf("  %zu runs through the disk; %.1f s in total: %.1f s in prepare(), %.1f s on the disk\n\n", runner.disk_runs,
              watch.elapsed_s(), runner.prepare_s, runner.disk_s);
  print_table("converted indentation, by rung", by_rung, true);
  std::printf("\n  totals: %zu success, %zu clean failure, %zu silent: %zu misplaced, %zu with other indentation of new lines\n\n",
              all.success, all.not_found + all.ambiguous, all.silent + all.deviation, all.silent, all.deviation);

  // The property: both kinds of wrong content are silent misapplications.
  CHECK_EQ(all.silent, std::size_t{0});
  CHECK_EQ(all.deviation, std::size_t{0});
  CHECK_EQ(all.refused, std::size_t{0});
  CHECK_EQ(all.total(), kCases);
  CHECK_EQ(runner.parser_errors, std::size_t{0});
  CHECK_EQ(runner.disk_mismatches + runner.disk_errors, std::size_t{0});
  CHECK(converted_files > 700u);
  CHECK(converted_targets > 300u);
  // Indentation that the SEARCH text shows is translated, the rest is refused.
  CHECK(by_rung["3 indentation"].success > 300u);
  CHECK(all.not_found > 50u);
}

// ---- typing errors under the fuzzy rung ----------------------------------------------------
//
// The fuzzy rung is enabled and the SEARCH text of at least four lines gets
// typing errors, fewer than the rung tolerates. The file is not disturbed.
PRAG_TEST(s1_typing_errors_under_the_fuzzy_rung) {
  constexpr std::size_t kCases = 600;
  const Stopwatch watch;
  Runner runner("s1fuzzy");
  runner.options.match.allow_fuzzy = true;
  const std::vector<Source>& sources = load_sources();
  Random random(0x5EED0003);
  Table by_rung;

  for (std::size_t id = 0; id < kCases; ++id) {
    Instance instance = make_case(sources, id, random, 4, 10);
    std::size_t characters = instance.search.size() - 1;
    for (const std::string& line : instance.search) characters += line.size();
    const std::size_t bound = characters / 10;
    const std::size_t errors = random.between(1, std::max<std::size_t>(1, bound / 2));
    for (std::size_t k = 0; k < errors; ++k) {
      std::string& line = instance.search[random.below(instance.search.size())];
      if (is_blank(line)) continue;
      // Inside the text of the line, so that no blank at either end is touched.
      const std::size_t low = indentation(line).size();
      const std::size_t at = low + random.below(line.size() - low);
      const char letter = static_cast<char>('a' + random.below(26));
      switch (random.below(3)) {
        case 0:
          line[at] = line[at] == letter ? '#' : letter;
          break;
        case 1:
          line.insert(line.begin() + static_cast<std::ptrdiff_t>(at), letter);
          break;
        default:
          if (line.size() - low > 1) line.erase(line.begin() + static_cast<std::ptrdiff_t>(at));
          break;
      }
    }
    if (std::any_of(instance.search.begin(), instance.search.end(), like_a_marker)) continue;
    instance.disturbances |= kBlockTypos;
    if (random.chance(0.25)) choose_scope(&instance, random);
    const Result result = runner.run(instance, id, random);
    by_rung[rung_name(result)].add(result.outcome);
  }

  const Tally all = sum(by_rung);
  std::printf("\n  typing errors, fuzzy rung enabled: %zu cases\n", all.total());
  std::printf("  %zu runs through the disk; %.1f s in total: %.1f s in prepare(), %.1f s on the disk\n\n", runner.disk_runs,
              watch.elapsed_s(), runner.prepare_s, runner.disk_s);
  print_table("typing errors, by rung", by_rung);
  std::printf("\n  totals: %zu success, %zu clean failure, %zu silent\n\n", all.success, all.not_found + all.ambiguous,
              all.silent);

  CHECK_EQ(all.silent, std::size_t{0});
  CHECK_EQ(all.refused, std::size_t{0});
  CHECK_EQ(runner.parser_errors, std::size_t{0});
  CHECK_EQ(runner.disk_mismatches + runner.disk_errors, std::size_t{0});
  CHECK(all.total() > 500u);
  CHECK(by_rung["4 fuzzy"].success > 300u);
}

