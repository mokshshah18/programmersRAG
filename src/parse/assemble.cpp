// From an outline to units.
//
// Whatever the front end found, the result is a partition of the file: every
// byte belongs to exactly one unit. The steps are
//   1. comments: one that shares a line with the end of a piece belongs to that
//      piece, a block directly above a piece is its documentation;
//   2. boundaries: a piece owns the rest of its last line, the piece after it
//      owns the blank lines between them;
//   3. grouping: functions, types and macros are units, the residue between
//      them is packed into regions of at most max_region_bytes;
//   4. splitting: what is still too large is cut at statement boundaries.
#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>

#include "lexer.hpp"
#include "outline.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"

namespace prag::parse::detail {

namespace {

inline bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v'; }

std::string_view slice(std::string_view text, std::size_t begin, std::size_t end) {
  end = std::min(end, text.size());
  begin = std::min(begin, end);
  return text.substr(begin, end - begin);
}

// Number of newlines in [begin, end), or npos when the range holds anything but white space.
std::size_t blank_lines_between(std::string_view text, std::size_t begin, std::size_t end) {
  std::size_t newlines = 0;
  for (const char c : slice(text, begin, end)) {
    if (!is_blank(c)) return std::string_view::npos;
    if (c == '\n') ++newlines;
  }
  return newlines;
}

struct Item {
  Piece piece;
  std::uint32_t lead = 0;   // first byte of the documentation comment, or of the piece
  std::uint32_t first = 0;  // first byte of the item: what opens it (`#if` lines above it), else `lead`
  std::uint32_t start = 0;  // the item owns [start, start of the next item)
  bool unit = false;        // becomes a unit of its own
};

// Structural residue that stands directly between units joins them: what
// closes goes to the unit before it, what opens to the unit after it.
std::vector<Item> join_residue(std::vector<Item> items) {
  const auto closes = [](const Item& item) {
    const Glue glue = item.piece.glue;
    return !item.unit && (glue == Glue::Endif || glue == Glue::Else || glue == Glue::Closer);
  };
  const auto opens = [](const Item& item) {
    const Glue glue = item.piece.glue;
    return !item.unit && (glue == Glue::If || glue == Glue::Else || glue == Glue::Opener);
  };

  // A brace inside a conditional of its own, as in
  //   #ifdef __cplusplus
  //   extern "C" {
  //   #endif
  // opens or closes as the brace does.
  std::vector<Item> whole;
  whole.reserve(items.size());
  for (std::size_t k = 0; k < items.size(); ++k) {
    const bool guarded = k + 2 < items.size() && !items[k].unit && items[k].piece.glue == Glue::If &&
                         !items[k + 1].unit &&
                         (items[k + 1].piece.glue == Glue::Opener || items[k + 1].piece.glue == Glue::Closer) &&
                         items[k + 1].first == items[k + 1].piece.begin && !items[k + 2].unit &&
                         items[k + 2].piece.glue == Glue::Endif && items[k + 2].first == items[k + 2].piece.begin;
    if (guarded) {
      Item item = std::move(items[k]);
      item.piece.glue = items[k + 1].piece.glue;
      item.piece.end = items[k + 2].piece.end;
      whole.push_back(std::move(item));
      k += 2;
    } else {
      whole.push_back(std::move(items[k]));
    }
  }

  std::vector<Item> joined;
  joined.reserve(whole.size());
  for (Item& item : whole) {
    if (closes(item) && !joined.empty() && joined.back().unit) {
      joined.back().piece.end = item.piece.end;
    } else {
      // after residual text, what closes is packed with that text
      if (closes(item) && !joined.empty()) item.piece.owner = joined.back().piece.owner;
      joined.push_back(std::move(item));
    }
  }
  // The head of a type holds what precedes its first member function, so the
  // access specifier that directly follows the head stays with the head.
  const auto follows_head = [&joined](std::size_t k) {
    if (k == 0 || joined[k].piece.glue != Glue::Opener || joined[k].piece.owner == 0) return false;
    const Item& above = joined[k - 1];
    return above.unit && above.piece.head && above.piece.owner == joined[k].piece.owner;
  };

  // Back to front, so that a run of openers reaches the unit below it.
  std::vector<Item> result;
  result.reserve(joined.size());
  bool below_is_unit = false;
  for (std::size_t k = joined.size(); k-- > 0;) {
    Item& item = joined[k];
    if (below_is_unit && opens(item) && !follows_head(k)) {
      result.back().first = item.first;
      continue;
    }
    below_is_unit = item.unit;
    result.push_back(std::move(item));
  }
  std::reverse(result.begin(), result.end());
  return result;
}

// Steps 1 and 2.
std::vector<Item> prepare(std::string_view text, std::vector<Piece> pieces, const ParseOptions& options) {
  std::stable_sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) { return a.begin < b.begin; });

  // Pieces never overlap and never reach beyond the text.
  std::vector<Piece> ordered;
  ordered.reserve(pieces.size());
  std::uint32_t floor = 0;
  for (Piece& piece : pieces) {
    piece.end = static_cast<std::uint32_t>(std::min<std::size_t>(piece.end, text.size()));
    piece.begin = std::max(piece.begin, floor);
    if (piece.end <= piece.begin) continue;
    piece.signature_end = std::clamp(piece.signature_end, piece.begin, piece.end);
    // A comment on the line where the previous piece ends belongs to that piece.
    if (piece.kind == PieceKind::Comment && !ordered.empty() &&
        slice(text, ordered.back().end, piece.begin).find('\n') == std::string_view::npos) {
      ordered.back().end = piece.end;
    } else {
      ordered.push_back(std::move(piece));
    }
    floor = ordered.back().end;
  }

  std::vector<Item> items;
  items.reserve(ordered.size());
  std::size_t comments = 0;  // comment pieces at the end of `items` that lie directly above one another
  for (Piece& piece : ordered) {
    Item item;
    item.lead = piece.begin;
    if (piece.kind == PieceKind::Comment) {
      const bool adjacent = comments > 0 && blank_lines_between(text, items.back().piece.end, piece.begin) <= 1;
      comments = adjacent ? comments + 1 : 1;
      piece.kind = PieceKind::Other;  // unless the piece below takes it
    } else {
      if (options.attach_leading_comment && comments > 0 &&
          blank_lines_between(text, items.back().piece.end, piece.begin) <= 1) {
        item.lead = items[items.size() - comments].piece.begin;
        items.resize(items.size() - comments);
      }
      comments = 0;
    }
    item.first = item.lead;
    const bool merged = options.merge_small_definitions && piece.small && !piece.callable;
    item.unit = piece.kind == PieceKind::Function ||
                ((piece.kind == PieceKind::Type || piece.kind == PieceKind::Macro) && !merged);
    item.piece = std::move(piece);
    items.push_back(std::move(item));
  }
  items = join_residue(std::move(items));

  for (std::size_t k = 1; k < items.size(); ++k) {
    const std::uint32_t end = items[k - 1].piece.end;
    const std::size_t newline = slice(text, end, items[k].first).find('\n');
    items[k].start = newline != std::string_view::npos ? static_cast<std::uint32_t>(end + newline + 1)
                                                       : items[k].first;
  }
  return items;
}

UnitKind unit_kind(PieceKind kind) {
  switch (kind) {
    case PieceKind::Function: return UnitKind::Function;
    case PieceKind::Type: return UnitKind::Type;
    case PieceKind::Macro: return UnitKind::Macro;
    default: return UnitKind::Region;
  }
}

struct Group {
  UnitKind kind = UnitKind::Region;
  std::string name;
  std::size_t first = 0;  // items [first, last]
  std::size_t last = 0;
  std::uint32_t owner = 0;  // the split type the group is a part of
  bool described = false;   // signature and documentation of items[first] apply
};

// ---- text clean-up ------------------------------------------------------------

// "----", "====", "****": decoration, not documentation.
bool is_rule(std::string_view s) {
  if (s.size() < 3) return false;
  for (const char c : s) {
    if (c != '-' && c != '=' && c != '*' && c != '/' && c != '#' && c != '_' && c != '~' && c != '+' && c != ' ') {
      return false;
    }
  }
  return true;
}

// Removes runs of three or more decoration characters from both ends.
std::string_view strip_rules(std::string_view s) {
  const auto decoration = [](char c) { return c == '-' || c == '=' || c == '*' || c == '#' || c == '~'; };
  std::size_t front = 0;
  while (front < s.size() && decoration(s[front]) && s[front] == s[0]) ++front;
  if (front >= 3) s.remove_prefix(front);
  std::size_t back = 0;
  while (back < s.size() && decoration(s[s.size() - 1 - back]) && s[s.size() - 1 - back] == s.back()) ++back;
  if (back >= 3) s.remove_suffix(back);
  return text::trim(s);
}

std::string join_lines(const std::vector<std::string_view>& lines) {
  std::size_t first = 0;
  std::size_t last = lines.size();
  while (first < last && lines[first].empty()) ++first;
  while (last > first && lines[last - 1].empty()) --last;
  std::string out;
  bool previous_empty = false;
  for (std::size_t i = first; i < last; ++i) {
    if (lines[i].empty() && previous_empty) continue;
    if (!out.empty()) out.push_back('\n');
    out.append(lines[i]);
    previous_empty = lines[i].empty();
  }
  return out;
}

// Comment text without comment markers and decoration.
std::string clean_comment(std::string_view raw) {
  std::vector<std::string_view> lines;
  bool in_block = false;
  for (std::string_view line : text::split_lines(raw)) {
    std::string_view s = text::trim(line);
    if (!s.empty() && s.back() == '\\') s = text::trim(s.substr(0, s.size() - 1));
    if (!in_block) {
      if (s.substr(0, 2) == "/*") {
        in_block = true;
        s.remove_prefix(2);
        while (!s.empty() && (s.front() == '*' || s.front() == '!') && s.substr(0, 2) != "*/") s.remove_prefix(1);
      } else if (s.substr(0, 2) == "//") {
        s.remove_prefix(2);
        while (!s.empty() && (s.front() == '/' || s.front() == '!')) s.remove_prefix(1);
        if (!s.empty() && s.front() == '<') s.remove_prefix(1);
      } else if (!s.empty() && s.front() == '#') {
        while (!s.empty() && (s.front() == '#' || s.front() == '!')) s.remove_prefix(1);
      }
    } else if (!s.empty() && s.front() == '*' && s.substr(0, 2) != "*/") {
      s.remove_prefix(1);
    }
    if (in_block) {
      const std::size_t close = s.find("*/");
      if (close != std::string_view::npos) {
        s = s.substr(0, close);
        in_block = false;
      }
    }
    s = text::trim(s);
    s = is_rule(s) ? std::string_view() : strip_rules(s);
    lines.push_back(s);
  }
  return join_lines(lines);
}

// Text of a Python documentation string without prefix and quotes.
std::string clean_docstring(std::string_view raw) {
  std::string_view s = text::trim(raw);
  while (!s.empty() && s.front() != '"' && s.front() != '\'') s.remove_prefix(1);  // r, b, f, u
  for (const std::string_view quote : {"\"\"\"", "'''", "\"", "'"}) {
    if (s.size() >= 2 * quote.size() && s.substr(0, quote.size()) == quote &&
        s.substr(s.size() - quote.size()) == quote) {
      s = s.substr(quote.size(), s.size() - 2 * quote.size());
      break;
    }
  }
  std::vector<std::string_view> lines;
  for (const std::string_view line : text::split_lines(s)) lines.push_back(text::trim(line));
  return join_lines(lines);
}

// Declaration text on one line: white space collapsed, comments and line splices removed.
std::string clean_signature(std::string_view raw, bool c_style) {
  std::string out;
  bool space = false;
  for (std::size_t i = 0; i < raw.size();) {
    const char c = raw[i];
    if (c_style && c == '/' && i + 1 < raw.size() && raw[i + 1] == '*') {
      const std::size_t close = raw.find("*/", i + 2);
      i = close == std::string_view::npos ? raw.size() : close + 2;
      space = true;
    } else if (c_style && c == '/' && i + 1 < raw.size() && raw[i + 1] == '/') {
      const std::size_t newline = raw.find('\n', i);
      i = newline == std::string_view::npos ? raw.size() : newline;
      space = true;
    } else if (c == '\\' && i + 1 < raw.size() && (raw[i + 1] == '\n' || raw[i + 1] == '\r')) {
      ++i;
      space = true;
    } else if (is_blank(c)) {
      ++i;
      space = true;
    } else {
      if (space && !out.empty()) out.push_back(' ');
      out.push_back(c);
      space = false;
      ++i;
    }
  }
  return out;
}

// ---- splitting --------------------------------------------------------------------

// Largest offset <= `at` that does not fall inside a UTF-8 sequence.
std::uint32_t character_boundary(std::string_view text, std::uint32_t at, std::uint32_t floor) {
  std::uint32_t p = at;
  for (int steps = 0; steps < 3 && p > floor && p < text.size(); ++steps) {
    if ((static_cast<unsigned char>(text[p]) & 0xC0) != 0x80) break;
    --p;
  }
  return p > floor ? p : at;
}

// Cuts [begin, end) into parts of at most `limit` bytes. Cut points are taken
// from the first level that offers one: statement boundaries of the body,
// nested statement boundaries, line starts, and as a last resort any character.
// Among the points of a level the one closest to an even division is chosen.
std::vector<std::uint32_t> cut_points(std::string_view text, std::uint32_t begin, std::uint32_t end, std::size_t limit,
                                      std::vector<std::uint32_t> levels[2]) {
  std::vector<std::uint32_t> cuts;
  if (end - begin <= limit) return cuts;
  std::vector<std::uint32_t> line_starts;
  for (std::uint32_t i = begin; i + 1 < end; ++i) {
    if (text[i] == '\n') line_starts.push_back(i + 1);
  }
  for (int level = 0; level < 2; ++level) {
    std::sort(levels[level].begin(), levels[level].end());
  }
  const std::vector<std::uint32_t>* sources[3] = {&levels[0], &levels[1], &line_starts};

  std::uint32_t pos = begin;
  while (end - pos > limit) {
    const std::size_t remaining = end - pos;
    const std::size_t parts = (remaining + limit - 1) / limit;
    const auto target = static_cast<std::uint32_t>(pos + (remaining + parts - 1) / parts);
    const auto high = static_cast<std::uint32_t>(pos + limit);
    std::uint32_t best = 0;
    // A part should not be much smaller than the others; give that up before giving up the level.
    for (const std::uint32_t low : {static_cast<std::uint32_t>(pos + limit / 4), pos}) {
      for (const std::vector<std::uint32_t>* source : sources) {
        const auto at = std::lower_bound(source->begin(), source->end(), target);
        std::uint32_t above = 0;
        std::uint32_t below = 0;
        if (at != source->end() && *at <= high && *at > low) above = *at;
        if (at != source->begin() && *(at - 1) > low && *(at - 1) <= high) below = *(at - 1);
        if (above != 0 && below != 0) {
          best = above - target < target - below ? above : below;
        } else {
          best = above != 0 ? above : below;
        }
        if (best != 0) break;
      }
      if (best != 0) break;
    }
    if (best == 0) best = character_boundary(text, high, pos);
    cuts.push_back(best);
    pos = best;
  }
  return cuts;
}

// ---- units ------------------------------------------------------------------------

// Name without the part number and without qualification: "A::f#2" -> "f".
std::string_view base_name(std::string_view name) {
  const std::size_t part = name.find('#');
  if (part != std::string_view::npos) name = name.substr(0, part);
  if (name.substr(0, 8) == "operator") return name;
  const std::size_t scope = name.rfind("::");
  if (scope != std::string_view::npos) return name.substr(scope + 2);
  const std::size_t dot = name.rfind('.');
  if (dot != std::string_view::npos) return name.substr(dot + 1);
  return name;
}

class Assembler {
 public:
  Assembler(std::string_view path, std::string_view language, std::string_view text, Outline outline,
            const ParseOptions& options)
      : path_(path),
        language_(language),
        text_(text),
        options_(options),
        lines_(text),
        calls_(std::move(outline.calls)),
        owners_(std::move(outline.owners)),
        parts_of_owner_(owners_.size() + 1, 1),
        items_(prepare(text, std::move(outline.pieces), options)) {}

  std::vector<SourceUnit> run() {
    if (text_.empty()) return {};
    if (items_.empty()) {  // white space only
      Item item;
      item.piece.end = static_cast<std::uint32_t>(text_.size());
      items_.push_back(std::move(item));
    }
    for (const Group& group : make_groups()) add_group(group);
    return std::move(units_);
  }

 private:
  std::uint32_t start_of(std::size_t k) const {
    return k < items_.size() ? items_[k].start : static_cast<std::uint32_t>(text_.size());
  }
  std::size_t size_of(std::size_t k) const { return start_of(k + 1) - start_of(k); }

  std::string part_name(std::uint32_t owner) {
    return owners_[owner - 1] + "#" + std::to_string(++parts_of_owner_[owner]);
  }

  // Step 3.
  std::vector<Group> make_groups() {
    std::vector<Group> groups;
    const std::size_t n = items_.size();
    std::size_t k = 0;
    while (k < n) {
      const Piece& piece = items_[k].piece;
      if (items_[k].unit) {
        Group group;
        group.kind = unit_kind(piece.kind);
        group.name = piece.name;
        group.first = group.last = k;
        group.described = true;
        if (piece.head) {
          // what precedes the first member function stays with the head of the type
          group.owner = piece.owner;
          std::size_t total = size_of(k);
          while (group.last + 1 < n && !items_[group.last + 1].unit &&
                 items_[group.last + 1].piece.owner == piece.owner &&
                 total + size_of(group.last + 1) <= options_.max_unit_bytes) {
            total += size_of(++group.last);
          }
        }
        k = group.last + 1;
        groups.push_back(std::move(group));
        continue;
      }
      std::size_t run_end = k;
      while (run_end < n && !items_[run_end].unit && items_[run_end].piece.owner == piece.owner) ++run_end;
      pack(k, run_end, piece.owner, groups);
      k = run_end;
    }
    return groups;
  }

  // Packs the residual items [first, last) into regions of about equal size.
  void pack(std::size_t first, std::size_t last, std::uint32_t owner, std::vector<Group>& groups) {
    const std::size_t limit = options_.max_region_bytes;
    const std::size_t total = start_of(last) - start_of(first);
    const std::size_t count = (total + limit - 1) / limit;
    const std::size_t target = count == 0 ? total : (total + count - 1) / count;
    const auto close = [&](std::size_t from, std::size_t to) {
      Group group;
      group.first = from;
      group.last = to;
      group.owner = owner;
      groups.push_back(std::move(group));
    };
    std::size_t from = first;
    std::size_t filled = 0;
    for (std::size_t k = first; k < last; ++k) {
      const std::size_t size = size_of(k);
      if (filled > 0 && (filled + size > limit || filled >= target)) {
        close(from, k - 1);
        from = k;
        filled = 0;
      }
      filled += size;
    }
    close(from, last - 1);
  }

  // Step 4 and the units themselves.
  void add_group(const Group& group) {
    const std::uint32_t begin = start_of(group.first);
    const std::uint32_t end = start_of(group.last + 1);
    const std::size_t limit = group.kind == UnitKind::Region ? options_.max_region_bytes : options_.max_unit_bytes;
    const Item* described = group.described ? &items_[group.first] : nullptr;

    std::string name = group.name;
    if (group.kind == UnitKind::Region && group.owner != 0) name = part_name(group.owner);

    std::vector<std::uint32_t> levels[2];
    if (end - begin > limit) {
      for (std::size_t k = group.first; k <= group.last; ++k) {
        for (int level = 0; level < 2; ++level) {
          for (const std::uint32_t at : items_[k].piece.breaks[level]) {
            if (at > begin && at < end) levels[level].push_back(at);
          }
        }
        if (k > group.first) levels[0].push_back(start_of(k));
      }
    }
    const std::vector<std::uint32_t> cuts = cut_points(text_, begin, end, limit, levels);

    std::uint32_t from = begin;
    for (std::size_t part = 0; part <= cuts.size(); ++part) {
      const std::uint32_t to = part < cuts.size() ? cuts[part] : end;
      if (part == 0) {
        add_unit(group.kind, name, from, to, described);
      } else if (group.owner != 0) {
        add_unit(UnitKind::Region, part_name(group.owner), from, to, nullptr);
      } else if (group.kind != UnitKind::Region && !group.name.empty()) {
        add_unit(UnitKind::Region, group.name + "#" + std::to_string(part + 1), from, to, nullptr);
      } else {
        add_unit(UnitKind::Region, "", from, to, nullptr);
      }
      from = to;
    }
  }

  void add_unit(UnitKind kind, std::string name, std::uint32_t begin, std::uint32_t end, const Item* described) {
    if (end <= begin) return;
    SourceUnit unit;
    unit.id = static_cast<UnitId>(units_.size());
    unit.path = std::string(path_);
    unit.language = std::string(language_);
    unit.kind = kind;
    unit.name = std::move(name);
    unit.text = std::string(slice(text_, begin, end));
    unit.start_byte = begin;
    unit.end_byte = end;
    unit.start_line = lines_.line_of(begin);
    unit.end_line = lines_.line_of(end - 1);
    if (described != nullptr) {
      const Piece& piece = described->piece;
      const bool c_style = language_ != "python";
      if (piece.signature_end > piece.begin) {
        unit.signature = clean_signature(slice(text_, piece.begin, piece.signature_end), c_style);
      }
      if (described->lead < piece.begin) unit.doc = clean_comment(slice(text_, described->lead, piece.begin));
      if (piece.doc_end > piece.doc_begin) {
        const std::string inner = clean_docstring(slice(text_, piece.doc_begin, piece.doc_end));
        if (!inner.empty()) {
          if (!unit.doc.empty()) unit.doc.push_back('\n');
          unit.doc += inner;
        }
      }
    }
    unit.callees = callees_in(begin, end, unit.name);
    unit.content_hash = hash64(unit.text);
    unit.tokens = text::estimate_tokens(unit.text);
    units_.push_back(std::move(unit));
  }

  std::vector<std::string> callees_in(std::uint32_t begin, std::uint32_t end, std::string_view own) const {
    std::vector<std::string> names;
    std::unordered_set<std::string_view> seen;
    const std::size_t part = own.find('#');
    const std::string_view full = part == std::string_view::npos ? own : own.substr(0, part);
    const std::string_view base = base_name(own);
    auto it = std::lower_bound(calls_.begin(), calls_.end(), begin,
                               [](const CallSite& call, std::uint32_t offset) { return call.offset < offset; });
    for (; it != calls_.end() && it->offset < end; ++it) {
      if (it->name.empty() || it->name == full || it->name == base) continue;
      if (seen.insert(it->name).second) names.emplace_back(it->name);
    }
    return names;
  }

  std::string_view path_;
  std::string_view language_;
  std::string_view text_;
  const ParseOptions& options_;
  LineIndex lines_;
  std::vector<CallSite> calls_;
  std::vector<std::string> owners_;
  std::vector<std::uint32_t> parts_of_owner_;
  std::vector<Item> items_;
  std::vector<SourceUnit> units_;
};

}  // namespace

std::vector<SourceUnit> assemble_units(std::string_view path, std::string_view language, std::string_view text,
                                       Outline outline, const ParseOptions& options) {
  return Assembler(path, language, text, std::move(outline), options).run();
}

}  // namespace prag::parse::detail
