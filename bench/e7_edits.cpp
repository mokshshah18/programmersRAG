// E7: inline edits. (a) the matching engine against constructed targets,
// (b) mechanical tasks with the serving model, (c) a build of the edited copy.
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>

#include "common.hpp"
#include "prag/core/clock.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/hash.hpp"
#include "prag/core/text.hpp"
#include "prag/edit/edit.hpp"
#include "prag/parse/parser.hpp"

namespace fsys = std::filesystem;

namespace bench {

namespace {

namespace ed = prag::edit;

// ---- line model ------------------------------------------------------------

struct Lines {
  std::vector<std::string> body, term;
  std::size_t size() const { return body.size(); }
};

Lines split_keep(const std::string& s) {
  Lines out;
  std::size_t i = 0;
  while (i < s.size()) {
    const std::size_t j = s.find('\n', i);
    if (j == std::string::npos) {
      out.body.push_back(s.substr(i));
      out.term.push_back("");
      break;
    }
    std::size_t e = j;
    std::string t = "\n";
    if (e > i && s[e - 1] == '\r') {
      --e;
      t = "\r\n";
    }
    out.body.push_back(s.substr(i, e - i));
    out.term.push_back(t);
    i = j + 1;
  }
  return out;
}

std::string join_all(const Lines& l) {
  std::string out;
  for (std::size_t i = 0; i < l.size(); ++i) out += l.body[i] + l.term[i];
  return out;
}

std::string join_range(const Lines& l, std::size_t first, std::size_t count, const char* eol = "\n") {
  std::string out;
  for (std::size_t i = 0; i < count; ++i) {
    if (i) out += eol;
    out += l.body[first + i];
  }
  return out;
}

bool blank(const std::string& s) { return prag::text::trim(s).empty(); }

std::string indent_more(const std::string& line) { return blank(line) ? line : "  " + line; }

std::string tabify(const std::string& line) {
  std::size_t spaces = 0;
  while (spaces < line.size() && line[spaces] == ' ') ++spaces;
  return std::string(spaces / 2, '\t') + line.substr(spaces / 2 * 2);
}

const char* kMarker = " /* bench */";

// ---- (a) matching engine ---------------------------------------------------

struct Outcome {
  int clean_success = 0, clean_failure = 0, misapplied = 0;
  std::map<std::string, std::array<int, 3>> by_rung;  // rung -> counts
  std::size_t cases = 0;
  void add(const std::string& rung, int kind) {
    ++cases;
    (kind == 0 ? clean_success : kind == 1 ? clean_failure : misapplied)++;
    by_rung[rung][static_cast<std::size_t>(kind)]++;
  }
};

json outcome_json(const Outcome& o) {
  json rungs = json::object();
  for (const auto& [rung, c] : o.by_rung) {
    const std::size_t n = static_cast<std::size_t>(c[0] + c[1] + c[2]);
    rungs[rung] = json{{"cases", n},
                       {"clean_success", to_json(wilson(static_cast<std::size_t>(c[0]), n))},
                       {"clean_failure", to_json(wilson(static_cast<std::size_t>(c[1]), n))},
                       {"silent_misapplication", to_json(wilson(static_cast<std::size_t>(c[2]), n))}};
  }
  return json{{"cases", o.cases},
              {"clean_success", to_json(wilson(static_cast<std::size_t>(o.clean_success), o.cases))},
              {"clean_failure", to_json(wilson(static_cast<std::size_t>(o.clean_failure), o.cases))},
              {"silent_misapplication", to_json(wilson(static_cast<std::size_t>(o.misapplied), o.cases))},
              {"by_deciding_rung", rungs}};
}

ed::ApplyOptions matching_options(const std::string& root, bool fuzzy) {
  ed::ApplyOptions o;
  o.allowed_roots = {root};
  o.match.allow_fuzzy = fuzzy;
  o.dry_run = true;
  o.keep_backup = false;
  o.unit_locator = [](std::string_view path, std::string_view text, std::string_view name, std::size_t* b, std::size_t* e) {
    return prag::parse::find_unit(path, text, name, b, e);
  };
  return o;
}

json matching_protocol(Context& ctx) {
  const auto& c = *ctx.corpus;
  const std::size_t want = ctx.quick() ? 12 : 250;
  const std::string case_dir = ctx.repo_dir + "/build/bench_edit/cases";
  std::error_code ec;
  fsys::remove_all(case_dir, ec);
  fsys::create_directories(case_dir);

  std::vector<UnitId> pool;
  for (const auto& u : c.units)
    if (u.kind == prag::UnitKind::Function && !u.name.empty() && u.name.find('#') == std::string::npos &&
        u.end_line >= u.start_line + 8 && u.tokens <= 1200)
      pool.push_back(u.id);
  shuffle(pool, ctx.seed + 71);

  const std::vector<std::string> perturbations = {"none", "unscoped", "reindent", "tabs", "trailing_blanks",
                                                  "crlf", "insert_above", "drift", "duplicate_target"};
  const std::vector<std::string> unit_perturbations = {"none", "reindent", "crlf", "insert_above"};
  std::map<std::string, Outcome> tally;  // key: form/config/perturbation
  json misapplications = json::array();
  std::size_t used = 0, skipped_units = 0;
  std::uint64_t state = ctx.seed + 72;

  for (const UnitId id : pool) {
    if (used >= want) break;
    const prag::SourceUnit& u = c.units[id];
    if (id + 1 < c.units.size() && c.units[id + 1].name.compare(0, u.name.size() + 1, u.name + "#") == 0) {
      ++skipped_units;  // a split function: the unit rewrite covers all its parts
      continue;
    }
    const auto text = prag::fs::read_file(u.path);
    if (!text) {
      ++skipped_units;
      continue;
    }
    const std::string base_text = *text;
    if (u.end_byte > base_text.size() || base_text.compare(u.start_byte, u.end_byte - u.start_byte, u.text) != 0) {
      ++skipped_units;
      continue;
    }
    const Lines base = split_keep(base_text);
    // A span of four lines inside the body, target = its second line.
    std::vector<std::size_t> starts;
    const std::size_t first_line = u.start_line - 1, last_line = u.end_line - 1;
    bool after_brace = false;
    for (std::size_t i = first_line; i <= last_line && i < base.size(); ++i) {
      if (!after_brace) {
        after_brace = base.body[i].find('{') != std::string::npos;
        continue;
      }
      if (i + 3 >= last_line) break;
      bool good = true;
      for (std::size_t k = 0; k < 4; ++k) {
        const std::string& line = base.body[i + k];
        if (!line.empty() && line.back() == '\\') good = false;
      }
      int braces = 0, parens = 0;
      for (std::size_t k = 0; k < 4; ++k)
        for (const char ch : base.body[i + k]) {
          braces += ch == '{' ? 1 : ch == '}' ? -1 : 0;
          parens += ch == '(' ? 1 : ch == ')' ? -1 : 0;
        }
      const std::string t = std::string(prag::text::trim(base.body[i + 1]));
      if (good && braces == 0 && parens == 0 && t.size() >= 6 && !blank(base.body[i])) starts.push_back(i);
    }
    if (starts.empty()) {
      ++skipped_units;
      continue;
    }
    const std::size_t s = starts[static_cast<std::size_t>(rng_below(state, starts.size()))];
    const std::size_t t = s + 1;
    ++used;
    const std::string fname = fsys::path(u.path).filename().string();

    auto run = [&](const std::string& form, const std::string& perturbation, const Lines& file, const ed::EditBlock& block,
                   const std::string& intended, bool expect_refusal) {
      const std::string path = case_dir + "/" + fname;
      prag::fs::write_file_atomic(path, join_all(file));
      for (const bool fuzzy : {false, true}) {
        ed::EditBlock b = block;
        b.path = fname;
        const ed::Transaction tx = ed::prepare({b}, matching_options(case_dir, fuzzy));
        int kind;
        std::string rung = "none";
        if (tx.status == ed::Status::Ok && tx.files.size() == 1) {
          const auto& f = tx.files[0];
          if (!f.matches.empty()) rung = ed::to_string(f.matches[0].kind);
          kind = (!expect_refusal && f.updated == intended) ? 0 : 2;
        } else {
          kind = 1;
        }
        tally[form + "/" + (fuzzy ? "fuzzy_on" : "fuzzy_off") + "/" + perturbation].add(rung, kind);
        if (kind == 2 && misapplications.size() < 40) {
          json m{{"function", u.name}, {"file", fname}, {"form", form}, {"fuzzy", fuzzy}, {"perturbation", perturbation}, {"rung", rung}};
          if (tx.files.size() == 1) {
            const auto& f = tx.files[0];
            if (!f.matches.empty()) m["matched_lines"] = json::array({f.matches[0].first_line, f.matches[0].last_line});
            std::string d = ed::unified_diff(intended, f.updated, "intended", "applied", 1);
            if (d.size() > 1200) d.resize(1200);
            m["diff_intended_vs_applied"] = d;
            m["block_search"] = block.search;
          }
          misapplications.push_back(std::move(m));
        }
      }
    };

    // Search/replace blocks.
    for (const std::string& p : perturbations) {
      Lines file = base;
      std::size_t span = s, target = t;
      if (p == "crlf") {
        for (auto& term : file.term)
          if (term == "\n") term = "\r\n";
      } else if (p == "insert_above") {
        for (int k = 0; k < 3; ++k) {
          file.body.insert(file.body.begin() + static_cast<std::ptrdiff_t>(span), "  // unrelated line " + std::to_string(k));
          file.term.insert(file.term.begin() + static_cast<std::ptrdiff_t>(span), base.term[s].empty() ? "\n" : base.term[s]);
        }
        span += 3;
        target += 3;
      } else if (p == "duplicate_target") {
        for (std::size_t k = 0; k < 4; ++k) {
          file.body.insert(file.body.begin() + static_cast<std::ptrdiff_t>(s + 4 + k), base.body[s + k]);
          file.term.insert(file.term.begin() + static_cast<std::ptrdiff_t>(s + 4 + k), base.term[s + k].empty() ? "\n" : base.term[s + k]);
        }
      }
      Lines search_lines, replace_lines;
      for (std::size_t k = 0; k < 4; ++k) {
        search_lines.body.push_back(file.body[span + k]);
        replace_lines.body.push_back(file.body[span + k]);
      }
      replace_lines.body[1] += kMarker;
      if (p == "reindent" || p == "tabs") {
        for (auto* l : {&search_lines, &replace_lines})
          for (auto& line : l->body) line = p == "reindent" ? indent_more(line) : tabify(line);
      } else if (p == "trailing_blanks") {
        for (auto& line : search_lines.body)
          if (!blank(line)) line += "  ";
      } else if (p == "drift") {
        std::string& line = search_lines.body[1];
        std::size_t best = std::string::npos, best_len = 0;
        for (std::size_t i = 0; i < line.size();) {
          if (!std::isalpha(static_cast<unsigned char>(line[i]))) { ++i; continue; }
          std::size_t j = i;
          while (j < line.size() && (std::isalnum(static_cast<unsigned char>(line[j])) || line[j] == '_')) ++j;
          if (j - i > best_len) { best = i; best_len = j - i; }
          i = j;
        }
        if (best != std::string::npos && best_len >= 4) line[best + 1] = line[best + 1] == 'q' ? 'z' : 'q';
      }
      Lines intended_lines = file;
      intended_lines.body[target] += kMarker;
      ed::EditBlock block;
      block.kind = ed::EditKind::SearchReplace;
      block.scope = p == "unscoped" ? "" : u.name;
      block.search = join_range(search_lines, 0, 4);
      block.replace = join_range(replace_lines, 0, 4);
      run("search_replace", p, file, block, join_all(intended_lines), p == "duplicate_target");
    }

    // Unit rewrite blocks.
    {
      std::size_t first = first_line, last = last_line;
      while (first < last && blank(base.body[first])) ++first;
      while (last > first && blank(base.body[last])) --last;
      for (const std::string& p : unit_perturbations) {
        Lines file = base;
        std::size_t shift = 0;
        if (p == "crlf") {
          for (auto& term : file.term)
            if (term == "\n") term = "\r\n";
        } else if (p == "insert_above") {
          // Three unrelated comment lines and a blank line, so that the comment does not attach to the unit.
          const std::string eol = base.term[first_line].empty() ? "\n" : base.term[first_line];
          file.body.insert(file.body.begin() + static_cast<std::ptrdiff_t>(first_line), "");
          file.term.insert(file.term.begin() + static_cast<std::ptrdiff_t>(first_line), eol);
          for (int k = 2; k >= 0; --k) {
            file.body.insert(file.body.begin() + static_cast<std::ptrdiff_t>(first_line), "// unrelated line " + std::to_string(k));
            file.term.insert(file.term.begin() + static_cast<std::ptrdiff_t>(first_line), eol);
          }
          shift = 4;
        }
        Lines fresh;
        for (std::size_t i = first; i <= last; ++i) fresh.body.push_back(file.body[i + shift]);
        fresh.body[t - first] += kMarker;
        if (p == "reindent")
          for (auto& line : fresh.body) line = indent_more(line);
        Lines intended_lines = file;
        intended_lines.body[t + shift] += kMarker;
        ed::EditBlock block;
        block.kind = ed::EditKind::UnitRewrite;
        block.scope = u.name;
        block.replace = join_range(fresh, 0, fresh.size());
        run("unit_rewrite", p, file, block, join_all(intended_lines), false);
      }
    }
    if (used % 25 == 0) progress("E7a %zu / %zu functions", used, want);
  }

  json cells = json::object();
  for (const auto& [key, outcome] : tally) cells[key] = outcome_json(outcome);
  json totals = json::object();
  for (const std::string form : {"search_replace", "unit_rewrite"})
    for (const std::string config : {"fuzzy_off", "fuzzy_on"}) {
      Outcome all;
      for (const auto& [key, outcome] : tally)
        if (key.rfind(form + "/" + config + "/", 0) == 0) {
          all.clean_success += outcome.clean_success;
          all.clean_failure += outcome.clean_failure;
          all.misapplied += outcome.misapplied;
          all.cases += outcome.cases;
          for (const auto& [rung, counts] : outcome.by_rung)
            for (std::size_t k = 0; k < 3; ++k) all.by_rung[rung][k] += counts[k];
        }
      totals[form + "/" + config] = outcome_json(all);
    }
  return json{{"functions", used},
              {"functions_skipped", skipped_units},
              {"span_lines", 4},
              {"change", "append a block comment to the second line of the span (unit rewrite: to the same line of the unit)"},
              {"classification", json{{"clean_success", "applied and the file equals the intended result"},
                                       {"clean_failure", "refused, nothing written"},
                                       {"silent_misapplication", "applied but the file differs from the intended result, or applied where refusal was the only correct answer (duplicate_target)"}}},
              {"perturbations", json{{"none", "exact block"},
                                     {"unscoped", "no scope line"},
                                     {"reindent", "SEARCH and REPLACE indented by two more spaces"},
                                     {"tabs", "SEARCH and REPLACE indented with tabs"},
                                     {"trailing_blanks", "two trailing blanks on every SEARCH line"},
                                     {"crlf", "file converted to CRLF, block written with LF"},
                                     {"insert_above", "three unrelated lines inserted above the target (for unit rewrites: above the unit, followed by a blank line)"},
                                     {"drift", "one letter of the longest word of the second SEARCH line changed"},
                                     {"duplicate_target", "the span occurs twice in the function; refusal is the only correct outcome"}}},
              {"totals", totals},
              {"cells", cells},
              {"silent_misapplications", misapplications}};
}

// ---- (b)/(c) the copy of the corpus ----------------------------------------

void copy_tree(const fsys::path& from, const fsys::path& to) {
  fsys::create_directories(to);
  for (auto it = fsys::recursive_directory_iterator(from); it != fsys::recursive_directory_iterator(); ++it) {
    const fsys::path rel = fsys::relative(it->path(), from);
    const std::string top = rel.begin()->string();
    if (top == ".git" || top == "build") {
      if (it->is_directory()) it.disable_recursion_pending();
      continue;
    }
    if (it->is_directory())
      fsys::create_directories(to / rel);
    else
      fsys::copy_file(it->path(), to / rel, fsys::copy_options::overwrite_existing);
  }
}

struct Copy {
  std::string dir;
  std::vector<std::string> roots;  // corpus roots inside the copy
};

Copy make_copy(Context& ctx) {
  Copy copy;
  copy.dir = ctx.repo_dir + "/build/bench_edit/m";
  std::error_code ec;
  fsys::remove_all(copy.dir, ec);
  copy_tree(ctx.demo_dir, copy.dir);
  for (const std::string& r : ctx.corpus->roots) {
    if (r.compare(0, ctx.demo_dir.size(), ctx.demo_dir) == 0) copy.roots.push_back(copy.dir + r.substr(ctx.demo_dir.size()));
  }
  if (copy.roots.empty()) copy.roots.push_back(copy.dir);
  return copy;
}

ed::ApplyOptions task_options(const Copy& copy, bool dry_run) {
  ed::ApplyOptions o;
  o.allowed_roots = copy.roots;
  o.dry_run = dry_run;
  o.keep_backup = true;
  o.backup_dir = copy.dir + "/.edit_backup";
  o.syntax_check = [](std::string_view path, std::string_view text) {
    return prag::parse::count_syntax_errors(prag::parse::detect_language(path, text), text);
  };
  o.unit_locator = [](std::string_view path, std::string_view text, std::string_view name, std::size_t* b, std::size_t* e) {
    return prag::parse::find_unit(path, text, name, b, e);
  };
  return o;
}

// ---- tasks -----------------------------------------------------------------

struct Task {
  std::string type;
  UnitId unit = prag::kNoUnit;
  std::string instruction;
  std::string file_path;     // in the original corpus
  std::string relative;      // path shown to the model
  std::string target_text;   // intended content of the whole file
};

bool word_char(char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; }

std::string replace_word(const std::string& text, const std::string& from, const std::string& to, bool call_only) {
  std::string out;
  std::size_t i = 0;
  while (i < text.size()) {
    if (text.compare(i, from.size(), from) == 0 && (i == 0 || !word_char(text[i - 1])) &&
        (i + from.size() >= text.size() || !word_char(text[i + from.size()]))) {
      std::size_t j = i + from.size();
      if (!call_only || (j < text.size() && text[j] == '(')) {
        out += to;
        i = j;
        continue;
      }
    }
    out += text[i++];
  }
  return out;
}

std::size_t count_word(const std::string& text, const std::string& word, bool call_only) {
  std::size_t n = 0;
  for (std::size_t i = 0; i + word.size() <= text.size(); ++i) {
    if (text.compare(i, word.size(), word) == 0 && (i == 0 || !word_char(text[i - 1])) &&
        (i + word.size() >= text.size() || !word_char(text[i + word.size()])) &&
        (!call_only || (i + word.size() < text.size() && text[i + word.size()] == '('))) ++n;
  }
  return n;
}

// Offset just after the line that ends in the opening brace of the body; npos when the layout is unusual.
std::size_t body_start(const std::string& text) {
  const std::string rest = strip_leading_comment(text, "c");
  const std::size_t skip = text.size() - rest.size();
  const std::size_t brace = text.find('{', skip);
  if (brace == std::string::npos) return std::string::npos;
  const std::size_t eol = text.find('\n', brace);
  if (eol == std::string::npos) return std::string::npos;
  for (std::size_t i = brace + 1; i < eol; ++i)
    if (!std::isspace(static_cast<unsigned char>(text[i]))) return std::string::npos;
  return eol + 1;
}

bool make_task(const prag::SourceUnit& u, const std::string& type, const std::string& relative, Task* task) {
  const std::string& text = u.text;
  const std::size_t body = body_start(text);
  if (body == std::string::npos) return false;
  const std::string head = text.substr(0, body), tail = text.substr(body);
  std::string modified;
  task->type = type;
  task->unit = u.id;
  task->file_path = u.path;
  task->relative = relative;
  if (type == "comment") {
    std::size_t k = 0;
    while (k < tail.size() && (tail[k] == ' ' || tail[k] == '\t')) ++k;
    const std::string indent = tail.substr(0, k);
    const std::string eol = text.find("\r\n") != std::string::npos ? "\r\n" : "\n";
    modified = head + indent + "// reviewed" + eol + tail;
    task->instruction = "Insert the comment line `// reviewed` as the first line of the body of the function `" + u.name + "`.";
  } else if (type == "rename_local") {
    static const std::regex decl(R"(\b(?:unsigned\s+)?(?:int|size_t|bool|char|long|uint\d+_t|intptr_t|uintptr_t|mi_\w+_t)\s*\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;))");
    std::smatch m;
    std::string name;
    for (auto it = std::sregex_iterator(tail.begin(), tail.end(), decl); it != std::sregex_iterator(); ++it) {
      const std::string candidate = (*it)[1];
      if (count_word(tail, candidate, false) >= 2 && tail.find(candidate + "_v") == std::string::npos) {
        name = candidate;
        break;
      }
    }
    if (name.empty()) return false;
    modified = head + replace_word(tail, name, name + "_v", false);
    task->instruction = "In the function `" + u.name + "`, rename the local variable `" + name + "` to `" + name + "_v`.";
  } else if (type == "numeric_literal") {
    const Lines lines = split_keep(tail);
    bool done = false;
    for (std::size_t li = 0; li < lines.size() && !done; ++li) {
      const std::string& line = lines.body[li];
      const std::string trimmed(prag::text::trim(line));
      if (trimmed.rfind("//", 0) == 0 || trimmed.rfind("*", 0) == 0 || trimmed.rfind("/*", 0) == 0 || trimmed.rfind("#", 0) == 0) continue;
      std::vector<std::pair<std::size_t, std::size_t>> literals;
      for (std::size_t i = 0; i < line.size();) {
        if (!std::isdigit(static_cast<unsigned char>(line[i]))) { ++i; continue; }
        std::size_t j = i;
        while (j < line.size() && std::isdigit(static_cast<unsigned char>(line[j]))) ++j;
        const bool left = i == 0 || !(word_char(line[i - 1]) || line[i - 1] == '.');
        const bool right = j >= line.size() || !(word_char(line[j]) || line[j] == '.');
        if (left && right && j - i <= 3 && line.substr(i, j - i) != "0") literals.emplace_back(i, j);
        i = j;
      }
      std::size_t occurrences = 0;
      for (std::size_t lj = 0; lj < lines.size(); ++lj) occurrences += lines.body[lj] == line ? 1 : 0;
      if (literals.size() != 1 || occurrences != 1) continue;
      const auto [b, e] = literals[0];
      const std::string old_value = line.substr(b, e - b);
      const std::string new_value = std::to_string(std::stoi(old_value) + 1);
      Lines changed = lines;
      changed.body[li] = line.substr(0, b) + new_value + line.substr(e);
      modified = head + join_all(changed);
      task->instruction = "In the function `" + u.name + "`, change the numeric literal " + old_value + " to " + new_value +
                          " on the line `" + trimmed + "`.";
      done = true;
    }
    if (!done) return false;
  } else {  // replace_call
    std::string callee;
    for (const std::string& c : u.callees)
      if (c.size() >= 3 && count_word(tail, c, true) >= 1 && tail.find(c + "_checked") == std::string::npos &&
          count_word(head, c, false) == 0) {
        callee = c;
        break;
      }
    if (callee.empty()) return false;
    modified = head + replace_word(tail, callee, callee + "_checked", true);
    task->instruction = "In the function `" + u.name + "`, replace every call to `" + callee + "` by a call to `" + callee +
                        "_checked` and keep the arguments.";
  }
  const auto file_text = prag::fs::read_file(u.path);
  if (!file_text || u.end_byte > file_text->size() || file_text->compare(u.start_byte, u.end_byte - u.start_byte, u.text) != 0) return false;
  task->target_text = file_text->substr(0, u.start_byte) + modified + file_text->substr(u.end_byte);
  return task->target_text != *file_text;
}

struct TaskResult {
  Task task;
  bool grammar = false;
  bool parsed = false, applied = false, equal = false, regression = false;
  std::string status;
  double tokens = 0, ms = 0;
  std::string answer;
};

json tasks_with_model(Context& ctx, const Copy& copy, std::vector<TaskResult>* results) {
  const auto chat = ctx.chat();
  if (!chat) return skipped("no chat model loaded");
  const auto& c = *ctx.corpus;
  const std::size_t per_type = ctx.quick() ? 2 : 25;
  const std::string front = c.roots.empty() ? std::string() : c.roots.front() + "/";
  std::vector<UnitId> pool;
  for (const auto& u : c.units)
    if (u.kind == prag::UnitKind::Function && !u.name.empty() && u.name.find('#') == std::string::npos && u.tokens <= 300 &&
        u.end_line >= u.start_line + 4 && !front.empty() && u.path.compare(0, front.size(), front) == 0)
      pool.push_back(u.id);
  shuffle(pool, ctx.seed + 81);
  std::vector<Task> tasks;
  std::vector<char> used(c.units.size(), 0);
  for (const std::string type : {"comment", "rename_local", "numeric_literal", "replace_call"}) {
    std::size_t made = 0;
    for (const UnitId id : pool) {
      if (made >= per_type) break;
      if (used[id]) continue;
      Task task;
      if (make_task(c.units[id], type, c.units[id].path.substr(front.size()), &task)) {
        tasks.push_back(std::move(task));
        used[id] = 1;
        ++made;
      }
    }
  }
  progress("E7b: %zu tasks", tasks.size());
  const ed::ApplyOptions options = task_options(copy, true);
  std::size_t done = 0;
  for (const Task& task : tasks) {
    for (const bool grammar : {false, true}) {
      TaskResult r;
      r.task = task;
      r.grammar = grammar;
      prag::llm::GenerateOptions go;
      go.max_tokens = 768;
      go.temperature = 0.0f;
      if (grammar) go.grammar = std::string(ed::edit_grammar());
      const auto messages = messages_for(ctx, task.instruction, {task.unit}, true);
      const prag::llm::GenerateStats g = chat->generate(messages, go, [&](std::string_view piece) {
        r.answer += piece;
        return true;
      });
      r.tokens = g.completion_tokens;
      r.ms = g.prompt_ms + g.completion_ms;
      const ed::ParseResult parsed = ed::parse_edit_blocks(r.answer);
      r.parsed = !parsed.blocks.empty();
      if (r.parsed) {
        const ed::Transaction tx = ed::prepare(parsed.blocks, options);
        r.status = ed::to_string(tx.status);
        r.applied = tx.status == ed::Status::Ok;
        r.regression = tx.status == ed::Status::SyntaxRegression;
        for (const auto& f : tx.files) r.regression = r.regression || f.status == ed::Status::SyntaxRegression;
        r.equal = r.applied && tx.files.size() == 1 && tx.files[0].updated == task.target_text;
      } else {
        r.status = "unparsed";
      }
      results->push_back(std::move(r));
    }
    progress("E7b task %zu / %zu", ++done, tasks.size());
  }
  json cells = json::object();
  for (const std::string type : {"comment", "rename_local", "numeric_literal", "replace_call"}) {
    for (const bool grammar : {false, true}) {
      std::size_t n = 0, parsed = 0, applied = 0, equal = 0, regression = 0;
      std::vector<double> tokens, ms;
      std::map<std::string, std::size_t> statuses;
      for (const TaskResult& r : *results) {
        if (r.task.type != type || r.grammar != grammar) continue;
        ++n;
        parsed += r.parsed;
        applied += r.applied;
        equal += r.equal;
        regression += r.regression;
        tokens.push_back(r.tokens);
        ms.push_back(r.ms);
        ++statuses[r.status];
      }
      if (n == 0) continue;
      cells[type + (grammar ? "/grammar" : "/free")] =
          json{{"tasks", n},
               {"parses", to_json(wilson(parsed, n))},
               {"applies", to_json(wilson(applied, n))},
               {"equals_target", to_json(wilson(equal, n))},
               {"syntax_regression_caught", to_json(wilson(regression, n))},
               {"tokens_generated_mean", to_json(bootstrap_mean(tokens, ctx.seed + 82))},
               {"time_ms_mean", to_json(bootstrap_mean(ms, ctx.seed + 82))},
               {"status_counts", statuses}};
    }
  }
  json details = json::array();
  for (const TaskResult& r : *results)
    details.push_back(json{{"type", r.task.type}, {"unit", r.task.unit}, {"grammar", r.grammar}, {"instruction", r.task.instruction},
                           {"status", r.status}, {"parsed", r.parsed}, {"applied", r.applied}, {"equals_target", r.equal},
                           {"tokens", r.tokens}, {"ms", r.ms}, {"answer_head", r.answer.substr(0, 600)}});
  return json{{"tasks", tasks.size()}, {"cells", cells}, {"details", details}};
}

// ---- (c) build check -------------------------------------------------------

int run_command(const std::string& command, const std::string& log) {
  const std::string full = command + " > \"" + log + "\" 2>&1";
  return std::system(full.c_str());
}

std::string tail_of(const std::string& path, std::size_t lines) {
  std::ifstream f(path, std::ios::binary);
  std::vector<std::string> all;
  std::string line;
  while (std::getline(f, line)) all.push_back(line);
  std::string out;
  for (std::size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); ++i) out += all[i] + "\n";
  return out;
}

json build_copy(const std::string& dir, const std::string& label) {
  const std::string build = dir + "/build";
  std::error_code ec;
  fsys::remove_all(build, ec);
  fsys::create_directories(build);
  const std::string log1 = dir + "/configure_" + label + ".log", log2 = dir + "/build_" + label + ".log";
  prag::Stopwatch watch;
  const int configure = run_command("cmake -S \"" + dir + "\" -B \"" + build + "\" -DCMAKE_BUILD_TYPE=Release -DMI_BUILD_TESTS=OFF", log1);
  const double configure_s = watch.elapsed_s();
  json out{{"configure_exit_code", configure}, {"configure_seconds", configure_s}};
  if (configure != 0) {
    out["built"] = false;
    out["configure_log_tail"] = tail_of(log1, 15);
    return out;
  }
  watch.reset();
  const int compile = run_command("cmake --build \"" + build + "\" --config Release", log2);
  out["build_exit_code"] = compile;
  out["build_seconds"] = watch.elapsed_s();
  out["built"] = compile == 0;
  if (compile != 0) out["build_log_tail"] = tail_of(log2, 25);
  return out;
}

json build_check(Context& ctx, const Copy& copy, const std::vector<TaskResult>& results) {
  // Distinct cases whose answer equalled the target, applied one after the other.
  std::map<UnitId, const TaskResult*> winners;
  for (const TaskResult& r : results)
    if (r.equal && !winners.count(r.task.unit)) winners[r.task.unit] = &r;
  const ed::ApplyOptions options = task_options(copy, false);
  std::size_t applied = 0, failed = 0;
  json failures = json::array();
  for (const auto& [unit, r] : winners) {
    const ed::ParseResult parsed = ed::parse_edit_blocks(r->answer);
    ed::Transaction tx = ed::prepare(parsed.blocks, options);
    if (tx.status == ed::Status::Ok && ed::commit(tx, options) == ed::Status::Ok) {
      ++applied;
    } else {
      ++failed;
      failures.push_back(json{{"unit", unit}, {"status", ed::to_string(tx.status)}});
    }
  }
  json out{{"edits_equal_to_target_in_b", winners.size()}, {"edits_applied_to_copy", applied}, {"edits_failed_to_apply", failed},
           {"apply_failures", failures},
           {"configure_options", "cmake -DCMAKE_BUILD_TYPE=Release -DMI_BUILD_TESTS=OFF"}};
  if (std::system("cmake --version > nul 2>&1") != 0 && std::system("cmake --version > /dev/null 2>&1") != 0) {
    out["skipped"] = "cmake is not available on PATH";
    return out;
  }
  out["edited_copy"] = build_copy(copy.dir, "edited");
  if (!out["edited_copy"].value("built", false) && applied > 0) {
    // Distinguish a broken edit from a broken environment: revert nothing, build a pristine copy.
    Copy pristine;
    pristine.dir = ctx.repo_dir + "/build/bench_edit/p";
    std::error_code ec;
    fsys::remove_all(pristine.dir, ec);
    copy_tree(ctx.demo_dir, pristine.dir);
    out["pristine_copy"] = build_copy(pristine.dir, "pristine");
  }
  return out;
}

}  // namespace

int run_e7(Context& ctx) {
  const std::string started = iso_now();
  json body = json::object();
  body["a_matching"] = guarded("E7a matching engine", [&] { return matching_protocol(ctx); });
  std::vector<TaskResult> results;
  Copy copy;
  bool have_copy = false;
  json copy_error;
  try {
    progress("E7: copying the corpus under build/ ...");
    copy = make_copy(ctx);
    have_copy = true;
  } catch (const std::exception& e) {
    copy_error = json{{"error", e.what()}};
  }
  if (have_copy) {
    body["b_model"] = guarded("E7b model tasks", [&] { return tasks_with_model(ctx, copy, &results); });
    body["c_build"] = guarded("E7c build check", [&] { return build_check(ctx, copy, results); });
  } else {
    body["b_model"] = copy_error;
    body["c_build"] = copy_error;
  }
  write_result(ctx, "e7_edits", started,
               json{{"a_cases_per_function", "9 search/replace perturbations x 2 fuzzy settings, 4 unit-rewrite perturbations x 2"},
                    {"a_functions", ctx.quick() ? 12 : 250},
                    {"b_tasks_per_type", ctx.quick() ? 2 : 25},
                    {"b_temperature", 0.0},
                    {"b_max_tokens", 768},
                    {"b_grammar", "edit::edit_grammar() on and off"},
                    {"copy_directory", "build/bench_edit/m"},
                    {"demo_repository_modified", false}},
               std::move(body));
  return 0;
}

}  // namespace bench
