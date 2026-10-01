#include "prag/parse/parser.hpp"

#include <algorithm>
#include <exception>
#include <new>

#include "outline.hpp"
#include "prag/core/fs.hpp"
#include "prag/core/log.hpp"
#include "prag/core/text.hpp"
#include "syntax_tree.hpp"

namespace prag::parse {

namespace {

// Offsets are 32 bit wide (SourceUnit, tree-sitter).
constexpr std::size_t kMaxTextBytes = 0xFFFFFFF0u;

std::string extension_of(std::string_view path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::string_view file = slash == std::string_view::npos ? path : path.substr(slash + 1);
  const std::size_t dot = file.rfind('.');
  if (dot == std::string_view::npos || dot == 0) return {};
  return text::to_lower(file.substr(dot));
}

ParseOptions checked(ParseOptions options) {
  // A limit of zero would ask for units without content.
  options.max_region_bytes = std::max<std::size_t>(options.max_region_bytes, 16);
  options.max_unit_bytes = std::max<std::size_t>(options.max_unit_bytes, 16);
  return options;
}

// "A::f#2" and "A.f" are parts and members of "A".
bool belongs_to(std::string_view name, std::string_view owner) {
  if (name.size() <= owner.size() || name.substr(0, owner.size()) != owner) return false;
  const std::string_view rest = name.substr(owner.size());
  return rest[0] == '#' || rest[0] == '.' || rest.substr(0, 2) == "::";
}

std::string_view last_component(std::string_view name) {
  if (name.substr(0, 8) == "operator") return name;
  const std::size_t scope = name.rfind("::");
  if (scope != std::string_view::npos) return name.substr(scope + 2);
  const std::size_t dot = name.rfind('.');
  if (dot != std::string_view::npos) return name.substr(dot + 1);
  return name;
}

}  // namespace

std::string detect_language(std::string_view path, std::string_view text) {
  const std::string extension = extension_of(path);
  if (extension == ".c") return "c";
  if (extension == ".h") return detail::has_cpp_constructs(text) ? "cpp" : "c";
  if (extension == ".cc" || extension == ".cpp" || extension == ".cxx" || extension == ".hpp" || extension == ".hh") {
    return "cpp";
  }
  if (extension == ".py") return "python";
  return {};
}

ParsedFile parse_source(std::string_view path, std::string_view text, const ParseOptions& requested) {
  const ParseOptions options = checked(requested);
  ParsedFile file;
  file.path = fs::normalize(path);
  if (text.size() > kMaxTextBytes) {
    PRAG_WARN("parse: %s is larger than 4 GiB and was not parsed", file.path.c_str());
    file.had_syntax_errors = true;
    return file;
  }
  file.language = detect_language(path, text);
  try {
    detail::Outline outline;
    if (file.language.empty()) {
      outline = detail::outline_plain(text);
    } else {
      const detail::SyntaxTree tree(file.language, text);
      file.had_syntax_errors = tree.has_errors();
      outline = file.language == "python" ? detail::outline_python(text, tree, options)
                                          : detail::outline_c_family(text, file.language == "cpp", tree, options);
    }
    file.units = detail::assemble_units(file.path, file.language, text, std::move(outline), options);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    // Not expected. The text still has to end up in units.
    PRAG_WARN("parse: %s: %s; falling back to line regions", file.path.c_str(), e.what());
    file.had_syntax_errors = true;
    file.units = detail::assemble_units(file.path, file.language, text, detail::outline_plain(text), options);
  }
  return file;
}

std::size_t count_syntax_errors(std::string_view language, std::string_view text) {
  if (detail::grammar_for(language) == nullptr) return 0;
  return detail::SyntaxTree(language, text).count_errors();
}

bool find_unit(std::string_view path, std::string_view text, std::string_view name, std::size_t* begin,
               std::size_t* end) {
  if (name.empty()) return false;
  const ParsedFile file = parse_source(path, text);
  const std::vector<SourceUnit>& units = file.units;

  // An exact name before the name of a member, a function before anything else.
  // The first tier with a match decides; when it holds more than one unit (for
  // example two definitions in alternative branches of #if), the name does not
  // say which one is meant, and the lookup fails rather than guess.
  const SourceUnit* found = nullptr;
  std::size_t matches = 0;
  for (const bool exact : {true, false}) {
    for (const bool functions : {true, false}) {
      if (matches != 0) break;
      for (const SourceUnit& unit : units) {
        if (unit.name.empty() || unit.name.find('#') != std::string::npos) continue;
        if (functions != (unit.kind == UnitKind::Function)) continue;
        if (exact ? unit.name == name : last_component(unit.name) == name) {
          if (found == nullptr) found = &unit;
          ++matches;
        }
      }
    }
    if (matches != 0) break;
  }
  if (found == nullptr || matches > 1) return false;

  // A unit that was split, or a type whose members are units, extends over its parts.
  std::size_t last = found->id;
  while (last + 1 < units.size() && belongs_to(units[last + 1].name, found->name)) ++last;
  if (begin != nullptr) *begin = found->start_byte;
  if (end != nullptr) *end = units[last].end_byte;
  return true;
}

}  // namespace prag::parse
