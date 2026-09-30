#include "common.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "prag/core/hash.hpp"
#include "prag/core/log.hpp"
#include "prag/exec/cache_topology.hpp"
#include "prag/llm/model.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <intrin.h>
#else
#include <unistd.h>
#include <sys/utsname.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#endif

namespace fsys = std::filesystem;

namespace bench {

// ---- logging ---------------------------------------------------------------

void progress(const char* format, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof buffer, format, args);
  va_end(args);
  std::fprintf(stderr, "[bench] %s\n", buffer);
  std::fflush(stderr);
}

std::string iso_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

// ---- random numbers --------------------------------------------------------

std::uint64_t rng_next(std::uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  return prag::mix64(state);
}
double rng_unit(std::uint64_t& state) { return static_cast<double>(rng_next(state) >> 11) * (1.0 / 9007199254740992.0); }
std::uint64_t rng_below(std::uint64_t& state, std::uint64_t n) { return n == 0 ? 0 : rng_next(state) % n; }

// ---- statistics ------------------------------------------------------------

double mean_of(const std::vector<double>& v) {
  double sum = 0.0;
  std::size_t n = 0;
  for (const double x : v) {
    if (!std::isfinite(x)) continue;
    sum += x;
    ++n;
  }
  return n ? sum / D(n) : kNaN;
}

double quantile(std::vector<double> v, double q) {
  v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
  if (v.empty()) return kNaN;
  std::sort(v.begin(), v.end());
  const double pos = std::clamp(q, 0.0, 1.0) * D(v.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (pos - D(lo)) * (v[hi] - v[lo]);
}

double median(std::vector<double> v) { return quantile(std::move(v), 0.5); }

Ci bootstrap_mean(const std::vector<double>& values, std::uint64_t seed, std::size_t reps) {
  std::vector<double> v;
  for (const double x : values)
    if (std::isfinite(x)) v.push_back(x);
  Ci ci;
  ci.n = v.size();
  if (v.empty()) return ci;
  ci.mean = mean_of(v);
  std::uint64_t state = seed * 0x9e3779b97f4a7c15ULL + 99;
  std::vector<double> means;
  means.reserve(reps);
  for (std::size_t r = 0; r < reps; ++r) {
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) sum += v[rng_below(state, v.size())];
    means.push_back(sum / D(v.size()));
  }
  ci.lo = quantile(means, 0.025);
  ci.hi = quantile(means, 0.975);
  return ci;
}

json to_json(const Ci& ci) { return json{{"mean", ci.mean}, {"lo", ci.lo}, {"hi", ci.hi}, {"n", ci.n}}; }

Wilson wilson(std::size_t k, std::size_t n, double z) {
  Wilson w;
  w.k = k;
  w.n = n;
  if (n == 0) return w;
  const double nn = D(n), p = D(k) / nn, z2 = z * z;
  const double denom = 1.0 + z2 / nn;
  const double center = (p + z2 / (2.0 * nn)) / denom;
  const double half = z * std::sqrt(p * (1.0 - p) / nn + z2 / (4.0 * nn * nn)) / denom;
  w.p = p;
  w.lo = std::max(0.0, center - half);
  w.hi = std::min(1.0, center + half);
  return w;
}

json to_json(const Wilson& w) { return json{{"p", w.p}, {"lo", w.lo}, {"hi", w.hi}, {"k", w.k}, {"n", w.n}}; }

namespace {
std::vector<double> ranks(const std::vector<double>& v) {
  std::vector<std::size_t> order(v.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
  std::vector<double> r(v.size());
  for (std::size_t i = 0; i < order.size();) {
    std::size_t j = i;
    while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) ++j;
    const double rank = 0.5 * D(i + j) + 1.0;
    for (std::size_t k = i; k <= j; ++k) r[order[k]] = rank;
    i = j + 1;
  }
  return r;
}
}  // namespace

double spearman(const std::vector<double>& x, const std::vector<double>& y) {
  std::vector<double> a, b;
  for (std::size_t i = 0; i < std::min(x.size(), y.size()); ++i) {
    if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
    a.push_back(x[i]);
    b.push_back(y[i]);
  }
  if (a.size() < 3) return kNaN;
  const std::vector<double> ra = ranks(a), rb = ranks(b);
  const double ma = mean_of(ra), mb = mean_of(rb);
  double sab = 0.0, saa = 0.0, sbb = 0.0;
  for (std::size_t i = 0; i < ra.size(); ++i) {
    sab += (ra[i] - ma) * (rb[i] - mb);
    saa += (ra[i] - ma) * (ra[i] - ma);
    sbb += (rb[i] - mb) * (rb[i] - mb);
  }
  return saa > 0.0 && sbb > 0.0 ? sab / std::sqrt(saa * sbb) : kNaN;
}

json timing_json(const std::vector<double>& samples) {
  const double q1 = quantile(samples, 0.25), q3 = quantile(samples, 0.75);
  return json{{"median", median(samples)}, {"q1", q1},
              {"q3", q3},                  {"iqr", q3 - q1},
              {"min", quantile(samples, 0.0)}, {"max", quantile(samples, 1.0)},
              {"reps", samples.size()}};
}

// ---- Platt scaling ---------------------------------------------------------

PlattFit fit_platt(const std::vector<prag::retrieve::LabeledScore>& samples, double prior) {
  PlattFit fit;
  std::size_t n1 = 0;
  for (const auto& s : samples)
    if (s.relevant && std::isfinite(s.z)) ++n1;
  std::size_t n = 0;
  for (const auto& s : samples)
    if (std::isfinite(s.z)) ++n;
  if (n1 == 0 || n1 == n || !(prior > 0.0 && prior < 1.0)) return fit;
  auto sigmoid = [](double t) { return 1.0 / (1.0 + std::exp(-t)); };
  auto nll = [&](double a, double b) {
    double total = 0.0;
    for (const auto& s : samples) {
      if (!std::isfinite(s.z)) continue;
      const double t = a * s.z + b;
      total += s.relevant ? std::log1p(std::exp(-t)) : std::log1p(std::exp(t));
      if (!std::isfinite(total)) return std::numeric_limits<double>::infinity();
    }
    return total + 0.5 * 1e-6 * (a * a + b * b);
  };
  double a = 0.0, b = std::log(D(n1) / D(n - n1));
  double current = nll(a, b);
  for (int iteration = 0; iteration < 100; ++iteration) {
    double ga = 1e-6 * a, gb = 1e-6 * b, haa = 1e-6, hab = 0.0, hbb = 1e-6;
    for (const auto& s : samples) {
      if (!std::isfinite(s.z)) continue;
      const double p = sigmoid(a * s.z + b), r = p - (s.relevant ? 1.0 : 0.0), w = p * (1.0 - p);
      ga += r * s.z;
      gb += r;
      haa += w * s.z * s.z;
      hab += w * s.z;
      hbb += w;
    }
    const double det = haa * hbb - hab * hab;
    if (!(std::fabs(det) > 1e-300)) break;
    const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
    double step = 1.0;
    bool moved = false;
    for (int halving = 0; halving < 30; ++halving) {
      const double next = nll(a - step * da, b - step * db);
      if (next <= current) {
        a -= step * da;
        b -= step * db;
        current = next;
        moved = true;
        break;
      }
      step *= 0.5;
    }
    if (!moved || std::fabs(step * da) + std::fabs(step * db) < 1e-10) break;
  }
  const double pi_sample = D(n1) / D(n);
  fit.a = a;
  fit.b = b + std::log(prior / (1.0 - prior)) - std::log(pi_sample / (1.0 - pi_sample));
  fit.ok = std::isfinite(fit.a) && std::isfinite(fit.b);
  return fit;
}

// ---- spherical k-means -----------------------------------------------------

std::vector<std::uint32_t> spherical_kmeans(const float* x, std::size_t n, std::size_t m, std::size_t k,
                                            std::uint64_t seed, int iterations) {
  std::vector<std::uint32_t> label(n, 0);
  if (n == 0 || k <= 1) return label;
  k = std::min(k, n);
  std::uint64_t state = seed * 0x9e3779b97f4a7c15ULL + 7;
  auto dot = [&](const float* a, const float* b) {
    double s = 0.0;
    for (std::size_t j = 0; j < m; ++j) s += static_cast<double>(a[j]) * static_cast<double>(b[j]);
    return s;
  };
  std::vector<float> centers(k * m);
  std::vector<double> nearest(n, 2.0);  // 1 - best cosine to a chosen centre
  std::size_t first = static_cast<std::size_t>(rng_below(state, n));
  std::copy(x + first * m, x + (first + 1) * m, centers.begin());
  for (std::size_t c = 1; c < k; ++c) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      nearest[i] = std::min(nearest[i], std::max(0.0, 1.0 - dot(x + i * m, &centers[(c - 1) * m])));
      total += nearest[i];
    }
    std::size_t pick = n - 1;
    if (total > 0.0) {
      double target = rng_unit(state) * total, run = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        run += nearest[i];
        if (run >= target) {
          pick = i;
          break;
        }
      }
    } else {
      pick = static_cast<std::size_t>(rng_below(state, n));
    }
    std::copy(x + pick * m, x + (pick + 1) * m, centers.begin() + static_cast<std::ptrdiff_t>(c * m));
  }
  std::vector<double> sums(k * m);
  std::vector<std::size_t> counts(k);
  for (int it = 0; it < iterations; ++it) {
    bool changed = false;
    std::fill(counts.begin(), counts.end(), std::size_t{0});
    std::fill(sums.begin(), sums.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t best = 0;
      double best_cos = -2.0;
      for (std::size_t c = 0; c < k; ++c) {
        const double d = dot(x + i * m, &centers[c * m]);
        if (d > best_cos) {
          best_cos = d;
          best = static_cast<std::uint32_t>(c);
        }
      }
      if (label[i] != best) changed = true;
      label[i] = best;
      ++counts[best];
      for (std::size_t j = 0; j < m; ++j) sums[best * m + j] += static_cast<double>(x[i * m + j]);
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) {  // re-seed an empty cluster with a random point
        const std::size_t pick = static_cast<std::size_t>(rng_below(state, n));
        std::copy(x + pick * m, x + (pick + 1) * m, centers.begin() + static_cast<std::ptrdiff_t>(c * m));
        changed = true;
        continue;
      }
      double norm = 0.0;
      for (std::size_t j = 0; j < m; ++j) norm += sums[c * m + j] * sums[c * m + j];
      norm = std::sqrt(norm);
      if (norm > 0.0)
        for (std::size_t j = 0; j < m; ++j) centers[c * m + j] = static_cast<float>(sums[c * m + j] / norm);
    }
    if (!changed) break;
  }
  return label;
}

// ---- reliability -----------------------------------------------------------

json reliability_json(const std::function<double(double)>& posterior,
                      const std::vector<prag::retrieve::LabeledScore>& samples, std::size_t bins) {
  if (bins == 0) bins = 1;
  std::vector<std::size_t> count(bins, 0), positives(bins, 0);
  std::vector<double> confidence(bins, 0.0);
  double brier = 0.0, loss = 0.0;
  std::size_t n = 0, total_positive = 0;
  for (const auto& s : samples) {
    double p = posterior(s.z);
    if (!std::isfinite(p)) p = 0.0;
    p = std::clamp(p, 0.0, 1.0);
    const double y = s.relevant ? 1.0 : 0.0;
    std::size_t b = static_cast<std::size_t>(p * D(bins));
    if (b >= bins) b = bins - 1;
    ++count[b];
    confidence[b] += p;
    if (s.relevant) {
      ++positives[b];
      ++total_positive;
    }
    brier += (p - y) * (p - y);
    const double q = std::clamp(p, 1e-12, 1.0 - 1e-12);
    loss += -(s.relevant ? std::log(q) : std::log(1.0 - q));
    ++n;
  }
  json bin_conf = json::array(), bin_acc = json::array(), bin_count = json::array();
  double ece = 0.0;
  for (std::size_t b = 0; b < bins; ++b) {
    const double conf = count[b] ? confidence[b] / D(count[b]) : 0.0;
    const double acc = count[b] ? D(positives[b]) / D(count[b]) : 0.0;
    bin_conf.push_back(conf);
    bin_acc.push_back(acc);
    bin_count.push_back(count[b]);
    if (n) ece += D(count[b]) / D(n) * std::fabs(acc - conf);
  }
  return json{{"n", n},
              {"positives", total_positive},
              {"ece", n ? ece : kNaN},
              {"brier", n ? brier / D(n) : kNaN},
              {"log_loss", n ? loss / D(n) : kNaN},
              {"bins", json{{"confidence", bin_conf}, {"accuracy", bin_acc}, {"count", bin_count}}}};
}

// ---- machine and git -------------------------------------------------------

std::string git_head(const std::string& directory) {
#if defined(_WIN32)
  const std::string command = "git -C \"" + directory + "\" rev-parse HEAD 2>nul";
  FILE* pipe = _popen(command.c_str(), "r");
#else
  const std::string command = "git -C \"" + directory + "\" rev-parse HEAD 2>/dev/null";
  FILE* pipe = popen(command.c_str(), "r");
#endif
  if (!pipe) return "";
  char buffer[256] = {0};
  std::string out;
  if (std::fgets(buffer, sizeof buffer, pipe)) out = buffer;
#if defined(_WIN32)
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
  return out;
}

std::size_t peak_working_set_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) return counters.PeakWorkingSetSize;
#endif
  return 0;
}

namespace {
std::string cpu_brand() {
  char brand[49] = {0};
#if defined(_WIN32)
  int regs[4] = {0, 0, 0, 0};
  __cpuid(regs, static_cast<int>(0x80000000));
  if (static_cast<unsigned>(regs[0]) >= 0x80000004u) {
    for (unsigned i = 0; i < 3; ++i) {
      __cpuid(regs, static_cast<int>(0x80000002u + i));
      std::memcpy(brand + 16 * i, regs, 16);
    }
  }
#elif defined(__x86_64__) || defined(__i386__)
  unsigned int a = 0, b = 0, c = 0, d = 0;
  if (__get_cpuid(0x80000000u, &a, &b, &c, &d) && a >= 0x80000004u) {
    for (unsigned i = 0; i < 3; ++i) {
      __get_cpuid(0x80000002u + i, &a, &b, &c, &d);
      unsigned int r[4] = {a, b, c, d};
      std::memcpy(brand + 16 * i, r, 16);
    }
  }
#endif
  std::string s = brand;
  const std::size_t first = s.find_first_not_of(' ');
  return first == std::string::npos ? std::string("unknown") : s.substr(first);
}
}  // namespace

json machine_json() {
  const prag::exec::CacheTopology& topology = prag::exec::cache_topology();
  json l1d = json::array();
  for (const auto& c : topology.classes)
    if (c.level == 1 && c.type == prag::exec::CacheType::Data) l1d.push_back(c.size_bytes);
  double ram_gb = 0.0;
  std::string os;
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof status;
  if (GlobalMemoryStatusEx(&status)) ram_gb = static_cast<double>(status.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
  os = "Windows";
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
    if (auto fn = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(ntdll, "RtlGetVersion")))) {
      OSVERSIONINFOW info{};
      info.dwOSVersionInfoSize = sizeof info;
      if (fn(&info) == 0)
        os += " " + std::to_string(info.dwMajorVersion) + "." + std::to_string(info.dwMinorVersion) + "." +
              std::to_string(info.dwBuildNumber);
    }
  }
#else
  ram_gb = static_cast<double>(sysconf(_SC_PHYS_PAGES)) * static_cast<double>(sysconf(_SC_PAGE_SIZE)) /
           (1024.0 * 1024.0 * 1024.0);
  utsname u{};
  if (uname(&u) == 0) os = std::string(u.sysname) + " " + u.release;
#endif
  return json{{"cpu", cpu_brand()},
              {"logical_cpus", topology.logical_cpus},
              {"l1d_bytes", l1d},
              {"ram_gb", ram_gb},
              {"os", os},
              {"cache_detected", topology.detected},
              {"block_bytes", prag::exec::recommended_block_bytes(topology)},
              {"simd_kernel_note", "see e5_execution for the kernel in use"}};
}

// ---- meta and results ------------------------------------------------------

json make_meta(Context& ctx, const std::string& experiment, const std::string& started, const json& settings) {
  std::size_t functions = 0;
  for (const auto& u : ctx.corpus->units)
    if (u.kind == prag::UnitKind::Function) ++functions;
  const auto chat = ctx.chat();
  json effective = settings.is_object() ? settings : json::object();
  effective["quick"] = ctx.quick();
  effective["engine_settings"] = ctx.engine->settings().snapshot();
  effective["block_bytes"] = ctx.engine->block_bytes();
  effective["threads"] = ctx.engine->executor().threads();
  return json{{"experiment", experiment},
              {"started", started},
              {"finished", iso_now()},
              {"git_commit", git_head(ctx.repo_dir)},
              {"corpus",
               json{{"path", ctx.demo_dir},
                    {"roots", ctx.corpus->roots},
                    {"commit", git_head(ctx.demo_dir)},
                    {"files", ctx.corpus->files.size()},
                    {"units", ctx.corpus->units.size()},
                    {"functions", functions}}},
              {"machine", machine_json()},
              {"models",
               json{{"embedding", ctx.corpus->embedder_id}, {"chat", chat ? chat->id() : std::string("none")}}},
              {"settings", effective},
              {"seed", ctx.seed}};
}

void write_result(Context& ctx, const std::string& name, const std::string& started, const json& settings,
                  json body) {
  using ordered = nlohmann::ordered_json;
  const auto dump = [](const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); };
  ordered root = ordered::object();
  root["meta"] = ordered::parse(dump(make_meta(ctx, name, started, settings)));
  if (ctx.quick()) root["quick"] = true;
  if (body.is_object())
    for (auto it = body.begin(); it != body.end(); ++it) root[it.key()] = ordered::parse(dump(it.value()));
  fsys::create_directories(ctx.out_dir);
  const std::string path = ctx.out_dir + "/" + name + ".json";
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + tmp);
    f << root.dump(1, ' ', false, ordered::error_handler_t::replace) << "\n";
  }
  std::error_code ec;
  fsys::remove(path, ec);
  fsys::rename(tmp, path);
  progress("wrote %s", path.c_str());
}

json guarded(const std::string& part, const std::function<json()>& f) {
  try {
    progress("part %s ...", part.c_str());
    return f();
  } catch (const std::exception& e) {
    progress("part %s failed: %s", part.c_str(), e.what());
    return json{{"error", e.what()}};
  }
}

json skipped(const std::string& reason) { return json{{"skipped", reason}}; }

// ---- context ---------------------------------------------------------------

namespace {
std::string normalize_slashes(std::string s) {
  std::replace(s.begin(), s.end(), '\\', '/');
  return s;
}

std::string resolve(const fsys::path& base, const std::string& p) {
  if (p.empty()) return p;
  fsys::path path(p);
  if (path.is_relative()) path = base / path;
  std::error_code ec;
  const fsys::path canonical = fsys::weakly_canonical(path, ec);
  return normalize_slashes((ec ? path : canonical).string());
}
}  // namespace

std::unique_ptr<Context> open_context(const Args& args) {
  auto ctx = std::make_unique<Context>();
  ctx->args = args;
  ctx->repo_dir = normalize_slashes(PRAG_SOURCE_DIR);
  ctx->demo_dir = normalize_slashes(PRAG_DEMO_REPO);

  // Settings: the given file when it exists, built-in defaults otherwise.
  fsys::path settings_file(args.settings);
  if (!fsys::exists(settings_file) && settings_file.is_relative()) settings_file = fsys::path(ctx->repo_dir) / settings_file;
  json settings = json::object();
  fsys::path base = fsys::current_path();
  if (fsys::exists(settings_file)) {
    std::ifstream f(settings_file, std::ios::binary);
    settings = json::parse(f, nullptr, false);
    if (settings.is_discarded() || !settings.is_object()) throw std::runtime_error("settings file is not valid JSON");
    base = fsys::absolute(settings_file).parent_path();
    ctx->settings_path = normalize_slashes(fsys::absolute(settings_file).string());
  } else {
    progress("settings file %s not found; using built-in defaults (hashing embedder, extractive answers)",
             args.settings.c_str());
    ctx->settings_path = "(defaults)";
  }
  json roots = json::array();
  if (settings.contains("corpus") && settings["corpus"].contains("roots"))
    for (const auto& r : settings["corpus"]["roots"]) roots.push_back(resolve(base, r.get<std::string>()));
  if (roots.empty()) {
    const fsys::path demo(ctx->demo_dir);
    if (fsys::exists(demo / "src")) roots.push_back(ctx->demo_dir + "/src");
    if (fsys::exists(demo / "include")) roots.push_back(ctx->demo_dir + "/include");
    if (roots.empty()) roots.push_back(ctx->demo_dir);
  }
  settings["corpus"]["roots"] = roots;
  for (const char* key : {"embedding", "chat", "tuning_base"}) {
    if (settings.contains("models") && settings["models"].contains(key) && settings["models"][key].is_string())
      settings["models"][key] = resolve(base, settings["models"][key].get<std::string>());
  }
  const std::string embedding = settings.contains("models") ? settings["models"].value("embedding", std::string()) : "";
  std::string label = "hashing";
  if (!embedding.empty() && prag::llm::available()) label = fsys::path(embedding).stem().string();
  ctx->data_dir = ctx->repo_dir + "/build/bench_data/" + label;
  settings["paths"]["data"] = ctx->data_dir;
  ctx->out_dir = resolve(fsys::current_path(), args.out);
  ctx->settings = settings;

  fsys::create_directories(ctx->data_dir);
  const std::string effective = ctx->data_dir + "/bench_settings.json";
  {
    std::ofstream f(effective, std::ios::binary);
    f << settings.dump(2);
  }
  prag::log::set_level(prag::log::Level::Warn);
  ctx->engine = std::make_unique<prag::engine::Engine>(effective);
  ctx->engine->open();

  auto usable = [&] {
    const auto c = ctx->engine->corpus();
    return c && ctx->engine->index_ready() && ctx->engine->embedder() &&
           c->embedder_id == ctx->engine->embedder()->id();
  };
  if (!usable()) {
    progress("building the index of %s ...", ctx->demo_dir.c_str());
    std::string last_stage;
    const auto report = ctx->engine->build_index([&](const prag::engine::IndexProgress& p) {
      if (p.stage != last_stage) {
        last_stage = p.stage;
        progress("index stage %s (%zu/%zu) %s", p.stage.c_str(), p.done, p.total, p.message.c_str());
      }
      return true;
    });
    if (!report.error.empty()) throw std::runtime_error("index build failed: " + report.error);
  }
  ctx->corpus = ctx->engine->corpus();
  if (!ctx->corpus) throw std::runtime_error("no index available");
  progress("corpus: %zu files, %zu units, embedder %s, chat model %s", ctx->corpus->files.size(),
           ctx->corpus->units.size(), ctx->corpus->embedder_id.c_str(),
           ctx->chat() ? ctx->chat()->id().c_str() : "none");
  return ctx;
}

}  // namespace bench
