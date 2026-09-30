# programmersRAG

Ask questions about a code base and apply edits to it, using language models
that run entirely on your own machine.

## Requirements

| | Windows | Linux / macOS |
|---|---|---|
| Compiler | Visual Studio 2022 or newer with the "Desktop development with C++" workload | GCC 11+ or Clang 15+ |
| Build tools | CMake 3.24+ (Ninja is included with Visual Studio) | CMake 3.24+, Ninja |
| Other | Git, curl | Git, curl |
| Disk | about 10 GB (models 3.5 GB, build 3 GB) | same |
| Memory | 8 GB minimum, 16 GB recommended; fine-tuning needs about 4 GB free | same |

A CPU with AVX2 is assumed. No GPU is needed.

## Installation

```
git clone https://github.com/mokshshah18/programmersRAG.git
cd programmersRAG
```

1. Download the models (about 3.5 GB) into `models/`:

   ```
   powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1      # Windows
   scripts/download-models.sh                                                # Linux / macOS
   ```

   Add `-Large` (Windows) or `--large` (Linux / macOS) to also fetch the
   1.5B chat model.

2. Build. The first build fetches the dependencies and compiles llama.cpp,
   which takes several minutes.

   ```
   scripts\build.bat          # Windows, run from any terminal
   scripts/build.sh           # Linux / macOS
   ```

   Optional: run the tests with `scripts\msvc.bat ctest --preset release`
   (Windows) or `ctest --preset release` (Linux / macOS).

3. Create your settings file and point it at the code you want to index:

   ```
   copy prag.example.json prag.json      # Windows
   cp prag.example.json prag.json        # Linux / macOS
   ```

   Edit `corpus.roots` in `prag.json`. Paths are relative to the settings file
   or absolute. The example indexes the demo repository in `Demo repo/mimalloc`.

The program is `build/release/src/cli/prag` (`prag.exe` on Windows). The
commands below assume it is on your `PATH` or called with that path.

## Running

Check the installation:

```
prag doctor
```

Build the index (the first run embeds every unit and takes a while; later runs
only re-embed what changed):

```
prag index
```

Start the web interface and open http://127.0.0.1:8000 in a browser:

```
prag serve
```

Ask from the command line instead:

```
prag ask "How does mi_free find the page a pointer belongs to?"
prag ask --edit "Add a comment explaining the fast path in mi_free"
```

Edit answers are shown as a preview. Apply them from the web interface, or with
`prag edit <file-with-edit-blocks> --yes`.

### Other commands

| Command | Purpose |
|---|---|
| `prag find-files` | list the files the current settings would index |
| `prag groups` | show the groups of similar functions |
| `prag calibrate` | re-estimate the relevance calibration |
| `prag tune status\|on\|off\|run` | background fine-tuning of the local model |
| `prag models` | list the configured model files |

All commands accept `--settings <file>` (default: `prag.json` in the current
directory).

### Fine-tuning

Fine-tuning is off by default. Turn it on with the switch in the web
interface or `prag tune on`. It runs only while the computer is idle and on
mains power, stops as soon as you use the program, and replaces the model only
when the tuned model does better on held-out questions.

## Troubleshooting

* **Build fails with a path-length error on Windows**: clone into a short
  directory such as `C:\src\programmersRAG`.
* **`prag doctor` reports that llama.cpp is not available**: the program was
  built with `-DPRAG_WITH_LLAMA=OFF`; rebuild with `scripts\build.bat`.
* **Downloads fail with a certificate revocation error**: the Windows script
  already passes `--ssl-revoke-best-effort`; on other systems check the proxy
  settings.
* **Port 8000 is in use**: set `server.port` in `prag.json`.
