// Development server for the web UI: serves the frontend directory and
// implements every endpoint of docs/DESIGN.md section 9 with mock data, so the
// UI can be developed and tested before (and without) the C++ server.
//
//   node frontend/dev/mock-server.js [--port 8000] [--host 127.0.0.1] [--speed 1] [--latency 0] [--no-llama] [--quiet]
//
//   --port 0     lets the operating system choose a free port (the URL is printed)
//   --speed N    makes the index build, the tuning pipeline and the answer stream N times faster
//   --latency N  delays every API response by N milliseconds, to see the loading states of the UI
//   --no-llama   imitates a build without llama.cpp: extractive answers, tuning unavailable
//
// No dependencies; Node 18 or later.
//
// Like the real server it binds to 127.0.0.1 and sends no CORS headers.
//
// The query stream is cut into chunks at arbitrary byte positions, also inside
// multi-byte UTF-8 sequences, to exercise the event stream parser of the UI.
//
// Markers in a question select the special cases:
//   #error       the stream ends with an `error` event after a few tokens
//   #drop        the stream ends without `done`
//   #violation   the scan reports ordering violations
//   in edit mode: #ambiguous  #notfound  #syntax  #indent  #conflict (apply fails)

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createEngine, HttpError } from './mock-engine.js';
import { makeRandom } from './mock-corpus.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MAX_BODY = 1 << 20;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
};

// Timers are coarse (about 15 ms on Windows), so a pause shorter than two
// milliseconds only yields to the event loop. That is enough for the pieces of
// a response to be written separately.
function sleep(ms) {
  if (ms < 2) return new Promise((resolve) => setImmediate(resolve));
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function sendJson(res, status, value) {
  const body = Buffer.from(JSON.stringify(value), 'utf8');
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': body.length,
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY) {
        reject(new HttpError(413, 'The request body is too large.'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      const text = Buffer.concat(chunks).toString('utf8');
      if (text.trim() === '') {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(text));
      } catch {
        reject(new HttpError(400, 'The request body is not valid JSON.'));
      }
    });
    req.on('error', reject);
  });
}

function serveStatic(req, res, pathname) {
  let relative;
  try {
    relative = decodeURIComponent(pathname);
  } catch {
    sendJson(res, 400, { error: 'Malformed path.' });
    return;
  }
  if (relative.endsWith('/')) relative += 'index.html';
  const file = path.resolve(ROOT, '.' + relative);
  // Containment: the resolved path must stay inside the frontend directory.
  if (file !== ROOT && !file.startsWith(ROOT + path.sep)) {
    sendJson(res, 403, { error: 'Outside the frontend directory.' });
    return;
  }
  fs.stat(file, (error, stat) => {
    if (error || !stat.isFile()) {
      sendJson(res, 404, { error: `Not found: ${pathname}` });
      return;
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Content-Length': stat.size,
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    if (req.method === 'HEAD') res.end();
    else fs.createReadStream(file).pipe(res);
  });
}

export function createMockServer(options = {}) {
  const speed = options.speed > 0 ? options.speed : 1;
  const quiet = options.quiet === true;
  const latency = options.latency > 0 ? options.latency : 0;
  // The time of the engine. Tests pass a clock that they advance themselves,
  // which makes the index build and the tuning pipeline independent of how
  // fast the machine is.
  const clock = typeof options.clock === 'function' ? options.clock : Date.now;
  const engine = createEngine({ speed, llama: options.llama, seed: options.seed, now: clock() });
  const cuts = makeRandom(7);
  const open = new Set();  // responses of running query streams

  // Writes `bytes` in one to three pieces, each in a turn of its own so that
  // they reach the client as separate chunks.
  async function writeInPieces(res, bytes) {
    const pieces = 1 + cuts.int(3);
    const points = [0];
    for (let i = 1; i < pieces; ++i) points.push(1 + cuts.int(Math.max(1, bytes.length - 1)));
    points.push(bytes.length);
    points.sort((a, b) => a - b);
    for (let i = 0; i + 1 < points.length; ++i) {
      if (points[i + 1] === points[i]) continue;
      if (res.destroyed || res.writableEnded) return;
      res.write(bytes.subarray(points[i], points[i + 1]));
      if (i + 2 < points.length) await sleep(0);
    }
  }

  async function streamQuery(req, res, body) {
    const result = engine.query(body, clock());  // throws HttpError before anything is sent
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-store',
      'X-Accel-Buffering': 'no',
      'X-Content-Type-Options': 'nosniff',
    });
    if (res.socket) res.socket.setNoDelay(true);
    open.add(res);
    let closed = false;
    res.on('close', () => {
      closed = true;
      open.delete(res);
    });
    const send = (event, data) =>
      writeInPieces(res, Buffer.from(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`, 'utf8'));
    const wait = (ms) => sleep(ms / speed);

    res.write(': stream open\n\n');  // a comment line, as servers send to keep a connection alive
    await wait(180);
    if (closed) return;
    await send('retrieval', result.retrieval);
    await wait(350);

    // The tokens are paced against a schedule rather than by a fixed pause:
    // timers are too coarse for a pause per token, and the answer should
    // take a second or two whatever its length.
    const duration = Math.min(2400, Math.max(1000, result.tokens.length * 8)) / speed;
    const began = Date.now();
    let sent = 0;
    for (const piece of result.tokens) {
      if (closed) return;
      await send('token', { text: piece });
      ++sent;
      if (result.failure !== null && sent === 12) {
        await send('error', { error: result.failure });
        res.end();
        return;
      }
      if (result.drop && sent === 12) {
        res.end();
        return;
      }
      const due = began + (duration * sent) / result.tokens.length;
      await sleep(Math.max(0, due - Date.now()));
    }
    if (closed) return;
    if (result.transaction) await send('edit', result.transaction);
    result.commit(clock());
    await send('done', result.done);
    res.end();
  }

  async function route(req, res) {
    const url = new URL(req.url, 'http://localhost');
    const pathname = url.pathname;
    const method = req.method || 'GET';

    if (!pathname.startsWith('/api/')) {
      if (method !== 'GET' && method !== 'HEAD') {
        sendJson(res, 405, { error: 'Method not allowed.' });
        return;
      }
      serveStatic(req, res, pathname);
      return;
    }

    if (latency > 0) await sleep(latency);
    const now = clock();
    const key = `${method} ${pathname}`;
    switch (key) {
      case 'GET /api/status':
        sendJson(res, 200, engine.status(now));
        return;
      case 'GET /api/settings':
        sendJson(res, 200, engine.settings());
        return;
      case 'POST /api/settings':
        sendJson(res, 200, engine.patchSettings(await readBody(req), now));
        return;
      case 'POST /api/index':
        await readBody(req);
        sendJson(res, 200, engine.startIndex(now));
        return;
      case 'GET /api/index/progress':
        sendJson(res, 200, engine.indexProgress(now));
        return;
      case 'POST /api/calibrate':
        await readBody(req);
        sendJson(res, 200, engine.calibrate(now));
        return;
      case 'POST /api/query':
        await streamQuery(req, res, await readBody(req));
        return;
      case 'GET /api/history':
        sendJson(res, 200, engine.history());
        return;
      case 'DELETE /api/history':
        sendJson(res, 200, engine.clearHistory());
        return;
      case 'POST /api/feedback':
        sendJson(res, 200, engine.feedback(await readBody(req), now));
        return;
      case 'GET /api/groups':
        sendJson(res, 200, engine.groups());
        return;
      case 'GET /api/unit':
        sendJson(res, 200, engine.unit(url.searchParams.get('id')));
        return;
      case 'GET /api/tune':
        sendJson(res, 200, engine.tune(now));
        return;
      case 'POST /api/tune': {
        const body = await readBody(req);
        if (typeof body.enabled !== 'boolean') throw new HttpError(400, '"enabled" must be true or false.');
        sendJson(res, 200, engine.setTune(body.enabled, now));
        return;
      }
      case 'POST /api/tune/run':
        await readBody(req);
        sendJson(res, 200, engine.runTune(now));
        return;
      case 'POST /api/edit/preview':
        sendJson(res, 200, engine.editPreview(await readBody(req), now));
        return;
      case 'POST /api/edit/apply':
        sendJson(res, 200, engine.editApply(await readBody(req), now));
        return;
      case 'POST /api/edit/revert':
        sendJson(res, 200, engine.editRevert(await readBody(req), now));
        return;
      default:
        sendJson(res, 404, { error: `No such endpoint: ${key}` });
    }
  }

  const server = http.createServer((req, res) => {
    const started = Date.now();
    res.on('finish', () => {
      if (!quiet) console.log(`${req.method} ${req.url} ${res.statusCode} ${Date.now() - started} ms`);
    });
    route(req, res).catch((error) => {
      const status = error instanceof HttpError ? error.status : 500;
      if (!(error instanceof HttpError)) console.error(error);
      if (res.headersSent) {
        res.end();
        return;
      }
      sendJson(res, status, { error: error instanceof HttpError ? error.message : 'Internal error of the mock server.' });
    });
  });

  return {
    server,
    engine,
    listen(port = 8000, host = '127.0.0.1') {
      return new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(port, host, () => {
          server.removeListener('error', reject);
          resolve({ host, port: server.address().port, url: `http://${host}:${server.address().port}` });
        });
      });
    },
    close() {
      for (const res of open) res.destroy();
      return new Promise((resolve) => {
        server.close(() => resolve());
        if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
      });
    },
  };
}

function parseArguments(argv) {
  const options = { port: 8000, host: '127.0.0.1', speed: 1, latency: 0, llama: true, quiet: false };
  for (let i = 0; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === '--port') options.port = Number(argv[++i]);
    else if (arg === '--host') options.host = String(argv[++i]);
    else if (arg === '--speed') options.speed = Number(argv[++i]);
    else if (arg === '--latency') options.latency = Number(argv[++i]);
    else if (arg === '--no-llama') options.llama = false;
    else if (arg === '--quiet') options.quiet = true;
    else if (arg === '--help' || arg === '-h') options.help = true;
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535) throw new Error('invalid --port');
  if (!(options.speed > 0)) throw new Error('invalid --speed');
  if (!(options.latency >= 0)) throw new Error('invalid --latency');
  return options;
}

const direct = process.argv[1] !== undefined && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (direct) {
  let options;
  try {
    options = parseArguments(process.argv.slice(2));
  } catch (error) {
    console.error(`${error.message}\nusage: node mock-server.js [--port N] [--host H] [--speed N] [--latency MS] [--no-llama] [--quiet]`);
    process.exit(2);
  }
  if (options.help) {
    console.log('usage: node mock-server.js [--port N] [--host H] [--speed N] [--latency MS] [--no-llama] [--quiet]');
    process.exit(0);
  }
  const mock = createMockServer(options);
  mock.listen(options.port, options.host).then(
    (address) => {
      console.log(`programmersRAG mock server: ${address.url}  (frontend: ${ROOT})`);
    },
    (error) => {
      console.error(`cannot listen on ${options.host}:${options.port}: ${error.message}`);
      process.exit(1);
    });
  const stop = () => mock.close().then(() => process.exit(0));
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
}
