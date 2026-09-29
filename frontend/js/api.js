// Client of the HTTP API (docs/DESIGN.md, sections 6.14 and 9).
//
// Imports nothing from the DOM: it needs `fetch`, which browsers and Node both
// provide, so the tests run it against the development mock server.
//
// Every call returns the parsed JSON body or throws an ApiError whose message
// is the server's {"error": "..."} text when there is one.

import { SseParser, parseEventData } from './lib/sse.js';

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;  // HTTP status, 0 when the request did not reach the server
  }
}

export function isAbort(error) {
  return error !== null && typeof error === 'object' && error.name === 'AbortError';
}

async function errorFrom(response) {
  let message = '';
  try {
    const body = await response.json();
    if (body && typeof body.error === 'string') message = body.error;
  } catch {
    // The body is not JSON; the status line has to do.
  }
  if (message === '') message = `${response.status} ${response.statusText || 'request failed'}`.trim();
  return new ApiError(message, response.status);
}

export function createApi(baseUrl = '', fetchImpl = (...args) => globalThis.fetch(...args)) {
  async function request(method, path, body, signal) {
    const init = { method, headers: { Accept: 'application/json' }, signal };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetchImpl(baseUrl + path, init);
    } catch (error) {
      if (isAbort(error)) throw error;
      throw new ApiError('The server cannot be reached.', 0);
    }
    if (!response.ok) throw await errorFrom(response);
    if (response.status === 204) return null;
    try {
      return await response.json();
    } catch {
      throw new ApiError(`${method} ${path}: the response is not valid JSON.`, response.status);
    }
  }

  // Sends a question and reads the event stream of the answer. Handlers:
  //   onRetrieval(object)  onToken(text)  onEdit(transaction)  onDone(object)
  // Resolves with the payload of the `done` event. Rejects with an ApiError
  // when the server reports an error or the stream ends without `done`, and
  // with the AbortError of fetch when `signal` is aborted.
  async function query(params, handlers = {}, signal) {
    const body = { question: params.question, edit: params.edit === true };
    if (typeof params.tau === 'number') body.tau = params.tau;

    let response;
    try {
      response = await fetchImpl(baseUrl + '/api/query', {
        method: 'POST',
        headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (isAbort(error)) throw error;
      throw new ApiError('The server cannot be reached.', 0);
    }
    if (!response.ok) throw await errorFrom(response);
    if (!response.body) throw new ApiError('The response has no body to read.', response.status);

    const parser = new SseParser();
    const reader = response.body.getReader();
    let done = null;
    let failure = null;

    const dispatch = (event) => {
      if (done !== null || failure !== null) return;
      const parsed = parseEventData(event);
      if (!parsed.ok) {
        failure = new ApiError(parsed.error, response.status);
        return;
      }
      const data = parsed.value;
      switch (event.event) {
        case 'retrieval':
          if (handlers.onRetrieval) handlers.onRetrieval(data);
          break;
        case 'token':
          if (handlers.onToken && data && typeof data.text === 'string') handlers.onToken(data.text);
          break;
        case 'edit':
          if (handlers.onEdit) handlers.onEdit(data);
          break;
        case 'done':
          done = data || {};
          break;
        case 'error':
          failure = new ApiError(data && typeof data.error === 'string' ? data.error : 'The query failed.',
                                 response.status);
          break;
        default:
          break;  // events added by a newer server
      }
    };

    try {
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        for (const event of parser.feed(chunk.value)) {
          // A chunk may hold many events. Once the caller has aborted, none
          // of those that are left is delivered.
          if (signal && signal.aborted) throw new DOMException('The request was aborted.', 'AbortError');
          dispatch(event);
        }
        if (done !== null || failure !== null) break;
      }
      if (done === null && failure === null) {
        for (const event of parser.end()) dispatch(event);
      }
    } catch (error) {
      if (isAbort(error)) throw error;
      throw new ApiError('The connection was lost while the answer was being received.', 0);
    } finally {
      try {
        await reader.cancel();
      } catch {
        // Already closed or aborted.
      }
    }
    if (failure !== null) throw failure;
    if (done === null) throw new ApiError('The answer ended before it was complete.', response.status);
    if (handlers.onDone) handlers.onDone(done);
    return done;
  }

  return {
    status: (signal) => request('GET', '/api/status', undefined, signal),
    settings: () => request('GET', '/api/settings'),
    saveSettings: (patch) => request('POST', '/api/settings', patch),
    startIndex: () => request('POST', '/api/index', {}),
    indexProgress: () => request('GET', '/api/index/progress'),
    calibrate: () => request('POST', '/api/calibrate', {}),
    history: () => request('GET', '/api/history'),
    clearHistory: () => request('DELETE', '/api/history'),
    feedback: (id, rating) => request('POST', '/api/feedback', { id, rating }),
    groups: () => request('GET', '/api/groups'),
    unit: (id) => request('GET', '/api/unit?id=' + encodeURIComponent(String(id))),
    tune: () => request('GET', '/api/tune'),
    setTune: (enabled) => request('POST', '/api/tune', { enabled: enabled === true }),
    runTune: () => request('POST', '/api/tune/run', {}),
    editPreview: (payload) => request('POST', '/api/edit/preview', payload),
    editApply: (id) => request('POST', '/api/edit/apply', { id }),
    editRevert: (id) => request('POST', '/api/edit/revert', { id }),
    query,
  };
}
