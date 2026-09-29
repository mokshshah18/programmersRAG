// Incremental parser for text/event-stream (server-sent events).
//
// EventSource cannot send a POST request, so the query stream is read with
// fetch and a ReadableStream, and the bytes are parsed here. The parser follows
// the event stream interpretation rules of the HTML standard:
//
//   * lines end with LF, CR or CRLF;
//   * "field: value" with one optional space after the colon; a line without a
//     colon is a field with an empty value; a line starting with a colon is a
//     comment;
//   * several data lines of one event are joined with LF;
//   * a blank line dispatches the event, unless it carries no data.
//
// Input arrives in chunks cut at arbitrary byte positions. Two things must
// therefore survive a chunk boundary: a multi-byte UTF-8 sequence (handled by
// the streaming TextDecoder) and a line, including a CRLF pair whose CR is the
// last byte of one chunk and whose LF is the first byte of the next.

export class SseParser {
  constructor() {
    this.decoder = new TextDecoder('utf-8');
    this.buffer = '';        // decoded text that does not yet end in a line terminator
    this.skipLf = false;     // the previous chunk ended with CR: a leading LF belongs to it
    this.started = false;    // false until the first character, to strip a byte order mark
    this.eventName = '';
    this.data = [];
    this.lastEventId = '';
    this.retry = null;       // last reconnection time sent by the server, in milliseconds
  }

  // Accepts a Uint8Array (or any ArrayBuffer view) or a string and returns the
  // events completed by it, in order: [{ event, data, id }].
  feed(chunk) {
    const text = typeof chunk === 'string' ? chunk : this.decoder.decode(chunk, { stream: true });
    return this.consume(text);
  }

  // Call once when the stream has ended. Flushes the decoder; an event that was
  // not terminated by a blank line is incomplete and is discarded, as the
  // standard requires.
  end() {
    const events = this.consume(this.decoder.decode());
    this.buffer = '';
    this.eventName = '';
    this.data = [];
    return events;
  }

  consume(text) {
    const events = [];
    if (text.length === 0) return events;
    if (!this.started) {
      this.started = true;
      if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    }
    if (this.skipLf) {
      this.skipLf = false;
      if (text.charCodeAt(0) === 0x0a) text = text.slice(1);
    }
    const input = this.buffer + text;
    let start = 0;
    for (let i = this.buffer.length; i < input.length; ++i) {
      const c = input.charCodeAt(i);
      if (c !== 0x0a && c !== 0x0d) continue;
      this.line(input.slice(start, i), events);
      if (c === 0x0d) {
        if (i + 1 === input.length) this.skipLf = true;
        else if (input.charCodeAt(i + 1) === 0x0a) ++i;
      }
      start = i + 1;
    }
    this.buffer = input.slice(start);
    return events;
  }

  line(line, events) {
    if (line === '') {
      if (this.data.length > 0) {
        events.push({ event: this.eventName || 'message', data: this.data.join('\n'), id: this.lastEventId });
      }
      this.eventName = '';
      this.data = [];
      return;
    }
    if (line.charCodeAt(0) === 0x3a) return;  // comment, used by servers as a keep-alive
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.charCodeAt(0) === 0x20) value = value.slice(1);
    switch (field) {
      case 'event':
        this.eventName = value;
        break;
      case 'data':
        this.data.push(value);
        break;
      case 'id':
        if (value.indexOf('\u0000') < 0) this.lastEventId = value;
        break;
      case 'retry':
        if (/^[0-9]+$/.test(value)) this.retry = Number(value);
        break;
      default:
        break;  // unknown fields are ignored
    }
  }
}

// Parses the JSON payload of an event. Returns { ok: true, value } or
// { ok: false, error }; never throws, because the payload is untrusted.
export function parseEventData(event) {
  try {
    return { ok: true, value: JSON.parse(event.data) };
  } catch (error) {
    return { ok: false, error: `event "${event.event}" does not carry valid JSON: ${error.message}` };
  }
}

// Reads a fetch response body to its end and calls onEvent for every event.
// `body` is a ReadableStream of Uint8Array. Resolves with the number of events.
export async function readEventStream(body, onEvent) {
  const parser = new SseParser();
  const reader = body.getReader();
  let count = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const event of parser.feed(value)) {
        ++count;
        onEvent(event);
      }
    }
    for (const event of parser.end()) {
      ++count;
      onEvent(event);
    }
  } finally {
    reader.releaseLock();
  }
  return count;
}
