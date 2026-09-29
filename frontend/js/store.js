// Application state: a few named values with change notification.
//
//   status        GET /api/status
//   tune          the tuning status (part of the status, or the answer of POST /api/tune)
//   progress      GET /api/index/progress while a build runs, else null
//   retrieval     the retrieval event shown in the inspector
//   transaction   the edit transaction shown in the inspector
//   groups        GET /api/groups
//   highlight     ids of the units to mark on the map (array)
//   focusUnit     id of the unit to reveal in the retrieval list
//   busy          true while an answer is streamed

export function createStore(initial = {}) {
  const state = { ...initial };
  const listeners = new Map();

  function on(key, listener) {
    if (!listeners.has(key)) listeners.set(key, new Set());
    listeners.get(key).add(listener);
    return () => listeners.get(key).delete(listener);
  }

  function set(key, value) {
    state[key] = value;
    const list = listeners.get(key);
    if (!list) return;
    for (const listener of Array.from(list)) {
      try {
        listener(value);
      } catch (error) {
        // One failing view must not keep the others from updating.
        console.error(`listener of "${key}" failed`, error);
      }
    }
  }

  return { get: (key) => state[key], set, on };
}
