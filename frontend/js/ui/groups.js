// Inspector, Groups tab: a map of all units.
//
// Position is the first two coordinates of the reduced embedding. Colour tells
// neighbouring density groups apart: there are more groups than colours that
// can be distinguished, so colours repeat the way they do on a political map,
// and the identity of a group comes from the tooltip and from the table below
// the map, which also selects a group. The units of the last retrieval carry a
// ring.

import * as geometry from '../lib/scatter.js';
import { formatCount, formatFixed, formatProbability, plural, shortenPath } from '../lib/format.js';
import { h, icon, byId, banner, emptyState, fact, nextFrame, replace } from './dom.js';
import { unitTitle } from './retrieval.js';

const SLOTS = 8;
const HIT_RADIUS = 16;
const TWO_PI = 2 * Math.PI;

export function createGroupsPanel({ api, store, toast }) {
  const panel = byId('panel-groups');
  let data = null;         // GET /api/groups
  let loadedFor = null;    // built_at of the index the data belongs to
  let loading = false;
  let slots = new Map();   // group id -> colour slot
  let view = null;
  let fitted = 1;          // scale of the fitted view
  let selected = null;     // group id
  let hover = -1;          // index into data.units
  let markRetrieval = true;
  let colours = null;
  let pending = false;  // a redraw is scheduled
  let width = 0;
  let height = 0;

  // Elements that live as long as the panel.
  const canvas = h('canvas', { class: 'map-canvas', tabIndex: 0, role: 'img' });
  const tooltip = h('div', { class: 'map-tooltip', hidden: true });
  const summary = h('p', { class: 'note' });
  const retrievalNote = h('span', { class: 'note' });
  const mark = h('input', { type: 'checkbox', id: 'mark-retrieval', checked: true });
  const zoomRetrieval = h('button', { type: 'button', class: 'btn btn-small', text: 'Zoom to retrieval' });
  const tableBody = h('tbody', null);
  const selection = h('div', { class: 'map-selection', 'aria-live': 'polite' });
  const surface = h('div', { class: 'map' }, canvas, tooltip);
  const context = canvas.getContext('2d');

  function readColours() {
    const style = getComputedStyle(panel);
    const value = (name) => style.getPropertyValue(name).trim();
    colours = {
      series: Array.from({ length: SLOTS }, (_, i) => value(`--series-${i + 1}`)),
      surface: value('--surface'),
      grid: value('--line'),
      ink: value('--ink'),
      muted: value('--ink-3'),
      dim: value('--line-strong'),
    };
  }

  function highlightSet() {
    const ids = store.get('highlight');
    return new Set(Array.isArray(ids) ? ids : []);
  }

  function resetView() {
    if (!data || width === 0) return;
    view = geometry.fitView(geometry.boundsOf(data.units), width, height, 20);
    fitted = view.scale;
    schedule();
  }

  function zoomTo(points) {
    if (points.length === 0 || width === 0) return;
    const bounds = geometry.boundsOf(points);
    const padded = geometry.fitView(bounds, width, height, 48);
    const scale = Math.min(padded.scale, fitted * 12);
    const cx = (bounds.minX + bounds.maxX) / 2;
    const cy = (bounds.minY + bounds.maxY) / 2;
    view = { scale, tx: width / 2 - scale * cx, ty: height / 2 + scale * cy };
    schedule();
  }

  function resize() {
    const rect = surface.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const hgt = Math.max(1, Math.floor(rect.height));
    if (w === width && hgt === height) return;
    const first = width === 0 || view === null;
    // Keep the centre of the view where it is when the panel is resized.
    const centre = first ? null : geometry.toWorld(view, width / 2, height / 2);
    const zoom = first ? 1 : view.scale / fitted;
    width = w;
    height = hgt;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(hgt * ratio);
    if (!data) return;
    const fit = geometry.fitView(geometry.boundsOf(data.units), width, height, 20);
    fitted = fit.scale;
    if (first) view = fit;
    else {
      const scale = fitted * zoom;
      view = { scale, tx: width / 2 - scale * centre.x, ty: height / 2 + scale * centre.y };
    }
    schedule();
  }

  function schedule() {
    if (pending) return;
    pending = true;
    nextFrame(() => {
      pending = false;
      draw();
    });
  }

  function draw() {
    if (!data || !view || width === 0) return;
    if (!colours) readColours();
    const ratio = window.devicePixelRatio || 1;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    // Grid: solid hairlines, one step off the surface.
    const topLeft = geometry.toWorld(view, 0, 0);
    const bottomRight = geometry.toWorld(view, width, height);
    const step = geometry.niceStep(bottomRight.x - topLeft.x, 6);
    context.strokeStyle = colours.grid;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = Math.ceil(topLeft.x / step) * step; x <= bottomRight.x; x += step) {
      const sx = Math.round(view.tx + view.scale * x) + 0.5;
      context.moveTo(sx, 0);
      context.lineTo(sx, height);
    }
    for (let y = Math.ceil(bottomRight.y / step) * step; y <= topLeft.y; y += step) {
      const sy = Math.round(view.ty - view.scale * y) + 0.5;
      context.moveTo(0, sy);
      context.lineTo(width, sy);
    }
    context.stroke();

    const marked = markRetrieval ? highlightSet() : new Set();
    const zoom = view.scale / fitted;
    const radius = Math.min(4.5, 2.2 + 0.5 * Math.log2(Math.max(1, zoom)));
    const units = data.units;
    const colourOf = (u) => colours.series[slots.get(u.group) % SLOTS] || colours.muted;

    // Units outside the selected group recede; the selected group and the
    // marked units are drawn on top.
    const layers = [[], [], []];
    for (let i = 0; i < units.length; ++i) {
      const u = units[i];
      if (marked.has(u.id)) layers[2].push(i);
      else if (selected !== null && u.group !== selected) layers[0].push(i);
      else layers[1].push(i);
    }
    for (const i of layers[0]) {
      const u = units[i];
      context.beginPath();
      context.arc(view.tx + view.scale * u.x, view.ty - view.scale * u.y, radius, 0, TWO_PI);
      context.fillStyle = colours.dim;
      context.globalAlpha = 0.55;
      context.fill();
    }
    context.globalAlpha = 0.9;
    for (const i of layers[1]) {
      const u = units[i];
      context.beginPath();
      context.arc(view.tx + view.scale * u.x, view.ty - view.scale * u.y, radius, 0, TWO_PI);
      context.fillStyle = colourOf(u);
      context.fill();
    }
    context.globalAlpha = 1;
    for (const i of layers[2]) {
      const u = units[i];
      const x = view.tx + view.scale * u.x;
      const y = view.ty - view.scale * u.y;
      // Ring in the surface colour, then the outline in ink, then the fill.
      context.beginPath();
      context.arc(x, y, radius + 5, 0, TWO_PI);
      context.fillStyle = colours.surface;
      context.fill();
      context.beginPath();
      context.arc(x, y, radius + 3.25, 0, TWO_PI);
      context.strokeStyle = colours.ink;
      context.lineWidth = 1.5;
      context.stroke();
      context.beginPath();
      context.arc(x, y, radius + 1, 0, TWO_PI);
      context.fillStyle = colourOf(u);
      context.fill();
    }

    // The density mode of the selected group.
    if (selected !== null) {
      const g = data.groups.find((x) => x.id === selected);
      if (g && Number.isFinite(g.x) && Number.isFinite(g.y)) {
        const x = Math.round(view.tx + view.scale * g.x) + 0.5;
        const y = Math.round(view.ty - view.scale * g.y) + 0.5;
        context.strokeStyle = colours.ink;
        context.lineWidth = 1.5;
        context.beginPath();
        context.moveTo(x - 7, y);
        context.lineTo(x + 7, y);
        context.moveTo(x, y - 7);
        context.lineTo(x, y + 7);
        context.stroke();
      }
    }

    if (hover >= 0 && hover < units.length) {
      const u = units[hover];
      context.beginPath();
      context.arc(view.tx + view.scale * u.x, view.ty - view.scale * u.y, radius + 4, 0, TWO_PI);
      context.strokeStyle = colours.ink;
      context.lineWidth = 2;
      context.stroke();
    }
  }

  // ---- tooltip --------------------------------------------------------------------

  function showTooltip(index, sx, sy) {
    const u = data.units[index];
    const g = data.groups.find((x) => x.id === u.group);
    const retrieval = store.get('retrieval');
    const traced = retrieval && highlightSet().has(u.id) ? retrieval.units.find((r) => r.id === u.id) : null;
    replace(tooltip,
      h('div', { class: 'map-tooltip-title', text: unitTitle(u) }),
      h('div', { class: 'map-tooltip-path', text: shortenPath(String(u.path || ''), 44) }),
      h('dl', { class: 'facts' },
        fact('Group', h('span', null, h('i', { class: 'swatch', style: { background: colours.series[slots.get(u.group) % SLOTS] } }),
                        ` ${u.group}`)),
        fact('Log density', formatFixed(u.log_density, 2)),
        traced ? fact('Probability', formatProbability(traced.p)) : null),
      g && g.label ? h('div', { class: 'map-tooltip-label', text: g.label }) : null);
    tooltip.hidden = false;
    const box = tooltip.getBoundingClientRect();
    let left = sx + 14;
    let top = sy + 14;
    if (left + box.width > width - 4) left = Math.max(4, sx - box.width - 14);
    if (top + box.height > height - 4) top = Math.max(4, sy - box.height - 14);
    tooltip.style.setProperty('left', left + 'px');
    tooltip.style.setProperty('top', top + 'px');
  }

  function setHover(index, sx, sy) {
    if (index !== hover) {
      hover = index;
      schedule();
    }
    if (index < 0) tooltip.hidden = true;
    else showTooltip(index, sx, sy);
  }

  // ---- selection and table ----------------------------------------------------------

  function renderSelection() {
    selection.replaceChildren();
    if (!data || selected === null) return;
    const g = data.groups.find((x) => x.id === selected);
    if (!g) return;
    const members = data.units.filter((u) => u.group === selected && u.name !== '')
      .sort((a, b) => b.log_density - a.log_density).slice(0, 8);
    selection.appendChild(h('div', { class: 'selection-card' },
      h('div', { class: 'selection-head' },
        h('strong', null, h('i', { class: 'swatch', style: { background: colours.series[slots.get(g.id) % SLOTS] } }),
          ` Group ${g.id}`),
        h('button', { type: 'button', class: 'btn btn-quiet btn-small', text: 'Clear selection', on: { click: () => select(null) } })),
      h('dl', { class: 'facts' },
        fact('Units', formatCount(g.size)),
        fact('Log density of the mode', formatFixed(g.log_density, 2)),
        fact('Densest units', members.map((u) => u.name).join(', ') || 'none'))));
  }

  function renderTable() {
    if (!data) return;
    if (!colours) readColours();
    const groups = data.groups.slice().sort((a, b) => b.size - a.size || a.id - b.id);
    tableBody.replaceChildren(...groups.map((g) => {
      const button = h('button', { type: 'button', class: 'group-button', 'aria-pressed': String(selected === g.id),
                                   title: 'Show this group on the map' },
        h('i', { class: 'swatch', style: { background: colours.series[slots.get(g.id) % SLOTS] } }), ` ${g.id}`);
      button.addEventListener('click', () => {
        select(selected === g.id ? null : g.id);
        if (selected !== null) zoomTo(data.units.filter((u) => u.group === selected));
      });
      return h('tr', { 'data-selected': String(selected === g.id) },
        h('th', { scope: 'row' }, button),
        h('td', { class: 'num', text: formatCount(g.size) }),
        h('td', { class: 'num', text: formatFixed(g.log_density, 2) }),
        h('td', { class: 'label', text: String(g.label || '') }));
    }));
  }

  function select(group) {
    selected = group;
    renderSelection();
    renderTable();
    schedule();
  }

  function describe() {
    if (!data) return;
    const marked = data.units.filter((u) => highlightSet().has(u.id));
    summary.textContent = `${plural(data.units.length, 'unit')} in ${plural(data.groups.length, 'density group')}; ` +
      `concentration ${formatFixed(data.kappa, 1)}, reduced to ${formatCount(data.reduced_dim)} dimensions. ` +
      'Drag to move, scroll or pinch to zoom, click a unit to select its group.';
    retrievalNote.textContent = marked.length === 0 ? 'No retrieval to mark yet.'
      : `${plural(marked.length, 'unit')} of the last retrieval ${marked.length === 1 ? 'is' : 'are'} marked with a ring.`;
    zoomRetrieval.disabled = marked.length === 0;
    canvas.setAttribute('aria-label',
      `Map of ${plural(data.units.length, 'unit')} in ${plural(data.groups.length, 'density group')}. ` +
      `${retrievalNote.textContent} The table below lists the groups. ` +
      'Arrow keys move the map, plus and minus zoom, zero resets the view.');
  }

  // ---- interaction ------------------------------------------------------------------

  const pointers = new Map();
  let moved = 0;
  let pinch = 0;

  function position(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  canvas.addEventListener('pointerdown', (event) => {
    if (!data) return;
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, position(event));
    moved = 0;
    if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      pinch = Math.hypot(a.x - b.x, a.y - b.y);
    }
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!data || !view) return;
    const at = position(event);
    const before = pointers.get(event.pointerId);
    if (!before) {
      if (event.pointerType === 'mouse' || event.pointerType === 'pen') {
        setHover(geometry.nearestPoint(data.units, view, at.x, at.y, HIT_RADIUS), at.x, at.y);
      }
      return;
    }
    pointers.set(event.pointerId, at);
    if (pointers.size === 1) {
      moved += Math.abs(at.x - before.x) + Math.abs(at.y - before.y);
      if (moved > 4) {
        setHover(-1);
        canvas.classList.add('dragging');
        view = geometry.panBy(view, at.x - before.x, at.y - before.y);
        schedule();
      }
    } else if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && distance > 0) {
        view = geometry.zoomAt(view, distance / pinch, (a.x + b.x) / 2, (a.y + b.y) / 2, fitted);
        schedule();
      }
      pinch = distance;
      moved = 99;
    }
  });

  function release(event) {
    const at = position(event);
    const was = pointers.size;
    pointers.delete(event.pointerId);
    canvas.classList.remove('dragging');
    if (!data || !view || was !== 1 || moved > 4 || event.type === 'pointercancel') return;
    const index = geometry.nearestPoint(data.units, view, at.x, at.y, HIT_RADIUS);
    if (index < 0) {
      select(null);
      setHover(-1);
    } else {
      select(data.units[index].group);
      setHover(index, at.x, at.y);  // touch has no hover: a tap shows the details
    }
  }
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => {
    if (pointers.size === 0) setHover(-1);
  });

  canvas.addEventListener('wheel', (event) => {
    if (!data || !view) return;
    event.preventDefault();
    const at = position(event);
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    view = geometry.zoomAt(view, Math.exp(-delta * 0.0015), at.x, at.y, fitted);
    setHover(-1);
    schedule();
  }, { passive: false });

  canvas.addEventListener('keydown', (event) => {
    if (!data || !view) return;
    const step = 48;
    let handled = true;
    if (event.key === 'ArrowLeft') view = geometry.panBy(view, step, 0);
    else if (event.key === 'ArrowRight') view = geometry.panBy(view, -step, 0);
    else if (event.key === 'ArrowUp') view = geometry.panBy(view, 0, step);
    else if (event.key === 'ArrowDown') view = geometry.panBy(view, 0, -step);
    else if (event.key === '+' || event.key === '=') view = geometry.zoomAt(view, 1.4, width / 2, height / 2, fitted);
    else if (event.key === '-' || event.key === '_') view = geometry.zoomAt(view, 1 / 1.4, width / 2, height / 2, fitted);
    else if (event.key === '0') resetView();
    else if (event.key === 'Escape' && selected !== null) select(null);
    else handled = false;
    if (handled) {
      event.preventDefault();
      schedule();
    }
  });

  mark.addEventListener('change', () => {
    markRetrieval = mark.checked;
    schedule();
  });
  zoomRetrieval.addEventListener('click', () => {
    const marked = highlightSet();
    if (data) zoomTo(data.units.filter((u) => marked.has(u.id)));
  });

  const tool = (name, label, action) => {
    const button = h('button', { type: 'button', class: 'btn-icon', 'aria-label': label, title: label }, icon(name));
    button.addEventListener('click', action);
    return button;
  };
  const centreZoom = (factor) => {
    if (!view) return;
    view = geometry.zoomAt(view, factor, width / 2, height / 2, fitted);
    schedule();
  };
  const toolbar = h('div', { class: 'map-tools', role: 'group', 'aria-label': 'Map view' },
    tool('plus', 'Zoom in', () => centreZoom(1.5)),
    tool('minus', 'Zoom out', () => centreZoom(1 / 1.5)),
    tool('reset', 'Reset the view', resetView));

  // ---- data -------------------------------------------------------------------------

  function renderLoaded() {
    replace(panel,
      summary,
      h('div', { class: 'list-tools' },
        h('label', { class: 'check', htmlFor: 'mark-retrieval' }, mark, 'Mark the last retrieval'),
        zoomRetrieval, toolbar),
      surface,
      h('p', { class: 'note' }, retrievalNote, ' Colours repeat: they tell neighbouring groups apart, the table names them.'),
      selection,
      h('h3', { class: 'section-title', text: 'Groups' }),
      h('div', { class: 'table-scroll table-scroll-tall' },
        h('table', { class: 'table groups' },
          h('caption', { class: 'visually-hidden', text: 'Density groups, largest first' }),
          h('thead', null, h('tr', null,
            h('th', { scope: 'col', text: 'Group' }), h('th', { scope: 'col', class: 'num', text: 'Units' }),
            h('th', { scope: 'col', class: 'num', text: 'Log density' }), h('th', { scope: 'col', text: 'Densest units' }))),
          tableBody)));
    describe();
    renderTable();
    renderSelection();
    width = 0;
    height = 0;
    resize();
  }

  async function load(force = false) {
    const status = store.get('status');
    const built = status && status.index ? String(status.index.built_at || '') : '';
    if (loading || (!force && data !== null && loadedFor === built)) return;
    loading = true;
    if (data === null) panel.replaceChildren(h('p', { class: 'note', text: 'Loading the map' }));
    try {
      const response = await api.groups();
      if (!response || !Array.isArray(response.units) || !Array.isArray(response.groups)) throw new Error('unexpected response');
      response.units = response.units.filter((u) => Number.isFinite(u.x) && Number.isFinite(u.y));
      data = response;
      loadedFor = built;
      slots = geometry.assignColorSlots(data.groups, SLOTS);
      if (selected !== null && !data.groups.some((g) => g.id === selected)) selected = null;
      hover = -1;
      view = null;
      store.set('groups', data);
      if (data.units.length === 0) {
        panel.replaceChildren(emptyState('The index is empty', 'Add corpus roots in the settings and rebuild the index.'));
      } else {
        renderLoaded();
      }
    } catch (error) {
      if (data === null) {
        const retry = h('button', { type: 'button', class: 'btn btn-small', text: 'Try again', on: { click: () => load(true) } });
        panel.replaceChildren(banner('bad', 'The map could not be loaded.', error.message), retry);
      } else {
        toast.error(`The map could not be refreshed: ${error.message}`);
      }
    } finally {
      loading = false;
    }
  }

  if (typeof ResizeObserver === 'function') new ResizeObserver(() => resize()).observe(surface);
  else window.addEventListener('resize', resize);

  const scheme = window.matchMedia('(prefers-color-scheme: dark)');
  const recolour = () => {
    colours = null;
    if (data) {
      readColours();
      renderTable();
      renderSelection();
      schedule();
    }
  };
  if (typeof scheme.addEventListener === 'function') scheme.addEventListener('change', recolour);

  store.on('highlight', () => {
    describe();
    schedule();
  });

  panel.replaceChildren(emptyState('Map of the index', 'The map is loaded when this tab is opened.'));

  return {
    // Called when the tab is shown and when the index has been rebuilt.
    show: () => load(false).then(() => resize()),
    reload: () => (data !== null ? load(true) : Promise.resolve()),
  };
}
