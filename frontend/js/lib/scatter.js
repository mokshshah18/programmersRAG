// Geometry of the density groups map: view transform, zooming, panning, hit
// testing and the assignment of colour slots to groups. No DOM; the canvas
// code in ui/groups.js draws what these functions compute.
//
// World coordinates are the first two coordinates of the reduced embedding in
// [-1, 1], with y pointing up. Screen coordinates are CSS pixels with y
// pointing down. A view is { scale, tx, ty }:
//
//     sx = tx + scale * x          sy = ty - scale * y

export const MIN_ZOOM = 0.5;   // relative to the fitted view
export const MAX_ZOOM = 64;

// Bounding box of points with finite coordinates; the unit square when there are none.
export function boundsOf(points) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (minX > maxX) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

// View that shows `bounds` centred in a width x height area with `padding`
// pixels free on every side. The scale is the same on both axes.
export function fitView(bounds, width, height, padding = 24) {
  const w = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const h = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const availableW = Math.max(width - 2 * padding, 1);
  const availableH = Math.max(height - 2 * padding, 1);
  const scale = Math.min(availableW / w, availableH / h);
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  return { scale, tx: width / 2 - scale * cx, ty: height / 2 + scale * cy };
}

export function toScreen(view, x, y) {
  return { x: view.tx + view.scale * x, y: view.ty - view.scale * y };
}

export function toWorld(view, sx, sy) {
  return { x: (sx - view.tx) / view.scale, y: (view.ty - sy) / view.scale };
}

// Zooms by `factor` keeping the world point under the screen point (sx, sy)
// where it is. `fitted` is the scale of the fitted view; the zoom relative to
// it is kept within [MIN_ZOOM, MAX_ZOOM].
export function zoomAt(view, factor, sx, sy, fitted) {
  const lo = fitted * MIN_ZOOM;
  const hi = fitted * MAX_ZOOM;
  const scale = Math.min(hi, Math.max(lo, view.scale * factor));
  const world = toWorld(view, sx, sy);
  return { scale, tx: sx - scale * world.x, ty: sy + scale * world.y };
}

export function panBy(view, dx, dy) {
  return { scale: view.scale, tx: view.tx + dx, ty: view.ty + dy };
}

// Index of the point nearest to the screen position, or -1 when none lies
// within `radius` pixels. The pointer only has to be closest to a point, not
// on it. Ties go to the lower index.
export function nearestPoint(points, view, sx, sy, radius = 14) {
  let best = -1;
  let bestD = radius * radius;
  for (let i = 0; i < points.length; ++i) {
    const p = points[i];
    const dx = view.tx + view.scale * p.x - sx;
    const dy = view.ty - view.scale * p.y - sy;
    const d = dx * dx + dy * dy;
    if (d < bestD || (d === bestD && best < 0)) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

// Assigns one of `slots` colour slots to every group, the way a political map
// is coloured: there are more groups than distinguishable colours, so colours
// repeat, and what matters is that groups lying close to each other differ.
// Groups are taken in order of decreasing size (ties by id). The first `slots`
// groups get the slots in order; every later group gets the slot whose nearest
// group of that colour is farthest away. The result depends only on the groups,
// so a group keeps its colour whatever is filtered or selected in the view.
// Returns a Map from group id to slot index.
export function assignColorSlots(groups, slots = 8) {
  const order = groups
    .filter((g) => g && Number.isFinite(g.x) && Number.isFinite(g.y))
    .slice()
    .sort((a, b) => (b.size || 0) - (a.size || 0) || a.id - b.id);
  const result = new Map();
  const members = Array.from({ length: slots }, () => []);
  for (const g of order) {
    let slot = -1;
    let bestDistance = -1;
    for (let s = 0; s < slots; ++s) {
      if (members[s].length === 0) {
        slot = s;
        break;
      }
      let nearest = Infinity;
      for (const other of members[s]) {
        const d = Math.hypot(other.x - g.x, other.y - g.y);
        if (d < nearest) nearest = d;
      }
      if (nearest > bestDistance) {
        bestDistance = nearest;
        slot = s;
      }
    }
    members[slot].push(g);
    result.set(g.id, slot);
  }
  // Groups without coordinates still need a colour.
  for (const g of groups) {
    if (g && !result.has(g.id)) result.set(g.id, Math.abs(g.id | 0) % slots);
  }
  return result;
}

// "Nice" step for grid lines: 1, 2 or 5 times a power of ten, giving about
// `target` lines over `span`.
export function niceStep(span, target = 6) {
  if (!(span > 0) || !(target > 0)) return 1;
  const raw = span / target;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const fraction = raw / power;
  const nice = fraction < 1.5 ? 1 : fraction < 3.5 ? 2 : fraction < 7.5 ? 5 : 10;
  return nice * power;
}
