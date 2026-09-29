// Tabs of the inspector, following the WAI-ARIA tabs pattern: one tab is in
// the tab order, the arrow keys move between tabs and select them, Home and
// End jump to the first and the last tab.

const FADE = 40;  // width of the faded edge in pixels, as in style.css

export function createTabs(tablist, onSelect) {
  const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
  const panelOf = (tab) => document.getElementById(tab.getAttribute('aria-controls'));
  const nameOf = (tab) => tab.id.replace(/^tab-/, '');

  function select(name, focus = false) {
    const target = tabs.find((tab) => nameOf(tab) === name);
    if (!target) return;
    for (const tab of tabs) {
      const selected = tab === target;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      panelOf(tab).hidden = !selected;
    }
    if (focus) target.focus({ preventScroll: true });
    reveal();
    if (onSelect) onSelect(name);
  }

  // On a narrow screen the tab list scrolls sideways. The style sheet fades
  // the edge behind which more tabs are waiting; this tells it which edge.
  function markEdges() {
    const hiddenRight = tablist.scrollWidth - tablist.clientWidth - tablist.scrollLeft > 1;
    const hiddenLeft = tablist.scrollLeft > 1;
    tablist.dataset.more = hiddenLeft && hiddenRight ? 'both' : hiddenLeft ? 'start' : hiddenRight ? 'end' : 'none';
  }

  // Scrolls the tab list so that the selected tab is visible and clear of the
  // faded edges. Does nothing while the list is not displayed.
  function reveal() {
    const target = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true');
    const box = tablist.getBoundingClientRect();
    if (target && box.width > 0 && tablist.scrollWidth > tablist.clientWidth) {
      const rect = target.getBoundingClientRect();
      if (rect.left < box.left + FADE) tablist.scrollLeft -= box.left + FADE - rect.left;
      else if (rect.right > box.right - FADE) tablist.scrollLeft += rect.right - (box.right - FADE);
    }
    markEdges();
  }
  tablist.addEventListener('scroll', markEdges, { passive: true });
  if (typeof ResizeObserver === 'function') new ResizeObserver(reveal).observe(tablist);
  else window.addEventListener('resize', reveal);
  reveal();

  tablist.addEventListener('click', (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) select(nameOf(tab));
  });

  tablist.addEventListener('keydown', (event) => {
    const at = tabs.indexOf(document.activeElement);
    if (at < 0) return;
    let next = -1;
    if (event.key === 'ArrowRight') next = (at + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (at + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    if (next < 0) return;
    event.preventDefault();
    select(nameOf(tabs[next]), true);
  });

  return {
    select,
    // To be called when the tab list has become visible or has changed its width.
    refresh: reveal,
    selected: () => nameOf(tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0]),
    // Small marker on a tab, used to show that its content is new.
    mark(name, on) {
      const tab = tabs.find((t) => nameOf(t) === name);
      if (tab) tab.classList.toggle('tab-new', on && tab.getAttribute('aria-selected') !== 'true');
    },
  };
}
