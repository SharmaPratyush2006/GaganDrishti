/** The DOM surface OrbitControls, first-person and measure use, on Node's EventTarget (no jsdom). */
export function fakeCanvas({ width = 800, height = 500 } = {}) {
  const el = new EventTarget();
  const doc = new EventTarget();
  Object.assign(el, {
    style: {}, ownerDocument: doc, getRootNode: () => doc, clientWidth: width, clientHeight: height,
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
    setPointerCapture() {}, releasePointerCapture() {},
  });
  return el;
}

/** A DOM-like event with extra fields (Node has no KeyboardEvent/PointerEvent). */
export function domEvent(type, fields = {}) {
  return Object.assign(new Event(type), fields);
}

/** A controllable frame scheduler and clock. */
export function manualFrames() {
  const queue = [];
  let t = 1000;
  return {
    queue,
    schedule: (fn) => queue.push(fn),
    now: () => t,
    advance(ms) { t += ms; },
    /** Run queued frames; each run may queue more (bounded). */
    flush(max = 100) {
      for (let i = 0; i < max && queue.length; i++) queue.shift()();
    },
    /** Run exactly what is queued now (one animation frame's worth). */
    runFrame() {
      queue.splice(0).forEach((fn) => fn());
    },
  };
}
