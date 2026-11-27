/* The live panel's stage text and elapsed timer (fix-kq49), against the assembled page. Only the
   network edge (window.fetch) is replaced. Frames go through the page's own tmApplyStreamFrame and
   the real tmActivityPanel / tmRenderTurn; restored turns go through the real stored path. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');

const srcDir = process.argv[3];
const names = fs.readdirSync(srcDir).filter((name) => !name.startsWith('.')).sort();
const html = names.map((name) => fs.readFileSync(path.join(srcDir, name))).join('');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (err) => {
  if (err.type !== 'css-parsing') errors.push(String(err.message || err));
});

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39132/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = () => new Promise(() => {});
  },
});
const w = dom.window;
const d = w.document;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sink = {output: () => {}, error: () => {}};

(async () => {
  await sleep(100);
  assert.equal(w.TM_PROGRESS_TICK_MS, 1000, 'the timer ticks once a second');
  w.TM_PROGRESS_TICK_MS = 30;            /* the test must not wait whole seconds */

  assert.equal(w.tmFormatElapsed(0), '0s');
  assert.equal(w.tmFormatElapsed(12900), '12s');
  assert.equal(w.tmFormatElapsed(65000), '1m 05s');
  assert.equal(w.tmFormatElapsed(600000), '10m 00s');

  const parts = (msg) => ({
    stage: msg.querySelector('.actStage'),
    elapsed: msg.querySelector('.actElapsed'),
  });

  /* ---- the timer appears with no progress event, and ticks -------------- */
  const msg = w.tmBubble('agent', '…');
  const activity = w.tmActivityPanel(msg);
  let p = parts(msg);
  assert.ok(p.stage && p.elapsed, 'the stage and timer are in the live panel');
  assert.match(p.elapsed.textContent, /^· 0s$/);
  assert.equal(p.stage.textContent, '', 'no stage until one is announced');
  assert.equal(p.stage.getAttribute('role'), 'status');
  assert.equal(p.stage.getAttribute('aria-live'), 'polite');
  assert.equal(p.elapsed.closest('[aria-live]'), null, 'the timer is not in a live region');
  assert.equal(p.elapsed.getAttribute('aria-live'), null);
  /* the timer reads the clock, so a stalled interval cannot under-report */
  const realNow = w.Date.now;
  let offset = 0;
  w.Date.now = () => realNow() + offset;
  offset = 14000;
  await sleep(120);
  assert.match(p.elapsed.textContent, /^· 14s$/);
  offset = 75000;
  await sleep(120);
  assert.match(p.elapsed.textContent, /^· 1m 15s$/);

  /* ---- the stage text follows progress frames --------------------------- */
  const frame = (type, data) => ({type, seq: null, data});
  w.tmApplyStreamFrame(frame('progress', {stage: 'split', label: 'Splitting the request'}), activity, sink);
  assert.equal(p.stage.textContent, '· Splitting the request…');
  w.tmApplyStreamFrame(frame('progress',
    {stage: 'match', label: 'Matching sub-task 2 of 3', index: 2, total: 3}), activity, sink);
  assert.equal(p.stage.textContent, '· Matching sub-task 2 of 3…');
  /* a stage frame is not an activity step */
  assert.equal(msg.querySelectorAll('.actRow').length, 0);
  assert.equal(msg.querySelector('.actCount').textContent, 'no steps yet');

  /* ---- unknown kinds and malformed payloads are ignored ----------------- */
  const before = msg.querySelector('summary').textContent;
  w.tmApplyStreamFrame(frame('something_new', {label: 'Never shown'}), activity, sink);
  w.tmApplyStreamFrame(frame('progress', null), activity, sink);
  w.tmApplyStreamFrame(frame('progress', {stage: 'x'}), activity, sink);
  w.tmApplyStreamFrame(frame('progress', {label: 42}), activity, sink);
  assert.equal(msg.querySelector('summary').textContent, before);
  assert.equal(msg.querySelectorAll('.actNote').length, 0);

  /* the label goes through textContent: markup in it is text */
  w.tmApplyStreamFrame(frame('progress', {label: '<img src=x onerror=alert(1)>'}), activity, sink);
  assert.equal(p.stage.querySelector('img'), null);
  w.tmApplyStreamFrame(frame('progress', {label: 'Drafting sub-task 1 of 2'}), activity, sink);

  /* ---- completion clears the stage and stops the timer ------------------ */
  w.tmApplyStreamFrame(frame('output', {status: 'completed', success: true, answer: 'Done.',
    command_outputs: []}), activity, {
      output: (out) => { activity.setState('completed'); w.tmRenderTurn(msg, out); },
      error: () => {}});
  assert.equal(p.stage.textContent, '');
  const frozen = p.elapsed.textContent;
  assert.match(frozen, /^· 1m 15s$/);
  offset = 200000;
  await sleep(120);
  assert.equal(p.elapsed.textContent, frozen, 'the timer stopped when the turn completed');
  w.tmApplyStreamFrame(frame('progress', {label: 'Late stage'}), activity, sink);
  assert.equal(p.stage.textContent, '', 'a stage after completion is not shown');

  /* ---- waiting on the user stops it too --------------------------------- */
  offset = 0;
  const waiting = w.tmBubble('agent', '…');
  const waitAct = w.tmActivityPanel(waiting);
  const wp = parts(waiting);
  w.tmApplyStreamFrame(frame('progress', {label: 'Preparing the sub-task review'}), waitAct, sink);
  assert.match(wp.stage.textContent, /Preparing/);
  waitAct.setState('waiting for your reply');
  assert.equal(wp.stage.textContent, '');
  const waitFrozen = wp.elapsed.textContent;
  offset = 90000;
  await sleep(120);
  assert.equal(wp.elapsed.textContent, waitFrozen);

  /* recovery states keep it running: the turn still runs server side */
  const rec = w.tmBubble('agent', '…');
  const recAct = w.tmActivityPanel(rec);
  recAct.setState('disconnected — recovering');
  recAct.setState('still working (past the 300s deadline)');
  offset = 90000 + 120000;
  await sleep(120);
  assert.match(parts(rec).elapsed.textContent, /^· 2m 00s$/);
  recAct.setState('failed');
  assert.equal(parts(rec).stage.textContent, '');

  /* ---- a source change or a removed panel cleans the timer up ----------- */
  const gone = w.tmBubble('agent', '…');
  const goneAct = w.tmActivityPanel(gone);
  w.tm.epoch += 1;
  await sleep(120);
  const goneText = parts(gone).elapsed.textContent;
  offset = 500000;
  await sleep(120);
  assert.equal(parts(gone).elapsed.textContent, goneText, 'the timer stopped when the source changed');
  const removed = w.tmBubble('agent', '…');
  const removedAct = w.tmActivityPanel(removed);
  const removedParts = parts(removed);
  removed.remove();
  await sleep(120);
  const removedText = removedParts.elapsed.textContent;
  offset = 900000;
  await sleep(120);
  assert.equal(removedParts.elapsed.textContent, removedText, 'the timer stopped when the panel left the page');

  /* ---- restored turns show nothing -------------------------------------- */
  const restored = w.tmBubble('agent', 'An earlier answer');
  restored.dataset.restored = '1';
  w.tmRenderTurn(restored, {status: 'completed', success: true, answer: 'An earlier answer', command_outputs: []});
  assert.equal(restored.querySelector('.actStage'), null);
  assert.equal(restored.querySelector('.actElapsed'), null);
  assert.equal(restored.querySelector('.activity'), null);

  w.Date.now = realNow;
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch((e) => { process.stderr.write(e.stack + '\n'); process.exit(1); });
