/* The chat answer footer (fix-weih.32): the results link (left), the turn status with
   "view trace" (centre) and the artifacts toggle (right) share ONE row per turn bubble.
   jsdom has no layout, so this asserts STRUCTURE: which slot of the one footer row holds
   each control, in DOM order, and that the page's CSS has the grid rule the slots rely on.
   The page's own tmRenderTurn, tmRenderStoredTurn and tmComposeCard run against the
   assembled page; only the network edge (window.fetch) is scripted. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');

const srcDir = process.argv[3];
const names = fs.readdirSync(srcDir).filter((name) => !name.startsWith('.')).sort();
const html = names.map((name) => fs.readFileSync(path.join(srcDir, name))).join('');
const css = fs.readFileSync(path.join(srcDir, '010-style.css'), 'utf8');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (err) => {
  if (err.type !== 'css-parsing') errors.push(String(err.message || err));
});
virtualConsole.on('error', (...args) => errors.push('console.error: ' + args.join(' ')));

let storedRecords = {};
let composeCard = null;
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url) => {
      url = String(url);
      if (url.endsWith('/compose_answer_card')) {
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({card: composeCard})});
      }
      const m = url.match(/\/api\/turn\/([^/?]+)/);
      if (m && storedRecords[decodeURIComponent(m[1])]) {
        const body = storedRecords[decodeURIComponent(m[1])];
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve(body)});
      }
      return new Promise(() => {});
    };
  },
});
const w = dom.window;
const d = w.document;

function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function until(fn, what) {
  for (let i = 0; i < 200; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
const book = {book_uid: 'u1', title: 'Dune', copies: 2};
function co(name, output, artifacts) {
  return {command_name: name, context: 'Library', command_parameters: null,
    command_response: Object.assign({success: true, response: 'ok'},
      output === undefined ? {} : {output}, artifacts ? {artifacts} : {})};
}
const WITH_RESULTS = () => co('show_book', book);
const WITH_ARTIFACTS = () => co('make_note', undefined, {note: 'an artifact'});
function turn(outputs, status, key) {
  return {turn_key: key === undefined ? 'k' + Math.random().toString(16).slice(2) : key,
    status: status || 'completed', success: true, answer: 'The answer.', command_outputs: outputs};
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
function kids(node) { return Array.from(node.children); }
/* The footer rows directly under the bubble message (never nested). */
function footers(msg) { return kids(msg).filter((n) => n.classList.contains('turnFooter')); }
function slots(msg) {
  const f = footers(msg);
  assert.equal(f.length, 1, 'exactly one footer row per turn: ' + f.length);
  const s = kids(f[0]);
  assert.deepEqual(s.map((n) => n.className),
    ['turnFooterSlot turnFooterLeft', 'turnFooterSlot turnFooterCentre', 'turnFooterSlot turnFooterRight'],
    'three slots in reading (and tab) order');
  return {row: f[0], left: s[0], centre: s[1], right: s[2]};
}
function expectInSlots(msg, want, label) {
  const s = slots(msg);
  assert.equal(s.left.querySelectorAll('.resultsLinkRow > button.resultsLink').length, want.results ? 1 : 0, label + ': results link in left');
  assert.equal(s.centre.querySelectorAll('.meta').length, want.status ? 1 : 0, label + ': status in centre');
  assert.equal(s.centre.querySelectorAll('.meta .traceLink').length, want.trace ? 1 : 0, label + ': view trace in centre');
  assert.equal(s.right.querySelectorAll('.artifactsLinkRow > button.artifactsLink').length, want.artifacts ? 1 : 0, label + ': artifacts toggle in right');
  /* no copy of any control outside its slot, and none twice */
  assert.equal(msg.querySelectorAll('.resultsLink').length, want.results ? 1 : 0, label + ': one results link');
  assert.equal(msg.querySelectorAll('.artifactsLink').length, want.artifacts ? 1 : 0, label + ': one artifacts toggle');
  assert.equal(msg.querySelectorAll('.meta').length, want.status ? 1 : 0, label + ': one status');
  return s;
}

(async () => {
  await until(() => typeof w.tmRenderTurn === 'function', 'page script ran');
  w.tm.commandSchemas = {};
  w.tm.baseUrl = 'http://127.0.0.1:39131';
  w.tm.connected = true;

  /* ---- all three present ------------------------------------------------- */
  const all = paint(turn([WITH_RESULTS(), WITH_ARTIFACTS()]));
  const s = expectInSlots(all, {results: true, status: true, trace: true, artifacts: true}, 'all three');
  assert.match(s.left.textContent, /^Show results \(1\)$/);
  assert.match(s.centre.textContent, /completed\s*view trace/);
  assert.match(s.right.textContent, /^1 artifact$/);
  assert.equal(all.lastElementChild, s.row, 'the footer is the last thing in the turn');
  const tab = Array.from(s.row.querySelectorAll('button')).map((b) => b.className);
  assert.deepEqual(tab, ['resultsLink', 'traceLink', 'artifactsLink'], 'tab order is left, centre, right');

  /* ---- the expanded artifact list and the results panel are outside the row */
  const artifactPanel = all.querySelector('.artifacts');
  assert.ok(artifactPanel && !s.row.contains(artifactPanel), 'the artifact list is not inside the footer row');
  assert.equal(artifactPanel.parentNode, all, 'it is a sibling of the footer row');
  s.right.querySelector('.artifactsLink').click();
  assert.equal(artifactPanel.hidden, false, 'the toggle still opens the list');
  assert.ok(!s.row.contains(artifactPanel));
  s.left.querySelector('.resultsLink').click();
  assert.equal(artifactPanel.hidden, true, 'opening results closes the artifact list as before');
  const panel = all.querySelector('.resultsPanel');
  assert.equal(panel.hidden, false);
  assert.ok(!s.row.contains(panel) && panel.parentNode === all, 'the results panel is a sibling of the bubble, not in the row');
  assert.equal(all.querySelector('.bubble').nextSibling, panel, 'and still follows the bubble');
  assert.ok(all.classList.contains('resultsOpen'));
  expectInSlots(all, {results: true, status: true, trace: true, artifacts: true}, 'results open');
  assert.equal(slots(all).row, s.row, 'opening the panel did not rebuild the row');
  assert.equal(s.left.querySelector('.resultsLink').textContent, 'Hide results (1)', 'link text is unchanged');
  s.left.querySelector('.resultsLink').click();
  expectInSlots(all, {results: true, status: true, trace: true, artifacts: true}, 'results closed again');

  /* ---- a repaint rebuilds one row, never two ------------------------------ */
  const out = turn([WITH_RESULTS(), WITH_ARTIFACTS()]);
  const again = paint(out);
  w.tmRenderTurn(again, out);
  w.tmRenderTurn(again, out);
  expectInSlots(again, {results: true, status: true, trace: true, artifacts: true}, 'repainted');
  w.tmRenderTurn(again, turn([]));
  expectInSlots(again, {status: true, trace: true}, 'repainted without results or artifacts');

  /* ---- missing pieces: every slot keeps its place -------------------------- */
  const onlyStatus = paint(turn([]));
  let o = expectInSlots(onlyStatus, {status: true, trace: true}, 'only status');
  assert.equal(o.left.childNodes.length, 0, 'empty left slot still exists');
  assert.equal(o.right.childNodes.length, 0, 'empty right slot still exists');
  expectInSlots(paint(turn([WITH_ARTIFACTS()])), {status: true, trace: true, artifacts: true}, 'status + artifacts');
  expectInSlots(paint(turn([WITH_RESULTS()])), {results: true, status: true, trace: true}, 'results + status');
  expectInSlots(paint(turn([], 'failed')), {status: true, trace: true}, 'error status');
  expectInSlots(paint(turn([], 'completed', '')), {status: true}, 'no turn key: no trace link');
  const waiting = paint(turn([WITH_ARTIFACTS()], 'awaiting_user'));
  expectInSlots(waiting, {status: true, trace: true, artifacts: true}, 'awaiting the user');
  assert.equal(waiting.lastElementChild, slots(waiting).row, 'the waiting note and cards sit above the footer');

  /* a running turn has painted nothing yet: no footer, no empty row */
  const running = w.tmBubble('agent', '...');
  assert.equal(footers(running).length, 0, 'a running turn has no footer row');
  assert.equal(running.querySelectorAll('.turnFooter, .turnFooterSlot').length, 0);

  /* ---- late pieces end in the right slots, with no duplicate rows ---------- */
  const lateOut = turn([WITH_RESULTS(), WITH_ARTIFACTS()]);
  const late = w.tmBubble('agent', 'x');
  w.tmRenderTurn(late, turn([]));
  expectInSlots(late, {status: true, trace: true}, 'status first');
  w.tmRenderResults(late, lateOut);
  expectInSlots(late, {results: true, status: true, trace: true}, 'results late');
  w.tmRenderArtifacts(late, lateOut.command_outputs);
  expectInSlots(late, {results: true, status: true, trace: true, artifacts: true}, 'artifacts later still');
  const late2 = w.tmBubble('agent', 'x');
  w.tmRenderTurn(late2, turn([]));
  w.tmRenderArtifacts(late2, lateOut.command_outputs);
  w.tmRenderResults(late2, lateOut);
  expectInSlots(late2, {results: true, status: true, trace: true, artifacts: true}, 'artifacts then results');
  assert.ok(late2.querySelector('.artifacts') && !slots(late2).row.contains(late2.querySelector('.artifacts')));

  /* ---- a restored turn: status first, results and artifacts with the record - */
  const key = 'stored-turn-1';
  const stored = {turn_key: key, status: 'completed', success: true, answer: 'The answer.',
    command_outputs: [WITH_RESULTS(), WITH_ARTIFACTS()]};
  storedRecords[key] = {turn: {record: {turn_output: stored}}};
  const restored = w.tmRenderStoredTurn({turn_key: key, status: 'completed', success: true,
    answer: 'The answer.', user_message: 'hi'}, true);
  expectInSlots(restored, {status: true, trace: true}, 'restored, before the record');
  await until(() => restored.querySelector('.artifactsLink'), 'restored artifacts');
  expectInSlots(restored, {results: true, status: true, trace: true, artifacts: true}, 'restored, after the record');
  assert.equal(restored.lastElementChild, slots(restored).row, 'the footer stays last on a restored turn');

  /* ---- the composed answer card replacing the answer keeps the row --------- */
  composeCard = {lead: 'Gist of it', blocks: []};
  const prose = 'A long paragraph of prose. '.repeat(21).slice(0, 550);
  const composed = w.tmBubble('agent', '...');
  composed.dataset.userMessage = 'what is going on';
  composed.dataset.deterministic = '';
  w.tmRenderTurn(composed, Object.assign(turn([WITH_ARTIFACTS()]), {answer: prose}));
  await until(() => composed.querySelector('.answerCard'), 'composed card drawn');
  expectInSlots(composed, {status: true, trace: true, artifacts: true}, 'composed card');
  assert.ok(!composed.querySelector('.cardStatusRow'), 'the composing status row is gone');

  /* ---- the CSS the slots rely on ------------------------------------------- */
  const rule = (sel) => {
    const m = css.match(new RegExp('(^|\\n)' + sel.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*\\{([^}]*)\\}'));
    assert.ok(m, 'CSS rule for ' + sel);
    return m[2].replace(/\s+/g, ' ');
  };
  assert.match(rule('.turnFooter'), /display:\s*grid/);
  assert.match(rule('.turnFooter'), /grid-template-columns:\s*minmax\(0, 1fr\) auto minmax\(0, 1fr\)/);
  assert.match(rule('.turnFooterLeft'), /justify-self:\s*start/);
  assert.match(rule('.turnFooterCentre'), /justify-self:\s*center/);
  assert.match(rule('.turnFooterRight'), /justify-self:\s*end/);
  const narrow = css.match(/@media \(max-width: 560px\) \{([\s\S]*?)\n\}/);
  assert.ok(narrow, 'a narrow-width rule exists');
  assert.match(narrow[1], /\.turnFooter \{[^}]*grid-template-columns:\s*minmax\(0, 1fr\)/, 'narrow: one column, same order');
  assert.ok(!/\.artifactsLinkRow\s*\{[^}]*margin:[^}]*12px/.test(css), 'the old indent on the artifacts toggle is gone');

  assert.deepEqual(errors, []);
  console.log('ok');
})().catch((e) => { console.error(e); process.exit(1); });
