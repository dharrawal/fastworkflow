/* The chat SPA's plan buttons under a part that no-agent mode did not run (bd fix-1ma8.6), against the assembled page.
   Only the network edge (window.fetch) is replaced: it records every request to the workflow server. */
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
virtualConsole.on('error', (...args) => errors.push('console.error: ' + args.join(' ')));

const BASE = 'http://127.0.0.1:9';
const requests = [];
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39133/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url, opts) => {
      const u = String(url);
      if (u.indexOf(BASE) !== 0) return new Promise(() => {});
      const body = opts && opts.body ? JSON.parse(opts.body) : null;
      requests.push({path: u.slice(BASE.length), body});
      return new Promise(() => {});
    };
  },
});
const w = dom.window;
const d = w.document;
const tm = w.tm;

function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }

const plan = (name, closest, description) => ({name, description: description === undefined ? 'does ' + name : description, closest});
const PLANS = () => [plan('Library/book_by_title', true), plan('Library/overdue_summary', true),
  plan('Library/author_books', false), plan('Library/book_loans', false)];
const entry = (request, plans, sub) => ({kind: 'not_run_here', command: null, phase: null, open_step: null, done: [],
  facts: {}, remaining: [], suggested_plan: '', means: 'from_scratch', sub_task: sub === undefined ? null : sub,
  reason: 'x', available_plans: plans, request});
const listText = (plans) => 'The closest planning commands available here:\n'
  + plans.filter((p) => p.closest).map((p) => '- `' + p.name + '`: ' + p.description).join('\n')
  + '\n\nOther planning commands available here:\n'
  + plans.filter((p) => !p.closest).map((p) => '- `' + p.name + '`: ' + p.description).join('\n');
const HEAD = 'No matching plan was found for this part. Free planning is not available on this server, because no agent '
  + 'model is set (LLM_AGENT).';
function turn(answer, entries, extra) {
  return Object.assign({turn_key: 'k' + Math.random().toString(16).slice(2), status: 'completed', success: true, answer,
    command_outputs: [{command_name: '', context: 'Plain', command_parameters: null,
      command_response: {success: true, response: answer, artifacts: {fw_not_run: entries}}}]}, extra || {});
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
const buttons = (msg, part) => Array.from(msg.querySelectorAll((part ? '.planGroup[data-part="' + part + '"] ' : '') + '.planBtn'));

async function main() {
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = {};
  idle();

  /* ---- 1. buttons from an fw_not_run entry: closest first and marked, description beside, list not shown twice --- */
  let mark = requests.length;
  const plans = PLANS();
  const answer = HEAD + '\n\n' + listText(plans) + '\n\nA suggested plan is in the data (fw_not_run).';
  const msg = paint(turn(answer, [entry('tell Ann a joke about cats', plans)]));
  let bs = buttons(msg);
  assert.deepEqual(bs.map((b) => b.dataset.plan), plans.map((p) => p.name));
  assert.deepEqual(bs.map((b) => b.classList.contains('closest')), [true, true, false, false]);
  assert.ok(text(bs[0]).indexOf('closest') !== -1 && text(bs[0]).indexOf('does Library/book_by_title') !== -1);
  assert.ok(bs.every((b) => !b.disabled), 'the last turn offers live buttons');
  const bubbleText = text(msg.querySelector('.bubble'));
  assert.ok(bubbleText.indexOf('No matching plan was found') !== -1, 'the answer text stays');
  assert.ok(bubbleText.indexOf('Library/author_books') === -1, 'the plan list is not shown twice');
  assert.ok(bubbleText.indexOf('A suggested plan is in the data') !== -1);
  mark = requests.length;   /* painting may ask the server for a composed card; a click is what is counted */

  /* ---- 2. a click sends ONLY the part's request with use_plan, and says so on the sent bubble --------------- */
  bs[2].click();
  assert.equal(requests.length, mark + 1);
  assert.equal(requests[mark].path, '/invoke_agent_stream');
  assert.equal(requests[mark].body.user_query, 'tell Ann a joke about cats');
  assert.equal(requests[mark].body.use_plan, 'Library/author_books');
  const sent = Array.from(d.querySelectorAll('.chatMsg.user')).pop();
  assert.ok(text(sent).indexOf('tell Ann a joke about cats') !== -1 && text(sent).indexOf('Run with the plan Library/author_books') !== -1);
  assert.ok(buttons(msg).every((b) => b.disabled), 'a click disables the buttons');
  assert.ok(msg.querySelector('.planChoices').classList.contains('disabled'));
  buttons(msg)[0].click();
  assert.equal(requests.length, mark + 1, 'a disabled button sends nothing');
  idle();

  /* ---- 3. a later message disables them; a turn that is not the last paints them disabled ------------------- */
  const first = paint(turn(answer, [entry('first request', plans)]));
  assert.ok(buttons(first).every((b) => !b.disabled));
  w.tmBubble('user', 'a later message');
  const later = paint(turn(answer, [entry('second request', plans)]));
  assert.ok(buttons(later).every((b) => !b.disabled));
  w.tmDisableLiveForms();
  assert.ok(buttons(first).every((b) => b.disabled) && buttons(later).every((b) => b.disabled));
  const stale = paint(turn(answer, [entry('old request', plans)]));
  w.tmBubble('user', 'and another');
  w.tmRenderTurn(stale, turn(answer, [entry('old request', plans)]));
  assert.ok(buttons(stale).every((b) => b.disabled), 'a turn with a later message is painted disabled');
  idle();

  /* ---- 4. a split turn: each ad hoc part has its own group, and a click sends only that part ----------------- */
  mark = requests.length;
  const p2 = [plan('Library/count_books', true), plan('Library/book_loans', false)];
  const split = paint(turn('### 1. one\n' + HEAD + '\n\n### 2. two\n' + HEAD,
    [entry('write me a haiku', plans, 1), entry('count the books', p2, 2)]));
  assert.equal(split.querySelectorAll('.planGroup').length, 2);
  assert.deepEqual(buttons(split, 2).map((b) => b.dataset.plan), ['Library/count_books', 'Library/book_loans']);
  buttons(split, 2)[0].click();
  assert.equal(requests[mark].body.user_query, 'count the books');
  assert.equal(requests[mark].body.use_plan, 'Library/count_books');
  idle();

  /* ---- 5. more than the cap: the first twelve show, "show all" reveals the rest ----------------------------- */
  const many = Array.from({length: 15}, (_, i) => plan('Ctx/plan_' + String(i).padStart(2, '0'), i < 2));
  const capped = paint(turn(HEAD, [entry('x', many)]));
  const visible = () => buttons(capped).filter((b) => !b.hidden);
  assert.equal(buttons(capped).length, 15);
  assert.equal(visible().length, 12);
  const toggle = capped.querySelector('.planMore');
  assert.equal(text(toggle), 'show all (15)');
  toggle.click();
  assert.equal(visible().length, 15);
  assert.equal(text(toggle), 'show fewer');
  toggle.click();
  assert.equal(visible().length, 12);
  idle();

  /* ---- 6. nothing to offer: no entry, an entry with no plans, a stopped command, a bad shape ---------------- */
  [turn(HEAD, []), turn(HEAD, [entry('x', [])]), turn(HEAD, [Object.assign(entry('x', plans), {command: 'Library/c'})]),
   turn(HEAD, [Object.assign(entry('x', plans), {request: undefined})]), turn(HEAD, [null, {available_plans: 'x'}]),
   turn(HEAD, [entry('x', [null, {name: 3}, 'text'])])].forEach((out) => {
    assert.equal(paint(out).querySelectorAll('.planChoices').length, 0);
  });

  /* ---- 7. a repaint of the same turn does not stack a second set -------------------------------------------- */
  const out = turn(answer, [entry('once', plans)]);
  const twice = paint(out);
  w.tmRenderTurn(twice, out);
  assert.equal(twice.querySelectorAll('.planChoices').length, 1);

  /* ---- 8. a restored turn paints the same buttons from the stored record, and repaints its text without the
          plan list; with a later message in the log they are disabled -------------------------------------------- */
  const restored = w.tmBubble('agent', '...');
  restored.dataset.restored = '1';
  w.tmRenderTurn(restored, {turn_key: 'r1', status: 'completed', success: true, answer, command_outputs: []});
  assert.ok(text(restored.querySelector('.bubble')).indexOf('Library/author_books') !== -1, 'the first paint is the text');
  assert.equal(restored.querySelectorAll('.planChoices').length, 0);
  w.tmRenderPlanChoices(restored, turn(answer, [entry('stored request', plans)]));
  assert.equal(buttons(restored).length, 4);
  assert.ok(text(restored.querySelector('.bubble')).indexOf('Library/author_books') === -1);
  w.tmBubble('user', 'next');
  const late = w.tmBubble('agent', '...');
  late.dataset.restored = '1';
  const earlier = w.tmBubble('agent', '...');
  earlier.dataset.restored = '1';
  w.tmBubble('user', 'after');
  w.tmRenderPlanChoices(earlier, turn(answer, [entry('stored request', plans)]));
  assert.ok(buttons(earlier).every((b) => b.disabled), 'a stored turn that is not the last is disabled');

  assert.deepEqual(errors, []);
}

main().then(() => {
  console.log('ok');
  process.exit(0);
}).catch((err) => {
  console.error(err && err.stack || err);
  console.error(errors.join('\n'));
  process.exit(1);
});
