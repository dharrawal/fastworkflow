/* The chat SPA's review form on a step whose planning-command match was ambiguous (bd fix-1ma8.1), exercised against
   the assembled page. Only the network edge (window.fetch) is replaced: it records every request to the workflow
   server. The reply the select composes is the edit the server's own reader takes (tests/test_plan_match_ambiguity.py
   feeds the same text to the real review_engine parser). */
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
function sentAfter(mark) { return requests.slice(mark); }
function type(node, value) { node.value = value; node.dispatchEvent(new w.Event('input', {bubbles: true})); }
function pick(select, value) { select.value = value; select.dispatchEvent(new w.Event('change', {bubbles: true})); }
function toggle(box, on) { box.checked = on; box.dispatchEvent(new w.Event('change', {bubbles: true})); }

const val = (name, value) => ({name, value, from_subtask: null, drafted: true});
const ALTS = [
  {name: 'Library/author_books', description: 'Every book by one author'},
  {name: 'Library/book_loans', description: 'The loans of one book'},
];
const sub = (goal, inputs, means, alternatives) => {
  const item = {goal, outcome: '', inputs: inputs || [], means, check: '', drafted_means: '', parts: [],
    parts_fingerprint: ''};
  if (alternatives) item.alternatives = alternatives;
  return item;
};
const planning = (name) => ({kind: 'planning', name});
const ITEMS = () => [
  sub('Who wrote Emma', [val('title', 'Emma')], planning('Library/book_by_title'), ALTS),
  sub('Count the books', [], planning('Library/count_books')),
  sub('Loans of Dune', [val('title', 'Dune')], planning('Library/book_loans'), ALTS.slice(0, 1)),
];
const review = (items) => ({kind: 'sub_tasks', title: 'Sub-tasks', revision: 0, items, changes_data: items.map(() => false)});
const VERBATIM = 'Sub-tasks:\n1. Who wrote Emma (title: Emma)\n   other plans that fit: Library/author_books, '
  + 'Library/book_loans; reply "step 1: use Library/author_books" to switch\nReply yes to run this.';
function awaiting(rev) {
  return {turn_key: 'k' + Math.random().toString(16).slice(2), status: 'awaiting_user', success: true, answer: VERBATIM,
    command_outputs: [{command_name: 'ask_user', context: 'Plain', command_parameters: null,
      command_response: {success: true, response: VERBATIM,
        artifacts: {question_kind: 'review', review: rev}}}]};
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
const step = (msg, n) => msg.querySelector('.reviewStep[data-step="' + n + '"]');
const selectOf = (msg, n) => step(msg, n).querySelector('.reviewPlanField select');
const primary = (msg) => msg.querySelector('.reviewActions .primary');
const skipOf = (msg, n) => step(msg, n).querySelector('.reviewSkip input');
const goalOf = (msg, n) => step(msg, n).querySelector('.reviewGoal');

async function main() {
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = {};
  idle();

  /* ---- 1. only the steps that carry alternatives get a select ------------------------------- */
  let mark = requests.length;
  const msg = paint(awaiting(review(ITEMS())));
  assert.ok(msg.querySelector('.reviewForm'));
  assert.ok(selectOf(msg, 1), 'step 1 has the select');
  assert.equal(selectOf(msg, 2), null, 'an unambiguous step has none');
  assert.ok(selectOf(msg, 3), 'step 3 has the select');
  assert.equal(msg.querySelectorAll('.reviewPlanField').length, 2);
  const one = selectOf(msg, 1);
  assert.deepEqual(Array.from(one.options).map((o) => o.value), ['', 'Library/author_books', 'Library/book_loans']);
  assert.equal(text(one.options[0]), 'Keep Library/book_by_title', 'the first option keeps the plan shown');
  assert.equal(one.value, '', 'nothing is switched until the person picks');
  assert.ok(text(one.options[1]).indexOf('Every book by one author') !== -1, 'the description is on the option');
  const label = msg.querySelector('label[for="' + one.id + '"]');
  assert.ok(label && text(label) === 'Other plans that fit:', 'the select is labelled');
  assert.deepEqual(Array.from(selectOf(msg, 3).options).map((o) => o.value), ['', 'Library/author_books']);
  assert.equal(text(primary(msg)), 'Run as shown', 'untouched: run as shown');
  assert.equal(sentAfter(mark).length, 0);

  /* ---- 2. picking a plan composes the existing means edit ------------------------------------- */
  pick(one, 'Library/book_loans');
  assert.equal(text(primary(msg)), 'Apply changes');
  primary(msg).click();
  let sent = sentAfter(mark);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].body.user_query, 'change 1: means=planning:Library/book_loans');
  assert.ok(Array.from(msg.querySelectorAll('.reviewForm select, .reviewForm button')).every((n) => n.disabled),
    'sending disables the form');
  idle();

  /* ---- 3. picking and putting back is no change; a goal edit and a plan switch are two clauses of one step;
          two steps are two edits; a skipped step's select is disabled -------------------------------------- */
  mark = requests.length;
  const msg2 = paint(awaiting(review(ITEMS())));
  pick(selectOf(msg2, 1), 'Library/author_books');
  assert.equal(text(primary(msg2)), 'Apply changes');
  pick(selectOf(msg2, 1), '');
  assert.equal(text(primary(msg2)), 'Run as shown', 'putting the shown plan back is no change');
  pick(selectOf(msg2, 1), 'Library/author_books');
  type(goalOf(msg2, 1), 'Who wrote Emma and Dune');
  pick(selectOf(msg2, 3), 'Library/author_books');
  toggle(skipOf(msg2, 3), true);
  assert.equal(selectOf(msg2, 3).disabled, true, 'a skipped step cannot be switched');
  assert.equal(w.tmComposeReviewEdits([{step: 1, goal: 'Who wrote Emma and Dune', plan: 'Library/author_books'},
    {step: 3, skip: true}]), 'change 1: goal=Who wrote Emma and Dune; change 1: means=planning:Library/author_books; drop 3');
  primary(msg2).click();
  sent = sentAfter(mark);
  assert.equal(sent[0].body.user_query,
    'change 1: goal=Who wrote Emma and Dune; change 1: means=planning:Library/author_books; drop 3');
  idle();

  /* ---- 4. a review with no alternatives is unchanged: no select, "yes" ------------------------------ */
  mark = requests.length;
  const plain = ITEMS().map((item) => { delete item.alternatives; return item; });
  const msg3 = paint(awaiting(review(plain)));
  assert.equal(msg3.querySelectorAll('.reviewPlanField, .reviewPlanSelect').length, 0);
  primary(msg3).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'yes');
  idle();

  /* ---- 5. malformed alternatives are ignored ----------------------------------------------------------- */
  const odd = ITEMS();
  odd[0].alternatives = [null, {name: 3}, {description: 'x'}, 'text'];
  odd[2].alternatives = 'Library/author_books';
  const msg4 = paint(awaiting(review(odd)));
  assert.equal(msg4.querySelectorAll('.reviewPlanField').length, 0);
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
