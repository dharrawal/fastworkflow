/* The chat SPA's review form (fix-weih.13), exercised against the assembled
   page. Only the network edge (window.fetch) is replaced: it records every
   request to the workflow server and lets the streaming agent endpoint hang.
   tests/fixtures/review_form_edits.json is the contract shared with
   tests/test_review_reply_grammar.py, which feeds the same replies to the
   real review_engine parser. */
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
const storedRecords = {};
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39132/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url, opts) => {
      const u = String(url);
      const stored = u.match(/\/api\/turn\/([^/?]+)/);
      if (stored && storedRecords[decodeURIComponent(stored[1])]) {
        requests.push({path: u, body: null});
        const body = storedRecords[decodeURIComponent(stored[1])];
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve(body)});
      }
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

function plain(value) { return JSON.parse(JSON.stringify(value)); }
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function until(fn, what) {
  for (let i = 0; i < 100; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }
function sentAfter(mark) { return requests.slice(mark); }
function type(node, value) { node.value = value; node.dispatchEvent(new w.Event('input', {bubbles: true})); }
function blur(node) { node.dispatchEvent(new w.Event('change', {bubbles: true})); }
function toggle(box, on) { box.checked = on; box.dispatchEvent(new w.Event('change', {bubbles: true})); }

const val = (name, value, drafted) => ({name, value, from_subtask: null, drafted: !!drafted});
const ref = (name, n) => ({name, value: null, from_subtask: n, drafted: false});
const sub = (goal, inputs, means, outcome) => ({goal, outcome: outcome || '', inputs: inputs || [],
  means: means || {kind: 'from_scratch', name: ''}, check: '', drafted_means: '', parts: [], parts_fingerprint: ''});
const ITEMS = () => [
  sub('List loans for Ann Lee', [val('member', 'Ann Lee', true), val('book', 'Dune'), val('check', 'loose'),
    val('goal', 'g0')], {kind: 'plain', name: 'list_loans'}, 'Ann\'s loans are listed'),
  sub('Show those loans', [ref('member', 1), val('title', 'Loans')], {kind: 'planning', name: 'show_loans'}),
  sub('Remove the oldest loan', [], undefined, 'The oldest loan is gone'),
];
/* As the server sends it (prompt_payloads): the badges the form draws, and changes_data, which the page does not read. */
const review = (items, extra) => Object.assign({kind: 'sub_tasks', title: 'Sub-tasks', revision: 0, items,
  changes_data: items.map((_, i) => i === 2), badges: items.map((_, i) => (i === 2 ? 'changes data' : ''))},
  extra || {});
const VERBATIM = 'Sub-tasks:\n1. List loans for Ann Lee (member: Ann Lee)\nReply yes to run this.';
function awaiting(rev, extra, kind, status) {
  return Object.assign({turn_key: 'k' + Math.random().toString(16).slice(2), status: status || 'awaiting_user',
    success: true, answer: VERBATIM,
    command_outputs: [{command_name: 'ask_user', context: 'Plain', command_parameters: null,
      command_response: {success: true, response: VERBATIM,
        artifacts: {question_kind: kind || 'review', review: rev}}}]}, extra || {});
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
const step = (msg, n) => msg.querySelector('.reviewStep[data-step="' + n + '"]');
const goalOf = (msg, n) => step(msg, n).querySelector('.reviewGoal');
const inputOf = (msg, n, name) => step(msg, n).querySelector('.reviewField[data-input="' + name + '"] input, .reviewField[data-input="' + name + '"] textarea');
const skipOf = (msg, n) => step(msg, n).querySelector('.reviewSkip input');
const primary = (msg) => msg.querySelector('.reviewActions .primary');
const cancel = (msg) => Array.from(msg.querySelectorAll('.reviewActions button')).find((b) => b.textContent === 'Cancel');
const noteOf = (field) => field.querySelector('.reviewFieldNote');

async function main() {
  for (const fn of ['tmComposeReviewEdits', 'tmReviewEditProblems', 'tmRenderReviewForm', 'tmReviewSanitize']) {
    assert.equal(typeof w[fn], 'function', fn);
  }
  assert.ok(w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderReviewForm) !== -1, 'registered');
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = {};
  idle();
  const composer = d.getElementById('chatInput');

  /* ---- 1. a 3-step sub_tasks review: rendering, yes, no ---------------------------------- */
  let mark = requests.length;
  const msg1 = paint(awaiting(review(ITEMS())));
  assert.ok(msg1.classList.contains('hasForm'));
  const form1 = msg1.querySelector('.reviewForm');
  assert.ok(form1);
  assert.equal(text(form1.querySelector('.reviewTitle')), 'Sub-tasks — check the steps before they run');
  assert.equal(form1.querySelector('.reviewRevision'), null, 'no banner on revision 0');
  assert.deepEqual(Array.from(msg1.querySelectorAll('.reviewStep')).map((n) => n.getAttribute('data-step')),
    ['1', '2', '3']);
  assert.equal(goalOf(msg1, 1).value, 'List loans for Ann Lee');
  assert.equal(goalOf(msg1, 2).value, 'Show those loans');
  const member1 = inputOf(msg1, 1, 'member');
  assert.equal(member1.value, 'Ann Lee');
  assert.ok(text(member1.closest('.reviewField').querySelector('label')).indexOf('(drafted)') !== -1);
  assert.ok(text(inputOf(msg1, 1, 'book').closest('.reviewField')).indexOf('(drafted)') === -1);
  const fixed = step(msg1, 2).querySelector('.reviewField[data-input="member"]');
  assert.ok(fixed.classList.contains('readonly'));
  assert.ok(text(fixed).indexOf('from step 1') !== -1);
  assert.equal(fixed.querySelector('input, textarea'), null, 'a reference is not editable');
  assert.equal(text(step(msg1, 1).querySelector('.reviewMeans')), 'runs list_loans');
  assert.equal(text(step(msg1, 2).querySelector('.reviewMeans')), 'uses show_loans');
  assert.equal(text(step(msg1, 3).querySelector('.reviewMeans')), 'built from scratch');
  assert.equal(text(step(msg1, 1).querySelector('.reviewOutcome')), 'Ann\'s loans are listed');
  assert.equal(step(msg1, 2).querySelector('.reviewOutcome'), null, 'an empty outcome is not shown');
  assert.equal(msg1.querySelectorAll('.reviewNote').length, 0, 'no note without check_notes');
  assert.equal(msg1.querySelectorAll('.reviewBadge').length, 1);
  assert.equal(text(step(msg1, 3).querySelector('.reviewBadge')), 'Changes data');
  const details1 = msg1.querySelector('details.reviewDetails');
  assert.ok(details1 && !details1.open, 'Details is closed');
  assert.equal(text(details1.querySelector('summary')), 'Details');
  assert.ok(details1.textContent.indexOf('Reply yes to run this.') !== -1, 'the verbatim prompt is in Details');
  assert.equal(details1.contains(form1), false);
  assert.equal(text(primary(msg1)), 'Run as shown');
  assert.equal(skipOf(msg1, 1).disabled, false);
  assert.equal(text(step(msg1, 1).querySelector('.reviewSkip')), 'Skip this step');
  Array.from(msg1.querySelectorAll('.reviewForm input[type=text], .reviewForm textarea')).forEach((c) => {
    const lab = msg1.querySelector('label[for="' + c.id + '"]');
    assert.ok(lab && lab.textContent.length, 'label for ' + c.id);
  });
  assert.equal(msg1.querySelectorAll('.cmdOut').length, 0);
  primary(msg1).click();
  let sent = sentAfter(mark);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].body.user_query, 'yes');
  assert.ok(Array.from(msg1.querySelectorAll('.reviewForm button, .reviewForm input, .reviewForm textarea')).every((n) => n.disabled),
    'sending disables the form');
  assert.ok(Array.from(d.querySelectorAll('.chatMsg.user .bubble')).some((n) => n.textContent === 'yes'),
    'the reply is ordinary transcript text');
  idle();
  mark = requests.length;
  const msg1b = paint(awaiting(review(ITEMS())));
  cancel(msg1b).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'no');
  idle();

  /* ---- 2. edits: input, goal, skip; Apply changes; revert ----------------------------------- */
  mark = requests.length;
  const msg2 = paint(awaiting(review(ITEMS())));
  type(inputOf(msg2, 1, 'member'), 'Ann B. Lee');
  assert.equal(text(primary(msg2)), 'Apply changes');
  type(inputOf(msg2, 1, 'member'), 'Ann Lee');
  assert.equal(text(primary(msg2)), 'Run as shown', 'reverting the value is no change');
  type(goalOf(msg2, 2), 'x');
  type(goalOf(msg2, 2), 'Show those loans');
  assert.equal(text(primary(msg2)), 'Run as shown');
  toggle(skipOf(msg2, 3), true);
  assert.equal(text(primary(msg2)), 'Apply changes');
  toggle(skipOf(msg2, 3), false);
  assert.equal(text(primary(msg2)), 'Run as shown', 'unskipping reverts');
  type(inputOf(msg2, 1, 'member'), 'Ann B. Lee');
  type(goalOf(msg2, 2), 'Show Ann Lee\'s loans');
  toggle(skipOf(msg2, 3), true);
  assert.equal(text(primary(msg2)), 'Apply changes');
  assert.ok(step(msg2, 3).classList.contains('skipped'));
  assert.ok(goalOf(msg2, 3).disabled, 'a skipped step is not editable');
  primary(msg2).click();
  sent = sentAfter(mark);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].body.user_query,
    'change 1: member=Ann B. Lee; change 2: goal=Show Ann Lee\'s loans; drop 3');
  idle();

  /* ---- 3. reserved input name ------------------------------------------------------------------- */
  mark = requests.length;
  const msg3 = paint(awaiting(review(ITEMS())));
  type(inputOf(msg3, 1, 'check'), 'strict');
  primary(msg3).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: inputs=check=strict');
  idle();

  /* ---- 4. min_items = 1 ------------------------------------------------------------------------- */
  const msg4 = paint(awaiting(review(ITEMS())));
  assert.ok(msg4.querySelector('.reviewGuard').hidden, 'no guard note while three steps remain');
  toggle(skipOf(msg4, 2), true);
  assert.equal(skipOf(msg4, 1).disabled, false);
  assert.equal(skipOf(msg4, 3).disabled, false);
  toggle(skipOf(msg4, 3), true);
  assert.equal(skipOf(msg4, 1).disabled, true, 'the last unskipped step cannot be skipped');
  assert.equal(skipOf(msg4, 1).title, 'At least one step must remain; click Cancel to decline the whole plan');
  assert.equal(skipOf(msg4, 2).disabled, false, 'a skipped step can be unskipped');
  assert.equal(skipOf(msg4, 3).disabled, false);
  assert.ok(!msg4.querySelector('.reviewGuard').hidden, 'the guard is explained on the page');
  assert.ok(msg4.querySelector('.reviewGuard').textContent.indexOf('click Cancel') !== -1);
  toggle(skipOf(msg4, 2), false);
  assert.equal(skipOf(msg4, 1).disabled, false, 'unskipping step 2 re-enables step 1');
  assert.equal(skipOf(msg4, 1).hasAttribute('title'), false);
  assert.equal(skipOf(msg4, 2).disabled, false);
  assert.ok(msg4.querySelector('.reviewGuard').hidden);
  toggle(skipOf(msg4, 2), true);
  assert.equal(skipOf(msg4, 1).disabled, true);
  mark = requests.length;
  primary(msg4).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'drop 2; drop 3', 'N-1 of N is sent');
  idle();
  /* a one-step proposal starts with the box disabled */
  const msg4b = paint(awaiting(review([ITEMS()[0]])));
  assert.equal(skipOf(msg4b, 1).disabled, true);
  assert.equal(skipOf(msg4b, 1).title, 'At least one step must remain; click Cancel to decline the whole plan');
  type(goalOf(msg4b, 1), 'Only step');
  assert.equal(text(primary(msg4b)), 'Apply changes');
  /* the guard also holds while the form is disabled and enabled again */
  const rec4b = msg4b.tmPrompt;
  rec4b.setDisabled(true);
  assert.ok(skipOf(msg4b, 1).disabled);
  rec4b.setDisabled(false);
  assert.ok(skipOf(msg4b, 1).disabled, 'still the last step');
  assert.equal(goalOf(msg4b, 1).disabled, false);

  /* ---- 5. revision banner ------------------------------------------------------------------------ */
  const msg5 = paint(awaiting(review(ITEMS(), {revision: 1})));
  const banner = msg5.querySelector('.reviewRevision');
  assert.ok(banner, 'banner on revision 1');
  assert.equal(text(banner), 'Updated plan (revision 1) — review and click Run as shown to confirm.');
  assert.equal(banner.closest('details'), null, 'the banner is not folded into Details');
  assert.equal(banner.hidden, false);
  assert.equal(banner.getAttribute('role'), 'status');
  assert.equal(text(paint(awaiting(review(ITEMS(), {revision: 3}))).querySelector('.reviewRevision'))
    .indexOf('Updated plan (revision 3)'), 0);

  /* ---- 6. sanitising ---------------------------------------------------------------------------- */
  mark = requests.length;
  const msg6 = paint(awaiting(review(ITEMS())));
  type(inputOf(msg6, 1, 'member'), 'A; B');
  primary(msg6).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: member=A, B');
  idle();

  /* ---- 7. plan kind --------------------------------------------------------------------------- */
  mark = requests.length;
  const planItems = [{goal: 'Fix the address', steps: ['update_address street=Main', 'confirm_address']},
    {goal: 'Send a receipt', steps: ['send_receipt']}, {goal: 'Close the ticket', steps: ['close_ticket id=7']}];
  const msg7 = paint(awaiting({kind: 'plan', title: 'Plan', revision: 0, items: planItems,
    changes_data: [false, false, false]}));
  assert.equal(text(msg7.querySelector('.reviewTitle')), 'Plan — check the steps before they run');
  assert.deepEqual(Array.from(step(msg7, 1).querySelectorAll('.reviewPlanSteps li')).map((n) => n.textContent),
    ['update_address street=Main', 'confirm_address']);
  assert.equal(msg7.querySelectorAll('.reviewValue').length, 0, 'a plan has no input fields');
  assert.equal(msg7.querySelectorAll('.reviewBadge, .reviewMeans').length, 0);
  assert.equal(msg7.querySelectorAll('.reviewPlanSteps input, .reviewPlanSteps textarea').length, 0, 'plan steps are read-only');
  type(goalOf(msg7, 1), 'Fix the address on file');
  toggle(skipOf(msg7, 3), true);
  primary(msg7).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: goal=Fix the address on file; drop 3');
  idle();

  /* ---- 8. emptied values: visible, restored on blur, never sent ---------------------------------- */
  mark = requests.length;
  const msg8 = paint(awaiting(review(ITEMS())));
  type(goalOf(msg8, 1), '');
  assert.equal(text(primary(msg8)), 'Run as shown', 'an emptied goal is unchanged');
  assert.equal(text(noteOf(goalOf(msg8, 1).closest('.reviewField'))), 'Can\'t be empty — the original value is kept.');
  assert.equal(noteOf(goalOf(msg8, 1).closest('.reviewField')).hidden, false);
  blur(goalOf(msg8, 1));
  assert.equal(goalOf(msg8, 1).value, 'List loans for Ann Lee', 'the original is put back on blur');
  assert.equal(noteOf(goalOf(msg8, 1).closest('.reviewField')).hidden, true);
  type(inputOf(msg8, 1, 'book'), '   ');
  assert.equal(noteOf(inputOf(msg8, 1, 'book').closest('.reviewField')).hidden, false);
  blur(inputOf(msg8, 1, 'book'));
  assert.equal(inputOf(msg8, 1, 'book').value, 'Dune');
  type(inputOf(msg8, 1, 'book'), '');
  primary(msg8).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'yes', 'nothing but empties is a plain yes');
  idle();

  /* ---- 9. what cannot be said is refused on the field ---------------------------------------------- */
  mark = requests.length;
  const msg9 = paint(awaiting(review(ITEMS())));
  type(inputOf(msg9, 1, 'member'), 'None');
  assert.equal(primary(msg9).disabled, true);
  assert.equal(inputOf(msg9, 1, 'member').getAttribute('aria-invalid'), 'true');
  assert.ok(text(noteOf(inputOf(msg9, 1, 'member').closest('.reviewField'))).indexOf('removes a value') !== -1);
  assert.equal(msg9.querySelector('.reviewProblemLine').hidden, false);
  primary(msg9).click();
  assert.equal(sentAfter(mark).length, 0, 'a refused edit is not sent');
  type(inputOf(msg9, 1, 'member'), 'Zed');
  assert.equal(primary(msg9).disabled, false);
  assert.equal(inputOf(msg9, 1, 'member').hasAttribute('aria-invalid'), false);
  assert.equal(msg9.querySelector('.reviewProblemLine').hidden, true);
  /* fix-2vyv: the server no longer reads an accept cue inside an edit's value as an acceptance, nor drops
     check/goal/... beside another input edit, so the form composes both (they are cases of the shared contract). */
  type(goalOf(msg9, 1), 'run it as shown please');
  assert.equal(primary(msg9).disabled, false, 'wording that sounds like accepting is an ordinary goal now');
  assert.equal(noteOf(goalOf(msg9, 1).closest('.reviewField')).hidden, true);
  type(goalOf(msg9, 1), 'Loans');
  type(inputOf(msg9, 1, 'check'), 'strict');
  assert.equal(primary(msg9).disabled, false, 'check may go with another input edit');
  assert.equal(w.tmComposeReviewEdits([{step: 1, goal: 'Loans', inputs: {member: 'Zed', check: 'strict'}}]),
    'change 1: goal=Loans; change 1: member=Zed; change 1: inputs=check=strict');
  type(inputOf(msg9, 1, 'member'), 'Ann Lee');
  assert.equal(primary(msg9).disabled, false);
  skipOf(msg9, 1).disabled || toggle(skipOf(msg9, 1), true);
  assert.equal(text(primary(msg9)), 'Apply changes');
  assert.equal(primary(msg9).disabled, false, 'a skipped step is not checked');

  /* ---- 10. guards: no form ------------------------------------------------------------------------------ */
  const noForm = (msg, why) => {
    assert.equal(msg.querySelectorAll('.reviewForm, .reviewDetails').length, 0, why);
    assert.ok(!msg.classList.contains('hasForm'), why);
    assert.ok(text(msg.querySelector('.bubble')).indexOf('Reply yes to run this.') !== -1, why + ': plain prompt stays');
  };
  noForm(paint(awaiting(review(ITEMS(), {kind: 'other'}))), 'unknown kind');
  noForm(paint(awaiting(review([]))), 'no items');
  noForm(paint(awaiting(review(ITEMS(), {items: 'x'}))), 'items not a list');
  noForm(paint(awaiting(review([{inputs: []}]))), 'item without a goal');
  noForm(paint(awaiting(review([{goal: 5}]))), 'goal not a string');
  noForm(paint(awaiting(review(['text']))), 'item not an object');
  noForm(paint(awaiting(review([sub('g', ['x'])]))), 'input not an object');
  noForm(paint(awaiting({kind: 'plan', title: 'Plan', revision: 0, items: [{goal: 'g', steps: [1]}]})), 'plan steps');
  noForm(paint(awaiting(null)), 'no review');
  noForm(paint(awaiting('review')), 'review is a string');
  /* other prompt kinds may have forms of their own (other parts); this one stays out */
  ['approval', 'agent', 'unknown'].forEach((kind) => {
    assert.equal(paint(awaiting(review(ITEMS()), null, kind)).querySelectorAll('.reviewForm').length, 0, kind);
  });
  noForm(paint(awaiting(review(ITEMS()), null, 'review', 'completed')), 'not awaiting');
  const notLast = awaiting(review(ITEMS()));
  notLast.command_outputs.push({command_name: 'x', command_response: {success: true, response: 'fine', artifacts: {}}});
  assert.equal(paint(notLast).querySelectorAll('.reviewForm').length, 0, 'the review is not on the last output');
  const early = w.tmBubble('agent', '...');
  w.tmBubble('user', 'later');
  w.tmRenderTurn(early, awaiting(review(ITEMS())));
  assert.equal(early.querySelectorAll('.reviewForm').length, 0, 'a turn with a later message is not live');
  /* a missing changes_data and revision still make a form */
  const lean = paint(awaiting({kind: 'sub_tasks', items: [{goal: 'Only'}]}));
  assert.equal(lean.querySelectorAll('.reviewForm').length, 1);
  assert.equal(text(lean.querySelector('.reviewTitle')), 'Sub-tasks — check the steps before they run');
  assert.equal(lean.querySelector('.reviewRevision'), null);

  /* ---- 11. restored turns: only the most recent waiting turn ------------------------------------------------ */
  const record = (rev) => ({turn: {record: {turn_output: {status: 'awaiting_user', command_outputs: [
    {command_name: 'ask_user', command_response: {success: true, response: VERBATIM,
      artifacts: {question_kind: 'review', review: rev}}}]}}}});
  storedRecords.rvA = record(review(ITEMS()));
  storedRecords.rvB = record(review(ITEMS()));
  const older = w.tmRenderStoredTurn({turn_key: 'rvA', status: 'awaiting_user', success: true, answer: VERBATIM}, true);
  const latest = w.tmRenderStoredTurn({turn_key: 'rvB', status: 'awaiting_user', success: true, answer: VERBATIM}, true);
  await until(() => latest.querySelector('.reviewForm') && older.dataset.pendingRecord === '', 'stored records');
  assert.equal(latest.querySelectorAll('.reviewForm').length, 1, 'the latest restored turn gets the form');
  assert.equal(older.querySelectorAll('.reviewForm').length, 0, 'an older one does not');
  assert.equal(latest.dataset.restored, '1');

  /* ---- 12. repaint ------------------------------------------------------------------------------------------ */
  const out12 = awaiting(review(ITEMS()));
  const msg12 = paint(out12);
  type(inputOf(msg12, 1, 'member'), 'Typed');
  const first = msg12.querySelector('.reviewForm');
  w.tmRenderTurn(msg12, out12);
  assert.equal(msg12.querySelectorAll('.reviewForm').length, 1);
  assert.equal(msg12.querySelector('.reviewForm'), first, 'the same form is put back');
  assert.equal(inputOf(msg12, 1, 'member').value, 'Typed');
  assert.equal(msg12.querySelectorAll('details.reviewDetails').length, 1);
  w.tmRenderTurn(msg12, awaiting(review(ITEMS())));
  assert.equal(msg12.querySelector('.reviewForm'), first, 'an equal payload from another copy of the turn');
  assert.equal(inputOf(msg12, 1, 'member').value, 'Typed');
  /* the server applied an edit: revision 1, new items */
  const revised = ITEMS();
  revised[0].inputs[0].value = 'Typed';
  revised.pop();
  w.tmRenderTurn(msg12, awaiting(review(revised, {revision: 1})));
  assert.equal(msg12.querySelectorAll('.reviewForm').length, 1);
  assert.notEqual(msg12.querySelector('.reviewForm'), first, 'a new payload rebuilds the form');
  assert.equal(msg12.querySelectorAll('.reviewStep').length, 2);
  assert.equal(inputOf(msg12, 1, 'member').value, 'Typed');
  assert.equal(text(primary(msg12)), 'Run as shown', 'the new items are the baseline');
  assert.ok(msg12.querySelector('.reviewRevision'));
  /* a repaint that no longer waits drops the form */
  w.tmRenderTurn(msg12, {turn_key: 'z', status: 'completed', success: true, answer: 'all good', command_outputs: []});
  assert.equal(msg12.querySelectorAll('.reviewForm').length, 0);
  assert.equal(text(msg12.querySelector('.bubble')), 'all good');

  /* ---- 13. a typed message disables the form ---------------------------------------------------------------- */
  const typed = paint(awaiting(review(ITEMS())));
  composer.value = 'hello';
  w.tmSend();
  assert.ok(Array.from(typed.querySelectorAll('.reviewForm button, .reviewForm input, .reviewForm textarea')).every((n) => n.disabled));
  idle();
  const live = paint(awaiting(review(ITEMS())));
  w.tmDisableLiveForms();
  assert.ok(Array.from(live.querySelectorAll('.reviewForm button, .reviewForm input, .reviewForm textarea')).every((n) => n.disabled));

  /* ---- the duplicate rule's note (SubTask.check_notes) is a plain line under its step ------------------------ */
  const noted = ITEMS();
  noted[1].check_notes = {title: "title left empty: 'Loans' is written once in your request and is the member."};
  const msgNote = paint(awaiting(review(noted)));
  assert.equal(text(step(msgNote, 2).querySelector('.reviewNote')),
    "title left empty: 'Loans' is written once in your request and is the member.");
  assert.equal(step(msgNote, 1).querySelector('.reviewNote'), null);
  assert.equal(step(msgNote, 2).querySelector('.reviewNote input, .reviewNote textarea'), null, 'not an input');

  /* ---- 14. markup stays text ------------------------------------------------------------------------------- */
  const evil = '<img src=x onerror=alert(1)>';
  const markup = paint(awaiting({kind: 'sub_tasks', title: '<b>t</b>', revision: 2, changes_data: [true],
    items: [{goal: evil, outcome: '<i>o</i>', means: {kind: 'plain', name: '<u>m</u>'},
      inputs: [{name: 'x_1', value: evil, from_subtask: null, drafted: true}]}]}));
  assert.equal(markup.querySelectorAll('.bubble img, .bubble b, .bubble i, .bubble u').length, 0);
  assert.equal(goalOf(markup, 1).value, evil);
  assert.equal(inputOf(markup, 1, 'x_1').value, evil);
  assert.ok(text(markup.querySelector('.reviewTitle')).indexOf('<b>t</b>') === 0);
  assert.ok(text(markup.querySelector('.reviewMeans')).indexOf('<u>m</u>') !== -1);
  const markupPlan = paint(awaiting({kind: 'plan', title: 'P', revision: 0, changes_data: [false],
    items: [{goal: 'g', steps: [evil]}]}));
  assert.equal(markupPlan.querySelectorAll('.bubble img').length, 0);
  assert.equal(text(markupPlan.querySelector('.reviewPlanSteps li')), evil);

  /* names that are not identifiers, repeated names and unnamed inputs are shown but not editable */
  const odd = paint(awaiting({kind: 'sub_tasks', title: 'S', revision: 0, changes_data: [false],
    items: [sub('g', [val('', 'bare'), val('a b', 'spaced'), val('dup', 'one'), val('dup', 'two')])]}));
  assert.equal(odd.querySelectorAll('.reviewValue').length, 1, 'only the first of a name is editable');
  assert.equal(odd.querySelector('.reviewValue').value, 'one');
  assert.equal(odd.querySelectorAll('.reviewField.readonly').length, 3);

  /* ---- 15. composition: pure function and the shared contract ------------------------------------------------ */
  assert.equal(w.tmComposeReviewEdits([]), 'yes');
  assert.equal(w.tmComposeReviewEdits(undefined), 'yes');
  assert.equal(w.tmComposeReviewEdits([{step: 1, goal: ' ', inputs: {a: ''}}]), 'yes');
  assert.equal(w.tmComposeReviewEdits([{step: 2, goal: 'B'}, {step: 1, skip: true}]), 'drop 1; change 2: goal=B');
  assert.equal(w.tmReviewSanitize(' a ;; b\r\nc '), 'a, b, c');
  const contract = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'review_form_edits.json'), 'utf8'));
  assert.ok(contract.length >= 25);
  let accepted = 0;
  let refused = 0;
  for (const c of contract) {
    if ('reply' in c) {
      accepted += 1;
      assert.deepEqual(plain(w.tmReviewEditProblems(c.edits)), [], c.name + ': nothing refused');
      assert.equal(w.tmComposeReviewEdits(c.edits), c.reply, c.name);
    } else {
      refused += 1;
      const problems = plain(w.tmReviewEditProblems(c.edits));
      assert.deepEqual(Array.from(new Set(problems.map((p) => p.field))).sort(), c.refused_fields.slice().sort(), c.name);
      problems.forEach((p) => assert.ok(p.message.length > 10, c.name));
      assert.equal(w.tmComposeReviewEdits(c.edits), c.naive_reply, c.name + ': what would have been sent');
    }
  }
  assert.ok(accepted >= 23 && refused >= 3);

  /* ---- 16. the same cases through the form itself ------------------------------------------------------------- */
  let driven = 0;
  for (const c of contract) {
    if (c.edits.some((e) => JSON.stringify(e).indexOf('\\n') !== -1)) continue; /* a text input holds no newline */
    const rev = {kind: c.kind, title: 'T', revision: 0, items: c.items, changes_data: c.items.map(() => false)};
    const msg = paint(awaiting(rev));
    const before = requests.length;
    for (const e of c.edits) {
      if (e.skip) { toggle(skipOf(msg, e.step), true); continue; }
      if (e.goal !== undefined) type(goalOf(msg, e.step), e.goal);
      for (const name of Object.keys(e.inputs || {})) {
        const box = inputOf(msg, e.step, name);
        if (box) type(box, e.inputs[name]);
      }
    }
    if ('reply' in c) {
      assert.equal(primary(msg).disabled, false, c.name);
      assert.equal(text(primary(msg)), c.reply === 'yes' ? 'Run as shown' : 'Apply changes', c.name);
      primary(msg).click();
      const got = sentAfter(before);
      assert.equal(got.length, 1, c.name);
      assert.equal(got[0].body.user_query, c.reply, c.name);
    } else {
      assert.equal(primary(msg).disabled, true, c.name + ': refused on the form');
      primary(msg).click();
      assert.equal(sentAfter(before).length, 0, c.name);
    }
    driven += 1;
    idle();
  }
  assert.ok(driven >= 28, 'driven ' + driven);

  /* ---- 17. the goal field: "1. Goal:" on one line, wrapping, growing [fix-weih.31] ------------------------ */
  const msg17 = paint(awaiting(review(ITEMS())));
  const goal17 = goalOf(msg17, 1);
  const label17 = msg17.querySelector('label[for="' + goal17.id + '"]');
  assert.ok(label17, 'the goal is labelled');
  assert.equal(text(label17), '1. Goal:', 'number and label are one text');
  assert.equal(label17.querySelector('.reviewNum').textContent, '1.');
  assert.equal(label17.querySelector('.reviewLabelText').textContent, 'Goal:');
  assert.equal(label17.querySelector('.reviewNum').parentNode, label17.querySelector('.reviewLabelText').parentNode,
    'number and label share a parent');
  assert.equal(label17.querySelector('.reviewNum').parentNode, label17);
  assert.equal(step(msg17, 1).querySelectorAll('.reviewNum').length, 1, 'one number per item, no header line');
  assert.equal(step(msg17, 1).querySelector('.reviewStepHead .reviewNum'), null, 'the number is not on a line of its own');
  assert.equal(text(step(msg17, 2).querySelector('label[for="' + goalOf(msg17, 2).id + '"]')), '2. Goal:');
  assert.equal(text(step(msg17, 3).querySelector('.reviewGoalLabel')), '3. Goal:');
  assert.equal(step(msg17, 1).querySelector('.reviewStepHead'), null, 'no badge, no header line');
  assert.equal(text(step(msg17, 3).querySelector('.reviewStepHead')), 'Changes data');
  assert.equal(text(inputOf(msg17, 1, 'member').closest('.reviewField').querySelector('label')), 'member (drafted):');
  assert.equal(text(inputOf(msg17, 1, 'book').closest('.reviewField').querySelector('label')), 'book:');
  assert.equal(step(msg17, 1).querySelector('.reviewField[data-input="book"] .reviewNum'), null, 'only the first label has the number');
  assert.equal(goal17.tagName, 'TEXTAREA');
  assert.equal(goal17.rows, 1);
  assert.equal(goal17.wrap, 'soft');
  assert.equal(step(msg17, 1).querySelectorAll('input.reviewGoal').length, 0);
  assert.equal(goal17.getAttribute('aria-describedby'), goal17.id.replace('-g1', '-gn1'));
  assert.equal(msg17.querySelectorAll('label[for="' + goal17.id + '"]').length, 1);

  /* the height decision is a pure function of measured numbers */
  assert.equal(w.tmReviewFitHeight(0, 18, 8, 2, 12), null, 'an unmeasured (hidden) field is left alone');
  assert.deepEqual(plain(w.tmReviewFitHeight(26, 18, 8, 2, 12)), {height: 28, scroll: false});
  assert.deepEqual(plain(w.tmReviewFitHeight(90, 18, 8, 2, 12)), {height: 92, scroll: false});
  assert.deepEqual(plain(w.tmReviewFitHeight(5000, 18, 8, 2, 12)), {height: 18 * 12 + 8 + 2, scroll: true});

  /* the wiring: input, a width change, a repaint, and being shown after being hidden */
  const scroll = (node, value) => Object.defineProperty(node, 'scrollHeight', {configurable: true, get: () => value.v});
  const sh = {v: 26};
  scroll(goal17, sh);
  const longGoal = 'A very long goal that would need sideways scrolling in a single line field. '.repeat(3);
  type(goal17, 'Short goal');
  const shortH = parseFloat(goal17.style.height);
  assert.ok(shortH >= 26, 'a short goal is one line: ' + goal17.style.height);
  sh.v = 130;
  type(goal17, longGoal);
  const longH = parseFloat(goal17.style.height);
  assert.ok(longH > shortH, 'a long goal grows the field');
  assert.equal(goal17.style.overflowY, 'hidden', 'no inner scrollbar for an ordinary length');
  sh.v = 4000;
  type(goal17, longGoal);
  assert.ok(parseFloat(goal17.style.height) < 4000, 'the growth is capped');
  assert.equal(goal17.style.overflowY, 'auto', 'past the cap it scrolls vertically');
  const cappedH = goal17.style.height;
  sh.v = 9000;
  type(goal17, longGoal + ' more');
  assert.equal(goal17.style.height, cappedH);
  /* the window is resized: the text rewraps, the field is measured again */
  sh.v = 60;
  w.dispatchEvent(new w.Event('resize'));
  assert.ok(parseFloat(goal17.style.height) < parseFloat(cappedH), 'a resize re-measures');
  assert.equal(goal17.style.overflowY, 'hidden');
  /* hidden: measures 0, keeps no stale height, and is measured when shown */
  sh.v = 0;
  w.dispatchEvent(new w.Event('resize'));
  assert.equal(goal17.style.height, '', 'a hidden field gets no height');
  sh.v = 78;
  w.dispatchEvent(new w.Event('resize'));
  assert.ok(parseFloat(goal17.style.height) >= 78, 'measured once shown');
  /* a repaint of the same prompt puts the form back and re-measures it */
  sh.v = 52;
  const out17 = awaiting(review(ITEMS()));
  const msg17b = paint(out17);
  const goal17b = goalOf(msg17b, 1);
  const sh2 = {v: 90};
  scroll(goal17b, sh2);
  type(goal17b, 'Typed');
  sh2.v = 44;
  w.tmRenderTurn(msg17b, out17);
  assert.equal(goalOf(msg17b, 1), goal17b, 'the same field is put back');
  assert.ok(parseFloat(goal17b.style.height) < 90 && parseFloat(goal17b.style.height) >= 44, 'a repaint re-measures: ' + goal17b.style.height);
  assert.equal(goal17b.value, 'Typed');
  /* first render: the field is measured when it is mounted (the hook is on the record) */
  assert.equal(typeof msg17b.tmPrompt.refit, 'function');

  /* one line, as the text input was: Enter inserts nothing, a pasted newline is not kept */
  const enter = new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true});
  goal17.dispatchEvent(enter);
  assert.equal(enter.defaultPrevented, true, 'Enter does not add a line');
  const other = new w.KeyboardEvent('keydown', {key: 'a', bubbles: true, cancelable: true});
  goal17.dispatchEvent(other);
  assert.equal(other.defaultPrevented, false);
  const msg17c = paint(awaiting(review(ITEMS())));
  mark = requests.length;
  type(goalOf(msg17c, 1), 'Line one\r\nline two');
  assert.equal(goalOf(msg17c, 1).value, 'Line oneline two', 'a newline is removed as a text input removes it');
  primary(msg17c).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: goal=Line oneline two');
  idle();
  const msg17d = paint(awaiting(review(ITEMS())));
  mark = requests.length;
  const paste = new w.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {value: {getData: () => ' more\r\nwords'}});
  goalOf(msg17d, 1).setSelectionRange(22, 22);
  goalOf(msg17d, 1).dispatchEvent(paste);
  assert.equal(goalOf(msg17d, 1).value, 'List loans for Ann Lee more words');
  primary(msg17d).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: goal=List loans for Ann Lee more words');
  idle();

  /* submit payload and edited-detection are what they were with the input */
  const padded = review([sub('  Tidy the shelf  ', [val('member', 'Ann')])]);
  const msg17e = paint(awaiting(padded));
  assert.equal(goalOf(msg17e, 1).value, '  Tidy the shelf  ', 'the value is not trimmed on render');
  assert.equal(text(primary(msg17e)), 'Run as shown', 'an untouched goal is not an edit');
  type(goalOf(msg17e, 1), 'Tidy the shelf');
  assert.equal(text(primary(msg17e)), 'Run as shown', 'only whitespace differs: not an edit');
  type(goalOf(msg17e, 1), '   Tidy the shelf and the desk   ');
  mark = requests.length;
  primary(msg17e).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: goal=Tidy the shelf and the desk', 'trimmed as before');
  idle();
  const msg17f = paint(awaiting(review(ITEMS())));
  type(goalOf(msg17f, 1), '');
  assert.equal(noteOf(goalOf(msg17f, 1).closest('.reviewField')).hidden, false);
  blur(goalOf(msg17f, 1));
  assert.equal(goalOf(msg17f, 1).value, 'List loans for Ann Lee');
  w.tmRenderTurn(msg17f, awaiting(review(ITEMS())));

  /* ---- 18 (fix-weih.33): the input values wrap and grow like the goal; one shared helper ---- */
  assert.equal(w.tmFitHeight(26, 18, 8, 2, 12).height, 28, 'the shared helper');
  assert.equal(w.tmReviewFitHeight(26, 18, 8, 2, 12).height, 28, 'the old name is an alias');
  const msg18 = paint(awaiting(review([sub('Tidy the shelf', [val('member', 'Ann Lee'), val('book', 'Dune')])])));
  const mem18 = inputOf(msg18, 1, 'member');
  assert.equal(mem18.tagName, 'TEXTAREA');
  assert.ok(mem18.classList.contains('fitText') && mem18.classList.contains('reviewValue'));
  assert.equal(mem18.rows, 1);
  assert.equal(mem18.wrap, 'soft');
  assert.equal(mem18.value, 'Ann Lee');
  assert.equal(msg18.querySelectorAll('label[for="' + mem18.id + '"]').length, 1);
  assert.equal(mem18.getAttribute('aria-describedby'), mem18.id + '-n');
  assert.equal(msg18.querySelectorAll('input[type=text]').length, 0, 'no single-line text input left');
  const sh18 = {v: 26};
  Object.defineProperty(mem18, 'scrollHeight', {configurable: true, get: () => sh18.v});
  type(mem18, 'Ann');
  const one18 = parseFloat(mem18.style.height);
  sh18.v = 110;
  type(mem18, 'Ann Lee '.repeat(30));
  assert.ok(parseFloat(mem18.style.height) > one18, 'a long value grows the field');
  assert.equal(mem18.style.overflowY, 'hidden');
  sh18.v = 9000;
  type(mem18, 'Ann Lee '.repeat(31));
  assert.equal(mem18.style.overflowY, 'auto');
  /* Enter did nothing in the input; a newline is removed or, pasted, a space */
  const enter18 = new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true});
  mark = requests.length;
  mem18.dispatchEvent(enter18);
  assert.equal(enter18.defaultPrevented, true);
  assert.equal(requests.length, mark);
  type(mem18, 'Bo\r\nb');
  assert.equal(mem18.value, 'Bob');
  type(mem18, 'Bo');
  mem18.setSelectionRange(2, 2);
  const paste18 = new w.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste18, 'clipboardData', {value: {getData: () => ' Lee\nJr'}});
  mem18.dispatchEvent(paste18);
  assert.equal(mem18.value, 'Bo Lee Jr');
  /* the reply is unchanged, spaces trimmed as before; an emptied value is still not an edit */
  type(mem18, '   Bo Lee   ');
  mark = requests.length;
  primary(msg18).click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'change 1: member=Bo Lee');
  idle();
  const msg18b = paint(awaiting(review([sub('Tidy the shelf', [val('member', 'Ann Lee')])])));
  type(inputOf(msg18b, 1, 'member'), '');
  assert.equal(noteOf(inputOf(msg18b, 1, 'member').closest('.reviewField')).hidden, false, 'an emptied value is still flagged');
  blur(inputOf(msg18b, 1, 'member'));
  assert.equal(inputOf(msg18b, 1, 'member').value, 'Ann Lee', 'and restored on leaving it');

  /* bd fix-ftdl.4.3: a part the user asked only to look up carries the payload's badge, as the review text words it */
  const msgLook = paint(awaiting(review(ITEMS(), {changes_data: [false, false, true], badges: ['look up only', '', 'changes data']})));
  /* The page reads only badges: a payload with changes_data and no badges draws no badge (the old fallback is gone). */
  const msgNoBadges = paint(awaiting(review(ITEMS(), {changes_data: [true, true, true], badges: undefined})));
  assert.equal(msgNoBadges.querySelectorAll('.reviewBadge').length, 0);
  assert.equal(text(step(msgLook, 1).querySelector('.reviewBadge')), 'Look up only');
  assert.equal(step(msgLook, 2).querySelector('.reviewBadge'), null);
  assert.equal(text(step(msgLook, 3).querySelector('.reviewBadge')), 'Changes data');

  assert.deepEqual(errors, [], 'jsdom reported errors');
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
