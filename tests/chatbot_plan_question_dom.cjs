/* The chat SPA's question "which plan should run this part?" (bd fix-yu3l.5), against the assembled page.
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
  plan('Library/author_books', false), plan('branch_names', false)];
const QUESTION = "No planning command matched this request: 'tell Ann a joke about cats'. Which one should run it?\n"
  + '1. Library/book_by_title: does Library/book_by_title\n'
  + "Reply with a name or number, 'none' if none of these fits, or cancel.";
let seq = 0;
function turn(payload, extra) {
  return Object.assign({turn_key: 'q' + (++seq), status: 'awaiting_user', success: false, answer: QUESTION,
    command_outputs: [{command_name: 'ask_user', ask_user_entry: true, context: 'Library', command_parameters: null,
      command_response: {success: false, response: QUESTION,
        artifacts: {question_kind: 'plan_choice', plan_choice: payload}}}]}, extra || {});
}
const choice = (plans, sub, request) => ({request: request === undefined ? 'tell Ann a joke about cats' : request,
  sub_task: sub === undefined ? null : sub, choices: plans, none_reply: 'none', decline: 'cancel'});
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
const buttons = (msg) => Array.from(msg.querySelectorAll('.planQuestion .planBtn'));
const control = (msg, cls) => msg.querySelector('.planQuestion .' + cls);
const lastUser = () => Array.from(d.querySelectorAll('.chatMsg.user')).pop();

async function main() {
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = {};
  idle();

  /* ---- 1. the question paints the plans as buttons: closest first and marked, the description beside, the request
          quoted, and the two other replies as controls; the server's text is kept, folded ---------------------- */
  const plans = PLANS();
  const msg = paint(turn(choice(plans)));
  let bs = buttons(msg);
  assert.deepEqual(bs.map((b) => b.dataset.plan), plans.map((p) => p.name));
  assert.deepEqual(bs.map((b) => b.classList.contains('closest')), [true, true, false, false]);
  assert.ok(text(bs[0]).indexOf('does Library/book_by_title') !== -1);
  assert.ok(bs.every((b) => !b.disabled));
  const root = msg.querySelector('.planQuestion');
  assert.ok(text(root).indexOf('No plan matched this request. Which one should run it?') !== -1);
  assert.ok(text(root).indexOf('tell Ann a joke about cats') !== -1);
  assert.equal(text(control(msg, 'planNone')), 'None of these');
  assert.equal(text(control(msg, 'planCancel')), 'Cancel');
  assert.ok(msg.classList.contains('hasForm'));
  assert.ok(text(msg.querySelector('details.paramDetails')).indexOf('Reply with a name or number') !== -1);
  /* The not-run buttons (a new turn with use_plan) are a different thing and are not painted for a question. */
  assert.equal(msg.querySelectorAll('.planChoices').length, 0);
  /* The question's data is page protocol, not an artifact to show. */
  assert.ok(Array.from(msg.querySelectorAll('.artifacts, .artifactsLink')).every((n) => text(n).indexOf('plan_choice') === -1));
  let mark = requests.length;

  /* ---- 2. a click REPLIES to the pending question with the plan's name: plain text, no use_plan --------------- */
  bs[2].click();
  assert.equal(requests.length, mark + 1);
  assert.equal(requests[mark].path, '/invoke_agent_stream');
  assert.equal(requests[mark].body.user_query, 'Library/author_books');
  assert.ok(!('use_plan' in requests[mark].body) || requests[mark].body.use_plan == null, 'a reply carries no use_plan');
  assert.ok(text(lastUser()).indexOf('Library/author_books') !== -1);
  assert.ok(text(lastUser()).indexOf('Run with the plan') === -1);
  assert.ok(buttons(msg).every((b) => b.disabled), 'a reply disables the buttons');
  assert.ok(control(msg, 'planNone').disabled && control(msg, 'planCancel').disabled);
  assert.ok(root.classList.contains('disabled'));
  buttons(msg)[0].click();
  control(msg, 'planNone').click();
  assert.equal(requests.length, mark + 1, 'a disabled control sends nothing');
  idle();

  /* ---- 3. "None of these" and Cancel send the replies the data names ------------------------------------------ */
  mark = requests.length;
  const none = paint(turn(choice(plans)));
  control(none, 'planNone').click();
  assert.equal(requests[mark].body.user_query, 'none');
  idle();
  const cancelled = paint(turn(choice(plans)));
  control(cancelled, 'planCancel').click();
  assert.equal(requests[mark + 1].body.user_query, 'cancel');
  idle();
  const other = paint(turn(Object.assign(choice(plans), {none_reply: 'nothing fits', decline: 'stop'})));
  control(other, 'planNone').click();
  assert.equal(requests[mark + 2].body.user_query, 'nothing fits');
  idle();
  const other2 = paint(turn(Object.assign(choice(plans), {none_reply: 'nothing fits', decline: 'stop'})));
  control(other2, 'planCancel').click();
  assert.equal(requests[mark + 3].body.user_query, 'stop');
  idle();

  /* ---- 4. a part of a split turn says which part ---------------------------------------------------------------- */
  const split = paint(turn(choice(plans, 2, 'write me a haiku')));
  assert.equal(split.querySelector('.planQuestion').dataset.part, '2');
  assert.ok(text(split.querySelector('.planQuestion')).indexOf('No plan matched part 2. Which one should run it?') !== -1);
  assert.ok(text(split.querySelector('.planQuestion')).indexOf('write me a haiku') !== -1);

  /* ---- 5. more than the cap: the first twelve show, "show all" reveals the rest and still works when disabled --- */
  const many = Array.from({length: 15}, (_, i) => plan('Ctx/plan_' + String(i).padStart(2, '0'), i < 2));
  const capped = paint(turn(choice(many)));
  const visible = () => buttons(capped).filter((b) => !b.hidden);
  assert.equal(buttons(capped).length, 15);
  assert.equal(visible().length, 12);
  const toggle = capped.querySelector('.planMore');
  assert.equal(text(toggle), 'show all (15)');
  toggle.click();
  assert.equal(visible().length, 15);
  mark = requests.length;
  buttons(capped)[14].click();
  assert.equal(requests[mark].body.user_query, 'Ctx/plan_14');
  idle();

  /* ---- 6. a later message disables it; a turn that is not the last gets no live question ------------------------ */
  const first = paint(turn(choice(plans)));
  assert.ok(buttons(first).every((b) => !b.disabled));
  w.tmDisableLiveForms();
  assert.ok(buttons(first).every((b) => b.disabled));
  const stale = paint(turn(choice(plans)));
  w.tmBubble('user', 'and another');
  w.tmRenderTurn(stale, turn(choice(plans)));
  assert.equal(stale.querySelectorAll('.planQuestion').length, 0, 'a turn with a later message shows the text only');
  idle();

  /* ---- 7. a repaint of the same question keeps one set; nothing to offer leaves the plain question -------------- */
  const out = turn(choice(plans));
  const twice = paint(out);
  w.tmRenderTurn(twice, out);
  assert.equal(twice.querySelectorAll('.planQuestion').length, 1);
  [turn(choice([])), turn(choice([null, {name: 3}, 'text'])), turn(null), turn('x'),
   Object.assign(turn(choice(plans)), {command_outputs: [{command_name: 'ask_user', ask_user_entry: true,
     command_response: {success: false, response: QUESTION, artifacts: {question_kind: 'agent', plan_choice: choice(plans)}}}]})
  ].forEach((bad) => {
    const node = paint(bad);
    assert.equal(node.querySelectorAll('.planQuestion').length, 0);
    idle();
  });

  /* ---- 8. the deterministic ("/") path asks the same question as a failed COMPLETED turn (bd fix-o1lg S7) ------ */
  const cme = (payload, success) => ({turn_key: 'c' + (++seq), status: 'completed', success: false,
    answer: 'Ambiguous intent error for command x',
    command_outputs: [{command_name: 'wildcard', context: 'Library', command_parameters: null,
      command_response: {success: success === true, response: 'Ambiguous intent error for command x',
        artifacts: {question_kind: 'plan_choice', plan_choice: payload}}}]});
  const ambiguous = Object.assign(choice([plan('Library/find_book', true), plan('Library/find_member', true)]),
    {ambiguous: true, misunderstood: false});
  const completed = paint(cme(ambiguous));
  assert.equal(buttons(completed).length, 2, 'a completed failed turn shows the question');
  assert.ok(text(completed.querySelector('.planQuestion .planHead')).indexOf('More than one command fits this request') !== -1);
  mark = requests.length;
  buttons(completed)[0].click();
  assert.equal(requests[mark].body.user_query, 'Library/find_book', 'a click is the reply to the pending question');
  idle();
  const everything = paint(cme(Object.assign(choice([plan('IntentDetection/what_can_i_do', false)]),
    {ambiguous: false, misunderstood: true})));
  assert.ok(text(everything.querySelector('.planQuestion .planHead')).indexOf('No command matched this request') !== -1);
  assert.equal(paint(cme(ambiguous, true)).querySelectorAll('.planQuestion').length, 0, 'a successful output asks nothing');
  idle();

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
