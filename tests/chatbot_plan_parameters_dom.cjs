/* The chat SPA's parameter form for a CHOSEN plan (bd fix-1ma8.6): after a plan button is clicked the server asks for the
   plan's parameter values (a planning_parameter question with parameter_request.confirm), not for a review. The form shows the
   plan's name and description read-only and one field per parameter with the drafted values prefilled; there is no goal
   text and no step, drop, add or switch-plan control. Only the network edge (window.fetch) is replaced. */
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
  url: 'http://127.0.0.1:39137/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url, opts) => {
      const u = String(url);
      if (u.indexOf(BASE) !== 0) return new Promise(() => {});
      requests.push({path: u.slice(BASE.length), body: opts && opts.body ? JSON.parse(opts.body) : null});
      return new Promise(() => {});
    };
  },
});
const w = dom.window;
const d = w.document;
const tm = w.tm;

function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }
function type(node, value) { node.value = value; node.dispatchEvent(new w.Event('input', {bubbles: true})); }
const streamCalls = () => requests.filter((r) => r.path === '/invoke_agent_stream');

const props = (obj, required) => ({properties: obj, required});
const PLAN = 'Library/note_member';
const SCHEMAS = {
  [PLAN]: {kind: 'planning', output: null, input: props({
    member: {type: 'string', title: 'Member', description: "the member's name"},
    text: {type: 'string', title: 'Text'},
    days: {type: 'integer', title: 'Days', minimum: 1},
  }, ['member', 'text'])},
};
const param = (name, required, value, state) => ({name, required, value: value === undefined ? null : value,
  state: state || null, description: ''});
const request = (parameters) => ({command: PLAN, description: "Add a note to a member's record", confirm: true,
  parameters, missing: parameters.filter((p) => p.state === 'missing').map((p) => p.name),
  invalid: parameters.filter((p) => p.state === 'invalid').map((p) => p.name),
  values: Object.fromEntries(parameters.filter((p) => p.value !== null && !p.state).map((p) => [p.name, String(p.value)]))});
let seq = 0;
function turn(req, question) {
  const q = question || 'You chose the planning command Library/note_member. Its parameters: ...';
  return {turn_key: 'p' + (++seq), status: 'awaiting_user', success: false, answer: q,
    command_outputs: [{command_name: 'ask_user', ask_user_entry: true, context: 'Library', command_parameters: null,
      command_response: {success: false, response: q,
        artifacts: {question_kind: 'planning_parameter', parameter_request: req}}}]};
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  msg.dataset.userMessage = 'put a note on a record';
  w.tmRenderTurn(msg, out);
  return msg;
}
const field = (msg, name) => msg.querySelector('.paramField[data-field="' + name + '"]');
const button = (root, label) => {
  const b = Array.from(root.querySelectorAll('button')).find((x) => x.textContent === label);
  assert.ok(b, 'no button ' + label);
  return b;
};

async function main() {
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = SCHEMAS;
  idle();

  /* ---- 1. the chosen plan's form: name and description, one field per parameter, values prefilled, nothing editable
          about the intent ---------------------------------------------------------------------------------------- */
  const msg = paint(turn(request([param('member', true, 'Zed Quill'), param('text', true, null, 'missing'),
    param('days', false, 3)])));
  assert.equal(text(msg.querySelector('.planChosenName')), 'note_member');
  assert.equal(text(msg.querySelector('.planChosenDesc')), "Add a note to a member's record");
  assert.deepEqual(Array.from(msg.querySelectorAll('.paramField')).map((n) => n.getAttribute('data-field')),
    ['member', 'text', 'days'], 'every parameter, in schema order');
  assert.equal(field(msg, 'member').querySelector('textarea.fitText, input').value, 'Zed Quill', 'a drafted value is prefilled');
  assert.equal(field(msg, 'text').querySelector('textarea.fitText, input').value, '', 'a missing one is empty');
  assert.ok(field(msg, 'text').classList.contains('asked'), 'a missing required one is marked');
  assert.equal(field(msg, 'days').querySelector('input').value, '3', 'a number is prefilled');
  assert.equal(msg.querySelectorAll('.reviewForm, .reviewStep, .reviewGoal, .reviewSkip, .reviewPlanField').length, 0,
    'no review control');
  const labels = Array.from(msg.querySelectorAll('button')).map((b) => b.textContent);
  assert.ok(labels.includes('Run') && labels.includes('Cancel'));
  assert.ok(!labels.some((l) => /drop|add|switch|Edit in composer|Send answer|Apply changes/i.test(l)), labels.join(','));
  assert.ok(!/goal|prompt/i.test(Array.from(msg.querySelectorAll('label')).map(text).join(' ')), 'no goal field');

  /* ---- 2. Run refuses while a required value is empty, then sends one "name: value" line per filled field ------------- */
  let mark = requests.length;
  button(msg, 'Run').click();
  assert.equal(requests.length, mark, 'the empty required field blocks Run');
  type(field(msg, 'text').querySelector('textarea.fitText, input'), 'rehung the banner');
  button(msg, 'Run').click();
  const sent = streamCalls().slice(-1)[0];
  assert.equal(sent.body.user_query, 'member: Zed Quill\ntext: rehung the banner\ndays: 3');
  assert.equal(sent.body.use_plan, undefined, 'the reply is a plain answer; the plan stays pinned on the server');
  assert.ok(Array.from(msg.querySelectorAll('.paramForm button, .paramForm input, .paramForm textarea')).every((n) => n.disabled),
    'sending disables the form');
  idle();

  /* ---- 3. every value drafted: the same form once, prefilled; Run confirms ------------------------------------------- */
  const ok = paint(turn(request([param('member', true, 'Zed Quill'), param('text', true, 'rehung the banner'),
    param('days', false)])));
  assert.equal(field(ok, 'text').querySelector('textarea.fitText, input').value, 'rehung the banner');
  assert.ok(!field(ok, 'text').classList.contains('asked'));
  mark = streamCalls().length;
  button(ok, 'Run').click();
  assert.equal(streamCalls().length, mark + 1);
  assert.equal(streamCalls().slice(-1)[0].body.user_query, 'member: Zed Quill\ntext: rehung the banner');
  idle();

  /* ---- 4. Cancel sends cancel --------------------------------------------------------------------------------------- */
  const stop = paint(turn(request([param('member', true, 'Zed Quill'), param('text', true, 'x'), param('days', false)])));
  mark = streamCalls().length;
  button(stop, 'Cancel').click();
  assert.equal(streamCalls().slice(-1)[0].body.user_query, 'cancel');
  idle();

  /* ---- 5. a field the form cannot draw: the plain question stays -------------------------------------------------- */
  const odd = paint(turn(request([param('member', true, 'Zed Quill'), param('nope', true, 'x')]), 'Plain question text'));
  assert.equal(odd.querySelectorAll('.paramForm').length, 0);
  assert.ok(text(odd.querySelector('.bubble')).indexOf('Plain question text') !== -1);

  /* ---- 6. an ordinary parameter question (no confirm) is unchanged: only the asked fields ----------------------- */
  const plainAsk = paint(turn({command: PLAN, missing: ['text'], invalid: [], values: {member: 'Zed Quill'}}));
  assert.deepEqual(Array.from(plainAsk.querySelectorAll('.paramField')).map((n) => n.getAttribute('data-field')), ['text']);
  assert.equal(plainAsk.querySelectorAll('.planChosen').length, 0);

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
