/* The chat SPA's question forms for the agent's clarifications (fix-weih.11), exercised against
   the assembled page: path A, a deterministic form for a refused planning command input
   (question_kind "planning_parameter", no model call), and path B, a model-composed form for an
   open-ended ask_user (question_kind "agent", POST /compose_question_form). Only the network edge
   (window.fetch) is replaced: it records every request, answers /compose_question_form as the test
   scripts it (a form, null, HTTP 500, a hang that honours the abort signal, a deferred answer), and
   lets the streaming agent endpoint hang. */
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
const requests = [];        /* every request to the workflow server: {path, body, signal} */
const storedRecords = {};
let composeHandler = null;  /* (call) => Promise<response> */
let alerted = false;
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.alert = () => { alerted = true; };
    window.fetch = (url, opts) => {
      const u = String(url);
      const stored = u.match(/\/api\/turn\/([^/?]+)/);
      if (stored && storedRecords[decodeURIComponent(stored[1])]) {
        const body = storedRecords[decodeURIComponent(stored[1])];
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve(body)});
      }
      if (u.indexOf(BASE) !== 0) return new Promise(() => {});
      const call = {path: u.slice(BASE.length), body: opts && opts.body ? JSON.parse(opts.body) : null,
        signal: opts && opts.signal};
      requests.push(call);
      if (call.path === '/compose_question_form') {
        return composeHandler ? composeHandler(call) : new Promise(() => {});
      }
      return new Promise(() => {});
    };
  },
});
const w = dom.window;
const d = w.document;
const tm = w.tm;

function plain(value) { return JSON.parse(JSON.stringify(value)); }
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function flush() { for (let i = 0; i < 8; i++) await tick(); }
async function until(fn, what) {
  for (let i = 0; i < 200; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }
const composeCalls = () => requests.filter((r) => r.path === '/compose_question_form');
const streamCalls = () => requests.filter((r) => r.path === '/invoke_agent_stream');
function replyForm(form) {
  return () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({form})});
}
function replyReason(reason) {
  return () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({form: null, reason})});
}
function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return {promise, resolve: (form) => resolve({ok: true, status: 200, json: () => Promise.resolve({form})})};
}

function props(obj, required) { return {properties: obj, required: required || Object.keys(obj)}; }
const SCHEMAS = {
  'Library/member_report': {kind: 'planning', output: null, input: props({
    member: {type: 'string', title: 'Member', description: 'the member to report on'},
    scope: {type: 'string', enum: ['active', 'overdue'], title: 'Scope',
      classifier_confirms: {meanings: {active: 'still borrowing', overdue: 'late returns'}}},
    note: {type: 'string', title: 'Note'},
    box: {type: 'object', title: 'Box'},
    pw: {type: 'string', format: 'password', title: 'Pw'},
    days: {type: 'integer', title: 'Days', minimum: 1},
  }, ['member'])},
  'Library/card_report': {kind: 'planning', output: null, input: props({
    member: {type: 'string', title: 'Library card holder'},
  })},
  get_user_details: {kind: 'execution', input: {properties: {}}, output: {properties: {
    user_details: {type: 'string', title: 'User Details'}}}},
};

const USER = {orders: ['#W1994898', '#W2041120'],
  payment_methods: {credit_card_4421890: {source: 'credit_card', last_four: '4421'}}};
const QUESTION = 'Which order would you like to exchange items from, and which payment method?';
const EXCHANGE = {fields: [
  {name: 'order_id', label: 'Order', kind: 'choice', required: true,
    options: [{value: '#W1994898', label: '#W1994898 (delivered)'}, {value: '#W2041120', label: '#W2041120'}]},
  {name: 'items', label: 'Items to exchange', kind: 'multichoice', required: true,
    options: [{value: '1008292230', label: 'Mechanical Keyboard'}, {value: '4983901480', label: 'Smart Thermostat'}]},
  {name: 'payment', label: 'Payment method', kind: 'choice', required: true,
    options: [{value: 'credit_card_4421890', label: 'Credit card ending 4421890'}]},
]};

function userDetailsEntry() {
  return {command_name: 'get_user_details', context: 'Retail', command_parameters: {user_id: 'u1'},
    command_response: {success: true, response: 'ok', output: {user_details: JSON.stringify(USER)}}};
}
function askEntry(question, artifacts) {
  return {command_name: 'ask_user', ask_user_entry: true, context: 'Retail', command_parameters: null,
    command_response: {success: false, response: question, artifacts: artifacts}};
}
let seq = 0;
function waiting(question, artifacts, extra) {
  return Object.assign({turn_key: 'q' + (++seq), status: 'awaiting_user', success: false, answer: question,
    command_outputs: [userDetailsEntry(), askEntry(question, artifacts)]}, extra || {});
}
const agentTurn = (extra) => waiting(QUESTION, {question_kind: 'agent'}, extra);
function planningRequest(missing, invalid) {
  return {command: 'Library/member_report', missing: missing || [], invalid: invalid || [], values: {}};
}
const planningTurn = (req, extra) => waiting('Which member should the report cover?',
  {question_kind: 'planning_parameter', parameter_request: req}, extra);
function paint(out, opts) {
  const msg = w.tmBubble('agent', '...');
  msg.dataset.userMessage = (opts && opts.message) !== undefined ? opts.message : 'exchange my items';
  msg.dataset.deterministic = opts && opts.deterministic ? '1' : '';
  w.tmRenderTurn(msg, out);
  return msg;
}
const field = (msg, name) => msg.querySelector('.paramField[data-field="' + name + '"]');
function button(root, label) {
  const b = Array.from(root.querySelectorAll('button')).find((x) => x.textContent === label);
  assert.ok(b, 'no button ' + label);
  return b;
}
function type(node, value) { node.value = value; node.dispatchEvent(new w.Event('input', {bubbles: true})); }
function pick(msg, name, value) {
  const radio = Array.from(field(msg, name).querySelectorAll('input[type=radio]')).find((r) => r.value === value);
  assert.ok(radio, 'no radio ' + value);
  radio.checked = true;
}
function tick2(msg, name, value) {
  const box = Array.from(field(msg, name).querySelectorAll('input[type=checkbox]')).find((r) => r.value === value);
  assert.ok(box, 'no checkbox ' + value);
  box.checked = true;
}
const noForm = (msg, what) => {
  assert.equal(msg.querySelectorAll('.paramForm').length, 0, what + ': no form');
  assert.equal(msg.querySelectorAll('.questionStatus').length, 0, what + ': no status');
};
function resetFlags() {
  tm.connected = true;
  tm.composeUnavailable = false;
  d.getElementById('richAnswers').checked = true;
  idle();
}

async function main() {
  for (const fn of ['tmRenderQuestionForm', 'tmQuestionReply', 'tmQuestionComposedFields']) {
    assert.equal(typeof w[fn], 'function', fn);
  }
  assert.ok(w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderQuestionForm) !== -1, 'registered');
  assert.equal(w.TM_QUESTION_TIMEOUT_MS, 25000);
  tm.baseUrl = BASE;
  tm.commandSchemas = SCHEMAS;
  resetFlags();
  const composer = d.getElementById('chatInput');

  /* ---- case 1: path A, no model, "Rich answers" off, no LLM_RESPONSE_GEN ---- */
  d.getElementById('richAnswers').checked = false;
  tm.composeUnavailable = true;
  let mark = requests.length;
  const a1 = paint(planningTurn(planningRequest(['member'])));
  assert.equal(a1.querySelectorAll('.paramField').length, 1, 'only the asked field');
  assert.equal(field(a1, 'member').querySelector('label').textContent.indexOf('Member'), 0);
  assert.equal(text(field(a1, 'member').querySelector('.paramState')), 'Required');
  assert.equal(text(field(a1, 'member').querySelector('.paramHelp')), 'the member to report on');
  assert.ok(text(a1.querySelector('.bubble .questionText')).indexOf('Which member should the report cover?') === 0,
    'the question stays visible above the form');
  assert.equal(d.activeElement, field(a1, 'member').querySelector('textarea.fitText'), 'the asked field has focus');
  assert.equal(requests.length, mark, 'no request at all');
  assert.equal(composeCalls().length, 0);
  assert.deepEqual(Array.from(a1.querySelectorAll('.paramBtn')).map((b) => b.textContent),
    ['Send answer', 'Edit in composer']);
  button(a1, 'Send answer').click();                 /* required: blocked */
  assert.equal(streamCalls().length, 0);
  assert.ok(field(a1, 'member').classList.contains('asked'));
  type(field(a1, 'member').querySelector('textarea.fitText'), 'Ann Lee');
  button(a1, 'Send answer').click();
  assert.equal(streamCalls().length, 1);
  assert.equal(streamCalls()[0].body.user_query, 'member: Ann Lee');
  assert.ok(Array.from(a1.querySelectorAll('.paramForm input, .paramForm textarea, .paramForm button')).every((n) => n.disabled),
    'sending disables the form');
  assert.ok(Array.from(d.querySelectorAll('.chatMsg.user .bubble')).some((n) => n.textContent === 'member: Ann Lee'));
  assert.equal(composeCalls().length, 0);
  resetFlags();
  d.getElementById('richAnswers').checked = false;
  tm.composeUnavailable = true;

  /* several asked fields: missing first, then invalid; enum has no Other…, meanings in labels */
  const a2 = paint(planningTurn(planningRequest(['member'], ['scope'])));
  assert.deepEqual(Array.from(a2.querySelectorAll('.paramField')).map((n) => n.getAttribute('data-field')),
    ['member', 'scope']);
  assert.equal(text(field(a2, 'scope').querySelector('.paramState')), 'Not accepted — see Details');
  assert.equal(field(a2, 'scope').querySelectorAll('input[type=radio]').length, 2, 'no Other… on a strict enum');
  assert.ok(text(field(a2, 'scope')).indexOf('overdue — late returns') !== -1);
  assert.equal(a2.querySelectorAll('input[type=text].paramOther').length, 0);
  type(field(a2, 'member').querySelector('textarea.fitText'), 'Ann Lee');
  pick(a2, 'scope', 'overdue');
  button(a2, 'Edit in composer').click();
  assert.equal(composer.value, 'member: Ann Lee\nscope: overdue', 'path A: one schema-name line per field');
  assert.equal(streamCalls().length, 1, 'Edit in composer sends nothing');
  assert.ok(Array.from(a2.querySelectorAll('.paramForm input, .paramForm textarea, .paramForm button')).every((n) => !n.disabled),
    'the form stays enabled');
  assert.equal(d.activeElement, composer, 'the composer has focus');
  composer.value = '';
  /* number and optional fields: Days is optional-style asked as invalid, a number reply */
  const a3 = paint(planningTurn(planningRequest([], ['days', 'member'])));
  assert.deepEqual(Array.from(a3.querySelectorAll('.paramField')).map((n) => n.getAttribute('data-field')),
    ['member', 'days'], 'schema order within a state group');
  type(field(a3, 'member').querySelector('textarea.fitText'), 'Bo');
  type(field(a3, 'days').querySelector('input'), '30');
  button(a3, 'Edit in composer').click();
  assert.equal(composer.value, 'member: Bo\ndays: 30');
  composer.value = '';

  /* a custom title shows on screen; the wire line keeps the schema field name */
  const ct = paint(waiting('Who?', {question_kind: 'planning_parameter',
    parameter_request: {command: 'Library/card_report', missing: ['member'], invalid: [], values: {}}}));
  assert.equal(field(ct, 'member').querySelector('label').textContent.indexOf('Library card holder'), 0);
  type(field(ct, 'member').querySelector('textarea.fitText'), 'Ann Lee');
  button(ct, 'Edit in composer').click();
  assert.equal(composer.value, 'member: Ann Lee');
  composer.value = '';
  /* without this renderer the "/" parameter form still does not take a tagged prompt */
  const at = w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderQuestionForm);
  w.TM_PROMPT_RENDERERS.splice(at, 1);
  try {
    const bare = paint(planningTurn(planningRequest(['member'])));
    assert.equal(bare.querySelectorAll('.paramForm').length, 0, 'no "/" form for a planning_parameter turn');
    assert.ok(!Array.from(bare.querySelectorAll('button')).some((b) => b.textContent === 'Run'));
  } finally {
    w.TM_PROMPT_RENDERERS.splice(at, 0, w.tmRenderQuestionForm);
  }

  /* no widget for an asked field, or an unknown one: the plain question stays, nothing is sent */
  mark = requests.length;
  ['box', 'pw', 'nope'].forEach((name) => {
    const msg = paint(planningTurn(planningRequest([name])));
    noForm(msg, 'no widget for ' + name);
    assert.ok(text(msg.querySelector('.bubble')).indexOf('Which member should the report cover?') !== -1);
  });
  noForm(paint(planningTurn(planningRequest([]))), 'nothing asked');
  assert.equal(w.tmRenderQuestionForm(d.createElement('div'), planningTurn(planningRequest(['member']), {status: 'completed'})),
    null, 'a completed turn is not this renderer\'s');
  noForm(paint(planningTurn({command: 'Library/nowhere', missing: ['member'], invalid: [], values: {}})),
    'a command with no schema');
  noForm(paint(planningTurn(null)), 'no parameter_request');
  assert.equal(requests.length, mark, 'no request for any of them');

  /* a schema listing that is still loading: the form comes when it does */
  tm.commandSchemas = null;
  let release;
  tm.commandSchemasPromise = new Promise((r) => { release = r; });
  const late = paint(planningTurn(planningRequest(['member'])));
  noForm(late, 'schemas pending');
  tm.commandSchemas = SCHEMAS;
  release(SCHEMAS);
  await until(() => late.querySelector('.paramForm'), 'the form never came after the schemas loaded');
  assert.equal(late.querySelectorAll('.paramForm').length, 1);
  tm.commandSchemasPromise = null;

  /* ---- case 2: path B, the retail exchange ---------------------------------- */
  resetFlags();
  const gate = deferred();
  composeHandler = () => gate.promise;
  const calls0 = composeCalls().length;
  const out2 = agentTurn();
  const b1 = paint(out2);
  assert.equal(composeCalls().length, calls0 + 1);
  const call = composeCalls()[calls0];
  assert.equal(call.body.question, QUESTION);
  assert.equal(call.body.user_message, 'exchange my items');
  assert.equal(call.body.use_dspy_cache, true);
  assert.equal(call.body.turn_key, out2.turn_key, 'the turn key is sent so the server files its compose span under the turn');
  assert.equal(call.body.results.length, 1, 'the ask_user entry is not a result');
  assert.equal(call.body.results[0].id, 'r1');
  assert.equal(call.body.results[0].command, 'get_user_details');
  assert.deepEqual(call.body.results[0].output, plain(w.tmResultsOf(out2.command_outputs, SCHEMAS)[0].output));
  assert.deepEqual(call.body.results[0].output.orders, USER.orders, 'the JSON string arrives unwrapped');
  assert.ok(call.body.results[0].output.payment_methods.credit_card_4421890);
  assert.deepEqual(Object.keys(call.body.results[0].titles).sort(), Object.keys(call.body.results[0].output).sort());
  const status = b1.querySelector('.questionStatus');
  assert.ok(status && status.getAttribute('role') === 'status', 'Preparing a form… is a status');
  assert.equal(text(status), 'Preparing a form…');
  assert.ok(text(b1.querySelector('.bubble')).indexOf(QUESTION) !== -1, 'the plain question stays visible');
  assert.ok(!composer.disabled, 'the composer stays usable');
  gate.resolve(EXCHANGE);
  await until(() => b1.querySelector('.paramForm'), 'the composed form never rendered');
  assert.equal(b1.querySelectorAll('.questionStatus').length, 0, 'the status is gone');
  assert.equal(text(b1.querySelector('.questionText')), QUESTION, 'the question stays above the form');
  assert.equal(field(b1, 'order_id').querySelectorAll('input[type=radio]').length, 3, '2 options + Other…');
  assert.equal(field(b1, 'order_id').querySelectorAll('.paramOptValue').length, 1,
    'the raw id sits muted beside a label that says something else');
  assert.equal(text(field(b1, 'order_id').querySelector('.paramOptValue')), '#W1994898');
  assert.equal(field(b1, 'items').querySelectorAll('input[type=checkbox]').length, 3, '2 options + Other…');
  assert.ok(field(b1, 'items').querySelector('input[type=text].paramOther'), 'allowOther on multichoice');
  assert.ok(field(b1, 'order_id').querySelector('input[type=text].paramOther'), 'allowOther on choice');
  assert.ok(!b1.contains(d.activeElement), 'a late form takes no focus');
  /* required fields block */
  mark = requests.length;
  button(b1, 'Send answer').click();
  assert.equal(streamCalls().length, 1, 'a required field is empty: nothing sent');
  ['order_id', 'items', 'payment'].forEach((n) => assert.ok(field(b1, n).classList.contains('asked'), n));
  pick(b1, 'order_id', '#W1994898');
  tick2(b1, 'items', '1008292230');
  tick2(b1, 'items', '4983901480');
  pick(b1, 'payment', 'credit_card_4421890');
  button(b1, 'Send answer').click();
  assert.equal(streamCalls().length, 2);
  assert.equal(streamCalls()[1].body.user_query,
    'Order: #W1994898\nItems to exchange: 1008292230, 4983901480\nPayment method: credit_card_4421890');
  assert.ok(Array.from(b1.querySelectorAll('.paramForm input, .paramForm textarea, .paramForm button')).every((n) => n.disabled));
  idle();

  /* ---- case 3: Other…, Edit in composer, drafts ------------------------------ */
  composeHandler = replyForm(EXCHANGE);
  const b2 = paint(agentTurn());
  await until(() => b2.querySelector('.paramForm'), 'form b2');
  type(field(b2, 'order_id').querySelector('input[type=text].paramOther'), '#W9999999');
  assert.ok(Array.from(field(b2, 'order_id').querySelectorAll('input[type=radio]')).pop().checked,
    'typing in Other… selects it');
  tick2(b2, 'items', '1008292230');
  pick(b2, 'payment', 'credit_card_4421890');
  composer.value = 'my own draft';
  button(b2, 'Edit in composer').click();
  assert.equal(composer.value, 'Order: #W9999999\nItems to exchange: 1008292230\nPayment method: credit_card_4421890');
  assert.equal(streamCalls().length, 2, 'nothing sent');
  assert.equal(d.activeElement, composer);
  assert.ok(Array.from(b2.querySelectorAll('.paramForm input, .paramForm textarea, .paramForm button')).every((n) => !n.disabled),
    'the form stays enabled');
  composer.value = composer.value + '\nAlso gift wrap.';
  composer.dispatchEvent(new w.Event('input', {bubbles: true}));
  w.tmSend();                                   /* typing and sending from the composer */
  assert.equal(streamCalls().length, 3);
  assert.ok(streamCalls()[2].body.user_query.endsWith('Also gift wrap.'));
  assert.ok(Array.from(b2.querySelectorAll('.paramForm input, .paramForm textarea, .paramForm button')).every((n) => n.disabled),
    'sending from the composer disables the form');
  idle();
  /* nothing filled: Edit in composer only focuses, and keeps a draft */
  const b3 = paint(agentTurn());
  await until(() => b3.querySelector('.paramForm'), 'form b3');
  composer.value = 'keep me';
  d.body.focus();
  button(b3, 'Edit in composer').click();
  assert.equal(composer.value, 'keep me');
  assert.equal(d.activeElement, composer);
  composer.value = '';
  /* Other… alone */
  composeHandler = replyForm({fields: [EXCHANGE.fields[0]]});
  const b4 = paint(agentTurn());
  await until(() => b4.querySelector('.paramForm'), 'form b4');
  type(field(b4, 'order_id').querySelector('input[type=text].paramOther'), '#W9999999');
  button(b4, 'Send answer').click();
  assert.equal(streamCalls()[3].body.user_query, 'Order: #W9999999');
  idle();

  /* reply composition rules */
  const reply = (fields, values) => w.tmQuestionReply(fields, values);
  assert.equal(reply([{name: 'a', label: 'A'}, {name: 'b', label: ' B  b '}, {name: 'c', label: 'C'},
    {name: 'd', label: 'D'}, {name: 'e', label: 'E'}, {name: 'f', label: 'F'}],
  {a: 'x\n  y', b: ['1', '2'], c: '', d: null, e: 7, f: []}),
  'A: x y\nB b: 1, 2\nE: 7', 'newlines collapse, lists join, empties are left out');

  /* ---- composed field mapping ------------------------------------------------ */
  const mapped = plain(w.tmQuestionComposedFields({fields: [
    {name: 'ok', label: 'Ok', kind: 'text'},
    {name: '', label: 'No name', kind: 'longtext'},
    {name: 'when', label: 'When', kind: 'date', required: true},
    {name: 'n', label: 'N', kind: 'number'},
    {name: 'yes', label: 'Yes?', kind: 'boolean'},
    {name: 'lonely', label: 'Lonely', kind: 'choice', options: []},
    {name: 'weird', label: 'Weird', kind: 'sparkles'},
    {name: 'dup', label: 'Dup', kind: 'text'}, {name: 'dup', label: 'Dup 2', kind: 'text'},
    {name: 'x', label: '', kind: 'text'}, 'junk', null,
    {name: 'o', label: 'O', kind: 'choice', options: [{value: 'a'}, {value: 'a', label: 'again'}, {value: ''}, 5]},
  ]}));
  assert.deepEqual(mapped.map((f) => [f.name, f.kind]), [['ok', 'text'], ['No name', 'longtext'], ['when', 'date'],
    ['n', 'number'], ['yes', 'choice'], ['lonely', 'text'], ['weird', 'text'], ['dup', 'text'], ['dup#8', 'text'],
    ['o', 'choice']]);
  assert.deepEqual(mapped[4].options, [{value: 'Yes', label: 'Yes'}, {value: 'No', label: 'No'}]);
  assert.ok(mapped.every((f) => f.enforceRequired === true));
  assert.equal(mapped[2].required, true);
  assert.equal(mapped[9].allowOther, true);
  assert.deepEqual(mapped[9].options, [{value: 'a', label: 'a'}], 'duplicates, empty values and non-objects are dropped');
  assert.equal(w.tmQuestionComposedFields(null).length, 0);

  /* ---- guards: no request, no form ------------------------------------------- */
  composeHandler = replyForm(EXCHANGE);
  mark = requests.length;
  const other = (kind, extraArtifacts) => paint(waiting('Please confirm.', Object.assign({question_kind: kind},
    extraArtifacts || {})));
  const apv = other('approval', {approval: {command: 'Record/add_note', parameters: {}, origin: 'agent',
    irreversible: true, can_always: true}});
  const rev = other('review', {review: {kind: 'plan', title: 't', revision: 0, items: [], changes_data: []}});
  const unk = other('unknown');
  const untagged = paint(waiting('Please confirm.', {}));
  const noArtifacts = paint(waiting('Please confirm.', undefined));
  [apv, rev, unk, untagged, noArtifacts].forEach((m, i) => {
    noForm(m, ['approval', 'review', 'unknown', 'untagged', 'no artifacts'][i]);
    assert.equal(w.tmRenderQuestionForm(m, {status: 'awaiting_user', command_outputs: [askEntry('q', {question_kind: 'approval'})]}), null);
  });
  assert.equal(w.tmRenderQuestionForm(apv, waiting('q', {question_kind: 'review'})), null, 'direct call: review');
  assert.equal(w.tmRenderQuestionForm(apv, waiting('q', {})), null, 'direct call: untagged');
  noForm(paint(agentTurn({status: 'completed'})), 'completed turn');
  const notAsk = agentTurn();
  notAsk.command_outputs[1].ask_user_entry = false;
  notAsk.command_outputs[1].command_name = 'something_else';
  noForm(paint(notAsk), 'last entry is not an ask_user');
  const answered = agentTurn();
  answered.command_outputs[1].command_response.success = true;
  noForm(paint(answered), 'ask_user already answered');
  noForm(paint(agentTurn(), {deterministic: true}), 'a "/" turn');
  noForm(paint({turn_key: 'e1', status: 'awaiting_user', success: false, answer: '', command_outputs: [
    askEntry('', {question_kind: 'agent'})]}), 'empty question');
  const huge = 'x'.repeat(20001);
  noForm(paint(waiting(huge, {question_kind: 'agent'})), 'a question over 20000 characters');
  /* not the latest message */
  const early = w.tmBubble('agent', '...');
  early.dataset.userMessage = 'earlier';
  w.tmBubble('user', 'something later');
  w.tmRenderTurn(early, agentTurn());
  noForm(early, 'not the latest message');
  /* switch off, disconnected, unavailable */
  d.getElementById('richAnswers').checked = false;
  noForm(paint(agentTurn()), 'Rich answers off');
  d.getElementById('richAnswers').checked = true;
  tm.connected = false;
  noForm(paint(agentTurn()), 'disconnected');
  tm.connected = true;
  tm.composeUnavailable = true;
  noForm(paint(agentTurn()), 'compose unavailable');
  tm.composeUnavailable = false;
  assert.equal(requests.length, mark, 'no request for any of them');
  assert.equal(composeCalls().length, calls0 + 4, 'only the four composed forms asked');

  /* ---- fallbacks: the plain question stays, the composer is ready -------------- */
  const failures = [
    ['null form', replyReason('failed')],
    ['HTTP 500', () => Promise.resolve({ok: false, status: 500, json: () => Promise.resolve({})})],
    ['network error', () => Promise.reject(new Error('down'))],
    ['empty fields', replyForm({fields: []})],
    ['not json', () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve('nope')})],
    ['server timeout', replyReason('timeout')],
    ['not needed (the gate chose the plain question)', replyReason('not_needed')],
  ];
  for (const [what, handler] of failures) {
    composeHandler = handler;
    const msg = paint(agentTurn());
    assert.ok(msg.querySelector('.questionStatus') || what === 'network error', what + ': status while waiting');
    await until(() => !msg.querySelector('.questionStatus'), what + ': the status never left');
    noForm(msg, what);
    assert.ok(text(msg.querySelector('.bubble')).indexOf(QUESTION) !== -1, what + ': plain question');
    assert.ok(!tm.composeUnavailable, what + ': not a no_model');
    assert.ok(!composer.disabled);
  }
  /* client timeout aborts the request */
  w.TM_QUESTION_TIMEOUT_MS = 30;
  composeHandler = () => new Promise(() => {});
  const slow = paint(agentTurn());
  const slowCall = composeCalls()[composeCalls().length - 1];
  assert.ok(slow.querySelector('.questionStatus'));
  await until(() => !slow.querySelector('.questionStatus'), 'the timeout never removed the status');
  assert.equal(slowCall.signal.aborted, true, 'the request was aborted');
  noForm(slow, 'timeout');
  w.TM_QUESTION_TIMEOUT_MS = 25000;

  /* no_model: one notice, shared flag, no second request */
  composeHandler = replyReason('no_model');
  const nm1 = paint(agentTurn());
  await until(() => tm.composeUnavailable, 'composeUnavailable was never set');
  await flush();
  const notice = nm1.querySelector('.questionNotice');
  assert.ok(notice && text(notice).indexOf('LLM_RESPONSE_GEN') !== -1, 'the first answer shows the notice');
  assert.equal(notice.getAttribute('role'), 'status');
  const before = composeCalls().length;
  const nm2 = paint(agentTurn());
  await flush();
  assert.equal(composeCalls().length, before, 'no second request once unavailable');
  assert.equal(nm2.querySelectorAll('.questionNotice, .questionStatus').length, 0, 'and no second notice');
  /* the answer card set the flag while this request was in flight: no second notice */
  tm.composeUnavailable = false;
  let finish;
  composeHandler = () => new Promise((r) => { finish = r; });
  const nm3 = paint(agentTurn());
  assert.ok(nm3.querySelector('.questionStatus'));
  tm.composeUnavailable = true;
  finish({ok: true, status: 200, json: () => Promise.resolve({form: null, reason: 'no_model'})});
  await until(() => !nm3.querySelector('.questionStatus'), 'the status never left');
  assert.equal(nm3.querySelectorAll('.questionNotice').length, 0, 'no second notice');
  resetFlags();

  /* ---- staleness ----------------------------------------------------------------- */
  /* a message sent while waiting: the late form never appears */
  const g1 = deferred();
  composeHandler = () => g1.promise;
  const s1 = paint(agentTurn());
  composer.value = 'actually, the first one';
  w.tmSend();
  g1.resolve(EXCHANGE);
  await flush();
  noForm(s1, 'a message was sent');
  assert.ok(text(s1.querySelector('.bubble')).indexOf(QUESTION) !== -1);
  idle();
  /* an epoch change */
  const g2 = deferred();
  composeHandler = () => g2.promise;
  const s2 = paint(agentTurn());
  tm.epoch += 1;
  g2.resolve(EXCHANGE);
  await flush();
  noForm(s2, 'the source changed');
  resetFlags();
  /* a repaint of the same turn with other content: one form, from the answer of its own request */
  const g3 = deferred();
  const g4 = deferred();
  const queue = [g3, g4];
  composeHandler = () => queue.shift().promise;
  const s3 = paint(agentTurn());
  const repainted = agentTurn();
  w.tmRenderTurn(s3, repainted);
  assert.equal(s3.querySelectorAll('.questionStatus').length, 1, 'a repaint leaves one status');
  g3.resolve({fields: [{name: 'first', label: 'First', kind: 'text'}]});
  await flush();
  assert.ok(!field(s3, 'first'), 'the first request belongs to an older paint');
  g4.resolve({fields: [{name: 'second', label: 'Second', kind: 'text'}]});
  await until(() => field(s3, 'second'), 'the repaint\'s own answer never rendered');
  assert.equal(s3.querySelectorAll('.paramForm').length, 1);
  /* the same turn painted again: cached, no request, the typed text survives */
  const cached = composeCalls().length;
  type(field(s3, 'second').querySelector('textarea.fitText'), 'typed');
  w.tmRenderTurn(s3, repainted);
  assert.equal(composeCalls().length, cached, 'served from the per-turn cache');
  assert.equal(field(s3, 'second').querySelector('textarea.fitText').value, 'typed', 'a repaint reuses the mounted form');
  assert.equal(s3.querySelectorAll('.paramForm').length, 1);
  /* an epoch change drops the cache */
  tm.epoch += 1;
  composeHandler = replyForm({fields: [{name: 'third', label: 'Third', kind: 'text'}]});
  const s4 = paint(repainted);
  assert.equal(composeCalls().length, cached + 1, 'a new source asks again');
  await until(() => field(s4, 'third'), 'form after the epoch change');
  resetFlags();

  /* ---- restored turns -------------------------------------------------------------- */
  function stored(key, artifacts, status) {
    storedRecords[key] = {turn: {record: {turn_output: {status: status || 'awaiting_user', answer: QUESTION,
      command_outputs: [userDetailsEntry(), askEntry(QUESTION, artifacts)]}}}};
  }
  stored('rA', {question_kind: 'agent'});
  stored('rB', {question_kind: 'agent'});
  composeHandler = replyForm(EXCHANGE);
  mark = composeCalls().length;
  const olderR = w.tmRenderStoredTurn({turn_key: 'rA', status: 'awaiting_user', success: false, answer: QUESTION,
    user_message: 'exchange'}, true);
  const latestR = w.tmRenderStoredTurn({turn_key: 'rB', status: 'awaiting_user', success: false, answer: QUESTION,
    user_message: 'exchange'}, true);
  await until(() => latestR.querySelector('.paramForm') && olderR.dataset.pendingRecord === '' &&
    latestR.dataset.pendingRecord === '', 'the restored turns never settled');
  await flush();
  assert.equal(composeCalls().length, mark + 1, 'only the latest restored turn composes');
  noForm(olderR, 'an older restored turn');
  assert.equal(latestR.querySelectorAll('.paramForm').length, 1);
  assert.equal(composeCalls()[mark].body.user_message, 'exchange');
  /* the latest restored planning turn gets path A with no request */
  stored('pA', {question_kind: 'planning_parameter', parameter_request: planningRequest(['member'])});
  stored('pB', {question_kind: 'planning_parameter', parameter_request: planningRequest(['member'])});
  const requestsBefore = requests.length;
  const pOlder = w.tmRenderStoredTurn({turn_key: 'pA', status: 'awaiting_user', success: false, answer: 'Which?'}, true);
  const pLatest = w.tmRenderStoredTurn({turn_key: 'pB', status: 'awaiting_user', success: false, answer: 'Which?'}, true);
  await until(() => pLatest.querySelector('.paramForm') && pOlder.dataset.pendingRecord === '',
    'the restored planning turns never settled');
  noForm(pOlder, 'an older restored planning turn');
  assert.equal(pLatest.querySelectorAll('.paramField').length, 1);
  assert.equal(requests.length, requestsBefore, 'path A restored: no request');

  /* ---- text only [R22] ---------------------------------------------------------------- */
  const evil = '<img src=x onerror=alert(1)>';
  composeHandler = replyForm({fields: [
    {name: evil, label: evil, kind: 'choice', required: true, help: evil, options: [{value: evil, label: evil},
      {value: '<b>v</b>', label: '<b>l</b>'}]},
    {name: 'm', label: 'M', kind: 'multichoice', options: [{value: '<i>x</i>', label: evil}]},
  ]});
  const xss = paint(waiting(evil, {question_kind: 'agent'}));
  await until(() => xss.querySelector('.paramForm'), 'xss form');
  assert.equal(xss.querySelectorAll('img, b, i, script').length, 0, 'no element from model text');
  assert.ok(text(xss.querySelector('.paramForm')).indexOf(evil) !== -1, 'shown as literal text');
  assert.ok(text(xss.querySelector('.questionText')).indexOf(evil) !== -1);
  Array.from(xss.querySelectorAll('input[type=radio]')).find((r) => r.value === evil).checked = true;
  tick2(xss, 'm', '<i>x</i>');
  button(xss, 'Edit in composer').click();
  assert.equal(composer.value, evil + ': ' + evil + '\nM: <i>x</i>');
  composer.value = '';
  const xssA = paint(planningTurn(planningRequest(['member'])));
  assert.equal(xssA.querySelectorAll('img').length, 0);
  assert.equal(alerted, false);

  /* ---- fix-weih.33: the free-text answer wraps and grows; the reply is unchanged ---- */
  const h33 = paint(planningTurn(planningRequest(['member'], ['days'])));
  const mem = field(h33, 'member').querySelector('textarea.fitText');
  assert.ok(mem, 'free text is a wrapping, growing field');
  assert.equal(mem.rows, 1);
  assert.equal(mem.required, true);
  assert.equal(mem.getAttribute('aria-required'), 'true');
  assert.equal(mem.getAttribute('aria-invalid'), 'true');
  assert.equal(d.querySelectorAll('label[for="' + mem.id + '"]').length, 1);
  const daysCtl = field(h33, 'days').querySelector('input');
  assert.equal(daysCtl.type, 'number', 'a number keeps its native input');
  assert.equal(daysCtl.min, '1');
  assert.equal(h33.querySelectorAll('.paramField[data-field="pw"]').length, 0, 'a secret is still not asked');
  const hq = {v: 26};
  Object.defineProperty(mem, 'scrollHeight', {configurable: true, get: () => hq.v});
  type(mem, 'Ann');
  const oneLine = parseFloat(mem.style.height);
  hq.v = 120;
  type(mem, 'Ann Lee, who borrowed a very long list of books and returned them late. '.repeat(3));
  assert.ok(parseFloat(mem.style.height) > oneLine, 'a long answer grows the field');
  assert.equal(mem.style.overflowY, 'hidden');
  hq.v = 9000;
  type(mem, 'again');
  assert.equal(mem.style.overflowY, 'auto', 'capped, then it scrolls');
  /* Enter did nothing in the input: it adds no line and sends nothing */
  mark = requests.length;
  const enterQ = new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true});
  mem.dispatchEvent(enterQ);
  assert.equal(enterQ.defaultPrevented, true);
  assert.equal(requests.length, mark, 'Enter sends nothing');
  type(mem, 'Ann\nLee');
  assert.equal(mem.value, 'AnnLee', 'a newline is removed as an input removed it');
  type(mem, 'Ann');
  mem.setSelectionRange(3, 3);
  const pasteQ = new w.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(pasteQ, 'clipboardData', {value: {getData: () => ' Lee\r\nJr'}});
  mem.dispatchEvent(pasteQ);
  assert.equal(mem.value, 'Ann Lee Jr', 'a pasted newline is a space');
  /* the reply: spaces around the answer are trimmed exactly as before */
  type(daysCtl, '30');
  type(mem, '   Ann   Lee  ');
  button(h33, 'Edit in composer').click();
  assert.equal(composer.value, 'member: Ann   Lee\ndays: 30');
  composer.value = '';
  /* required: an empty free-text answer still blocks Send */
  type(mem, '');
  mark = requests.length;
  button(h33, 'Send answer').click();
  assert.equal(requests.length, mark);
  assert.equal(text(field(h33, 'member').querySelector('.paramState')), 'Required');
  type(mem, '  Ann Lee  ');
  idle();
  mark = requests.length;
  button(h33, 'Send answer').click();
  assert.equal(streamCalls().pop().body.user_query, 'member: Ann Lee\ndays: 30');
  idle();

  assert.deepEqual(errors, [], 'jsdom reported errors');
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
