/* The chat SPA's parameter form, the which-command question and shared form
   renderer (fix-weih.5), exercised against the assembled page. Only the
   network edge (window.fetch) is replaced: it records every request to the
   workflow server, answers /invoke_assistant with a finished turn, and lets
   the streaming agent endpoint hang. */
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
  url: 'http://127.0.0.1:39131/',
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
      if (u === BASE + '/invoke_assistant') {
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({
          turn_key: 'done', status: 'completed', success: true, answer: 'ran', command_outputs: []})});
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
async function until(fn, what) {
  for (let i = 0; i < 100; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }
function sentAfter(mark) { return requests.slice(mark); }

function props(obj, required) { return {properties: obj, required: required || Object.keys(obj)}; }
const SCHEMAS = {
  'Plain/record_values': {kind: 'execution', output: null, input: props({
    name: {type: 'string', title: 'Name', description: 'a name', examples: ['Ann']},
    count: {type: 'integer', title: 'Count', minimum: 0, maximum: 99},
    ratio: {type: 'number', title: 'Ratio'},
    ids: {type: 'array', items: {type: 'string'}, title: 'Ids'},
  })},
  'Plain/pick_seat': {kind: 'execution', output: null, input: props({
    seat: {type: 'string', enum: ['a1', 'b2'], title: 'Seat'},
    note: {type: 'string', title: 'Note'},
  }, ['seat'])},
  'Plain/set_flags': {kind: 'execution', output: null, input: props({
    flag: {type: 'boolean', title: 'Flag', default: false},
    maybe: {anyOf: [{type: 'boolean'}, {type: 'null'}], default: null, title: 'Maybe',
      description: 'optional switch'},
  }, [])},
  'Plain/widgets': {kind: 'execution', output: null, input: props({
    story: {type: 'string', title: 'Story', ui: {widget: 'textarea'}},
    color: {type: 'string', enum: ['r', 'g', 'b', 'c', 'm', 'y'], title: 'Color', ui: {widget: 'radio'}},
    size: {type: 'string', enum: ['s', 'm', 'l', 'xl', 'xxl', 'xxxl'], title: 'Size'},
    shelf: {type: 'string', enum: ['top', 'low'], title: 'Shelf',
      classifier_confirms: {meanings: {top: 'one shelf', low: 'the other'}}},
    when: {type: 'string', format: 'date', title: 'When'},
    secret: {type: 'string', writeOnly: true, title: 'Secret'},
    pw: {type: 'string', format: 'password', title: 'Pw'},
  }, ['story'])},
  'Plain/login': {kind: 'execution', output: null, input: props({
    user: {type: 'string', title: 'User'},
    token: {type: 'string', writeOnly: true, title: 'Token'},
  }, ['user', 'token'])},
  'Plain/tricky': {kind: 'execution', output: null, input: props({
    box: {type: 'object', title: 'Box'},
  })},
  'Library/find_book': {kind: 'execution', output: null, input: {properties: {},
    description: 'Find a book by title.'}},
  'Library/find_member': {kind: 'execution', output: null, input: null,
    output_: null},
  'Plain/with_markup': {kind: 'execution', output: null, input: props({
    q: {type: 'string', title: 'Q', description: '<img src=x onerror=alert(1)>'},
  })},
};

function request(command, missing, invalid, values) {
  return {command, missing: missing || [], invalid: invalid || [], values: values || {}};
}
function failedTurn(artifacts, response, extra) {
  return Object.assign({turn_key: 'k' + Math.random().toString(16).slice(2), status: 'completed',
    success: false, answer: response || 'CME instructions',
    command_outputs: [{command_name: 'wildcard', context: 'Plain', command_parameters: null,
      command_response: {success: false, response: response || 'CME instructions', artifacts}}]},
  extra || {});
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
const field = (msg, name) => msg.querySelector('.paramField[data-field="' + name + '"]');
const control = (msg, name) => field(msg, name).querySelector('input, select, textarea');
function button(root, label) {
  const b = Array.from(root.querySelectorAll('button')).find((x) => x.textContent === label);
  assert.ok(b, 'no button ' + label);
  return b;
}
function type(node, value) { node.value = value; node.dispatchEvent(new w.Event('input', {bubbles: true})); }

async function main() {
  for (const fn of ['tmRenderForm', 'tmDisableLiveForms', 'tmRenderParameterForm', 'tmRenderPlanQuestion',
    'tmComposeCommandReply', 'tmRenderPromptForms']) {
    assert.equal(typeof w[fn], 'function', fn);
  }
  assert.deepEqual(plain(w.TM_PROMPT_ARTIFACT_KEYS),
    ['question_kind', 'approval', 'review', 'parameter_request', 'plan_choice']);
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = SCHEMAS;
  idle();
  const composer = d.getElementById('chatInput');

  /* ---- case 1: record_values, prefill, asked marks, focus, Run -------------- */
  let mark = requests.length;
  const msg1 = paint(failedTurn({parameter_request: request('Plain/record_values',
    ['ratio', 'ids'], [], {name: 'Bob', count: 3})}, 'Missing: ratio, ids. Enter them.'));
  assert.deepEqual(Array.from(msg1.querySelectorAll('.paramField')).map((n) => n.getAttribute('data-field')),
    ['name', 'count', 'ratio', 'ids']);
  assert.equal(control(msg1, 'name').value, 'Bob');
  assert.equal(control(msg1, 'count').value, '3');
  assert.equal(control(msg1, 'name').placeholder, 'Ann', 'examples[0] is the placeholder');
  assert.equal(control(msg1, 'count').max, '99');
  ['ratio', 'ids'].forEach((n) => {
    assert.ok(field(msg1, n).classList.contains('asked'), n);
    assert.equal(control(msg1, n).getAttribute('aria-invalid'), 'true', n);
    assert.equal(text(field(msg1, n).querySelector('.paramState')), 'Required', n);
  });
  ['name', 'count'].forEach((n) => {
    assert.ok(!field(msg1, n).classList.contains('asked'));
    assert.ok(!control(msg1, n).hasAttribute('aria-invalid'));
  });
  assert.equal(d.activeElement, control(msg1, 'ratio'), 'the first asked field has focus');
  const details1 = msg1.querySelector('details.paramDetails');
  assert.ok(details1 && !details1.open, 'Details closed when nothing was rejected');
  assert.equal(text(details1.querySelector('summary')), 'Details');
  assert.ok(text(details1).indexOf('Missing: ratio, ids. Enter them.') !== -1);
  assert.ok(text(msg1.querySelector('.bubble')).indexOf('Missing: ratio') !== -1);
  assert.equal(msg1.querySelectorAll('.cmdOut').length, 0, 'the CME line is not repeated under the form');
  assert.equal(msg1.querySelectorAll('.artifactsLink').length, 0, 'prompt keys are not artifacts');
  /* every input has a programmatic label */
  Array.from(msg1.querySelectorAll('.paramField input, .paramField textarea')).forEach((c) => {
    const lab = msg1.querySelector('label[for="' + c.id + '"]');
    assert.ok(lab && lab.textContent.length, 'label for ' + c.id);
  });
  assert.equal(msg1.querySelector('.paramField[data-field="ratio"] label').textContent.indexOf('Ratio'), 0);
  type(control(msg1, 'ratio'), '1.5');
  type(control(msg1, 'ids'), 'a\nb\n');
  button(msg1, 'Run').click();
  let sent = sentAfter(mark);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].path, '/invoke_assistant');
  assert.equal(sent[0].body.user_query, '/record_values name=Bob count=3 ratio=1.5 ids=["a","b"]');
  assert.equal(composer.value, '', 'the composer is left as it was');
  assert.ok(Array.from(msg1.querySelectorAll('.paramForm button, .paramForm input, .paramForm textarea')).every((n) => n.disabled),
    'sending disables the form');
  const userBubbles = Array.from(d.querySelectorAll('.chatMsg.user .bubble')).map((n) => n.textContent);
  assert.ok(userBubbles.indexOf('/record_values name=Bob count=3 ratio=1.5 ids=["a","b"]') !== -1,
    'the reply is ordinary transcript text');
  await until(() => !tm.busy, 'the assistant turn never settled');

  /* ---- case 2: Edit as text, Cancel ----------------------------------------- */
  mark = requests.length;
  const msg2 = paint(failedTurn({parameter_request: request('Plain/record_values',
    ['ratio', 'ids'], [], {name: 'Bob', count: 3})}));
  composer.value = 'a draft';
  button(msg2, 'Edit as text').click();
  assert.equal(composer.value, '/record_values name=Bob count=3');
  assert.equal(requests.length, mark, 'Edit as text sends nothing');
  composer.value = 'my draft';
  button(msg2, 'Cancel').click();
  sent = sentAfter(mark);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].body.user_query, 'abort');
  assert.equal(composer.value, 'my draft', 'a draft typed meanwhile survives a send');
  composer.value = '';
  idle();

  /* ---- case 3: quoting ------------------------------------------------------- */
  const cmd = (value) => w.tmComposeCommandReply('Plain/record_values', [{name: 'name', value}]);
  assert.equal(cmd('Ann (EU)'), '/record_values name="Ann (EU)"');
  assert.equal(cmd('say "hi" (x)'), '/record_values name=\'say "hi" (x)\'');
  assert.equal(cmd('a, b,'), '/record_values name="a, b,"');
  assert.equal(cmd('a;b'), '/record_values name="a;b"');
  assert.equal(cmd(' pad '), '/record_values name=" pad "');
  assert.equal(cmd('one\ntwo'), '/record_values name="one two"');
  assert.equal(cmd('"x"'), '/record_values name=\'"x"\'');
  assert.equal(cmd('null'), '/record_values name=null', 'quotes cannot make None/null literal for the parser');
  assert.equal(cmd('Bob'), '/record_values name=Bob');
  assert.equal(cmd('two words'), '/record_values name=two words');
  assert.equal(cmd('say "hi"'), '/record_values name=say "hi"');
  assert.equal(w.tmComposeCommandReply('Ctx/cmd', []), '/cmd');
  assert.equal(w.tmComposeCommandReply('cmd', [{name: 'n', value: 3}, {name: 'b', value: false},
    {name: 'l', value: ['a', 'b (c)']}]), '/cmd n=3 b=false l=["a","b (c)"]');

  /* ---- the shared reply contract (also bound by the Python parser) ---- */
  const contract = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'parameter_form_replies.json'), 'utf8'));
  assert.ok(contract.length >= 10);
  contract.forEach((c) => {
    assert.equal(w.tmComposeCommandReply(c.command, c.pairs), c.reply, c.name);
  });

  /* ---- case 4: invalid field --------------------------------------------------- */
  const msg4 = paint(failedTurn({parameter_request: request('Plain/pick_seat', [], ['seat'],
    {note: 'window side'})}, 'seat: not an option'));
  assert.equal(text(field(msg4, 'seat').querySelector('.paramState')), 'Not accepted — see Details');
  assert.ok(msg4.querySelector('details.paramDetails').open, 'rejected values open Details');
  assert.equal(control(msg4, 'note').value, 'window side');
  assert.equal(d.activeElement.closest('.paramField').getAttribute('data-field'), 'seat');
  assert.equal(w.document.querySelector('.chatMsg:last-child .paramField[data-field="seat"] .paramChoices')
    .getAttribute('aria-invalid'), 'true');

  /* ---- case 5: tri-state Optional[bool] vs bool --------------------------------- */
  mark = requests.length;
  const msg5 = paint(failedTurn({parameter_request: request('Plain/set_flags', ['flag'], [], {maybe: null})}));
  const flagBox = control(msg5, 'flag');
  assert.equal(flagBox.type, 'checkbox');
  const maybe = field(msg5, 'maybe');
  assert.deepEqual(Array.from(maybe.querySelectorAll('input[type=radio]')).map((r) => r.value), ['', 'true', 'false']);
  assert.deepEqual(Array.from(maybe.querySelectorAll('.paramOpt')).map((r) => text(r)).slice(0, 3),
    ['—', 'yes true', 'no false']);
  assert.ok(maybe.querySelector('input[value=""]').checked, 'the dash is selected');
  assert.equal(maybe.querySelector('.paramLabel').textContent.indexOf('Maybe'), 0);
  button(msg5, 'Edit as text').click();
  assert.equal(composer.value, '/set_flags flag=false', 'a bool is always sent; an untouched Optional[bool] is omitted');
  flagBox.checked = true;
  button(msg5, 'Edit as text').click();
  assert.equal(composer.value, '/set_flags flag=true');
  maybe.querySelector('input[value="false"]').checked = true;
  button(msg5, 'Edit as text').click();
  assert.equal(composer.value, '/set_flags flag=true maybe=false');
  maybe.querySelector('input[value="true"]').checked = true;
  button(msg5, 'Edit as text').click();
  assert.equal(composer.value, '/set_flags flag=true maybe=true');
  composer.value = '';
  /* a given true prefills the choice */
  const msg5b = paint(failedTurn({parameter_request: request('Plain/set_flags', ['flag'], [], {maybe: true})}));
  assert.ok(field(msg5b, 'maybe').querySelector('input[value="true"]').checked);

  /* ---- case 6: ui.widget hints, meanings --------------------------------------- */
  const msg6 = paint(failedTurn({parameter_request: request('Plain/widgets', ['story'])}));
  assert.equal(control(msg6, 'story').tagName, 'TEXTAREA');
  assert.equal(field(msg6, 'color').querySelectorAll('input[type=radio]').length, 7, '6 values + the dash');
  assert.ok(!field(msg6, 'color').querySelector('select'));
  assert.equal(control(msg6, 'size').tagName, 'SELECT', '6 values without a hint is a select');
  assert.ok(field(msg6, 'shelf').querySelector('input[type=radio]'), '2 values are radios');
  assert.ok(text(field(msg6, 'shelf')).indexOf('top — one shelf') !== -1);
  assert.ok(text(field(msg6, 'shelf')).indexOf('low — the other') !== -1);
  assert.equal(control(msg6, 'when').type, 'date');
  /* a secret parameter has no control at all, only a labelled notice row */
  assert.equal(msg6.querySelectorAll('input[type=password]').length, 0, 'no masked input');
  for (const name of ['secret', 'pw']) {
    assert.equal(field(msg6, name).querySelectorAll('input, select, textarea').length, 0, name + ' has no control');
    const row = field(msg6, name);
    assert.equal(row.getAttribute('role'), 'group');
    const labelNode = msg6.ownerDocument.getElementById(row.getAttribute('aria-labelledby'));
    assert.ok(labelNode && row.contains(labelNode) && text(labelNode).indexOf(name === 'pw' ? 'Pw' : 'Secret') === 0);
    const note = msg6.ownerDocument.getElementById(row.getAttribute('aria-describedby'));
    assert.ok(note && text(note).indexOf('Not collected here: a value typed into this form is sent and recorded as chat text.') === 0);
  }
  type(control(msg6, 'story'), 'line one\nline two');
  field(msg6, 'shelf').querySelector('input[value="low"]').checked = true;
  button(msg6, 'Edit as text').click();
  assert.equal(composer.value, '/widgets story="line one line two" shelf=low');
  composer.value = '';

  /* ---- password: never prefilled, never echoed, never copied to the composer ---- */
  const msg6b = paint(failedTurn({parameter_request: request('Plain/widgets', ['story'], [],
    {secret: 'hunter2', pw: 'hunter3'})}, 'plain CME text'));
  assert.equal(control(msg6b, 'secret'), null, 'a given secret has no control to prefill');
  assert.equal(control(msg6b, 'pw'), null);
  assert.equal(msg6b.querySelectorAll('input[type=password]').length, 0);
  assert.equal(msg6b.querySelectorAll('.paramField[data-field="secret"] input').length, 0);
  assert.ok(msg6b.textContent.indexOf('hunter') === -1);
  assert.ok(!msg6b.innerHTML.includes('hunter'), 'a given secret is not in the page');
  type(control(msg6b, 'story'), 'tale');
  button(msg6b, 'Edit as text').click();
  assert.equal(composer.value, '/widgets story=tale', 'Edit as text leaves the password out');
  composer.value = '';
  assert.equal(msg6b.querySelector('.paramStatus'), null, 'an optional secret blocks nothing');
  assert.equal(button(msg6b, 'Run').disabled, false);
  mark = requests.length;
  button(msg6b, 'Run').click();
  assert.equal(sentAfter(mark)[0].body.user_query, '/widgets story=tale', 'Run omits an optional secret');
  assert.ok(!JSON.stringify(sentAfter(mark)).includes('hunter'));
  await until(() => !tm.busy, 'password run');
  assert.ok(!w.document.getElementById('chatLog').textContent.includes('hunter'), 'no secret in a user bubble');
  assert.ok(!w.document.getElementById('chatLog').textContent.includes('pass1'));

  /* a required secret: Run disabled with a status line, other buttons work */
  const msg6c = paint(failedTurn({parameter_request: request('Plain/login', ['user', 'token'], [],
    {token: 'hunter9'})}));
  assert.equal(msg6c.querySelectorAll('input[type=password]').length, 0);
  assert.ok(!msg6c.textContent.includes('hunter9'));
  assert.ok(field(msg6c, 'token').textContent.indexOf('Not collected here') !== -1);
  assert.equal(button(msg6c, 'Run').disabled, true, 'a required secret disables Run');
  const status6c = msg6c.querySelector('.paramStatus[role="status"]');
  assert.ok(status6c && text(status6c).indexOf('Token') !== -1 && /another way/.test(text(status6c)));
  assert.equal(button(msg6c, 'Edit as text').disabled, false);
  assert.equal(button(msg6c, 'Cancel').disabled, false);
  type(control(msg6c, 'user'), 'ann');
  mark = requests.length;
  button(msg6c, 'Run').click();
  assert.equal(sentAfter(mark).length, 0, 'a disabled Run sends nothing');
  button(msg6c, 'Edit as text').click();
  assert.equal(composer.value, '/login user=ann', 'Edit as text leaves a required secret out');
  composer.value = '';

  /* ---- case 7: the which-command question of the "/" path (the one plan_choice shape) ------------------------ */
  const planChoice = (names) => ({question_kind: 'plan_choice', plan_choice: {request: 'x', sub_task: null,
    choices: names.map((n) => ({name: n, description: n === 'Library/find_book' ? 'Find a book by title.' : '',
      closest: true})), none_reply: 'none', decline: 'cancel', ambiguous: true, misunderstood: false}});
  const ambiguous = () => failedTurn(planChoice(['Library/find_book', 'Library/find_member']),
    'Ambiguous intent error for command x');
  mark = requests.length;
  const msg7 = paint(ambiguous());
  const bar = msg7.querySelector('div.planQuestion');
  assert.ok(bar);
  assert.equal(text(bar.querySelector('.planHead')), 'More than one command fits this request. Which one is meant?');
  const pills = Array.from(bar.querySelectorAll('button.planBtn'));
  assert.deepEqual(pills.map((p) => p.dataset.plan), ['Library/find_book', 'Library/find_member']);
  assert.equal(text(pills[0].querySelector('.planDesc')), 'Find a book by title.');
  button(bar, 'None of these');
  button(bar, 'Cancel');
  const dets = msg7.querySelector('details.paramDetails');
  assert.ok(dets && !dets.open);
  assert.ok(text(dets).indexOf('Ambiguous intent error for command x') !== -1);
  pills[0].click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'Library/find_book');
  assert.equal(sentAfter(mark)[0].path, '/invoke_agent_stream');
  assert.ok(Array.from(bar.querySelectorAll('button')).every((b) => b.disabled), 'a click disables the bar');
  idle();
  mark = requests.length;
  button(paint(ambiguous()), 'None of these').click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'none');
  idle();
  mark = requests.length;
  button(paint(ambiguous()), 'Cancel').click();
  assert.equal(sentAfter(mark)[0].body.user_query, 'cancel');
  idle();
  /* tmDisableLiveForms disables bars and forms that were never clicked */
  const liveBar = paint(ambiguous());
  const liveForm = paint(failedTurn({parameter_request: request('Plain/record_values', ['ratio'])}));
  w.tmDisableLiveForms();
  assert.ok(Array.from(liveBar.querySelectorAll('.planQuestion button')).every((b) => b.disabled));
  assert.ok(Array.from(liveForm.querySelectorAll('.paramForm button, .paramForm input, .paramForm textarea')).every((b) => b.disabled));
  /* typing a message disables them as well */
  const typedForm = paint(failedTurn({parameter_request: request('Plain/record_values', ['ratio'])}));
  const typedBar = paint(ambiguous());
  composer.value = 'hello';
  w.tmSend();
  assert.ok(Array.from(typedForm.querySelectorAll('.paramForm button, .paramForm input, .paramForm textarea')).every((b) => b.disabled));
  assert.ok(Array.from(typedBar.querySelectorAll('.planQuestion button')).every((b) => b.disabled));
  idle();

  /* ---- case 8: guards ----------------------------------------------------------- */
  const noForm = (msg, why) => {
    assert.equal(msg.querySelectorAll('.paramForm, .planQuestion, .paramDetails').length, 0, why);
    assert.ok(!msg.classList.contains('hasForm'), why);
  };
  /* not the last output */
  const notLast = failedTurn({parameter_request: request('Plain/record_values', ['ratio'])});
  notLast.command_outputs.push({command_name: 'x', command_response: {success: true, response: 'fine', artifacts: {}}});
  noForm(paint(notLast), 'the request is on a non-last output');
  const notLastPills = failedTurn(planChoice(['Library/find_book']));
  notLastPills.command_outputs.push({command_name: 'x', command_response: {success: true, response: 'fine'}});
  noForm(paint(notLastPills), 'pills on a non-last output');
  /* no schema, no input schema, an empty list */
  noForm(paint(failedTurn({parameter_request: request('Plain/unknown_cmd', ['x'])})), 'no schema');
  noForm(paint(failedTurn({parameter_request: request('Library/find_member', ['x'])})), 'input is null');
  noForm(paint(failedTurn(planChoice([]))), 'empty choices');
  noForm(paint(failedTurn(planChoice([7, '']))), 'junk choices');
  /* an asked field with no widget leaves the CME text as the only help */
  noForm(paint(failedTurn({parameter_request: request('Plain/tricky', ['box'])})), 'object field');
  /* not the last message in the log */
  const early = w.tmBubble('agent', '...');
  w.tmBubble('user', 'later');
  w.tmRenderTurn(early, failedTurn({parameter_request: request('Plain/record_values', ['ratio'])}));
  noForm(early, 'a turn with a later message is not live');
  /* restored turns: the built-in prompts never show, a registered renderer does on the latest only */
  const record = (artifacts) => ({turn: {record: {turn_output: {status: 'completed', command_outputs: [
    {command_name: 'wildcard', command_response: {success: false, response: 'CME text', artifacts}}]}}}});
  const pr = {parameter_request: request('Plain/record_values', ['ratio']), test_prompt: 'r'};
  storedRecords.tkA = record(pr);
  storedRecords.tkB = record(pr);
  const seen = [];
  const restoredRender = (bubbleMsg, out) => {
    const info = w.tmLastPromptArtifacts(out);
    if (!info || !info.artifacts.test_prompt) return null;
    seen.push({live: w.tmIsLiveTurn(bubbleMsg), outputs: out.command_outputs.length, status: out.status,
      restored: bubbleMsg.dataset.restored});
    if (!w.tmIsLiveTurn(bubbleMsg)) return null;
    const node = d.createElement('div');
    node.className = 'testPrompt';
    return w.tmMountPrompt(bubbleMsg, {key: 'r', element: node, setDisabled() {}, nodes: [node]});
  };
  w.TM_PROMPT_RENDERERS.push(restoredRender);
  const mark2 = requests.length;
  const older = w.tmRenderStoredTurn({turn_key: 'tkA', status: 'awaiting_user', success: false,
    answer: 'CME text'}, true);
  const latest = w.tmRenderStoredTurn({turn_key: 'tkB', status: 'awaiting_user', success: false,
    answer: 'CME text'}, true);
  noForm(older, 'restored turn');
  await until(() => latest.querySelector('.testPrompt') && older.dataset.pendingRecord === '',
    'the stored records never loaded');
  assert.ok(!older.querySelector('.testPrompt'), 'an older restored turn is not mounted');
  assert.equal(latest.querySelectorAll('.testPrompt').length, 1, 'the latest restored turn is');
  assert.equal(latest.querySelectorAll('.paramForm, .planQuestion').length, 0, 'no built-in form on a restored turn');
  assert.equal(older.querySelectorAll('.paramForm, .planQuestion').length, 0);
  assert.deepEqual(seen.map((x) => [x.live, x.outputs, x.status, x.restored]).sort(),
    [[false, 1, 'awaiting_user', '1'], [true, 1, 'awaiting_user', '1']]);
  assert.deepEqual(requests.slice(mark2).map((r) => r.path).sort(), ['/api/turn/tkA', '/api/turn/tkB'],
    'the hook adds no request');
  w.TM_PROMPT_RENDERERS.splice(w.TM_PROMPT_RENDERERS.indexOf(restoredRender), 1);
  /* markup stays text */
  const msgM = paint(failedTurn({parameter_request: request('Plain/with_markup', ['q'])},
    '<b>bold</b> response'));
  assert.equal(msgM.querySelectorAll('img, b').length, 0);
  assert.ok(text(field(msgM, 'q')).indexOf('<img src=x onerror=alert(1)>') !== -1);
  const msgP = paint(failedTurn(planChoice(['Library/<i onclick=x>']), '<u>amb</u>'));
  assert.equal(msgP.querySelectorAll('i, u').length, 0);
  assert.equal(msgP.querySelector('.planBtn .planName').textContent, 'Library/<i onclick=x>');

  /* ---- repaint: no duplicate, typed text survives, a new request replaces it --------- */
  const repaintOut = failedTurn({parameter_request: request('Plain/record_values', ['ratio', 'ids'], [],
    {name: 'Bob'})});
  const msgR = paint(repaintOut);
  type(control(msgR, 'ratio'), '9.5');
  const firstForm = msgR.querySelector('.paramForm');
  w.tmRenderTurn(msgR, repaintOut);
  assert.equal(msgR.querySelectorAll('.paramForm').length, 1);
  assert.equal(msgR.querySelectorAll('details.paramDetails').length, 1);
  assert.equal(msgR.querySelector('.paramForm'), firstForm, 'the same form is put back');
  assert.equal(control(msgR, 'ratio').value, '9.5', 'what was typed survives the repaint');
  w.tmRenderTurn(msgR, JSON.parse(JSON.stringify(repaintOut)));
  assert.equal(control(msgR, 'ratio').value, '9.5', 'an equal request from another copy of the turn');
  const changed = failedTurn({parameter_request: request('Plain/record_values', ['ids'], [], {name: 'Bob'})});
  w.tmRenderTurn(msgR, changed);
  assert.equal(msgR.querySelectorAll('.paramForm').length, 1);
  assert.equal(control(msgR, 'ratio').value, '', 'a different request gets a fresh form');
  w.tmRenderTurn(msgR, {turn_key: 'z', status: 'completed', success: true, answer: 'all good',
    command_outputs: []});
  noForm(msgR, 'a repaint without a request drops the form');
  assert.equal(text(msgR.querySelector('.bubble')), 'all good');
  /* pills too */
  const pillOut = ambiguous();
  const msgRP = paint(pillOut);
  w.tmRenderTurn(msgRP, pillOut);
  assert.equal(msgRP.querySelectorAll('.planQuestion').length, 1);
  assert.equal(msgRP.querySelectorAll('details.paramDetails').length, 1);

  /* ---- a registered renderer is dispatched, mounted and reused ---- */
  let regCalls = 0;
  const regRender = (bubbleMsg, out) => {
    const entry = w.tmLastPromptArtifacts(out);
    const kind = entry && entry.artifacts.test_prompt;
    if (!kind) return null;
    regCalls += 1;
    assert.ok(entry.entry && entry.entry.command_name === 'wildcard', 'the output entry is returned');
    const prev = w.tmReusePrompt(bubbleMsg, 'test:' + kind);
    if (prev) return w.tmMountPrompt(bubbleMsg, prev);
    const node = d.createElement('div');
    node.className = 'testPrompt';
    return w.tmMountPrompt(bubbleMsg, {key: 'test:' + kind, element: node, setDisabled() {}, nodes: [node]});
  };
  const renderersBefore = w.TM_PROMPT_RENDERERS.length;
  const builtInAt = w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderParameterForm);
  const planAt = w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderPlanQuestion);
  assert.ok(builtInAt !== -1 && planAt !== -1 && planAt < builtInAt,
    'the two built-in renderers are registered, the which-command question in front');
  w.TM_PROMPT_RENDERERS.push(regRender);
  try {
    const regOut = failedTurn({test_prompt: 'one'});
    const regMsg = paint(regOut);
    assert.equal(regMsg.querySelectorAll('.testPrompt').length, 1);
    assert.ok(regMsg.classList.contains('hasForm'));
    const regNode = regMsg.querySelector('.testPrompt');
    w.tmRenderTurn(regMsg, regOut);
    assert.equal(regMsg.querySelector('.testPrompt'), regNode, 'a repaint reuses it');
    assert.equal(regMsg.querySelectorAll('.testPrompt').length, 1);
    const callsBefore = regCalls;
    const builtIn = paint(ambiguous());
    assert.ok(builtIn.querySelector('.planQuestion'));
    assert.equal(regCalls, callsBefore, 'a built-in prompt wins first and stops the search');
  } finally {
    w.TM_PROMPT_RENDERERS.splice(w.TM_PROMPT_RENDERERS.indexOf(regRender), 1);
  }
  assert.equal(w.TM_PROMPT_RENDERERS.length, renderersBefore);
  assert.equal(w.TM_PROMPT_RENDERERS.indexOf(regRender), -1);

  /* ---- the results view and the form do not fight over the bubble --------------------- */
  const mixed = failedTurn({parameter_request: request('Plain/record_values', ['ratio'])},
    'Missing: ratio.\n\nlong follow-up paragraph. '.repeat(8));
  mixed.command_outputs.unshift({command_name: 'show_book', context: 'Plain', command_parameters: null,
    command_response: {success: true, response: 'ok', output: {book_uid: 'b1', title: 'Dune', copies: 3}}});
  const msgX = paint(mixed);
  assert.equal(msgX.querySelectorAll('.paramForm').length, 1);
  assert.equal(msgX.querySelectorAll('.results').length, 0);
  assert.equal(msgX.querySelectorAll('.resultsLink, .resultsPanel, .resultsLinkRow').length, 0, 'a prompt turn shows no results link or panel [fix-weih.24]');
  assert.ok(!msgX.classList.contains('resultsOpen') && !msgX.classList.contains('hasResults'));
  assert.equal(msgX.querySelectorAll('.answerRest').length, 0);
  assert.ok(text(msgX.querySelector('details.paramDetails')).indexOf('long follow-up paragraph.') !== -1);

  /* ---- the "/" parameter question carries question_kind "planning_parameter" ----------- */
  const tagged = paint(failedTurn({question_kind: 'planning_parameter', parameter_request: request('Plain/record_values',
    ['ratio', 'ids'], [], {name: 'Bob', count: 3})}));
  assert.equal(tagged.querySelectorAll('.paramForm').length, 1, 'a completed "/" turn with the tag still gets the form');
  const waitingTagged = failedTurn({question_kind: 'planning_parameter', parameter_request: request('Plain/record_values',
    ['ratio', 'ids'], [], {name: 'Bob', count: 3})}, null, {status: 'awaiting_user'});
  assert.equal(w.tmRenderParameterForm(w.tmBubble('agent', '...'), waitingTagged), null,
    'a waiting turn with the tag is an agent turn question, not this form');

  /* ---- artifacts link skips prompt keys only ------------------------------------------- */
  const only = paint(failedTurn({question_kind: 'agent', approval: {a: 1}, review: {}, parameter_request: null,
    plan_choice: null}));
  assert.equal(only.querySelectorAll('.artifactsLink').length, 0, 'prompt keys alone make no link');
  const mix = paint(failedTurn({question_kind: 'agent', approval: {}, receipt: {id: 5}}));
  assert.equal(mix.querySelector('.artifactsLink').textContent, '1 artifact');
  const kept = paint(failedTurn({awaiting_user: true, reply_intent: 'wait', parameter_request: null}));
  assert.equal(kept.querySelector('.artifactsLink').textContent, '2 artifacts', 'other keys are untouched');
  assert.deepEqual(plain(w.tmArtifactsIn(failedTurn({question_kind: 'x', other: 1}).command_outputs)
    .map((a) => a.key)), ['other']);

  /* ---- direct tmRenderForm checks ------------------------------------------------------ */
  const direct = w.tmRenderForm({title: 'Direct', fields: [
    {name: 'who', label: 'Who', kind: 'choice', required: true, enforceRequired: true,
      options: [{value: 'a', label: 'A'}, {value: 'b', label: 'B'}], allowOther: true},
    {name: 'tags', label: 'Tags', kind: 'multichoice', allowOther: true,
      options: [{value: 'x', label: 'X'}, {value: 'y', label: 'Y'}], value: ['y', 'zz']},
    {name: 'day', label: 'Day', kind: 'date', value: '2025-02-03'},
    {name: 'qty', label: 'Qty', kind: 'integer', min: 1, max: 5},
    {name: 'many', label: 'Many', kind: 'choice', required: true,
      options: [1, 2, 3, 4, 5, 6].map((n) => ({value: 'v' + n, label: 'V' + n})), allowOther: true},
    {name: 'opt', label: 'Opt', kind: 'choice', options: [{value: 'p'}, {value: 'q'}]},
  ]}, [
    {label: 'Go', kind: 'primary', onClick: (v) => { direct.got = v; }},
    {label: 'Skip', kind: 'secondary', onClick: (v) => { direct.skipped = v; }},
  ]);
  assert.deepEqual(Object.keys(direct).sort(), ['element', 'focusFirstAsked', 'setDisabled', 'values']);
  const holder = d.createElement('div');
  d.body.appendChild(holder);
  holder.appendChild(direct.element);
  const root = direct.element;
  const who = root.querySelector('[data-field="who"]');
  assert.ok(Array.from(who.querySelectorAll('input[type=radio]')).every((r) => r.required),
    'enforceRequired sets the required attribute');
  assert.equal(who.querySelectorAll('input[type=radio]').length, 3, 'a, b and Other…, no dash when required');
  assert.deepEqual(plain(direct.values()).who, null);
  assert.equal(root.querySelector('[data-field="tags"] input[type=checkbox]:nth-child(1)') === null, false);
  assert.deepEqual(plain(direct.values()).tags, ['y', 'zz'], 'a value outside the options lands in Other…');
  assert.equal(plain(direct.values()).day, '2025-02-03');
  assert.equal(root.querySelector('[data-field="day"] input').type, 'date');
  assert.equal(plain(direct.values()).qty, null);
  assert.ok(root.querySelector('[data-field="many"] select'), 'more than five is a select');
  assert.equal(root.querySelector('[data-field="many"] select').options.length, 8, 'Select…, 6, Other…');
  assert.ok(root.querySelector('[data-field="opt"] input[type=radio][value=""]').checked, 'optional choice: dash');
  assert.equal(root.querySelector('[data-field="opt"] .paramOpt:nth-child(2)').textContent, 'p');
  /* a primary action refuses while a required field is empty */
  button(root, 'Go').click();
  assert.equal(direct.got, undefined);
  assert.ok(who.classList.contains('asked'));
  assert.equal(text(who.querySelector('.paramState')), 'Required');
  assert.equal(d.activeElement.closest('[data-field]').getAttribute('data-field'), 'who');
  button(root, 'Skip').click();
  assert.ok(direct.skipped, 'a secondary action is never blocked');
  /* Other… takes the typed text, and typing selects it */
  const otherRow = Array.from(who.querySelectorAll('.paramOpt')).pop();
  type(otherRow.querySelector('input[type=text]'), '  custom ');
  assert.ok(otherRow.querySelector('input[type=radio]').checked);
  assert.equal(plain(direct.values()).who, 'custom');
  button(root, 'Go').click();
  assert.equal(direct.got.who, 'custom');
  assert.ok(!who.classList.contains('asked'), 'the mark is cleared once it is filled');
  /* number, select + Other, multichoice */
  const qty = root.querySelector('[data-field="qty"] input');
  qty.value = '4';
  assert.equal(plain(direct.values()).qty, 4);
  const many = root.querySelector('[data-field="many"] select');
  many.selectedIndex = many.options.length - 1;
  type(root.querySelector('[data-field="many"] input[type=text]'), 'else');
  assert.equal(plain(direct.values()).many, 'else');
  many.selectedIndex = 2;
  assert.equal(plain(direct.values()).many, 'v2');
  /* option display: label then the muted value */
  const shown = w.tmRenderForm({fields: [{name: 'c', label: 'C', kind: 'choice', required: true,
    options: [{value: 'k1', label: 'First'}, {value: 'k2', label: 'k2'}]}]}, []);
  const rows = Array.from(shown.element.querySelectorAll('.paramOpt'));
  assert.equal(rows[0].querySelector('.paramOptValue').textContent, 'k1');
  assert.ok(!rows[1].querySelector('.paramOptValue'));
  /* setDisabled and focusFirstAsked */
  direct.setDisabled(true);
  assert.ok(Array.from(root.querySelectorAll('input, select, textarea, button')).every((n) => n.disabled));
  direct.setDisabled(false);
  assert.ok(Array.from(root.querySelectorAll('input, select, textarea, button')).every((n) => !n.disabled));
  const asked = w.tmRenderForm({fields: [{name: 'a', label: 'A', kind: 'text'},
    {name: 'b', label: 'B', kind: 'longtext', state: 'missing'}]}, []);
  holder.appendChild(asked.element);
  asked.focusFirstAsked();
  assert.equal(d.activeElement.tagName, 'TEXTAREA');
  holder.removeChild(asked.element);
  /* list kind, text values keep their spacing, numbers fall back to the raw text */
  const kinds = w.tmRenderForm({fields: [{name: 'l', label: 'L', kind: 'list', value: ['a', 'b']},
    {name: 't', label: 'T', kind: 'text', value: ' x '}, {name: 'bool', label: 'B', kind: 'boolean'}]}, []);
  assert.deepEqual(plain(kinds.values()), {l: ['a', 'b'], t: ' x ', bool: false});

  /* ---- fix-weih.33: free text wraps and grows; everything else is as it was --- */
  assert.equal(typeof w.tmFitHeight, 'function');
  assert.equal(w.tmFitHeight(0, 18, 8, 2, 12), null, 'an unmeasured (hidden) field is left alone');
  assert.deepEqual(plain(w.tmFitHeight(26, 18, 8, 2, 12)), {height: 28, scroll: false});
  assert.deepEqual(plain(w.tmFitHeight(5000, 18, 8, 2, 12)), {height: 18 * 12 + 8 + 2, scroll: true});
  assert.equal(w.tmReviewFitHeight(26, 18, 8, 2, 12).height, 28, 'the review name is an alias of the same code');

  const gm = paint(failedTurn({parameter_request: request('Plain/widgets', ['story'])}, 'x'));
  const gw = paint(failedTurn({parameter_request: request('Plain/record_values', [], [], {name: 'Bob'})}));
  const nameCtl = control(gw, 'name');
  assert.equal(nameCtl.tagName, 'TEXTAREA', 'free text is a wrapping field');
  assert.ok(nameCtl.classList.contains('fitText'));
  assert.equal(nameCtl.rows, 1);
  assert.equal(nameCtl.wrap, 'soft');
  assert.equal(nameCtl.placeholder, 'Ann', 'the placeholder carries over');
  assert.equal(nameCtl.value, 'Bob');
  assert.equal(w.document.querySelectorAll('label[for="' + nameCtl.id + '"]').length, 1, 'one label for the field');
  assert.equal(control(gw, 'count').tagName, 'INPUT', 'integers keep the native number input');
  assert.equal(control(gw, 'count').type, 'number');
  assert.equal(control(gw, 'ratio').type, 'number');
  assert.equal(control(gm, 'story').tagName, 'TEXTAREA');
  assert.ok(!control(gm, 'story').classList.contains('fitText'), 'the textarea widget stays a 4-row textarea');
  assert.equal(control(gm, 'story').rows, 4);
  assert.equal(control(gm, 'when').type, 'date');
  assert.equal(control(gm, 'color').tagName, 'INPUT', 'radios are untouched');
  assert.equal(gm.querySelectorAll('textarea.fitText').length, 0, 'no growing field among enum, date, secret, widget fields');

  /* grows with the text (scrollHeight is test data: jsdom has no layout) */
  const scrollOf = (node, value) => Object.defineProperty(node, 'scrollHeight', {configurable: true, get: () => value.v});
  const hh = {v: 26};
  scrollOf(nameCtl, hh);
  type(nameCtl, 'Short');
  const shortH = parseFloat(nameCtl.style.height);
  assert.ok(shortH >= 26, 'a short value is one line: ' + nameCtl.style.height);
  hh.v = 130;
  type(nameCtl, 'A long value that would need sideways scrolling in a single line field. '.repeat(3));
  assert.ok(parseFloat(nameCtl.style.height) > shortH, 'a long value grows the field');
  assert.equal(nameCtl.style.overflowY, 'hidden');
  hh.v = 4000;
  type(nameCtl, 'x');
  assert.ok(parseFloat(nameCtl.style.height) < 4000, 'growth is capped');
  assert.equal(nameCtl.style.overflowY, 'auto', 'past the cap it scrolls vertically');
  hh.v = 0;
  w.dispatchEvent(new w.Event('resize'));
  assert.equal(nameCtl.style.height, '', 'a hidden field gets no height and is measured when shown');
  hh.v = 52;
  w.dispatchEvent(new w.Event('resize'));
  assert.ok(parseFloat(nameCtl.style.height) >= 52);

  /* Enter does what it did in the input (nothing: the form has no submit on Enter) */
  const enter = new w.KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true});
  mark = requests.length;
  nameCtl.dispatchEvent(enter);
  assert.equal(enter.defaultPrevented, true, 'Enter adds no line');
  assert.equal(sentAfter(mark).length, 0, 'and sends nothing');
  const letter = new w.KeyboardEvent('keydown', {key: 'a', bubbles: true, cancelable: true});
  nameCtl.dispatchEvent(letter);
  assert.equal(letter.defaultPrevented, false);
  /* a newline typed or set is removed as an input removes it; a pasted one becomes a space */
  type(nameCtl, 'Line one\r\nline two');
  assert.equal(nameCtl.value, 'Line oneline two');
  type(nameCtl, 'Bob');
  nameCtl.setSelectionRange(3, 3);
  const paste = new w.Event('paste', {bubbles: true, cancelable: true});
  Object.defineProperty(paste, 'clipboardData', {value: {getData: () => ' and\r\nAnn'}});
  nameCtl.dispatchEvent(paste);
  assert.equal(nameCtl.value, 'Bob and Ann');

  /* the reply for the same input is what it was: no trimming added or removed */
  type(nameCtl, '  Bob Lee  ');
  assert.equal(plain(w.tmRenderForm({fields: [{name: 'n', label: 'N', kind: 'text', value: ' x '}]}, []).values()).n, ' x ');
  mark = requests.length;
  button(gw, 'Run').click();
  assert.equal(sentAfter(mark)[0].body.user_query, '/record_values name="  Bob Lee  "',
    'leading and trailing spaces are quoted exactly as they were for the input');
  idle();
  const gw2 = paint(failedTurn({parameter_request: request('Plain/record_values', [], [], {})}));
  type(control(gw2, 'name'), 'Plain value');
  mark = requests.length;
  button(gw2, 'Run').click();
  assert.equal(sentAfter(mark)[0].body.user_query, '/record_values name=Plain value');
  idle();
  /* required: the same refusal and mark */
  const req = w.tmRenderForm({fields: [{name: 'r', label: 'R', kind: 'text', required: true, enforceRequired: true}]},
    [{label: 'Go', kind: 'primary', onClick() { throw new Error('must not run'); }}]);
  holder.appendChild(req.element);
  const rc = req.element.querySelector('textarea');
  assert.equal(rc.required, true);
  assert.equal(rc.getAttribute('aria-required'), 'true');
  req.element.querySelector('button').click();
  assert.equal(rc.getAttribute('aria-invalid'), 'true');
  assert.equal(d.activeElement, rc, 'the empty required field takes focus');
  holder.removeChild(req.element);

  assert.deepEqual(errors, [], 'jsdom reported errors');
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
