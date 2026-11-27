/* The chat SPA's approval card (fix-weih.12), exercised against the assembled page. Only the network edge
   (window.fetch) is replaced: it records every request to the workflow server, answers /invoke_assistant with a
   finished turn, and lets the streaming agent endpoint hang. */
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
const assistantReplies = [];   /* queued /invoke_assistant answers; empty means a finished turn */
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
      if (u === BASE + '/invoke_assistant' && assistantReplies.length) {
        const reply = assistantReplies.shift();
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve(reply)});
      }
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

function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function until(fn, what) {
  for (let i = 0; i < 100; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function text(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
function idle() { tm.busy = false; w.tmComposerState(); }

const SCHEMAS = {
  'Orders/cancel_pending_order': {kind: 'execution', output: null, input: {
    properties: {
      order_id: {type: 'string', title: 'Order ID'},
      reason: {type: 'string', title: 'Reason'},
    }, required: ['order_id', 'reason']}},
  'Record/add_note': {kind: 'execution', output: null, input: {
    properties: {text: {type: 'string', title: 'Text'}}, required: ['text']}},
  'Account/set_login': {kind: 'execution', output: null, input: {
    properties: {user: {type: 'string', title: 'User'},
      token: {type: 'string', writeOnly: true, title: 'Token'},
      pw: {type: 'string', format: 'password', title: 'Pw'}}, required: ['user']}},
};
const PROMPT = 'The agent wants to run Orders/cancel_pending_order with order_id=#W1, reason=dup. ' +
  'Reply yes to approve, no to decline, or always to approve this command from now on.';

function approval(over) {
  return Object.assign({command: 'Orders/cancel_pending_order',
    parameters: {order_id: '#W1', reason: 'dup'}, origin: 'agent', can_always: true}, over || {});
}
function waitingTurn(artifacts, response, status) {
  return {turn_key: 'k' + Math.random().toString(16).slice(2), status: status || 'awaiting_user',
    success: false, answer: response || PROMPT,
    command_outputs: [{command_name: 'ask_user', context: 'Orders', command_parameters: null,
      command_response: {success: false, response: response || PROMPT, artifacts}}]};
}
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
function button(root, label) {
  const b = Array.from(root.querySelectorAll('button')).find((x) => x.textContent === label);
  assert.ok(b, 'no button ' + label);
  return b;
}
const labels = (msg) => Array.from(msg.querySelectorAll('.approvalCard button')).map((b) => b.textContent);

async function main() {
  assert.equal(typeof w.tmRenderApprovalCard, 'function');
  assert.deepEqual(JSON.parse(JSON.stringify(w.TM_APPROVAL_REPLIES)),
    {approve: 'yes', always: 'always', decline: 'no'});
  assert.ok(w.TM_PROMPT_RENDERERS.indexOf(w.tmRenderApprovalCard) !== -1, 'registered');
  tm.connected = true;
  tm.baseUrl = BASE;
  tm.commandSchemas = SCHEMAS;
  idle();
  const composer = d.getElementById('chatInput');

  /* ---- case 1: agent-originated approval, all three buttons send the exact words ----------------------- */
  for (const [label, word] of [['Approve', w.TM_APPROVAL_REPLIES.approve],
    ['Always allow cancel_pending_order', w.TM_APPROVAL_REPLIES.always], ['Decline', w.TM_APPROVAL_REPLIES.decline]]) {
    const mark = requests.length;
    const msg = paint(waitingTurn({question_kind: 'approval', approval: approval()}));
    const card = msg.querySelector('.approvalCard');
    assert.ok(card, 'card');
    assert.equal(text(card.querySelector('.approvalTitle')), 'Approve cancel_pending_order?');
    assert.equal(card.getAttribute('aria-labelledby'), card.querySelector('.approvalTitle').id);
    assert.deepEqual(Array.from(card.querySelectorAll('dl.rKv dt')).map(text), ['Order ID', 'Reason']);
    assert.deepEqual(Array.from(card.querySelectorAll('dl.rKv dd')).map(text), ['#W1', 'dup']);
    assert.deepEqual(labels(msg), ['Approve', 'Always allow cancel_pending_order', 'Decline']);
    const details = msg.querySelector('details.approvalDetails, details.paramDetails');
    assert.ok(details && !details.open, 'closed Details');
    assert.equal(text(details.querySelector('summary')), 'Details');
    assert.ok(text(details).indexOf('Reply yes to approve') !== -1, 'verbatim prompt kept in Details');
    assert.equal(msg.querySelectorAll('.cmdOut').length, 0);
    assert.equal(d.activeElement, button(card, 'Approve'), 'initial focus is Approve');
    button(card, label).click();
    assert.deepEqual(requests.slice(mark).map((r) => r.body && r.body.user_query), [word]);
    assert.equal(composer.value, '', 'composer untouched');
    assert.ok(Array.from(card.querySelectorAll('button')).every((b) => b.disabled), 'sending disables the card');
    assert.ok(Array.from(d.querySelectorAll('.chatMsg.user .bubble')).some((n) => n.textContent === word),
      'the reply is ordinary transcript text: ' + word);
    idle();
  }

  /* ---- case 2: user-originated "/" approval: status completed, artifacts.awaiting_user ----------------- */
  let mark = requests.length;
  const slash = paint(waitingTurn({awaiting_user: true, question_kind: 'approval',
    approval: approval({command: 'Record/add_note', parameters: {text: 'Ann'}, origin: 'user'})},
  'You asked to run Record/add_note with text=Ann. Reply yes to approve.', 'completed'));
  const slashCard = slash.querySelector('.approvalCard');
  assert.ok(slashCard, 'card on a completed turn that awaits the user');
  assert.equal(text(slashCard.querySelector('.approvalTitle')), 'Run add_note?');
  assert.deepEqual([text(slashCard.querySelector('dt')), text(slashCard.querySelector('dd'))], ['Text', 'Ann']);
  assert.equal(slashCard.querySelector('.approvalActions') !== null, true);
  button(slashCard, 'Always allow add_note');
  button(slashCard, 'Approve').click();
  assert.deepEqual(requests.slice(mark).map((r) => r.body && r.body.user_query), ['yes']);
  idle();
  /* a completed turn that does NOT await the user gets no card */
  assert.equal(paint(waitingTurn({question_kind: 'approval', approval: approval()}, 'x', 'completed'))
    .querySelectorAll('.approvalCard').length, 0, 'completed and not awaiting: no card');

  /* ---- case 3: can_always ------------------------------------------------------------------------------ */
  assert.deepEqual(labels(paint(waitingTurn({question_kind: 'approval', approval: approval({can_always: false})}))),
    ['Approve', 'Decline']);
  const older = approval();
  delete older.can_always;
  assert.deepEqual(labels(paint(waitingTurn({question_kind: 'approval', approval: older}))),
    ['Approve', 'Always allow cancel_pending_order', 'Decline'], 'a missing can_always still offers it');

  /* ---- case 4: fallbacks ------------------------------------------------------------------------------- */
  /* no parameters */
  const bare = paint(waitingTurn({question_kind: 'approval', approval: approval({parameters: {}})}));
  assert.equal(bare.querySelectorAll('dl.rKv').length, 0);
  assert.deepEqual(labels(bare), ['Approve', 'Always allow cancel_pending_order', 'Decline']);
  const noParams = approval();
  delete noParams.parameters;
  assert.equal(paint(waitingTurn({question_kind: 'approval', approval: noParams})).querySelectorAll('dl.rKv').length, 0);
  /* missing approval payload: verbatim text stays visible, three buttons beneath it */
  mark = requests.length;
  const missing = paint(waitingTurn({question_kind: 'approval'}));
  const mCard = missing.querySelector('.approvalCard');
  assert.ok(mCard, 'card without payload');
  assert.equal(text(mCard.querySelector('.approvalText')), text(d.createTextNode(PROMPT)));
  assert.equal(missing.querySelectorAll('details').length, 0, 'the text is not hidden');
  assert.equal(mCard.querySelector('.approvalTitle'), null);
  assert.deepEqual(labels(missing), ['Approve', 'Always allow', 'Decline']);
  button(mCard, 'Decline').click();
  assert.deepEqual(requests.slice(mark).map((r) => r.body && r.body.user_query), ['no']);
  idle();
  /* a payload with no command is as good as none */
  assert.ok(paint(waitingTurn({question_kind: 'approval', approval: {parameters: {a: 1}}})).querySelector('.approvalText'));
  /* unknown command schema: parameters still shown, raw names */
  const unknown = paint(waitingTurn({question_kind: 'approval', approval: approval({command: 'X/mystery', parameters: {a: 1}})}));
  assert.equal(text(unknown.querySelector('.approvalTitle')), 'Approve mystery?');
  assert.equal(unknown.querySelectorAll('dl.rKv dd').length, 1);

  /* no card for other kinds, or without a question_kind */
  for (const kind of ['agent', 'review', 'unknown', 'planning_parameter']) {
    assert.equal(paint(waitingTurn({question_kind: kind, approval: approval()}))
      .querySelectorAll('.approvalCard').length, 0, kind);
  }
  assert.equal(paint(waitingTurn({approval: approval()})).querySelectorAll('.approvalCard').length, 0, 'no kind');
  /* not the last output */
  const notLast = waitingTurn({question_kind: 'approval', approval: approval()});
  notLast.command_outputs.push({command_name: 'x', command_response: {success: true, response: 'fine', artifacts: {}}});
  assert.equal(paint(notLast).querySelectorAll('.approvalCard').length, 0, 'not the last output');
  /* not the latest message */
  const early = w.tmBubble('agent', '...');
  w.tmBubble('user', 'later');
  w.tmRenderTurn(early, waitingTurn({question_kind: 'approval', approval: approval()}));
  assert.equal(early.querySelectorAll('.approvalCard').length, 0, 'a turn with a later message is not live');
  /* the "Rich answers" switch does not matter */
  const rich = d.getElementById('richAnswers');
  if (rich) { rich.checked = false; }
  assert.ok(paint(waitingTurn({question_kind: 'approval', approval: approval()})).querySelector('.approvalCard'));

  /* a typed message disables the card */
  const typed = paint(waitingTurn({question_kind: 'approval', approval: approval()}));
  composer.value = 'hello';
  w.tmSend();
  assert.ok(Array.from(typed.querySelectorAll('.approvalCard button')).every((b) => b.disabled), 'typing disables');
  idle();
  composer.value = '';

  /* a repaint of the same payload reuses the element; a changed payload rebuilds it */
  const repaint = paint(waitingTurn({question_kind: 'approval', approval: approval()}));
  const first = repaint.querySelector('.approvalCard');
  const same = waitingTurn({question_kind: 'approval', approval: approval()});
  w.tmRenderTurn(repaint, same);
  assert.equal(repaint.querySelectorAll('.approvalCard').length, 1);
  assert.equal(repaint.querySelector('.approvalCard'), first, 'same payload reuses the element');
  w.tmRenderTurn(repaint, waitingTurn({question_kind: 'approval', approval: approval({can_always: false})}));
  assert.equal(repaint.querySelectorAll('.approvalCard').length, 1);
  assert.notEqual(repaint.querySelector('.approvalCard'), first, 'a changed payload rebuilds');
  assert.equal(labels(repaint).length, 2);
  idle();

  /* ---- secrets, markup, focus ------------------------------------------------------------------------- */
  const secret = paint(waitingTurn({question_kind: 'approval', approval: approval({command: 'Account/set_login',
    parameters: {user: 'ann', token: 'S3CR3T-tok', pw: 'hunter2'}})}));
  const secretCard = secret.querySelector('.approvalCard');
  assert.deepEqual(Array.from(secretCard.querySelectorAll('dl.rKv dd')).map(text), ['ann', '••••', '••••']);
  assert.equal(secretCard.textContent.indexOf('S3CR3T-tok'), -1, 'no secret in the card');
  assert.equal(secretCard.textContent.indexOf('hunter2'), -1);
  /* markup stays text */
  const evil = '<img src=x onerror=alert(1)>';
  const markup = paint(waitingTurn({question_kind: 'approval', approval: approval({
    command: 'Orders/cancel_pending_order', parameters: {order_id: evil, reason: '<b>bold</b>'}})}, evil + ' prompt'));
  assert.equal(markup.querySelectorAll('img, b').length, 0, 'no element from response text');
  assert.ok(text(markup.querySelector('.approvalCard')).indexOf(evil) !== -1);
  const markup2 = paint(waitingTurn({question_kind: 'approval'}, evil));
  assert.equal(markup2.querySelectorAll('img').length, 0);
  assert.equal(text(markup2.querySelector('.approvalText')), evil);
  /* a draft in the composer keeps the focus */
  composer.value = 'half typed';
  composer.focus();
  const drafted = paint(waitingTurn({question_kind: 'approval', approval: approval()}));
  await tick();
  assert.notEqual(d.activeElement, drafted.querySelector('.approvalCard button'), 'a draft keeps its focus');
  assert.equal(composer.value, 'half typed');
  composer.value = '';

  /* ---- restored turns through the stored path ---------------------------------------------------------- */
  const record = (status, artifacts) => ({turn: {record: {turn_output: {status, command_outputs: [
    {command_name: 'ask_user', command_response: {success: false, response: PROMPT, artifacts}}]}}}});
  storedRecords.tkA = record('awaiting_user', {question_kind: 'approval', approval: approval()});
  storedRecords.tkB = record('awaiting_user', {question_kind: 'approval', approval: approval()});
  const olderStored = w.tmRenderStoredTurn({turn_key: 'tkA', status: 'awaiting_user', success: false, answer: PROMPT}, true);
  const latestStored = w.tmRenderStoredTurn({turn_key: 'tkB', status: 'awaiting_user', success: false, answer: PROMPT}, true);
  await until(() => latestStored.querySelector('.approvalCard') && olderStored.dataset.pendingRecord === '',
    'the stored records never loaded');
  assert.equal(olderStored.querySelectorAll('.approvalCard').length, 0, 'an older restored turn has no card');
  assert.equal(latestStored.querySelectorAll('.approvalCard').length, 1, 'the latest restored turn does');
  idle();
  /* restored user-originated approval: status completed */
  storedRecords.tkC = record('completed', {awaiting_user: true, question_kind: 'approval',
    approval: approval({origin: 'user', command: 'Record/add_note', parameters: {text: 'Ann'}})});
  const restoredSlash = w.tmRenderStoredTurn({turn_key: 'tkC', status: 'completed', success: false, answer: PROMPT}, true);
  await until(() => restoredSlash.querySelector('.approvalCard'), 'restored "/" approval');
  assert.equal(text(restoredSlash.querySelector('.approvalTitle')), 'Run add_note?');

  /* ---- the approval token a host-surface server delivers (fix-rkr1) ------------------------------------- */
  const TOKEN = 'T0KEN-9f3a1c';
  const delivered = (over) => Object.assign(waitingTurn({question_kind: 'approval', approval: approval()}),
    {approval: {record_id: 'r1', token: TOKEN}}, over || {});
  const sendSlash = async (line) => {
    composer.value = line;
    w.tmSend();
    await until(() => !tm.busy, 'the assistant turn to settle');
  };
  const lastBody = () => requests[requests.length - 1].body;
  const noTokenAnywhere = () => {
    assert.equal(d.documentElement.outerHTML.indexOf(TOKEN), -1, 'the token is never rendered');
    assert.equal(JSON.stringify(Object.assign({}, w.localStorage)).indexOf(TOKEN), -1, 'never in localStorage');
    assert.equal(Array.from(d.querySelectorAll('.chatMsg.user .bubble')).some((n) => n.textContent.indexOf(TOKEN) !== -1),
      false, 'never in chat text');
  };

  /* delivered with the prompt: the card sends it back once, as approval_token on the one reply that answers it */
  const served = delivered();
  assistantReplies.push(served);
  await sendSlash('/cancel it');
  assert.equal(lastBody().user_query, '/cancel it');
  assert.equal('approval_token' in lastBody(), false, 'the first message carries no token');
  assert.equal(served.approval, undefined, 'taken off the turn output, so nothing can render it');
  const tokenCard = Array.from(d.querySelectorAll('.approvalCard')).pop();
  assert.ok(tokenCard, 'the approval card is painted');
  noTokenAnywhere();
  mark = requests.length;
  button(tokenCard, 'Approve').click();
  assert.deepEqual(requests.slice(mark).map((r) => [r.path, r.body.user_query, r.body.approval_token]),
    [['/invoke_agent_stream', 'yes', TOKEN]]);
  noTokenAnywhere();
  idle();
  /* once: the next reply, and the next message, carry none */
  mark = requests.length;
  w.tmSendChatText('yes');
  assert.equal('approval_token' in requests[mark].body, false, 'cleared after use');
  idle();

  /* a message that is not one of the three replies drops the token without sending it */
  assistantReplies.push(delivered());
  await sendSlash('/cancel it');
  mark = requests.length;
  w.tmSendChatText('something else');
  assert.equal('approval_token' in requests[mark].body, false);
  idle();
  mark = requests.length;
  w.tmSendChatText('yes');
  assert.equal('approval_token' in requests[mark].body, false, 'the token was dropped by the other message');
  idle();

  /* a new conversation, and a source (epoch) change, clear it */
  assistantReplies.push(delivered());
  await sendSlash('/cancel it');
  w.tmClearLog();
  mark = requests.length;
  w.tmSendChatText('yes');
  assert.equal('approval_token' in requests[mark].body, false, 'cleared on a conversation change');
  idle();
  assistantReplies.push(delivered());
  await sendSlash('/cancel it');
  tm.epoch++;
  mark = requests.length;
  w.tmSendChatText('yes');
  assert.equal('approval_token' in requests[mark].body, false, 'not sent across an epoch change');
  idle();
  noTokenAnywhere();

  /* no token delivered (a channel-surface server): the request body is what it was before */
  mark = requests.length;
  const plain = paint(waitingTurn({question_kind: 'approval', approval: approval()}));
  button(plain.querySelector('.approvalCard'), 'Approve').click();
  assert.deepEqual(Object.keys(requests[mark].body).sort(),
    ['timeout_seconds', 'unattended', 'use_dspy_cache', 'user_query']);
  assert.equal(requests[mark].body.user_query, 'yes');
  idle();

  assert.deepEqual(errors, [], 'jsdom reported errors');
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
