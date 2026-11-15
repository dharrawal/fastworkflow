/* The chat SPA's composed answer card (fix-weih.10). The page's own tmComposeCard,
   tmResolveCardRef, tmRenderTurn and tmRenderStoredTurn run against the assembled page.
   Only the network edge (window.fetch) is replaced: it records every request and answers
   /compose_answer_card as the test scripts it (a card, null, HTTP 500, a hang that honours
   the abort signal). Nothing else asks the network for anything. */
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

const calls = [];          /* every /compose_answer_card request: {url, body, signal} */
let composeHandler = null; /* (call) => Promise<response> */
let storedRecords = {};
let alerted = false;
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.alert = () => { alerted = true; };
    window.fetch = (url, opts) => {
      url = String(url);
      if (url.endsWith('/compose_answer_card')) {
        const call = {url, body: JSON.parse(opts.body), signal: opts.signal};
        calls.push(call);
        return composeHandler ? composeHandler(call) : new Promise(() => {});
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
const tm = w.tm;

function plain(value) { return JSON.parse(JSON.stringify(value)); }
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function flush() { for (let i = 0; i < 8; i++) await tick(); }
async function until(fn, what) {
  for (let i = 0; i < 200; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function reply(card) {
  return () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({card})});
}
function replyReason(reason) {
  return () => Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({card: null, reason})});
}
function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return {promise, resolve: (card) => resolve({ok: true, status: 200, json: () => Promise.resolve({card})})};
}
function co(name, output, params, extra) {
  return Object.assign({
    command_name: name, context: 'Library', command_parameters: params === undefined ? null : params,
    command_response: {success: true, response: 'ok', output},
  }, extra || {});
}
function turn(answer, outputs, status) {
  return {turn_key: 'k' + Math.random().toString(16).slice(2), status: status || 'completed',
    success: true, answer, command_outputs: outputs};
}
/* A live agent bubble, as tmSend leaves it, painted once. */
function paint(out, opts) {
  const msg = w.tmBubble('agent', '...');
  msg.dataset.userMessage = (opts && opts.message) || 'what is going on';
  msg.dataset.deterministic = (opts && opts.deterministic) ? '1' : '';
  w.tmRenderTurn(msg, out);
  return msg;
}
/* An eligible turn composes by itself (fix-weih.24): the bubble keeps the whole
   plain answer, with a "Composing answer..." status line next to the results link,
   until the card arrives. paintC is paint, kept so the call sites read as before. */
function paintC(out, opts) { return paint(out, opts); }
function noButton(msg) { assert.equal(msg.querySelectorAll('button.cardCompose').length, 0, 'no Compose card button'); }
function text(node) {
  const parts = [];
  const walk = (n) => {
    if (n.nodeType === 3) parts.push(n.nodeValue);
    else Array.from(n.childNodes).forEach(walk);
  };
  walk(node);
  return parts.join(' ').replace(/\s+/g, ' ').trim();
}
function tm_close(msg) { w.tmSetResultsOpen(msg, false, ''); }
function norm(s) { return s.replace(/\s+/g, ' ').trim(); }

const SCHEMAS = {
  report: {kind: 'execution', input: null, output: {properties: {
    internal_id: {type: 'string', title: 'Internal Id', ui: {hidden: true}},
    total: {type: 'number', title: 'Total', description: 'Order total including tax', ui: {hero: true, format: 'currency'}},
    rate: {type: 'number', title: 'Rate', description: 'x'.repeat(150), ui: {format: 'percent'}},
    lines: {type: 'array', title: 'Lines', items: {type: 'object', properties: {
      sku: {type: 'string', title: 'Sku'},
      secret: {type: 'string', title: 'Secret', ui: {hidden: true}},
      qty: {type: 'integer', title: 'Qty'},
    }}},
  }}},
  'Library/find_book': {kind: 'execution', input: {properties: {}}, output: {properties: {
    book_uids: {type: 'array', items: {type: 'string'}, title: 'Book Uids'},
    titles: {type: 'array', items: {type: 'string'}, title: 'Titles'},
    total: {type: 'integer', title: 'Total'},
  }}},
  get_user_details: {kind: 'execution', input: {properties: {}}, output: null},
  get_order_details: {kind: 'execution', input: {properties: {
    order_id: {type: 'string', available_from: ['get_user_details']}}},
    output: {properties: {order_details: {type: 'string', title: 'Order Details'}}}},
};

const QUOTE = 'the warehouse will close early on Friday so deliveries slip';
const ANSWER = 'Order summary is ready.\n\nNote: ' + QUOTE + ', sorry about that.\n\nThanks for waiting.';
const REPORT = {internal_id: 'x1', total: 186.32, rate: 12.5,
  lines: [{sku: 'A1', qty: 2, secret: 's1'}, {sku: 'B2', qty: 5, secret: 's2'}]};
const BOOKS = {book_uids: ['u1', 'u2'], titles: ['Dune', 'Emma'], total: 2};   /* a page, as the server defines one (fix-weih.20) */
function multi() { return [co('report', REPORT, null), co('find_book', BOOKS, {start: 0})]; }
const GOOD_CARD = {
  lead: 'Two results are ready',
  blocks: [
    {kind: 'stats', title: 'Headline', stats: [
      {label: 'Order total', ref: 'r1.total', tone: 'good'},
      {label: 'Rate', ref: 'r1.rate'},
      {label: 'Id', ref: 'r1.internal_id'},
      {label: 'Ghost', ref: 'r9.total'}]},
    {kind: 'table', title: 'Books', ref: 'r2', fields: []},
    {kind: 'table', title: 'Lines', ref: 'r1.lines', fields: ['sku', 'qty']},
    {kind: 'record', title: 'Report', ref: 'r1', fields: ['total', 'rate', 'internal_id']},
    {kind: 'quote', title: 'Heads up', quote: QUOTE, tone: 'warning'},
  ],
};


function visibleText(node) {
  const parts = [];
  const walk = (n) => {
    if (n.nodeType === 3) parts.push(n.nodeValue);
    else if (n.nodeType === 1 && !n.hidden) Array.from(n.childNodes).forEach(walk);
  };
  walk(node);
  return parts.join(' ').replace(/\s+/g, ' ').trim();
}
const LATER = 'Thanks for waiting.';
const GOOD_BLOCKS = [GOOD_CARD.blocks[0]];

async function main() {
  const box = d.getElementById('richAnswers');
  tm.commandSchemas = SCHEMAS;
  tm.baseUrl = 'http://127.0.0.1:39131';
  tm.connected = true;

  /* ---- 1 + 2: the text stays while composing; the card replaces it when drawn -------------- */
  const gate = deferred();
  composeHandler = () => gate.promise;
  const out = turn(ANSWER, multi());
  const msg = paint(out);
  await until(() => calls.length === 1, 'request');
  assert.equal(norm(visibleText(msg.querySelector('.bubble'))), norm(ANSWER), 'composing: the text answer shows');
  assert.ok(!msg.querySelector('.answerCard') && !msg.querySelector('.answerToggle'), 'composing: no card, no toggle');
  assert.ok(msg.querySelector('.cardStatus'), 'composing: the status line is there');
  gate.resolve(GOOD_CARD);
  await until(() => msg.querySelector('.answerCard'), 'card');
  const bubble = msg.querySelector('.bubble');
  assert.equal(msg.querySelectorAll('.answerCard').length, 1, 'exactly one card');
  assert.equal(msg.querySelector('.answerCard').hidden, false);
  assert.ok(!visibleText(bubble).includes(LATER), 'text answer still visible next to the card');
  assert.ok(!visibleText(bubble).includes('Order summary is ready'), 'text answer still visible next to the card');
  const rest = msg.querySelector('.answerRest');
  assert.ok(rest && rest.hidden === true && bubble.contains(rest), 'the text node stays in the DOM, hidden');
  assert.equal(norm(rest.textContent), norm(ANSWER), 'with the full text');
  /* the payload is what the server always got: the text of the turn, not the DOM */
  const wire = JSON.stringify(calls[0].body);
  const schemas = tm.commandSchemas;
  assert.equal(wire, JSON.stringify(w.tmCardPayload(msg, out, w.tmResultsOf(out.command_outputs, schemas), schemas)));
  assert.equal(calls[0].body.answer, ANSWER);
  assert.deepEqual(Object.keys(calls[0].body),
    ['user_message', 'answer', 'status', 'use_dspy_cache', 'turn_key', 'results']);

  /* ---- 3: the toggle ----------------------------------------------------------------------- */
  const toggle = msg.querySelector('button.answerToggle');
  assert.equal(msg.querySelectorAll('button.answerToggle').length, 1);
  assert.equal(toggle.tagName, 'BUTTON');
  assert.equal(toggle.type, 'button');
  assert.equal(toggle.textContent, 'Show full text');
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  const card = msg.querySelector('.answerCard');
  assert.equal(toggle.getAttribute('aria-controls'), card.id + ' ' + rest.id);
  assert.ok(toggle.parentNode === bubble && !toggle.closest('.turnFooter'), 'under the card, outside the footer');
  assert.equal(bubble.lastChild, toggle, 'the toggle is the last thing in the bubble');
  assert.equal(toggle.tabIndex, 0, 'keyboard reachable');
  const footer = msg.querySelector('.turnFooter');
  assert.deepEqual(Array.from(footer.children).map((c) => c.className),
    ['turnFooterSlot turnFooterLeft', 'turnFooterSlot turnFooterCentre', 'turnFooterSlot turnFooterRight'],
    'the footer keeps its three slots');
  assert.ok(!footer.querySelector('.answerToggle'));
  toggle.click();
  assert.equal(norm(visibleText(bubble).replace('Show summary', '')), norm(ANSWER), 'full text replaces the card');
  assert.equal(card.hidden, true);
  assert.equal(toggle.textContent, 'Show summary');
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  /* state survives a repaint (cached card) and a results panel open / close */
  w.tmRenderTurn(msg, out);
  await flush();
  assert.equal(msg.querySelectorAll('.answerCard').length, 1);
  assert.equal(msg.querySelector('.answerCard').hidden, true, 'the repaint keeps the full text');
  assert.equal(msg.querySelector('button.answerToggle').getAttribute('aria-expanded'), 'true');
  w.tmSetResultsOpen(msg, true, 'panel');
  assert.equal(msg.querySelector('.answerCard').hidden, true);
  w.tmSetResultsOpen(msg, false, '');
  assert.equal(msg.querySelector('.answerCard').hidden, true);
  msg.querySelector('button.answerToggle').click();
  assert.equal(msg.querySelector('.answerCard').hidden, false);
  assert.equal(msg.querySelector('.answerRest').hidden, true);
  assert.equal(msg.querySelector('button.answerToggle').textContent, 'Show full text');
  w.tmSetResultsOpen(msg, true, 'panel');
  assert.equal(msg.querySelector('.answerCard').hidden, false, 'the panel opening keeps the summary');
  w.tmSetResultsOpen(msg, false, '');
  assert.equal(calls.length, 1, 'no further request');

  /* ---- 4: a cached card re-renders as the card, never the text first ----------------------- */
  const cachedOut = turn(ANSWER, multi());
  tm.cardCache[cachedOut.turn_key] = GOOD_CARD;
  const before = calls.length;
  const cached = paint(cachedOut);
  await until(() => cached.querySelector('.answerCard'), 'cached card');
  assert.ok(!visibleText(cached.querySelector('.bubble')).includes(LATER));
  assert.equal(calls.length, before, 'no request for a cached card');
  assert.equal(cached.querySelectorAll('.answerCard').length, 1);

  /* ---- 5: each turn keeps its own state ------------------------------------------------------ */
  const other = paint(turn(ANSWER, multi()));
  await until(() => other.querySelector('.answerCard'), 'second card');
  other.querySelector('button.answerToggle').click();
  assert.equal(other.querySelector('.answerCard').hidden, true);
  assert.equal(msg.querySelector('.answerCard').hidden, false, 'the first turn did not follow');
  assert.equal(cached.querySelector('.answerCard').hidden, false);

  /* ---- 6: no card, the text stays and the notice shows ------------------------------------- */
  calls.length = 0;
  async function textStays(handler, label, noticeReason) {
    composeHandler = handler;
    const n = calls.length;
    const m = paint(turn(ANSWER, multi()));
    await until(() => calls.length === n + 1 && !m.querySelector('.cardStatus'), label + ' settles');
    await flush();
    assert.ok(!m.querySelector('.answerCard') && !m.querySelector('.answerToggle'), label + ': no card, no toggle');
    assert.equal(norm(visibleText(m.querySelector('.bubble'))), norm(ANSWER), label + ': the text stays');
    assert.ok(!m.querySelector('.answerRest'), label + ': nothing hidden');
    if (noticeReason) assert.equal(m.querySelector('.cardNotice').getAttribute('data-reason'), noticeReason, label);
    return m;
  }
  await textStays(replyReason('failed'), 'failed', 'failed');
  await textStays(replyReason('not_needed'), 'not needed', null);
  await textStays(reply({lead: 'Only a lead', blocks: []}), 'a lead with no block over a results panel', 'card_refused');
  await textStays(reply({lead: '', blocks: [{kind: 'stats', stats: [{label: 'x', ref: 'r9.total'}]}]}), 'a card with no nodes', 'card_refused');
  await textStays(() => Promise.reject(new Error('down')), 'network error', 'failed');

  /* a lead alone with no results panel: the page draws it and it replaces the text */
  composeHandler = reply({lead: 'Gist of it', blocks: []});
  const prose = 'A long paragraph of prose. '.repeat(21).slice(0, 550);
  const lone = paint(turn(prose, []));
  await until(() => lone.querySelector('.answerCard'), 'lead-only card, no panel');
  assert.equal(norm(visibleText(lone.querySelector('.bubble')).replace('Show full text', '')), 'Gist of it');
  assert.equal(norm(lone.querySelector('.answerRest').textContent), norm(prose));

  /* the empty-lead card (blocks only) replaces the text too */
  composeHandler = reply({lead: '', blocks: GOOD_BLOCKS});
  const noLead = paint(turn(ANSWER, multi()));
  await until(() => noLead.querySelector('.answerCard'), 'blocks-only card');
  assert.ok(!visibleText(noLead.querySelector('.bubble')).includes(LATER), 'text answer still visible next to the card');

  /* ---- 7: Rich answers switched off after a card: the shown card stays, a repaint shows the text -- */
  box.checked = false;
  assert.ok(msg.querySelector('.answerCard') && !msg.querySelector('.answerCard').hidden, 'switching off changes nothing on screen');
  w.tmRenderTurn(msg, out);
  await flush();
  assert.ok(!msg.querySelector('.answerCard') && !msg.querySelector('.answerToggle'));
  assert.equal(norm(visibleText(msg.querySelector('.bubble'))), norm(ANSWER), 'the repaint shows the text');
  box.checked = true;

  /* ---- 8: forms, "/" turns and a waiting turn are untouched ---------------------------------- */
  composeHandler = reply(GOOD_CARD);
  const n8 = calls.length;
  const asking = paint(turn(ANSWER, multi(), 'awaiting_user'));
  const slash = paint(turn(ANSWER, multi()), {deterministic: true, message: '/report'});
  await flush();
  assert.equal(calls.length, n8);
  [asking, slash].forEach((m) => {
    assert.ok(!m.querySelector('.answerCard') && !m.querySelector('.answerToggle') && !m.querySelector('.answerRest'));
    assert.equal(norm(visibleText(m.querySelector('.bubble'))), norm(ANSWER));
  });

  /* ---- 9: focus that sat in the replaced text goes to the bubble, never to the page ---------- */
  const gateF = deferred();
  composeHandler = () => gateF.promise;
  const linked = '**Note** See [the docs](https://example.com/docs) for the rest. ' + 'More words in the answer. '.repeat(20);
  const fm = paint(turn(linked, multi()));
  await until(() => fm.querySelector('.cardStatus'), 'pending');
  const anchor = fm.querySelector('.bubble a');
  assert.ok(anchor);
  anchor.focus();
  assert.equal(d.activeElement, anchor);
  gateF.resolve(GOOD_CARD);
  await until(() => fm.querySelector('.answerCard'), 'card for focus case');
  assert.equal(d.activeElement, fm.querySelector('.bubble'), 'focus moved to the bubble container');
  assert.ok(!fm.querySelector('[aria-live]'), 'nothing new is announced');

  if (alerted) throw new Error('script ran');
  if (errors.length) throw new Error(errors.join('\n'));
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
