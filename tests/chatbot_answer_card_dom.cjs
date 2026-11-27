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

async function main() {
  for (const fn of ['tmComposeCard', 'tmResolveCardRef', 'tmCardApply', 'tmCardQuoteGrounded']) {
    assert.equal(typeof w[fn], 'function', fn);
  }
  assert.equal(w.TM_CARD_TIMEOUT_MS, 35000);
  assert.equal(w.TM_CARD_MAX_RESULTS, 60);
  assert.equal(w.TM_CARD_MAX_ANSWER, 150000);       /* composer.MAX_ANSWER_CHARS */
  assert.equal(w.TM_CARD_MAX_SECTIONS, 24);         /* composer.MAX_SECTIONS */
  assert.equal(w.TM_CARD_MAX_BLOCKS, 48);           /* composer.MAX_CARD_BLOCKS_CEILING */
  assert.equal(w.TM_CARD_ROWS_SHOWN, 8);           /* composer.TABLE_ROWS_SHOWN: the row cap the completeness check reads */
  const box = d.getElementById('richAnswers');
  assert.ok(box && box.checked, 'the Rich answers switch is on by default');
  assert.equal(d.getElementById('richAnswersToggle').tagName, 'LABEL');
  tm.commandSchemas = SCHEMAS;
  tm.baseUrl = 'http://127.0.0.1:39131';
  tm.connected = true;

  /* ---- one definition of a page, shared with the server (fix-weih.20) --------- */
  const shapes = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'page_shape_cases.json'), 'utf8'));
  shapes.forEach((c) => {
    assert.equal(w.tmIsPageOutput(c.output), c.is_page, 'tmIsPageOutput: ' + c.name);
    assert.equal(w.tmCardIsPage(c.output), c.table_worthy, 'tmCardIsPage: ' + c.name);
  });
  assert.equal(w.tmCardIsPage({owners: ['Ann', 'Bo'], managers: ['Cy', 'Di']}), false, 'unrelated lists, no total');

  /* ---- pure ref resolution ---------------------------------------------- */
  const results = w.tmResultsOf(multi(), SCHEMAS);
  const NF = w.TM_CARD_NOT_FOUND;
  assert.equal(w.tmResolveCardRef('r1.total', results), 186.32);
  assert.equal(w.tmResolveCardRef('r1', results), results[0].output);
  assert.deepEqual(plain(w.tmResolveCardRef('r2.titles', results)), ['Dune', 'Emma']);
  ['r9.total', 'r1.nope', 'r1.lines.0', 'r1.lines.length', 'r1.__proto__', 'r1.total.x', 'r1.', '', '.r1',
    'total', 'r1..total'].forEach((ref) =>
    assert.equal(w.tmResolveCardRef(ref, results), NF, JSON.stringify(ref)));
  [null, undefined, 5, {}].forEach((ref) => assert.equal(w.tmResolveCardRef(ref, results), NF));
  assert.equal(w.tmCardQuoteGrounded(QUOTE, ANSWER), true);
  assert.equal(w.tmCardQuoteGrounded('**' + QUOTE.toUpperCase() + '**', ANSWER), true, 'emphasis and case ignored');
  assert.equal(w.tmCardQuoteGrounded('a sentence the answer never said at all', ANSWER), false);
  assert.equal(w.tmCardQuoteGrounded('too short', 'too short'), false);
  assert.equal(w.tmCardQuoteGrounded(ANSWER, ANSWER), false, 'a quote of the whole answer is no callout');

  /* ---- request parts: the recorded sub_task reaches the payload (fix-2nsr piece 2) ---------------- */
  const partMsg = w.tmBubble('agent', '...');
  partMsg.dataset.userMessage = 'two things';
  const payloadOf = (outputs) => w.tmCardPayload(partMsg, turn(ANSWER, outputs), w.tmResultsOf(outputs, SCHEMAS), SCHEMAS);
  const plainPayload = payloadOf(multi());
  plainPayload.results.forEach((r) => assert.ok(!('sub_task' in r), 'no part number: the key is absent, as before'));
  assert.equal(JSON.stringify(payloadOf(multi().map((o) => Object.assign({}, o, {sub_task: null}))).results),
    JSON.stringify(plainPayload.results), 'an output with a null part sends the very payload of one without the key');
  const partOut = [co('report', REPORT, null, {sub_task: 2}), co('find_book', BOOKS, {start: 0}, {sub_task: 1})];
  const partPayload = payloadOf(partOut);
  assert.deepEqual(plain(partPayload.results.map((r) => [r.id, r.sub_task])), [['r1', 2], ['r2', 1]]);
  ['x', 0, -1, 1.5, true, null].forEach((bad) =>
    assert.ok(!('sub_task' in payloadOf([co('report', REPORT, null, {sub_task: bad})]).results[0]), 'rejected: ' + String(bad)));
  const pagedParts = [co('find_book', BOOKS, {start: 0}, {sub_task: 3}), co('find_book', BOOKS, {start: 2}, {sub_task: 3})];
  assert.equal(payloadOf(pagedParts).results[0].sub_task, 3, 'a result read in pages keeps its part');

  /* over the limit the selection is fair across parts: a part with primary results never loses all of them */
  const fake = (id, subTask, supporting) => ({id, subTask, supporting: supporting === true, command: 'c' + id});
  const crowd = [];
  for (let i = 1; i <= 64; i++) crowd.push(fake('r' + i, 1, false));
  crowd.push(fake('r65', 2, false), fake('r66', 3, false), fake('r67', 3, true));
  const fairIds = w.tmCardSelect(crowd).map((r) => r.id);
  assert.equal(fairIds.length, 60);
  assert.ok(fairIds.includes('r65') && fairIds.includes('r66'), 'the first primary result of every part is kept');
  assert.ok(!fairIds.includes('r67'), 'a supporting result of a part is cut before a primary one');
  assert.deepEqual(fairIds.slice(0, 3), ['r1', 'r2', 'r3'], 'send order is kept');
  const unnumbered = crowd.map((r) => fake(r.id, null, r.supporting));
  assert.deepEqual(plain(w.tmCardSelect(unnumbered).map((r) => r.id)),
    plain(w.tmCardSendOrder(unnumbered).slice(0, 60).map((r) => r.id)), 'without part numbers: the first 60, as before');

  /* ---- case 1: multi-result turn ------------------------------------------ */
  const gate1 = deferred();
  composeHandler = () => gate1.promise;
  const out1 = turn(ANSWER, multi());
  const msg1 = w.tmBubble('agent', '...');
  msg1.dataset.userMessage = 'status of my order';
  w.tmRenderTurn(msg1, out1);
  const resultsNode = msg1.querySelector('.results');
  assert.ok(resultsNode && !msg1.querySelector('.bubble .results'), 'the results are not in the bubble');
  assert.ok(resultsNode.closest('.resultsPanel') && resultsNode.closest('.resultsPanel').hidden, 'they wait in the closed panel');
  const jsonBtn = resultsNode.querySelector('button.rJson');
  jsonBtn.click();                        /* user state in the deterministic view */
  noButton(msg1);
  assert.equal(norm(text(msg1.querySelector('.bubble'))), norm(ANSWER), 'the whole plain answer shows at once');
  assert.ok(!msg1.querySelector('.answerRest') && !msg1.querySelector('.answerToggle'));
  const link1 = msg1.querySelector('.resultsLinkRow button.resultsLink');
  assert.ok(link1);
  link1.focus();
  await until(() => calls.length === 1, 'the automatic request');
  const status1 = msg1.querySelector('.resultsLinkRow .cardStatus[role="status"][aria-live="polite"]');
  assert.ok(status1, 'a status line sits next to the results link');
  assert.equal(text(status1), 'Composing answer…');
  assert.equal(norm(text(msg1.querySelector('.bubble'))), norm(ANSWER), 'still the whole plain answer while pending');
  gate1.resolve(GOOD_CARD);
  await until(() => msg1.querySelector('.answerCard'), 'the card never rendered');
  assert.equal(calls.length, 1);
  assert.equal(d.activeElement, link1, 'focus stayed on the link');
  const body = calls[0].body;
  assert.equal(calls[0].url, 'http://127.0.0.1:39131/compose_answer_card');
  assert.equal(body.user_message, 'status of my order');
  assert.equal(body.answer, ANSWER);
  assert.equal(body.status, 'completed');
  assert.equal(body.use_dspy_cache, true);
  assert.equal(body.turn_key, out1.turn_key, 'the turn key is sent so the server files its compose span under the turn');
  assert.deepEqual(body.results.map((r) => r.id), ['r1', 'r2']);
  assert.deepEqual(body.results[0].titles,
    {internal_id: 'Internal ID', total: 'Total', rate: 'Rate', lines: 'Lines'});
  assert.deepEqual(body.results[1].titles, {book_uids: 'Book UIDs', titles: 'Titles', total: 'Total'});   /* total added with the page's total */
  assert.deepEqual(body.results[0].output, REPORT, 'the normalized output is what is sent');
  assert.deepEqual(body.results[0].descriptions, {total: 'Order total including tax', rate: 'x'.repeat(100)},
    'schema descriptions are sent, cut to 100 characters');
  assert.deepEqual(body.results[0].hints, {internal_id: 'hidden', total: 'hero,currency', rate: 'percent'});
  assert.deepEqual(body.results.map((r) => r.supporting), [false, false]);
  assert.deepEqual(body.results[1].descriptions, {}); assert.deepEqual(body.results[1].hints, {});
  assert.equal(body.results[1].command, 'find_book');
  assert.deepEqual(body.results[1].parameters, {start: 0});

  const card = msg1.querySelector('.bubble .answerCard');
  assert.equal(text(card.querySelector('p.cardLead')), 'Two results are ready');
  const stats = card.querySelector('dl.cardStats');
  const statText = Array.from(stats.querySelectorAll('.cardStat')).map((s) => text(s));
  assert.deepEqual(statText, ['Order total 186.32', 'Rate 12.5%'], 'formatted with ui.format; hidden and ghost skipped');
  assert.ok(stats.querySelector('.cardStat.tone-good'));
  assert.ok(!/x1/.test(text(card)), 'a hidden field never appears');
  const tables = card.querySelectorAll('table.rTable');
  assert.equal(tables.length, 2);
  assert.deepEqual(Array.from(tables[0].querySelectorAll('th')).map((t) => t.textContent), ['Book UIDs', 'Titles']);
  assert.deepEqual(Array.from(tables[0].querySelectorAll('tbody tr')).map((r) => text(r)), ['u1 Dune', 'u2 Emma']);
  assert.deepEqual(Array.from(tables[1].querySelectorAll('th')).map((t) => t.textContent), ['SKU', 'Qty']);
  assert.ok(!/s1/.test(text(tables[1])), 'ui.hidden column dropped');
  const record = card.querySelector('dl.rKv');
  assert.equal(text(record), 'Total 186.32 Rate 12.5%');
  assert.equal(text(card.querySelector('blockquote.cardQuote')), QUOTE);
  assert.ok(card.querySelector('blockquote.cardQuote.tone-warning'));
  assert.equal(text(card.querySelector('.cardBlockTitle')), 'Headline');
  assert.ok(!msg1.querySelector('.cardStatus'), 'the status is gone');

  /* the card REPLACES the agent's own text [fix-weih.36]: one toggle, one answerRest (hidden, the WHOLE answer inside it) */
  assert.equal(msg1.querySelectorAll('button.answerToggle').length, 1);
  assert.equal(msg1.querySelectorAll('.answerRest').length, 1);
  const rest = msg1.querySelector('.answerRest');
  assert.equal(rest.hidden, true);
  assert.equal(card.hidden, false, 'the card is what the bubble shows');
  assert.equal(norm(rest.textContent), norm(ANSWER));
  const toggle = msg1.querySelector('button.answerToggle');
  assert.equal(toggle.textContent, 'Show full text');
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  assert.equal(toggle.getAttribute('aria-controls'), card.id + ' ' + rest.id);
  toggle.click();
  assert.equal(rest.hidden, false);
  assert.equal(card.hidden, true, 'the full text replaces the card');
  assert.equal(toggle.textContent, 'Show summary');
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  toggle.click();
  assert.equal(rest.hidden, true);
  assert.equal(card.hidden, false);
  assert.ok(!msg1.querySelector('.answerLead'));

  /* the deterministic results are never pulled into the bubble: they stay in the panel */
  assert.ok(!msg1.querySelector('details.resultsAll'));
  assert.equal(msg1.querySelector('.results'), resultsNode, 'the same node, never moved');
  assert.ok(resultsNode.closest('.resultsPanel') && !msg1.querySelector('.bubble .results'));
  assert.equal(jsonBtn.getAttribute('aria-pressed'), 'true', 'its JSON toggle kept its state');
  assert.ok(msg1.classList.contains('hasResults'));
  link1.click();
  assert.equal(resultsNode.closest('.resultsPanel').hidden, false, 'the link still opens the results');
  link1.click();

  /* ---- case 5: a repaint of the same turn reuses the cache ---------------- */
  assert.ok(tm.cardCache[out1.turn_key], 'the validated card is cached by turn_key');
  w.tmRenderTurn(msg1, out1);
  assert.ok(msg1.querySelector('.answerCard p.cardLead'), 'repaint shows the card at once');
  assert.equal(msg1.querySelectorAll('.answerCard').length, 1);
  assert.equal(msg1.querySelectorAll('button.answerToggle').length, 1);
  assert.equal(msg1.querySelectorAll('.answerRest').length, 1);
  assert.equal(msg1.querySelectorAll('.resultsLink').length, 1);
  await flush();
  assert.equal(calls.length, 1, 'no second request');
  noButton(msg1);

  /* ---- focus on the results link survives the card and a failure ----------- */
  async function linkFocus(handler, label) {
    composeHandler = handler;
    const before = calls.length;
    const msg = paint(turn(ANSWER, multi()));
    const link = msg.querySelector('button.resultsLink');
    link.focus();
    await flush();
    assert.equal(calls.length, before + 1, label + ': one automatic request');
    assert.equal(d.activeElement, link, label + ': focus stays on the link');
    return msg;
  }
  assert.ok((await linkFocus(reply(GOOD_CARD), 'card lands')).querySelector('.answerCard'));
  assert.ok(!(await linkFocus(replyReason('failed'), 'compose fails')).querySelector('.answerCard'));
  d.body.focus();
  calls.length = 0;

  /* ---- repaints while pending and after share the one request ---------------- */
  const outPre = turn(ANSWER, multi());
  const gatePre = deferred();
  composeHandler = () => gatePre.promise;
  const msgPre = paint(outPre);
  w.tmRenderTurn(msgPre, outPre);
  w.tmRenderTurn(msgPre, outPre);
  assert.equal(msgPre.querySelectorAll('.cardStatus').length, 1, 'repaints keep one status line');
  assert.equal(msgPre.querySelectorAll('.resultsLink').length, 1, 'and one results link');
  gatePre.resolve(GOOD_CARD);
  await until(() => msgPre.querySelector('.answerCard'), 'repaints while pending');
  assert.equal(calls.length, 1, 'exactly one request for the turn');
  w.tmRenderTurn(msgPre, outPre);
  await flush();
  assert.equal(calls.length, 1, 'a repaint after the card does not ask again');
  assert.equal(msgPre.querySelectorAll('.answerCard').length, 1);
  calls.length = 0;

  /* a failed turn is not asked again by a repaint, and keeps the whole plain answer */
  composeHandler = replyReason('failed');
  const outFail = turn(ANSWER, multi());
  const msgFail = paint(outFail);
  await flush();
  assert.equal(calls.length, 1);
  w.tmRenderTurn(msgFail, outFail);
  await flush();
  assert.equal(calls.length, 1, 'a failure is not retried by a repaint');
  assert.equal(norm(text(msgFail.querySelector('.bubble'))), norm(ANSWER), 'the plain answer stays');
  assert.ok(!msgFail.querySelector('.cardStatus') && !msgFail.querySelector('.answerCard'));
  calls.length = 0;

  /* an answer of one paragraph also ends with one toggle and one rest */
  const flat = 'Flat answer without any paragraph break. '.repeat(12).trim();
  composeHandler = reply({lead: 'Short version', blocks: [GOOD_CARD.blocks[1]]});
  const msgFlat = paintC(turn(flat, multi()));
  assert.ok(!msgFlat.querySelector('.answerToggle'), 'nothing folds the answer before the card');
  await until(() => msgFlat.querySelector('.answerCard'), 'flat case');
  assert.equal(msgFlat.querySelectorAll('button.answerToggle').length, 1);
  assert.equal(norm(msgFlat.querySelector('.answerRest').textContent), norm(flat));

  /* ---- case 2: the selective trigger --------------------------------------- */
  calls.length = 0;
  composeHandler = reply(GOOD_CARD);
  const longText = 'A long prose answer that goes on. '.repeat(20);
  assert.ok(longText.length > 400);
  const slashA = paint(turn(ANSWER, multi()), {deterministic: true, message: '/report'});
  const slashB = paint(turn(longText, []), {deterministic: true, message: '/report'});
  await flush();
  noButton(slashA); noButton(slashB);
  assert.equal(calls.length, 0, '"/" turns make no request');

  const one = [co('get_user_details', {name: 'Ann'}, {user_id: 'u1'}),
    co('get_order_details', {order_details: JSON.stringify({status: 'delivered'})}, {order_id: '#W1'})];
  const oneResults = w.tmResultsOf(one, SCHEMAS);
  assert.deepEqual(plain(oneResults.map((r) => r.supporting)), [true, false]);
  const short180 = 'x'.repeat(179) + '.';
  assert.equal(short180.length, 180);
  const shortMsg = paint(turn(short180, one));
  paint(turn('short', [co('report', REPORT, null)]));
  await flush();
  assert.equal(calls.length, 0, 'one primary result with a short answer keeps the deterministic card');
  assert.ok(shortMsg.querySelector('.results'));
  assert.ok(!shortMsg.querySelector('.answerCard') && !shortMsg.querySelector('.cardStatus'));

  const prose550 = 'A long paragraph of prose. '.repeat(21).slice(0, 550);
  composeHandler = reply({lead: 'Gist of the long answer', blocks: []});
  const proseOne = paint(turn(prose550, [co('report', REPORT, null)]));
  await until(() => calls.length === 1, 'a long answer asks by itself');
  await until(() => !proseOne.querySelector('.cardStatus'), 'long answer with one result settles');
  assert.equal(calls.length, 1, 'a long answer asks, even with one result');
  /* a lead alone would hide the answer and show no data, and this turn has a results panel to open:
     the plain answer stays as it is [fix-weih.25] */
  assert.ok(!proseOne.querySelector('.answerCard') && !proseOne.querySelector('p.cardLead'), 'a lead-only card is not drawn');
  assert.equal(norm(text(proseOne.querySelector('.bubble'))), norm(prose550), 'the whole plain answer stays');
  assert.ok(proseOne.querySelector('.resultsPanel .results'), 'the results stay in the panel');
  const proseNone = paint(turn(prose550, []));
  assert.ok(proseNone.querySelector('.cardStatusRow .cardStatus'), 'no results link: the status sits in its own row');
  await until(() => proseNone.querySelector('.answerCard'), 'long answer with no result');
  assert.equal(calls.length, 2);
  assert.equal(proseNone.querySelectorAll('button.answerToggle').length, 1);
  assert.equal(norm(proseNone.querySelector('.answerRest').textContent), norm(prose550));
  assert.ok(proseNone.querySelector('.cardStatusRow') === null, 'its status row was cleaned up');

  /* ---- case 3: empty lead keeps the deterministic lead --------------------- */
  calls.length = 0;
  composeHandler = reply({lead: '', blocks: [GOOD_CARD.blocks[0]]});
  const msgEmpty = paintC(turn(ANSWER, multi()));
  await until(() => msgEmpty.querySelector('.answerCard'), 'empty lead case');
  assert.ok(!msgEmpty.querySelector('p.cardLead'));
  assert.equal(msgEmpty.querySelector('.bubble').firstChild, msgEmpty.querySelector('.answerCard'), 'the card replaces the answer: it comes first');
  assert.equal(msgEmpty.querySelectorAll('.answerRest').length, 1, 'the whole answer is kept, hidden, behind the toggle');
  assert.equal(msgEmpty.querySelector('.answerRest').hidden, true);
  assert.ok(norm(msgEmpty.querySelector('.answerRest').textContent).startsWith('Order summary is ready.'), 'the whole answer');
  assert.ok(msgEmpty.querySelector('.answerCard dl.cardStats'));
  assert.ok(msgEmpty.querySelector('.resultsPanel .results') && !msgEmpty.querySelector('details.resultsAll'));
  /* an empty lead and no block that resolves: nothing changes */
  composeHandler = reply({lead: '', blocks: [{kind: 'stats', stats: [{label: 'x', ref: 'r9.total'}]}]});
  const msgNothing = paintC(turn(ANSWER, multi()));
  await flush();
  assert.ok(!msgNothing.querySelector('.answerCard') && !msgNothing.querySelector('details.resultsAll'));
  assert.ok(!msgNothing.querySelector('.cardStatus'));

  /* ---- case 4: unresolvable and wrongly shaped refs are skipped alone ------ */
  composeHandler = reply({lead: 'Mixed bag', blocks: [
    {kind: 'stats', title: 'dead', stats: [{label: 'Gone', ref: 'r9.total'}]},
    {kind: 'table', title: 'dead', ref: 'r9.total'},
    {kind: 'table', title: 'dead scalar', ref: 'r1.total'},
    {kind: 'table', title: 'dead index', ref: 'r1.lines.0'},
    {kind: 'record', title: 'dead list', ref: 'r1.lines'},
    {kind: 'stats', title: 'dead list stat', stats: [{label: 'L', ref: 'r1.lines'}, {label: 'R', ref: 'r1'}]},
    {kind: 'quote', quote: 'a sentence the answer never said at all'},
    {kind: 'bogus', ref: 'r1'},
    null,
    {kind: 'stats', title: 'Alive', stats: [{label: 'Total', ref: 'r1.total', value: '999999'},
      {label: 'Bad', ref: 'r9.total'}]},
    {kind: 'record', title: 'Alive record', ref: 'r2'},
  ]});
  const msgRefs = paintC(turn(ANSWER, multi()));
  await until(() => msgRefs.querySelector('.answerCard'), 'refs case');
  const titles = Array.from(msgRefs.querySelectorAll('.cardBlockTitle')).map((t) => t.textContent);
  assert.deepEqual(titles, ['Alive', 'Alive record']);
  assert.ok(!/999999/.test(text(msgRefs.querySelector('.answerCard'))), 'a model-supplied value is never rendered');
  assert.equal(msgRefs.querySelectorAll('.answerCard .cardStat').length, 1);

  /* ---- case 5: failures leave the deterministic view untouched ------------ */
  async function stays(handler, label) {
    composeHandler = handler;
    const before = calls.length;
    const msg = paintC(turn(ANSWER, multi()));
    const node = msg.querySelector('.results');
    await until(() => calls.length > before, label + ': no request');
    await flush();
    assert.ok(!msg.querySelector('.answerCard'), label + ': no card');
    assert.ok(!msg.querySelector('details.resultsAll'), label);
    assert.ok(!msg.querySelector('.cardStatus[role="status"]'), label + ': status removed');
    assert.equal(msg.querySelector('.results'), node, label + ': results untouched');
    assert.equal(norm(text(msg.querySelector('.bubble'))), norm(ANSWER), label + ': the whole plain answer stays');
    return msg;
  }
  await stays(reply(null), 'null card');
  await stays(replyReason('failed'), 'failed');
  await stays(replyReason('timeout'), 'server timeout');
  await stays(() => Promise.resolve({ok: false, status: 500, json: () => Promise.resolve({detail: 'boom'})}), 'HTTP 500');
  await stays(() => Promise.reject(new Error('network down')), 'network error');
  await stays(() => Promise.resolve({ok: true, status: 200, json: () => Promise.reject(new Error('bad json'))}), 'bad body');
  await stays(reply({lead: '', blocks: []}), 'a card with nothing');
  /* "not_needed" (the composer chose to show nothing) is a null like any other: no notice, no breaker */
  const msgNotNeeded = await stays(replyReason('not_needed'), 'not_needed');
  assert.equal(tm.composeUnavailable, false, 'not_needed never trips the no_model breaker');
  assert.ok(!msgNotNeeded.querySelector('.cardNotice'), 'and shows no notice');   /* fix-weih.34: scoped to this bubble; the other failures above now leave a notice of their own */
  const before = calls.length;
  await stays(replyReason('not_needed'), 'not_needed again');
  assert.equal(calls.length, before + 1, 'the next turn still asks');

  /* a table that selects fewer than two columns shows every visible column (the server does the same) */
  composeHandler = reply({lead: '', blocks: [{kind: 'table', title: 'One', ref: 'r1.lines', fields: ['sku']}]});
  const msgOne = paintC(turn(ANSWER, multi()));
  await until(() => msgOne.querySelector('.answerCard table.rTable'), 'one-column table case');
  assert.deepEqual(Array.from(msgOne.querySelectorAll('.answerCard table.rTable th')).map((t) => t.textContent),
    ['SKU', 'Qty']);
  composeHandler = reply({lead: '', blocks: [{kind: 'table', title: 'Page', ref: 'r2', fields: ['titles']}]});
  const msgPageOne = paintC(turn(ANSWER, multi()));
  await until(() => msgPageOne.querySelector('.answerCard table.rTable'), 'one-column page case');
  assert.deepEqual(Array.from(msgPageOne.querySelectorAll('.answerCard table.rTable th')).map((t) => t.textContent),
    ['Book UIDs', 'Titles']);

  /* timeout: aborted after TM_CARD_TIMEOUT_MS */
  let aborted = false;
  w.TM_CARD_TIMEOUT_MS = 40;
  composeHandler = (call) => new Promise((_, reject) => {
    call.signal.addEventListener('abort', () => {
      aborted = true;
      reject(Object.assign(new Error('aborted'), {name: 'AbortError'}));
    });
  });
  const msgSlow = paintC(turn(ANSWER, multi()));
  assert.ok(msgSlow.querySelector('.cardStatus'));
  await until(() => aborted, 'the request was never aborted');
  await until(() => !msgSlow.querySelector('.cardStatus'), 'status never cleared after the abort');
  assert.ok(!msgSlow.querySelector('.answerCard'));
  w.TM_CARD_TIMEOUT_MS = 35000;

  /* stale: the source changed while the request was out */
  const gateEpoch = deferred();
  composeHandler = () => gateEpoch.promise;
  const outEpoch = turn(ANSWER, multi());
  const msgEpoch = paintC(outEpoch);
  await until(() => calls.length > 0 && msgEpoch.querySelector('.cardStatus'), 'epoch request');
  tm.epoch += 1;
  gateEpoch.resolve(GOOD_CARD);
  await flush();
  assert.ok(!msgEpoch.querySelector('.answerCard'), 'a response from an older epoch is ignored');
  assert.ok(!tm.cardCache[outEpoch.turn_key], 'and not cached');

  /* stale: the same bubble was repainted with another turn */
  const gateA = deferred();
  const gateB = deferred();
  const handlers = [gateA, gateB];
  let n = 0;
  composeHandler = () => handlers[n++].promise;
  const msgRe = w.tmBubble('agent', '...');
  msgRe.dataset.userMessage = 'q';
  w.tmRenderTurn(msgRe, turn(ANSWER, multi()));
  w.tmRenderTurn(msgRe, turn(ANSWER, multi()));
  await until(() => n === 2, 'two requests for two different turns');
  gateA.resolve({lead: 'FIRST', blocks: [GOOD_CARD.blocks[0]]});
  await flush();
  assert.ok(!msgRe.querySelector('.answerCard'), 'the first turn\'s late answer is ignored');
  gateB.resolve({lead: 'SECOND', blocks: [GOOD_CARD.blocks[0]]});
  await until(() => msgRe.querySelector('.answerCard'), 'second answer');
  assert.equal(text(msgRe.querySelector('p.cardLead')), 'SECOND');

  /* the same turn repainted while its request is out shares that one request */
  const gateShared = deferred();
  composeHandler = () => gateShared.promise;
  calls.length = 0;
  const outShared = turn(ANSWER, multi());
  const msgShared = paintC(outShared);
  w.tmRenderTurn(msgShared, outShared);
  assert.equal(msgShared.querySelectorAll('.cardStatus').length, 1, 'a repaint while the request is out joins it');
  gateShared.resolve(GOOD_CARD);
  await until(() => msgShared.querySelector('.answerCard'), 'shared flight');
  assert.equal(calls.length, 1, 'one request for one turn, however often it is painted');
  assert.equal(msgShared.querySelectorAll('.answerCard').length, 1);
  assert.equal(msgShared.querySelectorAll('button.answerToggle').length, 1);

  /* use_dspy_cache follows the "LLM cache" control */
  d.getElementById('useDspyCache').checked = false;
  calls.length = 0;
  composeHandler = reply(GOOD_CARD);
  paintC(turn(ANSWER, multi()));
  await until(() => calls.length === 1, 'cache request');
  assert.equal(calls[0].body.use_dspy_cache, false);
  d.getElementById('useDspyCache').checked = true;

  /* Rich answers unchecked: no new requests, and the plain card stays */
  calls.length = 0;
  box.checked = false;
  const msgOff = paint(turn(ANSWER, multi()));
  await flush();
  assert.equal(calls.length, 0);
  assert.ok(msgOff.querySelector('.results') && !msgOff.querySelector('.answerCard'));
  assert.ok(!msgOff.querySelector('.cardStatus'));
  assert.equal(norm(text(msgOff.querySelector('.bubble'))), norm(ANSWER), 'Rich answers off: the plain answer');
  box.checked = true;

  /* awaiting_user: none, by status and by artifact */
  const askA = paint(turn(ANSWER, multi(), 'awaiting_user'));
  const asking = multi();
  asking[1].command_response.artifacts = {awaiting_user: true};
  const askB = paint(turn(ANSWER, asking));
  await flush();
  assert.equal(calls.length, 0, 'an open question gets no card');

  /* not connected: none */
  tm.connected = false;
  paint(turn(ANSWER, multi()));
  await flush();
  assert.equal(calls.length, 0);
  tm.connected = true;

  /* ---- XSS: model text is text ---------------------------------------------- */
  const EVIL = '<img src=x onerror=alert(1)>';
  const evilAnswer = 'Report ready.\n\nWatch out: ' + EVIL + ' is how it was written, honestly.\n\nBye.';
  composeHandler = reply({lead: EVIL + ' lead', blocks: [
    {kind: 'stats', title: EVIL, stats: [{label: EVIL, ref: 'r1.total'}]},
    {kind: 'quote', title: '<b>t</b>', quote: EVIL + ' is how it was written, honestly.'},
  ]});
  const msgEvil = paintC(turn(evilAnswer, [
    co('report', Object.assign({}, REPORT, {lines: [{sku: '<script>x</script>', qty: 1}]}), null),
    co('find_book', BOOKS, null)]));
  await until(() => msgEvil.querySelector('.answerCard'), 'xss case');
  assert.equal(text(msgEvil.querySelector('p.cardLead')), EVIL + ' lead');
  assert.equal(text(msgEvil.querySelector('.cardStat dt')), EVIL);
  assert.equal(msgEvil.querySelector('.bubble img'), null);
  assert.equal(msgEvil.querySelectorAll('img, script').length, 0);
  assert.equal(alerted, false);


  /* ---- fix-weih.25 step 1: what is sent ----------------------------------------- */
  calls.length = 0;
  composeHandler = reply({lead: '', blocks: []});
  const longAnswer = 'A long prose answer that goes on. '.repeat(20);
  paintC(turn(longAnswer, one));
  await until(() => calls.length === 1, 'supporting order request');
  assert.deepEqual(calls[0].body.results.map((r) => [r.id, r.supporting]), [['r2', false], ['r1', true]],
    'primary results are sent first, each with its supporting flag');
  calls.length = 0;
  const many = [];
  for (let i = 0; i < 62; i++) many.push(co('report', REPORT, {n: i}));
  paintC(turn(ANSWER, many));
  await until(() => calls.length === 1, 'sixty results request');
  assert.equal(calls[0].body.results.length, 60, 'at most 60 results are sent');
  calls.length = 0;

  /* ---- fix-weih.25 step 1: the 8-row cap and the completeness footer ------------ */
  const BIG = Object.assign({}, REPORT, {lines: Array.from({length: 12}, (_, i) => ({sku: 'S' + i, qty: i}))});
  const bigTurn = () => turn(ANSWER, [co('report', BIG, null), co('find_book', BOOKS, {start: 0})]);
  composeHandler = reply({lead: 'Report lines', blocks: [{kind: 'table', title: 'Lines', ref: 'r1.lines'}]});
  const msgBig = paintC(bigTurn());
  await until(() => msgBig.querySelector('.answerCard'), 'row cap card');
  assert.equal(msgBig.querySelectorAll('.answerCard table.rTable tbody tr').length, 8, 'a card table shows 8 rows');
  const moreBtn = msgBig.querySelector('.answerCard button.cardMore');
  assert.ok(moreBtn, 'a control follows the table');
  assert.equal(text(moreBtn), '+4 more — open in results');
  const panelBig = msgBig.querySelector('.resultsPanel');
  assert.equal(panelBig.hidden, true);
  moreBtn.click();
  assert.equal(panelBig.hidden, false, '"+M more" opens the results panel');
  assert.equal(d.activeElement, panelBig.querySelector('.resultsClose'), 'focus goes to the panel');
  /* the second group (find_book, r2) is not on the card: one footer line, a button that opens the panel */
  const footerLines = Array.from(msgBig.querySelectorAll('.answerCard .cardFooter .cardFooterLine'));
  assert.deepEqual(footerLines.map((l) => text(l)), ['Find Book · Library (1 result) — open in results']);
  tm_close(msgBig);
  footerLines[0].querySelector('button').click();
  assert.equal(panelBig.hidden, false, 'a footer line opens the results panel');
  tm_close(msgBig);

  /* every group shown: no footer; a table of 8 or fewer rows has no "more" control */
  composeHandler = reply(GOOD_CARD);
  const msgAll = paintC(turn(ANSWER, multi()));
  await until(() => msgAll.querySelector('.answerCard'), 'all referenced');
  assert.ok(!msgAll.querySelector('.cardFooter') && !msgAll.querySelector('.cardMore'));

  /* a lead alone shows no block and the turn has a results panel: no card, the plain answer stays */
  calls.length = 0;
  composeHandler = reply({lead: 'Gist of the long answer', blocks: []});
  const msgLead = paintC(turn(longAnswer, one));
  await until(() => calls.length === 1 && !msgLead.querySelector('.cardStatus'), 'lead-only card settles');
  assert.ok(!msgLead.querySelector('.answerCard') && !msgLead.querySelector('.cardFooter'), 'a lead-only card is not drawn');
  assert.equal(norm(text(msgLead.querySelector('.bubble'))), norm(longAnswer), 'the whole plain answer stays');
  assert.ok(!msgLead.querySelector('button.answerToggle'), 'nothing is folded behind a toggle');
  composeHandler = reply({v: 2, lead: 'Gist', sections: [], blocks: []});
  const msgLeadV2 = paintC(turn(ANSWER, multi()));
  await until(() => !msgLeadV2.querySelector('.cardStatus'), 'lead-only v2 settles');
  assert.ok(!msgLeadV2.querySelector('.answerCard'), 'a v2 card with no section and no block is not drawn either');

  /* the footer: a group the card does not show is a line of its own; the supporting lookup never is */
  const manyOut = [co('report', REPORT, {n: 1}), co('report', REPORT, {n: 2}), co('find_book', BOOKS, null)];
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r3', title: 'Books'}]});
  const msgPlural = paintC(turn(ANSWER, manyOut));
  await until(() => msgPlural.querySelector('.answerCard'), 'plural footer');
  assert.deepEqual(Array.from(msgPlural.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['Report · Library (2 results) — open in results']);
  const supOut = [co('get_user_details', {name: 'Ann'}, {user_id: 'u1'}),
    co('get_order_details', {order_details: JSON.stringify({status: 'delivered'})}, {order_id: '#W1'}),
    co('find_book', BOOKS, null)];
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r3', title: 'Books'}]});
  const msgSup = paintC(turn(ANSWER, supOut));
  await until(() => msgSup.querySelector('.answerCard'), 'supporting footer');
  assert.deepEqual(Array.from(msgSup.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['Get Order Details · Library (1 result) — open in results'], 'the supporting lookup is not footered');

  /* the footer names a group by the record it ran for, as the results panel does, not by the command alone */
  const siblingOut = [co('report', REPORT, {n: 1}, {context: 'Identity: Ann'}), co('report', REPORT, {n: 2}, {context: 'Identity: Bob'}),
    co('find_book', BOOKS, null, {context: 'global'})];
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r3', title: 'Books'}]});
  const msgSibling = paintC(turn(ANSWER, siblingOut));
  await until(() => msgSibling.querySelector('.answerCard'), 'sibling footer');
  assert.deepEqual(Array.from(msgSibling.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['Report · Identity: Ann, Identity: Bob (2 results) — open in results']);
  const globalOut = [co('report', REPORT, {n: 1}, {context: 'global'}), co('find_book', BOOKS, null)];
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r2', title: 'Books'}]});
  const msgGlobal = paintC(turn(ANSWER, globalOut));
  await until(() => msgGlobal.querySelector('.answerCard'), 'global footer');
  assert.deepEqual(Array.from(msgGlobal.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['Report (1 result) — open in results'], 'a global context is no label');

  /* more than 3 groups unreferenced: ONE line, never a wall */
  const wide = ['c1', 'c2', 'c3', 'c4', 'c5'].map((name) => co(name, {value: name + ' value', count: 1}, null));
  const wideSchemas = Object.assign({}, SCHEMAS);
  wide.forEach((o) => { wideSchemas[o.command_name] = {kind: 'execution', input: null, output: {properties: {value: {type: 'string'}, count: {type: 'integer'}}}}; });
  const wideTurn = turn(ANSWER, wide.concat([co('find_book', BOOKS, null)]));
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r6', title: 'Books'}]});
  tm.commandSchemas = wideSchemas;
  const msgWide = paintC(wideTurn);
  await until(() => msgWide.querySelector('.answerCard'), 'wide footer');
  assert.deepEqual(Array.from(msgWide.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['5 more results — open in results'], 'five unreferenced groups are one line');
  const wideLine = msgWide.querySelector('.cardFooterLine button');
  const widePanel = msgWide.querySelector('.resultsPanel');
  wideLine.click();
  assert.equal(widePanel.hidden, false, 'the one line opens the results panel');
  tm_close(msgWide);
  tm.commandSchemas = SCHEMAS;
  /* exactly 3 groups unreferenced: three lines */
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'table', ref: 'r4', title: 'Books'}]});
  tm.commandSchemas = wideSchemas;
  const msgThree = paintC(turn(ANSWER, wide.slice(0, 3).concat([co('find_book', BOOKS, null)])));
  await until(() => msgThree.querySelector('.answerCard'), 'three footer');
  assert.equal(msgThree.querySelectorAll('.cardFooterLine').length, 3, 'three groups keep a line each');
  tm.commandSchemas = SCHEMAS;

  /* the framework's own review commands are no result: not sent, not footered, not numbered [fix-weih.25] */
  calls.length = 0;
  const reviewOut = [
    co('review_proposal', {round: 1, accepted: '{"items":[]}'}, {round: 1}, {context: 'global'}),
    co('review_subtasks', {items: ['a', 'b']}, null, {context: 'global'}),
  ].concat(multi());
  composeHandler = reply({lead: 'Gist', blocks: [{kind: 'stats', stats: [{label: 'Total', ref: 'r1.total'}]}]});
  const msgReview = paintC(turn(ANSWER, reviewOut));
  await until(() => msgReview.querySelector('.answerCard'), 'review plumbing card');
  assert.deepEqual(calls[0].body.results.map((r) => [r.id, r.command]), [['r1', 'report'], ['r2', 'find_book']],
    'review commands are not in the payload and do not shift the ids');
  assert.deepEqual(Array.from(msgReview.querySelectorAll('.cardFooterLine')).map((l) => text(l)),
    ['Find Book · Library (1 result) — open in results'], 'and are never a footer line');
  assert.ok(!/Review/.test(text(msgReview.querySelector('.resultsPanel'))), 'nor a chip of the results panel');
  /* with or without a schema listing */
  assert.deepEqual(plain(w.tmResultsOf(reviewOut, {}).map((r) => r.command)), ['report', 'find_book']);

  /* no results panel (no results link): no footer, and the "more" text is no button */
  const bare = w.tmBubble('agent', 'plain');
  const bareOut = bigTurn();
  const bareResults = w.tmResultsOf(bareOut.command_outputs, SCHEMAS);
  assert.equal(w.tmCardApply(bare, bareOut, {lead: 'L', blocks: [{kind: 'table', ref: 'r1.lines'}]}, bareResults, SCHEMAS), true);
  assert.ok(!bare.querySelector('.cardFooter'), 'no panel: no footer');
  assert.ok(!bare.querySelector('button.cardMore') && text(bare.querySelector('.cardMore')) === '+4 more');

  /* ---- fix-weih.25 step 2: sections ---------------------------------------------- */
  const V2 = {v: 2, lead: 'Two results are ready', sections: [
    {heading: 'Order status', verdict: 'The order total is 186.32.', blocks: [GOOD_CARD.blocks[0]]},
    {heading: 'dropped', verdict: '', blocks: []},
    {heading: 'Books', verdict: '', blocks: [GOOD_CARD.blocks[1]]},
    {heading: '', verdict: 'Only a verdict.', blocks: []},
  ], blocks: [GOOD_CARD.blocks[0], GOOD_CARD.blocks[1]]};
  composeHandler = reply(V2);
  const msgV2 = paintC(turn(ANSWER, multi()));
  await until(() => msgV2.querySelector('.answerCard'), 'sections card');
  const sections = Array.from(msgV2.querySelectorAll('.answerCard div.cardSection'));
  assert.equal(sections.length, 3, 'an empty section is not drawn');
  assert.deepEqual(sections.map((s) => s.querySelector('h3.cardSectionHeading') ? text(s.querySelector('h3')) : null),
    ['Order status', 'Books', null]);
  assert.equal(text(sections[0].querySelector('p.cardVerdict')), 'The order total is 186.32.');
  assert.ok(sections[0].querySelector('dl.cardStats') && sections[1].querySelector('table.rTable'));
  assert.ok(!sections[1].querySelector('p.cardVerdict') && text(sections[2].querySelector('p.cardVerdict')) === 'Only a verdict.');
  assert.equal(msgV2.querySelectorAll('.answerCard table.rTable').length, 1, 'the flat blocks are not drawn again');
  assert.equal(text(msgV2.querySelector('p.cardLead')), 'Two results are ready');
  assert.ok(!msgV2.querySelector('.cardFooter'), 'both groups are shown by the sections');

  /* a v2 response whose sections all fail draws its flat blocks; one with no `sections` is the v1 card */
  composeHandler = reply({v: 2, lead: 'Flat', sections: [{heading: 'x', verdict: '', blocks: [{kind: 'stats', stats: [{label: 'g', ref: 'r9.total'}]}]}],
    blocks: [GOOD_CARD.blocks[1]]});
  const msgFallback = paintC(turn(ANSWER, multi()));
  await until(() => msgFallback.querySelector('.answerCard'), 'sections fall back');
  assert.equal(msgFallback.querySelectorAll('.cardSection').length, 0);
  assert.equal(msgFallback.querySelectorAll('.answerCard table.rTable').length, 1);
  composeHandler = reply({v: 2, lead: 'Flat v2', sections: [], blocks: [GOOD_CARD.blocks[1]]});
  const msgEmptySections = paintC(turn(ANSWER, multi()));
  await until(() => msgEmptySections.querySelector('.answerCard'), 'empty sections');
  assert.equal(msgEmptySections.querySelectorAll('.cardSection').length, 0);
  assert.equal(msgEmptySections.querySelectorAll('.answerCard table.rTable').length, 1);

  /* the page mirrors the server's ceilings: 24 sections (one per part of the request, 5 at least; was 8), 4 blocks each */
  const stat = (i) => ({kind: 'stats', stats: [{label: 'L' + i, ref: i % 2 ? 'r1.total' : 'r1.rate'}]});
  const nineSections = Array.from({length: 25}, (_, i) => ({heading: 'S' + i, verdict: 'v' + i, blocks: [stat(i)]}));
  composeHandler = reply({v: 2, lead: 'Caps', sections: nineSections, blocks: []});
  const msgCaps = paintC(turn(ANSWER, multi()));
  await until(() => msgCaps.querySelector('.answerCard'), 'caps');
  assert.equal(msgCaps.querySelectorAll('.cardSection').length, 24);
  const fiveBlocks = [{heading: 'One', verdict: 'v', blocks: Array.from({length: 6}, (_, i) => ({kind: 'record', title: 'B' + i, ref: 'r1'}))}];
  composeHandler = reply({v: 2, lead: 'Caps', sections: fiveBlocks, blocks: []});
  const msgCaps2 = paintC(turn(ANSWER, multi()));
  await until(() => msgCaps2.querySelector('.answerCard'), 'block caps');
  assert.equal(msgCaps2.querySelectorAll('.cardSection .cardBlock').length, 4);

  /* text from the model stays text, also in a heading and a verdict */
  const EVIL2 = '<img src=x onerror=alert(2)>';
  composeHandler = reply({v: 2, lead: '', sections: [{heading: EVIL2, verdict: EVIL2 + ' done', blocks: [GOOD_CARD.blocks[0]]}], blocks: []});
  const msgEvil2 = paintC(turn(ANSWER, multi()));
  await until(() => msgEvil2.querySelector('.answerCard'), 'xss section');
  assert.equal(text(msgEvil2.querySelector('h3.cardSectionHeading')), EVIL2);
  assert.equal(text(msgEvil2.querySelector('p.cardVerdict')), EVIL2 + ' done');
  assert.equal(msgEvil2.querySelectorAll('img, script').length, 0);
  assert.equal(alerted, false);
  calls.length = 0;

  /* ---- restored turns -------------------------------------------------------- */
  calls.length = 0;
  composeHandler = reply(GOOD_CARD);
  const stored = multi();
  storedRecords = {
    'tk-old': {turn: {record: {turn_output: {command_outputs: stored}}}},
    'tk-new': {turn: {record: {turn_output: {command_outputs: stored}}}},
    'tk-det': {turn: {record: {turn_output: {command_outputs: stored}}}},
    'tk-lazy': {turn: {record: {turn_output: {command_outputs: stored}}}},
  };
  const det = w.tmRenderStoredTurn({turn_key: 'tk-det', status: 'completed', success: true,
    answer: ANSWER, user_message: '/report'}, true);
  assert.equal(det.dataset.deterministic, '1');
  assert.equal(det.dataset.userMessage, '/report');
  const old = w.tmRenderStoredTurn({turn_key: 'tk-old', status: 'completed', success: true,
    answer: ANSWER, user_message: 'what about my order'}, true);
  const fresh = w.tmRenderStoredTurn({turn_key: 'tk-new', status: 'completed', success: true,
    answer: ANSWER, user_message: 'and now'}, true);
  assert.equal(old.dataset.deterministic, '');
  assert.equal(calls.length, 0, 'the placeholder paint, with no record yet, asks for nothing');
  await until(() => fresh.querySelector('.results'), 'the latest restored record loaded');
  await until(() => old.querySelector('.results'), 'older record loaded');
  await until(() => fresh.querySelector('.answerCard') && old.querySelector('.answerCard'), 'loaded records compose');
  assert.equal(calls.length, 2, 'one request per loaded eligible turn, none for the "/" turn');
  assert.deepEqual(calls.map((c) => c.body.user_message).sort(), ['and now', 'what about my order']);
  assert.ok(!det.querySelector('.answerCard') && !det.querySelector('.cardStatus'), '"/" restored turn: nothing');
  noButton(fresh); noButton(old);
  assert.ok(fresh.querySelector('.resultsPanel .results') && !fresh.querySelector('.bubble .results'));
  /* a turn whose record is not loaded (not autoloaded) makes no request */
  const lazy = w.tmRenderStoredTurn({turn_key: 'tk-lazy', status: 'completed', success: true,
    answer: ANSWER, user_message: 'later'}, false);
  await flush();
  assert.equal(calls.length, 2, 'an unloaded restored turn asks nothing');
  assert.ok(!lazy.querySelector('.resultsLink') && !lazy.querySelector('.cardStatus'));
  assert.equal(norm(text(lazy.querySelector('.bubble'))), norm(ANSWER), 'it shows the plain answer');
  Array.from(lazy.querySelectorAll('button')).find((b) => b.textContent === 'show results and artifacts').click();
  await until(() => lazy.querySelector('.answerCard'), 'loading the record composes it');
  assert.equal(calls.length, 3);
  assert.ok(lazy.querySelector('.resultsLink'), 'a reopened stored turn gets the results link');

  /* ---- a record over list-valued fields prints readable text, never raw JSON (G5) ---- */
  composeHandler = reply({lead: 'Lists', blocks: [{kind: 'record', title: 'Item', ref: 'r1'}]});
  const listOut = {name: 'Widget', tags: ['a', 'b', 'c', 'd', 'e', 'f', 'g'], few: ['x', 'y'],
    parts: [{sku: 'A1'}, {sku: 'B2'}, {sku: 'C3'}], none: [], total: 4};
  const msgList = paintC(turn(ANSWER, [co('report', listOut, null), co('find_book', BOOKS, {start: 0})]));
  await until(() => msgList.querySelector('.answerCard'), 'list-valued record composes');
  const listText = text(msgList.querySelector('.answerCard dl.rKv'));
  assert.ok(listText.includes('a, b, c, d, e, +2 more'), listText);
  assert.ok(listText.includes('x, y'), listText);
  assert.ok(listText.includes('3 items'), listText);
  assert.ok(!/[\[\]{}"]/.test(listText), 'no raw JSON on the card: ' + listText);

  /* ---- fix-weih.34: a skip says so, unless it is by design ------------------------------- */
  const NOTICE_LONG = 'Summary card skipped: the answer is too long to summarise.';
  const NOTICE_FAILED = 'Summary card could not be composed.';
  const notices = (m) => m.querySelectorAll('.cardNotice');
  const bigText = (n) => ('Ten more words follow in this sentence of the long answer. ').repeat(Math.ceil(n / 58)).slice(0, n);
  calls.length = 0;
  composeHandler = reply(GOOD_CARD);
  const tooLong = turn(bigText(150001), multi());
  const msgLong = paint(tooLong);
  await flush();
  assert.equal(calls.length, 0, 'an answer over 150000 characters makes no request');
  assert.equal(notices(msgLong).length, 1);
  assert.equal(text(msgLong.querySelector('.cardNotice')), NOTICE_LONG);
  assert.equal(msgLong.querySelector('.cardNotice').getAttribute('data-reason'), 'too_long');
  assert.ok(msgLong.querySelector('.cardNotice').title.includes('too_long'), 'the reason code is in the tooltip');
  assert.ok(msgLong.querySelector('.resultsLinkRow .cardNotice'), 'it sits where the status line sits');
  assert.ok(!/too_long|digest/i.test(text(msgLong.querySelector('.cardNotice'))), 'no jargon in the text');
  assert.ok(!msgLong.querySelector('.cardStatus') && !msgLong.querySelector('.answerCard'));
  assert.ok(!msgLong.querySelector('.cardNotice[role="alert"]'), 'no error styling');
  assert.equal(norm(text(msgLong.querySelector('.bubble'))), norm(tooLong.answer), 'the plain answer is as before');
  assert.ok(msgLong.querySelector('.resultsPanel .results'), 'and so is the results panel');
  w.tmRenderTurn(msgLong, tooLong);
  w.tmRenderTurn(msgLong, tooLong);
  await flush();
  assert.equal(notices(msgLong).length, 1, 'a repaint does not duplicate the notice');
  assert.equal(calls.length, 0);
  /* exactly the limit still composes */
  const atLimit = paint(turn(bigText(150000), multi()));
  await until(() => calls.length === 1 && atLimit.querySelector('.answerCard'), 'an answer of 150000 characters composes');
  assert.equal(notices(atLimit).length, 0);
  /* a long answer with no results panel puts the notice in its own row, and it goes with a repaint */
  const outNone = turn(bigText(150001), []);
  const longNone = paint(outNone);
  assert.equal(text(longNone.querySelector('.cardStatusRow .cardNotice')), NOTICE_LONG);
  w.tmRenderTurn(longNone, outNone);
  assert.equal(longNone.querySelectorAll('.cardNotice').length, 1);
  assert.equal(longNone.querySelectorAll('.cardStatusRow').length, 1);
  assert.equal(calls.length, 1, 'still no request');

  /* each failure leaves one calm line, its reason in data-reason, the plain answer as it was */
  async function notified(handler, reason, expectText, label) {
    composeHandler = handler;
    const before = calls.length;
    const out = turn(ANSWER, multi());
    const msg = paintC(out);
    await until(() => calls.length === before + 1 && !msg.querySelector('.cardStatus'), label + ': settled');
    await flush();
    assert.equal(notices(msg).length, 1, label + ': one notice');
    assert.equal(text(msg.querySelector('.cardNotice')), expectText, label);
    assert.equal(msg.querySelector('.cardNotice').getAttribute('data-reason'), reason, label);
    assert.ok(msg.querySelector('.cardNotice').title.includes(reason), label + ': tooltip');
    assert.ok(!msg.querySelector('.answerCard'), label + ': no card');
    assert.equal(norm(text(msg.querySelector('.bubble'))), norm(ANSWER), label + ': the plain answer stays');
    assert.ok(msg.querySelector('.resultsPanel .results'), label + ': the results stay');
    w.tmRenderTurn(msg, out);
    w.tmRenderTurn(msg, out);
    await flush();
    assert.equal(calls.length, before + 1, label + ': a repaint asks nothing');
    assert.equal(notices(msg).length, 1, label + ': a repaint does not duplicate it');
    assert.equal(msg.querySelector('.cardNotice').getAttribute('data-reason'), reason, label + ': nor change it');
    return {msg, out};
  }
  const failedRun = await notified(replyReason('failed'), 'failed', NOTICE_FAILED, 'failed');
  await notified(replyReason('timeout'), 'timeout', 'Summary card could not be composed in time.', 'server timeout');
  await notified(() => Promise.resolve({ok: false, status: 422, json: () => Promise.resolve({detail: 'too long'})}),
    'http_422', NOTICE_FAILED, 'HTTP 422');
  await notified(() => Promise.resolve({ok: false, status: 500, json: () => Promise.resolve({})}), 'http_500', NOTICE_FAILED, 'HTTP 500');
  await notified(() => Promise.reject(new Error('network down')), 'failed', NOTICE_FAILED, 'network error');
  await notified(reply(null), 'failed', NOTICE_FAILED, 'null card');
  await notified(reply({lead: '', blocks: []}), 'card_refused', NOTICE_FAILED, 'a card the page cannot draw');
  await notified(reply({lead: 'Only a lead', blocks: []}), 'card_refused', NOTICE_FAILED, 'a lead alone over a results panel');
  /* the page's own timeout */
  let abortedHere = false;
  w.TM_CARD_TIMEOUT_MS = 40;
  await notified((call) => new Promise((_, reject) => {
    call.signal.addEventListener('abort', () => { abortedHere = true; reject(new Error('aborted')); });
  }), 'timeout', 'Summary card could not be composed in time.', 'page timeout');
  w.TM_CARD_TIMEOUT_MS = 35000;
  assert.ok(abortedHere);
  /* a card that appears later clears the notice: the same bubble, the card now cached */
  tm.cardCache[failedRun.out.turn_key] = GOOD_CARD;
  w.tmRenderTurn(failedRun.msg, failedRun.out);
  await flush();
  assert.ok(failedRun.msg.querySelector('.answerCard'), 'the cached card shows');
  assert.equal(notices(failedRun.msg).length, 0, 'and the notice is gone');
  /* a request that lands a card while the notice of an earlier paint is still there */
  const gateLate = deferred();
  composeHandler = () => gateLate.promise;
  const outLate = turn(ANSWER, multi());
  const msgLate = paintC(outLate);
  await until(() => msgLate.querySelector('.cardStatus'), 'late request');
  gateLate.resolve(GOOD_CARD);
  await until(() => msgLate.querySelector('.answerCard'), 'late card');
  assert.equal(notices(msgLate).length, 0, 'no notice beside a card');
  /* not_needed: nothing to say */
  const nn = await stays(replyReason('not_needed'), 'not_needed notice check');
  assert.equal(notices(nn).length, 0);

  /* by design: silent */
  calls.length = 0;
  composeHandler = reply(GOOD_CARD);
  box.checked = false;
  const quietOff = paint(turn(bigText(150001), multi()));
  box.checked = true;
  const quietAsk = paint(turn(bigText(150001), multi(), 'awaiting_user'));
  const quietSlash = paint(turn(bigText(150001), multi()), {deterministic: true, message: '/report'});
  const quietShort = paint(turn('short', [co('report', REPORT, null)]));
  const quietNone = paint(turn('short', []));
  tm.connected = false;
  const quietOff2 = paint(turn(bigText(150001), multi()));
  tm.connected = true;
  await flush();
  assert.equal(calls.length, 0);
  [['Rich answers off', quietOff], ['awaiting_user', quietAsk], ['"/" turn', quietSlash],
    ['one short result', quietShort], ['short answer, no results', quietNone], ['not connected', quietOff2]].forEach(([label, m]) => {
    assert.equal(notices(m).length, 0, label + ': no notice');
    assert.ok(!m.querySelector('.cardStatus'), label + ': no status');
  });

  /* a restored turn makes the same decision again once its record is loaded; none before */
  calls.length = 0;
  const storedLong = multi();
  storedRecords['tk-long'] = {turn: {record: {turn_output: {command_outputs: storedLong}}}};
  const restoredLong = w.tmRenderStoredTurn({turn_key: 'tk-long', status: 'completed', success: true,
    answer: bigText(150001), user_message: 'long one'}, true);
  assert.equal(notices(restoredLong).length, 0, 'no notice while the record is still loading');
  await until(() => restoredLong.querySelector('.results'), 'restored long record');
  await until(() => notices(restoredLong).length === 1, 'restored long notice');
  assert.equal(text(restoredLong.querySelector('.cardNotice')), NOTICE_LONG);
  assert.equal(calls.length, 0, 'no request for a restored long answer either');
  calls.length = 0;

  /* ---- no_model: a notice, and the page stops asking ------------------------- */
  calls.length = 0;
  composeHandler = replyReason('no_model');
  const msgNo = paintC(turn(ANSWER, multi()));
  await until(() => tm.composeUnavailable === true, 'no_model flag');
  await flush();
  assert.equal(text(msgNo.querySelector('.cardNotice')), 'Summary card unavailable (composer model not configured).');   /* fix-weih.34: was "no notice" */
  assert.ok(!msgNo.querySelector('.cardStatus'), 'no status');
  assert.equal(norm(text(msgNo.querySelector('.bubble'))), norm(ANSWER));
  const msgAfter = paint(turn(ANSWER, multi()));
  await flush();
  assert.equal(calls.length, 1, 'the circuit breaker stops further requests');
  assert.ok(!msgAfter.querySelector('.cardStatus'));
  assert.equal(msgAfter.querySelector('.cardNotice').getAttribute('data-reason'), 'no_model', 'the breaker still says why, with no request');
  const msgAfterShort = paint(turn('short', [co('report', REPORT, null)]));
  await flush();
  assert.ok(!msgAfterShort.querySelector('.cardNotice'), 'a turn that needed no card gets no notice');

  /* a source switch resets the breaker and the cache */
  w.resetSourceScopedState();
  assert.equal(tm.composeUnavailable, false);
  assert.deepEqual(plain(tm.cardCache), {});

  if (alerted) throw new Error('script ran');
  if (errors.length) throw new Error(errors.join('\n'));
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
