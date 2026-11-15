/* The chat SPA's card-grade results view (fix-weih.4). The page's own
   tmResultsOf, tmResultGroups, tmFormatValue, tmRenderTurn and
   tmRenderStoredTurn are exercised against the assembled page. Only the
   network edge (window.fetch) is replaced, and only to record and script the
   one stored-turn read; nothing else asks the network for anything. */
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

const requested = [];
let storedRecords = {};
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url) => {
      requested.push(String(url));
      const m = String(url).match(/\/api\/turn\/([^/?]+)/);
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

/* jsdom objects come from another realm; compare by value through JSON. */
function plain(value) { return JSON.parse(JSON.stringify(value)); }
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function until(fn, what) {
  for (let i = 0; i < 100; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
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
function paint(out) {
  const msg = w.tmBubble('agent', '...');
  w.tmRenderTurn(msg, out);
  return msg;
}
/* textContent glues sibling cells together; join the text nodes with a space. */
function text(node) {
  const parts = [];
  const walk = (n) => {
    if (n.nodeType === 3) parts.push(n.nodeValue);
    else Array.from(n.childNodes).forEach(walk);
  };
  walk(node);
  return parts.join(' ').replace(/\s+/g, ' ').trim();
}

const SCHEMAS = {
  whatever: {kind: 'execution', input: null, output: null},
  listing: {kind: 'execution', input: null, output: null},
  deepcmd: {kind: 'execution', input: null, output: null},
  'Library/add_two_numbers': {kind: 'execution', input: null, output: {
    properties: {sum_of_two_numbers: {type: 'number', title: 'Sum Of Two Numbers'}}}},
  'Library/show_book': {kind: 'execution',
    input: {properties: {book_uid: {type: 'string', available_from: ['find_book.book_uids']}}},
    output: {properties: {
      book_uid: {type: 'string', title: 'Book Uid'},
      title: {type: 'string', title: 'Title'},
      copies: {type: 'integer', title: 'Copies'},
    }}},
  'Library/find_book': {kind: 'execution', input: {properties: {}}, output: {properties: {
    book_uids: {type: 'array', items: {type: 'string'}, title: 'Book Uids', ui: {table: 'books'}},
    titles: {type: 'array', items: {type: 'string'}, title: 'Titles', ui: {table: 'books'}},
  }}},
  find_user_id_by_email: {kind: 'execution', input: {properties: {email: {type: 'string'}}}, output: null},
  get_user_details: {kind: 'execution', input: {properties: {
    user_id: {type: 'string', available_from: ['find_user_id_by_email.user_id']}}}, output: null},
  get_order_details: {kind: 'execution', input: {properties: {
    order_id: {type: 'string', available_from: ['get_user_details']}}},
    output: {properties: {order_details: {type: 'string', title: 'Order Details'}}}},
  report: {kind: 'execution', input: null, output: {properties: {
    internal_id: {type: 'string', title: 'Internal Id', ui: {hidden: true}},
    total: {type: 'number', title: 'Total', ui: {hero: true, format: 'currency'}},
    rate: {type: 'number', title: 'Rate', ui: {format: 'percent'}},
    ref: {type: 'string', title: 'Ref', ui: {format: 'code'}},
    mood: {type: 'string', title: 'Mood', ui: {format: 'mystery'}},
    lines: {type: 'array', title: 'Lines', items: {type: 'object', properties: {
      sku: {type: 'string', title: 'Sku'},
      secret: {type: 'string', title: 'Secret', ui: {hidden: true}},
      qty: {type: 'integer', title: 'Qty'},
    }}},
  }}},
};

async function main() {
  for (const fn of ['tmNormalizeOutputObject', 'tmResultsOf', 'tmResultGroups', 'tmRenderOutputView',
    'tmRenderRowsTable', 'tmRenderKeyValues', 'tmRenderStatStrip', 'tmFormatValue', 'tmAutoTone',
    'tmRenderResults', 'tmMountResults', 'tmSetResultsOpen']) {
    assert.equal(typeof w[fn], 'function', fn);
  }
  tm.commandSchemas = SCHEMAS;

  /* ---- pure value layer ------------------------------------------------ */
  assert.equal(w.tmFormatValue(null), '—');
  assert.equal(w.tmFormatValue(''), '—');
  assert.equal(w.tmFormatValue(true), 'yes');
  assert.equal(w.tmFormatValue(false), 'no');
  assert.equal(w.tmFormatValue(1234.5), '1,234.5');
  assert.equal(w.tmFormatValue(186.32, {ui: {format: 'currency'}}), '186.32', 'no symbol: the hint carries no currency code');
  assert.equal(w.tmFormatValue(1234567.5, {ui: {format: 'currency'}}), (1234567.5).toLocaleString(undefined,
    {minimumFractionDigits: 2, maximumFractionDigits: 2}), 'grouped and two decimals in the browser locale');
  assert.ok(!/[$€£¥]/.test(w.tmFormatValue(9, {ui: {format: 'currency'}})), 'never a hard-coded symbol');
  assert.equal(w.tmFormatValue(12.5, {ui: {format: 'percent'}}), '12.5%');
  assert.equal(w.tmFormatValue('abc', {ui: {format: 'code'}}).tagName, 'CODE');
  assert.equal(w.tmFormatValue('abc', {ui: {format: 'bogus'}}), 'abc');
  /* a field's name never decides how it is drawn: un-hinted status/state is plain text, a badge needs format: badge */
  assert.equal(w.tmFormatValue('Rejected', null, 'status'), 'Rejected', 'un-hinted status is plain text');
  assert.equal(w.tmFormatValue('Rejected', {}, 'state'), 'Rejected', 'un-hinted state is plain text');
  assert.equal(w.tmFormatValue('shipped', null, 'color'), 'shipped', 'no badge outside a format: badge hint');
  const badge = w.tmFormatValue('Rejected', {ui: {format: 'badge'}}, 'status');
  assert.equal(badge.className, 'rBadge tone-bad');
  assert.equal(w.tmFormatValue('Cancelled', {ui: {format: 'badge'}}, 'status').className, 'rBadge tone-neutral', 'a domain word is the app\'s to tone');
  assert.equal(w.tmFormatValue('hi', {ui: {format: 'badge'}}).className, 'rBadge tone-neutral');
  [['Success', 'good'], ['ok', 'good'], ['done', 'good'], ['Active', 'good'], ['enabled', 'good'], ['passed', 'good'],
    ['approved', 'good'], ['complete', 'good'], ['pending', 'warning'], ['waiting', 'warning'], ['in progress', 'warning'],
    ['warning', 'warning'], ['error', 'bad'], ['failed', 'bad'], ['denied', 'bad'], ['blocked', 'bad'],
    ['expired', 'bad'], ['rejected', 'bad'], ['whatever', 'neutral'],
    ['delivered', 'neutral'], ['paid', 'neutral'], ['overdue', 'neutral'], ['processing', 'neutral'],
    ['cancelled', 'neutral'], ['canceled', 'neutral']]
    .forEach(([v, t]) => assert.equal(w.tmAutoTone(v), t, v));
  const compact = w.tmFormatValue({a: 'z'.repeat(300)});
  assert.ok(compact.length <= 121 && compact.endsWith('…'));

  const nested = plain(w.tmNormalizeOutputObject({book: {title: 'Dune', year: 1965}}));
  assert.deepEqual(nested, {output: {title: 'Dune', year: 1965}, unwrappedFrom: 'book', unwrappedPath: ['book']});
  const twoDeep = plain(w.tmNormalizeOutputObject({order_details: '{"order": {"status": "delivered"}}'}));
  assert.deepEqual(twoDeep, {output: {status: 'delivered'}, unwrappedFrom: 'order_details',
    unwrappedPath: ['order_details', 'order']});
  [{k: 5}, {k: [1]}, {k: {}}, {k: '[1]'}, {k: 'text'}].forEach((o) =>
    assert.deepEqual(plain(w.tmNormalizeOutputObject(o)).unwrappedPath, [], JSON.stringify(o)));
  [{book: {title: 'Dune'}}, {order_details: '{"order": {"status": "x"}}'}, {a: 1, b: 2}].forEach((o) => {
    const once = plain(w.tmNormalizeOutputObject(o));
    const again = plain(w.tmNormalizeOutputObject(once.output));
    assert.deepEqual(again, {output: once.output, unwrappedFrom: null, unwrappedPath: []});
  });
  const deepResults = w.tmResultsOf([co('shelf_cmd', {book: {title: 'Dune', year: 1965}}, null)], {
    shelf_cmd: {output: {properties: {book: {'$ref': '#/$defs/Book'}}, '$defs': {Book: {
      type: 'object', title: 'Book', properties: {title: {type: 'string', title: 'Headline'}}}}}}});
  assert.deepEqual(plain(deepResults[0].unwrappedPath), ['book']);
  tm.commandSchemas = {shelf_cmd: {output: null}};
  const deepMsg = paint(turn('x', [co('shelf_cmd', {book: {title: 'Dune', year: 1965}}, null)]));
  tm.commandSchemas = {shelf_cmd: {output: {properties: {book: {'$ref': '#/$defs/Book'}}, '$defs': {Book: {
    type: 'object', title: 'Book', properties: {title: {type: 'string', title: 'Headline'}}}}}}};
  const deepMsg2 = paint(turn('x', [co('shelf_cmd', {book: {title: 'Dune', year: 1965}}, null)]));
  assert.match(text(deepMsg2.querySelector('.results')), /Headline/);
  assert.match(text(deepMsg.querySelector('.results')), /Title/);
  tm.commandSchemas = SCHEMAS;

  const norm = plain(w.tmNormalizeOutputObject({a: '{"x": 1}', b: '[1,2]', c: '{oops', d: 'plain', e: '5'}));
  assert.deepEqual(norm, {output: {a: {x: 1}, b: [1, 2], c: '{oops', d: 'plain', e: '5'}, unwrappedFrom: null, unwrappedPath: []});
  const wrapped = plain(w.tmNormalizeOutputObject({wrap: ' {"k": "v"} '}));
  assert.deepEqual(wrapped, {output: {k: 'v'}, unwrappedFrom: 'wrap', unwrappedPath: ['wrap']});
  assert.equal(plain(w.tmNormalizeOutputObject({wrap: '{}'})).unwrappedFrom, null, 'empty object is not unwrapped');
  assert.equal(plain(w.tmNormalizeOutputObject({wrap: '[1]'})).unwrappedFrom, null, 'an array is not unwrapped');

  /* ---- case 1: "/" turn, stat strip, whole answer, results behind a link ---- */
  const add = paint(turn('{"sum_of_two_numbers":5}', [
    co('add_two_numbers', {sum_of_two_numbers: 5}, {first_num: 2, second_num: 3})]));
  assert.ok(add.classList.contains('hasResults'));
  const stats = add.querySelector('.results dl.cardStats');
  assert.ok(stats, 'a single scalar renders as a stat strip');
  assert.equal(text(stats.querySelector('dt')), 'Sum Of Two Numbers');
  assert.equal(text(stats.querySelector('dd')), '5');
  assert.ok(!add.querySelector('.answerRest') && !add.querySelector('button.answerToggle'), 'the answer is not folded');
  assert.equal(text(add.querySelector('.bubble')), '{"sum_of_two_numbers":5}', 'the whole answer, JSON line included');
  const jsonBtn = add.querySelector('button.rJson');
  assert.equal(jsonBtn.getAttribute('aria-pressed'), 'false');
  jsonBtn.click();
  assert.equal(jsonBtn.getAttribute('aria-pressed'), 'true');
  assert.equal(add.querySelector('pre.json').hidden, false);
  assert.deepEqual(JSON.parse(add.querySelector('.results pre.json').textContent), {sum_of_two_numbers: 5});
  jsonBtn.click();
  assert.equal(add.querySelector('.results pre.json').hidden, true);

  /* the paging line stays visible as the lead; Python-style JSON spacing still matches */
  const book = {book_uid: 'b-17', title: 'Dune', copies: 3};
  const shown = paint(turn('{"book_uid": "b-17", "title": "Dune", "copies": 3}\nShowing 1 of 1; next start=1',
    [co('show_book', book, {book_uid: 'b-17'})]));
  assert.equal(text(shown.querySelector('.bubble')),
    '{"book_uid": "b-17", "title": "Dune", "copies": 3} Showing 1 of 1; next start=1', 'the whole answer stays visible');
  assert.ok(!shown.querySelector('.answerLead') && !shown.querySelector('.answerRest'));
  assert.equal(text(shown.querySelector('.cardStats dt')), 'Book UID');
  assert.equal(shown.querySelectorAll('.cardStats .cardStat').length, 3);

  /* an answer is never folded */
  const other = paint(turn('{"something": "else"}', [co('show_book', book, {book_uid: 'b-17'})]));
  assert.ok(!other.querySelector('.answerRest'));
  assert.equal(text(other.querySelector('.bubble')), '{"something": "else"}');

  /* ---- case 2: retail_workflow JSON-string unwrapping ------------------- */
  const orderRaw = {order_id: '#W1', status: 'delivered', total_paid: 186.32,
    items: [{name: 'Lamp', price: 12.5}]};
  const orderOut = [co('get_order_details', {order_details: JSON.stringify(orderRaw)},
    {order_id: '#W1'}, {context: '*'})];
  const results = w.tmResultsOf(orderOut, SCHEMAS);
  assert.equal(results.length, 1);
  assert.equal(results[0].id, 'r1');
  assert.equal(results[0].output.status, 'delivered');
  assert.ok(Array.isArray(results[0].output.items));
  assert.equal(results[0].unwrappedFrom, 'order_details');
  assert.equal(typeof results[0].outputs[0].order_details, 'string', 'raw output kept for the toggle');
  const order = paint(turn('Order #W1 is delivered.\n\nDetails follow.', orderOut));
  assert.ok(!order.querySelector('.results .rBadge'), 'an un-hinted status field is plain text, not a badge');
  assert.match(text(order.querySelector('.results')), /delivered/);
  assert.match(text(order.querySelector('.results dl.cardStats')), /Total Paid 186\.32/);
  assert.match(text(order.querySelector('.results dl.cardStats')), /Order ID #W1/);
  const priceHead = Array.from(order.querySelectorAll('.results th')).find((th) => th.textContent === 'Price');
  assert.ok(priceHead && priceHead.classList.contains('num'));
  assert.equal(order.querySelector('.results td.num').textContent, '12.5');
  assert.ok(!order.querySelector('.results td.num').previousSibling || true);
  order.querySelector('button.rJson').click();
  const orderJson = order.querySelector('.results pre.json');
  assert.equal(orderJson.hidden, false);
  assert.match(orderJson.textContent, /"order_details"/, 'the toggle shows the original raw output');
  assert.equal(order.querySelector('.rView').hidden, true);

  /* ---- case 3: upstream available_from lookups collapse ------------------ */
  const chain = [
    co('find_user_id_by_email', {user_id: 'u9'}, {email: 'a@b.c'}, {context: '*'}),
    co('get_user_details', {user_details: JSON.stringify({user_id: 'u9', order_ids: ['#W1']})},
      {user_id: 'u9'}, {context: '*'}),
    co('get_order_details', {order_details: JSON.stringify(orderRaw)}, {order_id: '#W1'}, {context: '*'}),
  ];
  const chainResults = w.tmResultsOf(chain, SCHEMAS);
  assert.deepEqual(plain(chainResults.map((r) => r.id)), ['r1', 'r2', 'r3']);
  assert.deepEqual(plain(chainResults.map((r) => r.supporting)), [true, true, false]);
  const plan = plain(w.tmResultGroups(chainResults));
  assert.deepEqual(plan.groups.map((g) => g.command), ['get_order_details']);
  assert.equal(plan.supporting.length, 2);
  const chained = paint(turn('Here is your order.', chain));
  const support = chained.querySelector('details.supportingResults');
  assert.ok(support);
  assert.equal(support.querySelector('summary').textContent, 'Supporting lookups (2)');
  assert.ok(!support.open);
  assert.equal(chained.querySelector('.rGroups .rTitle').textContent, 'get_order_details');
  assert.ok(!chained.querySelector('.rGroups').textContent.includes('u9'), 'lookup scaffolding stays out of the stage');
  assert.match(support.textContent, /find_user_id_by_email/);
  assert.match(support.textContent, /u9/);
  /* with no schema nothing is a lookup */
  assert.deepEqual(plain(w.tmResultsOf(chain, {}).map((r) => r.supporting)), [false, false, false]);

  /* ---- case 3b: data-driven lookups [fix-weih.25] --------------------------- */
  /* no schema edge: a result is still a lookup when a later call of another command was given an
     identifier it returned */
  const UID = '2c3d168291cf64aa7fd114c13a260aff';
  const fed = [
    co('find_person', {person_uids: [UID], labels: ['Ann'], total: 1}, {query: 'Ann'}),
    co('open_person', {person_uid: UID, label: 'Ann'}, {person_uid: UID}),
    co('list_rights', {right_uids: ['a1b2c3d4e5f6'], labels: ['Admin'], total: 1}, {start: 0}),
  ];
  assert.deepEqual(plain(w.tmResultsOf(fed, {}).map((r) => r.supporting)), [true, false, false],
    'find_person fed open_person; nothing fed list_rights');
  /* a short value (a name, a count) proves nothing, and the same command never feeds itself */
  const shortFeed = [co('find_person', {person_uids: ['u9'], total: 1}, null), co('open_person', {person_uid: 'u9'}, {person_uid: 'u9'})];
  assert.deepEqual(plain(w.tmResultsOf(shortFeed, {}).map((r) => r.supporting)), [false, false]);
  const sameCommand = [co('find_person', {person_uids: [UID], total: 1}, null), co('find_person', {person_uids: [UID + 'x'], total: 1}, {query: UID})];
  assert.deepEqual(plain(w.tmResultsOf(sameCommand, {}).map((r) => r.supporting)), [false, false]);

  /* ---- case 3c: the framework's review commands are no result [fix-weih.25] --- */
  const planned = [
    co('review_proposal', {round: 1, accepted: '{}'}, {round: 1}, {context: 'global'}),
    co('review_subtasks', {items: ['a']}, null, {context: 'global'}),
    co('find_book', {book_uids: ['u1'], titles: ['Dune'], total: 1}, {start: 0}),
  ];
  const plannedResults = plain(w.tmResultsOf(planned, SCHEMAS));
  assert.deepEqual(plannedResults.map((r) => [r.id, r.command]), [['r1', 'find_book']], 'plumbing is left out');
  assert.deepEqual(plain(w.tmResultsOf(planned, {}).map((r) => r.command)), ['find_book'], 'with or without a schema listing');
  assert.equal(plain(w.tmResultsOf([co('Shelf/review_plan', {a: 1}, null)], {})).length, 1, 'a workflow command of that leaf is not plumbing');
  /* the server's marker decides: `framework: true` is plumbing whatever the name, with or without a listing;
     `framework: false` is a result even under a name the legacy list holds; a record without the key (a stored
     turn) keeps the legacy name list (the `planned` records above) [fix-2nsr] */
  const marked = [
    co('a_future_framework_command', {x: 1}, null, {framework: true}),
    co('review_plan', {a: 1}, null, {framework: true, context: 'global'}),
    co('find_book', {book_uids: ['u1'], titles: ['Dune'], total: 1}, {start: 0}, {framework: false}),
  ];
  assert.deepEqual(plain(w.tmResultsOf(marked, {}).map((r) => r.command)), ['find_book'], 'framework:true is left out with no listing');
  assert.deepEqual(plain(w.tmResultsOf(marked, SCHEMAS).map((r) => r.command)), ['find_book'], 'and with one');
  const appReview = [co('review_plan', {verdict: 'ok', risks: ['r']}, {topic: 't'}, {framework: false})];
  assert.deepEqual(plain(w.tmResultsOf(appReview, {}).map((r) => r.command)), ['review_plan'],
    'an application command that shares a legacy name is shown when the key is present and false');
  assert.equal(plain(w.tmResultsOf([co('review_plan', {a: 1}, null, {framework: null})], {})).length, 0,
    'a null marker is "unknown": the legacy list applies');
  /* a command the listing does not name is still a result: the listing also omits the internal metadata-extraction
     commands and child-workflow commands, whose outputs are real */
  const unlisted = [co('ErrorCorrection/you_misunderstood', {valid_command_names: ['a', 'b']}, null),
    co('what_can_i_do', {valid_command_names: ['a']}, null), co('ProposalReview/add', {added: 'x'}, null),
    co('find_book', {book_uids: ['u1'], titles: ['Dune'], total: 1}, {start: 0})];
  assert.deepEqual(plain(w.tmResultsOf(unlisted, SCHEMAS).map((r) => r.command)),
    ['ErrorCorrection/you_misunderstood', 'what_can_i_do', 'ProposalReview/add', 'find_book'], 'absence from the listing hides nothing');
  const priorSchemas = tm.commandSchemas;
  tm.commandSchemas = SCHEMAS;
  const plannedMsg = paint(turn('Done.', planned.concat([co('show_book', book, {book_uid: 'z'})])));
  tm.commandSchemas = priorSchemas;
  assert.ok(!/review_/i.test(text(plannedMsg.querySelector('.resultsPanel') || plannedMsg)), 'no chip, title or row for review commands');

  /* ---- case 4: Library paged read, fan-out, exclusions, folds ------------ */
  const page0 = {book_uids: ['u1', 'u2'], titles: ['Dune', 'Emma'], total: 4};
  const page2 = {book_uids: ['u3', 'u4'], titles: ['Dune', 'Ulysses'], total: 4};
  const library = [
    co('find_book', page0, {query: 'x', start: 0}),
    co('find_book', page2, {query: 'x', start: 2}),
    co('find_book', page0, {query: 'x', start: 0}),
    co('find_book', {nope: 1}, {query: 'bad'}, {command_response: {success: false, response: 'no', output: {nope: 1}}}),
    co('ask_user', {q: 'which?'}, 'which one?', {ask_user_entry: true}),
    co('show_book', {book_uid: 'u1', title: 'Dune', copies: 2}, {book_uid: 'u1'}),
    co('show_book', {book_uid: 'u3', title: 'Dune', copies: 1}, {book_uid: 'u3'}),
    co('show_book', {book_uid: 'u2', title: 'Emma', copies: 5}, {book_uid: 'u2'}),
  ];
  const lib = w.tmResultsOf(library, {});
  assert.deepEqual(plain(lib.map((r) => r.id)), ['r1', 'r2', 'r3', 'r4'], 'failed and ask_user entries are excluded');
  assert.deepEqual(plain(lib[0].output.book_uids), ['u1', 'u2', 'u3', 'u4']);
  assert.deepEqual(plain(lib[0].output.titles), ['Dune', 'Emma', 'Dune', 'Ulysses'], 'no dedup across pages');
  assert.equal(lib[0].output.total, 4);
  assert.equal(lib[0].outputs.length, 2, 'the re-read replaced its earlier copy');
  assert.deepEqual(plain(w.tmResultsOf([co('f', {a: [1]}, null), co('f', {a: [2]}, {})], {}).map((r) => r.output)), [{a: [2]}]);
  /* the author declares the book columns as one table group; with no schema at all the lists are plain
     lists [fix-sush step 3] */
  tm.commandSchemas = {'Library/find_book': SCHEMAS['Library/find_book']};
  const libMsg = paint(turn('Found it.', library));
  const chips = Array.from(libMsg.querySelectorAll('.rChip'));
  assert.deepEqual(chips.map((c) => c.textContent), ['find_book', 'show_book ×3']);
  assert.deepEqual(chips.map((c) => c.getAttribute('aria-pressed')), ['false', 'true']);
  const panels = libMsg.querySelectorAll('.rPanel');
  assert.equal(panels[0].hidden, true);
  assert.equal(panels[1].hidden, false);
  const bookRows = Array.from(panels[0].querySelectorAll('tbody tr')).map((tr) =>
    Array.from(tr.children).map((td) => td.textContent));
  assert.deepEqual(bookRows, [['u1', 'Dune'], ['u2', 'Emma'], ['u3', 'Dune'], ['u4', 'Ulysses']],
    'one row per index, the two Dune rows keep their own uids');
  assert.equal(panels[1].querySelectorAll('tbody tr').length, 3);
  assert.ok(!Array.from(panels[1].querySelectorAll('th')).some((th) => th.textContent === 'Record'));
  chips[0].click();
  assert.equal(panels[0].hidden, false);
  assert.equal(panels[1].hidden, true);
  assert.equal(chips[0].getAttribute('aria-pressed'), 'true');
  /* different contexts add a Record column */
  const ctxResults = [co('show_book', book, {book_uid: 'a'}, {context: 'Shelf'}),
    co('show_book', book, {book_uid: 'b'}, {context: 'Cart'})];
  const ctxMsg = paint(turn('Both.', ctxResults));
  assert.equal(ctxMsg.querySelector('.results th').textContent, 'Record');
  assert.deepEqual(Array.from(ctxMsg.querySelectorAll('.results tbody tr td:first-child')).map((t) => t.textContent),
    ['Shelf', 'Cart']);
  tm.commandSchemas = SCHEMAS;
  /* with the real edges, find_book is a lookup of show_book */
  const edged = w.tmResultsOf(library, SCHEMAS);
  assert.deepEqual(plain(edged.map((r) => r.supporting)), [true, false, false, false]);
  const edgedMsg = paint(turn('Found it.', library));
  assert.equal(edgedMsg.querySelector('details.supportingResults summary').textContent, 'Supporting lookups (1)');
  assert.equal(edgedMsg.querySelectorAll('.rChip').length, 0, 'one group needs no chips');

  /* the whole answer shows however many paragraphs it has, and keeps its markup */
  const three = paint(turn('**Summary** one.\n\nSecond paragraph.\n\nThird paragraph.', [co('show_book', book, {book_uid: 'z'})]));
  assert.ok(!three.querySelector('.answerLead') && !three.querySelector('.answerRest') && !three.querySelector('.answerToggle'));
  assert.match(text(three.querySelector('.bubble')), /Summary one\. Second paragraph\. Third paragraph\./);
  assert.equal(three.querySelectorAll('.bubble p').length, 3);
  assert.ok(three.querySelector('.bubble').classList.contains('md'));
  assert.ok(!three.querySelector('.bubble[hidden], .bubble [hidden]'), 'nothing in the answer is hidden');
  const plainText = paint(turn('First part\n\nSecond part', [co('show_book', book, {book_uid: 'z'})]));
  assert.ok(!plainText.querySelector('.answerRest'));
  assert.equal(plainText.querySelector('.bubble').textContent, 'First part\n\nSecond part');
  const lit = paint(turn('Uses a_b_c and *not bold* here.\n\nSecond total_paid_usd, 2 * 3 * 4.',
    [co('show_book', book, {book_uid: 'z'})]));
  assert.equal(lit.querySelectorAll('.bubble em, .bubble strong').length, 0);
  assert.equal(lit.querySelector('.bubble').textContent,
    'Uses a_b_c and *not bold* here.\n\nSecond total_paid_usd, 2 * 3 * 4.');
  /* a turn that waits on the user still has its results behind the link */
  const awaiting = paint(turn('First\n\nSecond', [co('show_book', book, {book_uid: 'z'})], 'awaiting_user'));
  assert.ok(!awaiting.querySelector('.answerRest'));
  assert.ok(awaiting.querySelector('.results'), 'results still render');
  const awaitingArtifact = paint(turn('First\n\nSecond', [co('show_book', book, {book_uid: 'z'},
    {command_response: {success: true, output: book, artifacts: {awaiting_user: true}}})]));
  assert.ok(!awaitingArtifact.querySelector('.answerRest'));
  const noResults = paint(turn('First\n\nSecond', [co('add_todo', undefined)].map((c) => {
    c.command_response.output = null; return c; })));
  assert.ok(!noResults.querySelector('.results') && !noResults.querySelector('.answerRest'));
  assert.ok(!noResults.classList.contains('hasResults'));

  /* ---- case 5: UiOptions ------------------------------------------------- */
  const rep = paint(turn('Report ready.', [co('report', {
    internal_id: 'SECRET-123', total: 186.32, rate: 12.5, ref: 'abc123', mood: 'ok',
    lines: [{sku: 'A1', secret: 'HIDDEN-COL', qty: 4}],
  }, null, {context: '*'})]));
  const repStats = rep.querySelector('.results dl.cardStats');
  assert.equal(text(repStats), 'Total 186.32', 'hero field in the stat strip');
  const kv = rep.querySelector('.results dl.rKv');
  assert.match(text(kv), /Rate 12\.5%/);
  assert.equal(kv.querySelector('code.rCode').textContent, 'abc123');
  assert.match(text(kv), /Mood ok/, 'an unknown format is tolerated');
  assert.ok(!rep.querySelector('.results').textContent.includes('SECRET-123'), 'hidden field not on the card');
  assert.ok(!rep.querySelector('.results').textContent.includes('HIDDEN-COL'), 'hidden column not in the table');
  assert.ok(!Array.from(rep.querySelectorAll('.results th')).some((th) => th.textContent === 'Secret'));
  rep.querySelector('button.rJson').click();
  assert.match(rep.querySelector('.results pre.json').textContent, /SECRET-123/, 'still in the JSON toggle');
  assert.match(rep.querySelector('.results pre.json').textContent, /HIDDEN-COL/);

  /* zero hints: more than four scalars go to the key/value list */
  const many = paint(turn('x', [co('whatever', {a: 1, b: 2, c: 3, d: 4, e: 5, long: 'l'.repeat(80)}, null)]));
  assert.ok(!many.querySelector('.results dl.cardStats'));
  assert.equal(many.querySelectorAll('.results dl.rKv dt').length, 6);

  /* ---- case 6: security, idempotence, resilience, caps, restore ---------- */
  const evil = paint(turn('<img src=x onerror=alert(1)>', [co('show_book',
    {book_uid: '<img src=x onerror=alert(1)>', title: '<b>bold</b>', copies: 1}, {book_uid: 'e'})]));
  assert.equal(evil.querySelectorAll('img').length, 0);
  assert.equal(evil.querySelectorAll('b').length, 0);
  assert.match(evil.querySelector('.results').textContent, /<img src=x onerror=alert\(1\)>/);
  w.tm.commandSchemas = {};
  const bare = paint(turn('Plain.', [co('show_book', book, {book_uid: 'q'})]));
  assert.ok(bare.querySelector('.results dl.cardStats'), 'no schemas still renders');
  assert.equal(text(bare.querySelector('.cardStats dt')), 'Book UID', 'titles are polished without a schema');
  tm.commandSchemas = SCHEMAS;

  const twice = w.tmBubble('agent', '...');
  const twiceOut = turn('First\n\nSecond', [co('show_book', book, {book_uid: 'q'})]);
  w.tmRenderTurn(twice, twiceOut);
  w.tmRenderTurn(twice, twiceOut);
  assert.equal(twice.querySelectorAll('.results').length, 1);
  assert.equal(twice.querySelectorAll('.answerRest').length, 0);
  assert.equal(twice.querySelectorAll('.answerToggle').length, 0);
  assert.equal(twice.querySelectorAll('.resultsLink').length, 1, 'a repaint never stacks the link');
  assert.equal(twice.querySelectorAll('.resultsPanel').length, 1, 'nor the panel');
  assert.equal(twice.querySelectorAll('.meta').length, 1);
  w.tmRenderTurn(twice, turn('No results now', []));
  assert.equal(twice.querySelectorAll('.results').length, 0);
  assert.ok(!twice.classList.contains('hasResults'));
  assert.equal(twice.querySelectorAll('.answerRest').length, 0);
  assert.equal(twice.querySelectorAll('.resultsLink, .resultsPanel, .resultsLinkRow').length, 0, 'no results, no link');

  /* ---- the results link and its side panel [fix-weih.24] ------------------- */
  const pn = paint(turn('The whole answer.\n\nSecond paragraph.', [co('show_book', book, {book_uid: 'q'}),
    co('find_book', {book_uids: ['u1'], titles: ['Dune']}, {query: 'x'})]));
  const pLink = pn.querySelector('.resultsLinkRow button.resultsLink');
  const pPanel = pn.querySelector('.resultsPanel');
  assert.ok(pLink && pPanel);
  assert.equal(pLink.tagName, 'BUTTON');
  assert.equal(pLink.type, 'button');
  assert.equal(pLink.textContent, 'Show results (2)', 'the link carries the result count');
  assert.equal(pLink.getAttribute('aria-expanded'), 'false');
  assert.equal(pLink.getAttribute('aria-controls'), pPanel.id);
  assert.equal(pPanel.hidden, true, 'closed by default');
  assert.ok(!pn.querySelector('.bubble .results') && !pn.querySelector('.bubble .rChip'), 'no results or chips in the answer bubble');
  assert.ok(pPanel.querySelector('.results') && pPanel.querySelectorAll('.rChip').length === 2, 'chips live in the panel');
  assert.equal(pn.querySelector('.bubble').nextSibling, pPanel, 'the panel follows the bubble (the page lays it out beside it)');
  assert.ok(pPanel.compareDocumentPosition(pn.querySelector('.turnFooter')) & 4, 'the footer row sits below the panel');
  assert.equal(pn.querySelector('.resultsLinkRow').parentNode, pn.querySelector('.turnFooterLeft'), 'the link row is the footer left slot');
  assert.ok(!pn.querySelector('.answerRest[hidden]') && !pn.querySelector('.answerToggle'), 'the whole answer is visible');
  assert.equal(text(pn.querySelector('.bubble')), 'The whole answer. Second paragraph.');
  assert.ok(pn.classList.contains('hasResults') && !pn.classList.contains('resultsOpen'));
  pLink.click();
  assert.equal(pPanel.hidden, false, 'the link opens the panel');
  assert.equal(pLink.getAttribute('aria-expanded'), 'true');
  assert.equal(pLink.textContent, 'Hide results (2)');
  assert.ok(pn.classList.contains('resultsOpen'));
  assert.equal(d.activeElement, pPanel.querySelector('button.resultsClose'), 'focus moves into the panel');
  pLink.click();
  assert.equal(pPanel.hidden, true, 'the link toggles');
  assert.equal(pLink.getAttribute('aria-expanded'), 'false');
  assert.ok(!pn.classList.contains('resultsOpen'));
  pLink.click();
  pPanel.querySelector('button.resultsClose').click();
  assert.equal(pPanel.hidden, true, 'the close control closes');
  assert.equal(d.activeElement, pLink, 'and returns focus to the link');
  pLink.click();
  pPanel.querySelector('button.resultsClose').focus();
  pPanel.dispatchEvent(new w.KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true}));
  assert.equal(pPanel.hidden, true, 'Escape closes');
  assert.equal(d.activeElement, pLink, 'and returns focus to the link');
  /* a repaint of the same turn does not stack, and keeps an open panel open */
  const pOut = turn('Whole.', [co('show_book', book, {book_uid: 'q'})]);
  const rp = w.tmBubble('agent', '...');
  w.tmRenderTurn(rp, pOut);
  rp.querySelector('.resultsLink').click();
  w.tmRenderTurn(rp, pOut);
  w.tmRenderTurn(rp, pOut);
  assert.equal(rp.querySelectorAll('.resultsLink').length, 1);
  assert.equal(rp.querySelectorAll('.resultsPanel').length, 1);
  assert.equal(rp.querySelectorAll('.resultsLinkRow').length, 1);
  assert.equal(rp.querySelector('.resultsPanel').hidden, false, 'an open panel stays open across a repaint');
  assert.equal(rp.querySelector('.resultsLink').getAttribute('aria-expanded'), 'true');
  assert.ok(rp.classList.contains('resultsOpen'));
  rp.querySelector('.resultsLink').click();
  w.tmRenderTurn(rp, pOut);
  assert.equal(rp.querySelector('.resultsPanel').hidden, true, 'a closed panel stays closed');
  /* a turn with no results has no link, and a new paint without results drops the old one */
  rp.querySelector('.resultsLink').click();
  w.tmRenderTurn(rp, turn('Nothing structured.', []));
  assert.equal(rp.querySelectorAll('.resultsLink, .resultsPanel').length, 0);
  assert.ok(!rp.classList.contains('resultsOpen'));
  w.tmRenderTurn(rp, pOut);
  assert.equal(rp.querySelector('.resultsPanel').hidden, true, 'the open state does not leak to a later turn');
  /* the artifact panel and the results panel are never open together on one answer */
  const both = paint(turn('Both.', [co('show_book', book, {book_uid: 'q'},
    {command_response: {success: true, response: 'ok', output: book, artifacts: {note: 'an artifact'}}})]));
  both.querySelector('.resultsLink').click();
  assert.equal(both.querySelector('.resultsPanel').hidden, false);
  both.querySelector('.artifactsLink').click();
  assert.equal(both.querySelector('.resultsPanel').hidden, true, 'opening the artifacts closes the results panel');
  assert.equal(both.querySelector('.artifacts').hidden, false);
  both.querySelector('.resultsLink').click();
  assert.equal(both.querySelector('.artifacts').hidden, true, 'opening the results closes the artifacts panel');
  assert.equal(both.querySelector('.resultsPanel').hidden, false);
  both.querySelector('.resultsLink').click();

  const rows = [];
  for (let i = 0; i < 120; i++) rows.push({n: i, label: 'row' + i});
  const big = paint(turn('many', [co('listing', {rows}, null)]));
  assert.equal(big.querySelectorAll('.results tbody tr').length, 50);
  const more = big.querySelector('button.rMore');
  assert.equal(more.textContent, 'Show all 120 rows');
  more.click();
  assert.equal(big.querySelectorAll('.results tbody tr').length, 120);
  assert.ok(!big.querySelector('button.rMore'));
  assert.equal(big.querySelector('.results td.num').textContent, '0');
  const huge = [];
  for (let i = 0; i < 1500; i++) huge.push({n: i});
  const hugeMsg = paint(turn('many', [co('listing', {rows: huge}, null)]));
  assert.equal(hugeMsg.querySelector('button.rMore').textContent, 'Show first 1000 of 1500 rows');
  hugeMsg.querySelector('button.rMore').click();
  assert.equal(hugeMsg.querySelectorAll('.results tbody tr').length, 1000);

  /* nested objects, depth cap, scalar lists */
  const deep = paint(turn('deep', [co('deepcmd', {
    tags: ['a', 'b'], none: [], inner: {x: 1, inner: {y: 2, inner: {z: 3, inner: {w: 4}}}},
    cells: [{o: {k: 'v'}}],
  }, null)]));
  assert.equal(deep.querySelectorAll('.results ul.rList li').length, 2);
  assert.match(deep.querySelector('.results .rNone').textContent, /none/);
  assert.equal(deep.querySelectorAll('.results pre.json').length, 2, 'depth 4 prints as JSON (plus the toggle pre)');

  /* schema timing: render now, repaint once when the listing arrives */
  tm.commandSchemas = null;
  let release;
  tm.commandSchemasPromise = new Promise((r) => { release = r; });
  const late = w.tmBubble('agent', '...');
  const lateOut = turn('Here.', chain);
  w.tmRenderTurn(late, lateOut);
  assert.equal(late.querySelectorAll('.results').length, 1);
  assert.ok(!late.querySelector('details.supportingResults'), 'painted at once without schemas');
  assert.equal(late.querySelectorAll('.rChip').length, 3);
  const stale = late.querySelector('.results');
  w.tmRenderTurn(late, lateOut);
  assert.notEqual(late.querySelector('.results'), stale);
  tm.commandSchemas = SCHEMAS;
  release(SCHEMAS);
  await until(() => late.querySelector('details.supportingResults'), 'the late schemas never repainted');
  assert.equal(late.querySelectorAll('.results').length, 1, 'the repaint replaces, never stacks');
  assert.ok(late.querySelector('.resultsPanel .results'), 'the replacement is inside the panel');
  assert.equal(late.querySelectorAll('.resultsLink').length, 1);
  assert.equal(late.querySelectorAll('details.supportingResults').length, 1);
  tm.commandSchemasPromise = null;

  /* a listing that never settles hides no result, and review commands are out while it is pending */
  tm.commandSchemas = null;
  tm.commandSchemasPromise = new Promise(() => {});
  const hung = w.tmBubble('agent', '...');
  w.tmRenderTurn(hung, turn('Here.', planned));
  assert.equal(hung.querySelectorAll('.results').length, 1);
  assert.ok(!/review_/i.test(text(hung.querySelector('.resultsPanel'))), 'no review command while pending');
  tm.commandSchemas = SCHEMAS;
  tm.commandSchemasPromise = null;

  /* restored turns: results come from the one stored-turn read */
  const storedOutputs = [co('show_book', book, {book_uid: 'b-17'})];
  storedRecords = {
    'tk-new': {turn: {record: {turn_output: {command_outputs: storedOutputs}}}},
    'tk-old': {turn: {record: {turn_output: {command_outputs: storedOutputs}}}},
  };
  const from = requested.length;
  const fresh = w.tmRenderStoredTurn({turn_key: 'tk-new', status: 'completed', success: true,
    answer: '{"book_uid":"b-17","title":"Dune","copies":3}\nnote line', user_message: 'show it'}, true);
  await until(() => fresh.querySelector('.results'), 'restored results never rendered');
  assert.ok(!fresh.querySelector('.answerRest'), 'a restored answer is not folded either');
  assert.match(text(fresh.querySelector('.bubble')), /note line/);
  assert.equal(fresh.querySelector('.resultsLink').textContent, 'Show results (1)', 'a reopened turn gets the link');
  assert.ok(fresh.querySelector('.resultsPanel').hidden);
  assert.deepEqual(requested.slice(from), ['/api/turn/tk-new'], 'exactly the one stored-turn read');
  const olderFrom = requested.length;
  const older = w.tmRenderStoredTurn({turn_key: 'tk-old', status: 'completed', success: true,
    answer: 'older'}, false);
  assert.ok(!older.querySelector('.results'));
  assert.equal(requested.length, olderFrom, 'an older turn asks for nothing until clicked');
  const showBtn = Array.from(older.querySelectorAll('button')).find((b) => b.textContent === 'show results and artifacts');
  assert.ok(showBtn);
  showBtn.click();
  await until(() => older.querySelector('.results'), 'the button never loaded results');
  assert.ok(older.querySelector('.resultsLink'));
  assert.deepEqual(requested.slice(olderFrom), ['/api/turn/tk-old']);

  /* ---- bug B (fix-weih.18): only a page pairs its lists into one table ---- */
  const shapes = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'page_shape_cases.json'), 'utf8'));
  for (const c of shapes) {
    assert.equal(w.tmIsPageOutput(c.output), c.is_page, 'tmIsPageOutput: ' + c.name);
  }
  const render = (output) => w.tmRenderOutputView(output, null, 0);
  const cells = (view) => Array.from(view.querySelectorAll('tbody tr')).map((tr) =>
    Array.from(tr.children).map((td) => td.textContent));
  const unrelated = {owners: ['Ann', 'Bo'], managers: ['Cy', 'Di']};
  const separate = render(unrelated);
  assert.equal(separate.querySelectorAll('table').length, 0, 'unrelated equal-length lists make no table');
  assert.equal(separate.querySelectorAll('ul.rList').length, 2, 'each list renders on its own');
  assert.deepEqual(Array.from(separate.querySelectorAll('ul.rList')).map((ul) => text(ul)), ['Ann Bo', 'Cy Di']);
  /* A field named total decides nothing in the panel [fix-sush step 3]: with an integer total the same
     lists are still two plain lists; only a declared ui.table group makes the table. The server's
     output_page (tmIsPageOutput, still used by the answer card until step 4) is unchanged. */
  const paged = render(Object.assign({total: 2}, unrelated));
  assert.equal(paged.querySelectorAll('table').length, 0, 'an integer total no longer makes the panel draw a table');
  assert.equal(paged.querySelectorAll('ul.rList').length, 2, 'the lists are drawn as plain lists');
  const declaredSchema = {properties: {owners: {type: 'array', items: {type: 'string'}, ui: {table: 'people'}},
    managers: {type: 'array', items: {type: 'string'}, ui: {table: 'people'}}, total: {type: 'integer'}}};
  assert.deepEqual(cells(w.tmRenderOutputView(Object.assign({total: 2}, unrelated), declaredSchema, 0)),
    [['Ann', 'Cy'], ['Bo', 'Di']], 'a declared table group pairs its lists, one row per index');
  assert.equal(w.tmIsPageOutput(Object.assign({total: 2}, unrelated)), true, 'the card still reads the page rule');
  assert.equal(w.tmCardIsPage(Object.assign({total: 2}, unrelated)), true, 'the answer card still tables them (until fix-sush step 4)');
  assert.equal(render({owners: ['Ann', 'Bo'], managers: ['Cy'], total: 2}).querySelectorAll('table').length, 0,
    'lists that differ in length have no combined table');
  assert.equal(render({owners: ['Ann'], managers: ['Cy'], total: true}).querySelectorAll('table').length, 0,
    'a boolean total is not a table');
  assert.equal(render({owners: ['Ann'], managers: ['Cy'], total: '1'}).querySelectorAll('table').length, 0,
    'a string total is not a table');
  /* an array of objects keeps its own table either way */
  assert.equal(render({rows: [{a: 1}, {a: 2}], ids: ['x', 'y']}).querySelectorAll('table').length, 1);

  if (errors.length) throw new Error(errors.join('\n'));
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
