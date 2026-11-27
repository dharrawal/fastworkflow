/* ui.tones and ui.currency (fix-weih.29, fix-weih.30). The page's own tmFormatValue, tmAutoTone and
   tmRenderOutputView are exercised against the assembled page with schemas that carry the hints. An output
   with neither hint renders as it always did. */
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
const dom = new JSDOM(html, {
  runScripts: 'dangerously', url: 'http://127.0.0.1:39131/', virtualConsole,
  beforeParse(window) { window.fetch = () => new Promise(() => {}); },
});
const w = dom.window;

const badge = (val, ui) => w.tmFormatValue(val, {ui}, 'order_state');
const tone = (node) => (/tone-(\w+)/.exec(node.className) || [])[1];
const TONES = {delivered: 'good', cancelled: 'bad', 'on hold': 'warning', sent: 'neutral'};

/* an author's tone is used for a value the generic vocabulary does not know */
assert.equal(tone(badge('delivered', {format: 'badge'})), 'neutral', 'no map: neutral as today');
assert.equal(tone(badge('delivered', {format: 'badge', tones: TONES})), 'good');
assert.equal(tone(badge('cancelled', {format: 'badge', tones: TONES})), 'bad');
assert.equal(tone(badge('on hold', {format: 'badge', tones: TONES})), 'warning');
/* the author's tone wins over the generic vocabulary */
assert.equal(tone(badge('failed', {format: 'badge'})), 'bad', 'generic vocabulary as today');
assert.equal(tone(badge('failed', {format: 'badge', tones: {failed: 'neutral'}})), 'neutral');
assert.equal(tone(badge('pending', {format: 'badge', tones: {pending: 'good'}})), 'good');
/* a value not in the map falls back to the generic vocabulary */
assert.equal(tone(badge('failed', {format: 'badge', tones: TONES})), 'bad');
assert.equal(tone(badge('Done', {format: 'badge', tones: TONES})), 'good');
assert.equal(tone(badge('shipped', {format: 'badge', tones: TONES})), 'neutral');
/* keys match case-insensitively after trimming; the badge text is the value as given */
assert.equal(tone(badge('  DELIVERED ', {format: 'badge', tones: TONES})), 'good');
assert.equal(tone(badge('delivered', {format: 'badge', tones: {' Delivered ': 'good'}})), 'good');
assert.equal(badge('  DELIVERED ', {format: 'badge', tones: TONES}).textContent, '  DELIVERED ');
/* an unknown tone word in a hand-edited schema is ignored (generic vocabulary) */
assert.equal(tone(badge('failed', {format: 'badge', tones: {failed: 'purple'}})), 'bad');
/* tones without format badge change nothing: no badge is drawn, whatever the field is named */
assert.equal(w.tmFormatValue('delivered', {ui: {tones: TONES}}, 'status'), 'delivered');

/* an output field: the stat box and the badge carry the author's tone */
const out = {state: 'delivered', note: 'x'};
const schema = {type: 'object', properties: {
  state: {type: 'string', ui: {format: 'badge', tones: TONES}}, note: {type: 'string'}}};
let view = w.tmRenderOutputView(out, schema, 0);
assert.ok(view.querySelector('.rBadge.tone-good'), 'the output badge is toned by the author');
assert.equal(view.querySelectorAll('.cardStat.tone-good').length, 1, 'and so is its stat box');
/* a list of badges */
view = w.tmRenderOutputView({states: ['delivered', 'cancelled']}, {type: 'object', properties: {
  states: {type: 'array', items: {type: 'string'}, ui: {format: 'badge', tones: TONES}}}}, 0);
assert.deepEqual(Array.from(view.querySelectorAll('.rBadge')).map(tone), ['good', 'bad']);
/* a table cell (the table is declared: a field named total no longer makes one [fix-sush step 3]) */
view = w.tmRenderOutputView({total: 2, names: ['a', 'b'], states: ['delivered', 'on hold']}, {type: 'object',
  properties: {total: {type: 'integer'}, names: {type: 'array', items: {type: 'string'}, ui: {table: 't'}},
    states: {type: 'array', items: {type: 'string'}, ui: {format: 'badge', tones: TONES, table: 't'}}}}, 0);
assert.deepEqual(Array.from(view.querySelectorAll('table td .rBadge')).map(tone), ['good', 'warning']);

/* currency: a code gives the browser's currency format for that code, whatever the locale */
const plain = (n) => Number(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const money = (val, ui) => w.tmFormatValue(val, {ui}, 'total');
const intl = (code, n) => new Intl.NumberFormat(undefined, {style: 'currency', currency: code}).format(n);
assert.equal(money(1234.5, {format: 'currency'}), plain(1234.5), 'no code: grouped, two decimals, no symbol');
assert.equal(money(1234.5, {format: 'currency', currency: 'EUR'}), intl('EUR', 1234.5));
assert.equal(money(1234.5, {format: 'currency', currency: 'eur'}), intl('EUR', 1234.5), 'lower case is normalised');
assert.notEqual(money(1234.5, {format: 'currency', currency: 'EUR'}), plain(1234.5), 'the code adds a currency mark');
assert.equal(money(0, {format: 'currency', currency: 'USD'}), intl('USD', 0));
/* a code the browser rejects, or a malformed one: the plain number, no exception */
assert.equal(money(1234.5, {format: 'currency', currency: 'EU'}), plain(1234.5));
assert.equal(money(1234.5, {format: 'currency', currency: 'E1R'}), plain(1234.5));
assert.equal(money(1234.5, {format: 'currency', currency: 7}), plain(1234.5));
/* a code without format currency changes nothing */
assert.equal(money(1234.5, {currency: 'EUR'}), Number(1234.5).toLocaleString());
assert.equal(w.tmFormatValue(12, {ui: {format: 'percent', currency: 'EUR'}}, 'rate'), Number(12).toLocaleString() + '%');
/* a field, a list and a table cell */
view = w.tmRenderOutputView({total: 99.9, prices: [1, 2.5]}, {type: 'object', properties: {
  total: {type: 'number', ui: {format: 'currency', currency: 'USD'}},
  prices: {type: 'array', items: {type: 'number'}, ui: {format: 'currency', currency: 'USD'}}}}, 0);
assert.ok(view.textContent.indexOf(intl('USD', 99.9)) >= 0, 'the field shows the currency string');
assert.ok(view.textContent.indexOf(intl('USD', 2.5)) >= 0, 'a list item too');

/* neither hint: nothing changed */
view = w.tmRenderOutputView({total: 1234.5, status: 'delivered'}, {type: 'object', properties: {
  total: {type: 'number', ui: {format: 'currency'}}, status: {type: 'string'}}}, 0);
assert.ok(view.textContent.indexOf(plain(1234.5)) >= 0);
assert.ok(!view.querySelector('.rBadge'), 'no hint, no badge: a field name never decides');
assert.ok(view.textContent.indexOf('delivered') >= 0);

if (errors.length) throw new Error(errors.join('\n'));
