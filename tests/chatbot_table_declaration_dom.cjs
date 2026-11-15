/* ui.table: an author's declaration of which list fields of one output are the columns of one table
   (fix-weih.23). The page's own tmRenderOutputView is exercised against the assembled page with a schema
   that carries the declaration. Only a declared group makes a table in the panel; a field named total decides
   nothing [fix-sush step 3]. */
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

/* A schema as GET /command_schemas serves an output: properties in field order; ui is the hint. */
function schema(spec) {
  const properties = {};
  Object.keys(spec).forEach((key) => {
    const prop = {type: 'array', items: {type: 'string'}};
    if (spec[key]) prop.ui = spec[key];
    properties[key] = prop;
  });
  return {type: 'object', properties};
}
const render = (output, spec) => w.tmRenderOutputView(output, spec ? schema(spec) : null, 0);
const cells = (table) => Array.from(table.querySelectorAll('tbody tr')).map((tr) =>
  Array.from(tr.children).map((td) => td.textContent));
const heads = (table) => Array.from(table.querySelectorAll('th')).map((th) => th.textContent);
const bullets = (view) => Array.from(view.querySelectorAll('ul.rList')).map((ul) => ul.textContent);

/* two parallel lists, no total: one table */
let view = render({fields: ['a', 'b'], values: ['1', '2']}, {fields: {table: 't'}, values: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 1, 'two declared lists with no total make one table');
assert.deepEqual(cells(view.querySelector('table')), [['a', '1'], ['b', '2']]);
assert.deepEqual(heads(view.querySelector('table')), ['Fields', 'Values']);
assert.equal(view.querySelectorAll('ul.rList').length, 0);
/* the same output, undeclared: bullets, as today */
view = render({fields: ['a', 'b'], values: ['1', '2']}, {fields: null, values: null});
assert.equal(view.querySelectorAll('table').length, 0, 'no declaration, no table');
assert.equal(view.querySelectorAll('ul.rList').length, 2);

/* two groups of different lengths: two tables */
const two = {names: ['a', 'b', 'c'], hours: ['1', '2', '3'], codes: ['x', 'y'], notes: ['p', 'q']};
const twoSpec = {names: {table: 'one'}, hours: {table: 'one'}, codes: {table: 'two'}, notes: {table: 'two'}};
view = render(two, twoSpec);
let tables = view.querySelectorAll('table');
assert.equal(tables.length, 2, 'two groups of different lengths are two tables');
assert.deepEqual(cells(tables[0]), [['a', '1'], ['b', '2'], ['c', '3']]);
assert.deepEqual(cells(tables[1]), [['x', 'p'], ['y', 'q']]);

/* two groups of EQUAL length: still two tables, never one wrong table (the coincidence case) */
view = render({names: ['a', 'b'], hours: ['1', '2'], codes: ['x', 'y'], notes: ['p', 'q']}, twoSpec);
tables = view.querySelectorAll('table');
assert.equal(tables.length, 2, 'equal-length groups stay two tables');
assert.deepEqual(heads(tables[0]), ['Names', 'Hours']);
assert.deepEqual(heads(tables[1]), ['Codes', 'Notes']);
assert.deepEqual(cells(tables[1]), [['x', 'p'], ['y', 'q']]);

/* a hidden member is a column for alignment but is not drawn */
view = render({row_id: ['r1', 'r2'], name: ['a', 'b'], hours: ['1', '2']},
  {row_id: {table: 't', hidden: true}, name: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 1);
assert.deepEqual(heads(view.querySelector('table')), ['Name', 'Hours']);
assert.deepEqual(cells(view.querySelector('table')), [['a', '1'], ['b', '2']]);
assert.ok(!/r1/.test(view.textContent), 'the hidden column is not drawn');
/* a hidden member of another length still breaks the alignment: no table, hidden stays hidden */
view = render({row_id: ['r1'], name: ['a', 'b'], hours: ['1', '2']},
  {row_id: {table: 't', hidden: true}, name: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 0);
assert.ok(!/r1/.test(view.textContent));

/* unequal member lengths: bullets, nothing trimmed or padded */
view = render({names: ['a', 'b', 'c'], hours: ['1', '2']}, {names: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 0, 'unequal lengths make no table');
assert.deepEqual(bullets(view), ['abc', '12']);
/* a missing member, a null member, a non-list member, an empty group */
view = render({names: ['a', 'b']}, {names: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 0, 'a missing member makes no table');
assert.deepEqual(bullets(view), ['ab']);
view = render({names: ['a', 'b'], hours: null}, {names: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 0, 'a null member makes no table');
assert.ok(bullets(view).indexOf('ab') >= 0);
view = render({names: [], hours: []}, {names: {table: 't'}, hours: {table: 't'}});
assert.equal(view.querySelectorAll('table').length, 0, 'zero rows make no table');

/* a total beside lists split into two declared groups: two tables */
view = render({total: 4, names: ['a', 'b'], hours: ['1', '2'], codes: ['x', 'y'], notes: ['p', 'q']}, twoSpec);
assert.equal(view.querySelectorAll('table').length, 2, 'declared groups make the tables');

/* mixed: a declared group beside undeclared lists; the undeclared lists stay plain lists. */
view = render({names: ['a', 'b'], hours: ['1', '2'], owners: ['Ann', 'Bo'], managers: ['Cy', 'Di']},
  {names: {table: 't'}, hours: {table: 't'}, owners: null, managers: null});
assert.equal(view.querySelectorAll('table').length, 1, 'an undeclared pair of equal length is not tabled (no total)');
assert.deepEqual(bullets(view), ['AnnBo', 'CyDi']);
view = render({total: 2, names: ['a', 'b'], hours: ['1', '2'], owners: ['Ann', 'Bo'], managers: ['Cy', 'Di']},
  {names: {table: 't'}, hours: {table: 't'}, owners: null, managers: null});
tables = view.querySelectorAll('table');
assert.equal(tables.length, 1, 'a total makes the undeclared rest no table in the panel (it was a page before fix-sush step 3)');
assert.deepEqual(heads(tables[0]), ['Names', 'Hours']);
assert.deepEqual(bullets(view), ['AnnBo', 'CyDi'], 'a declared column is never paired with an undeclared one');
/* a declared column and a lone undeclared list of the same length are never paired */
view = render({total: 2, names: ['a', 'b'], hours: ['1', '2'], extra: ['u', 'v']},
  {names: {table: 't'}, hours: {table: 't'}, extra: null});
tables = view.querySelectorAll('table');
assert.equal(tables.length, 1);
assert.deepEqual(heads(tables[0]), ['Names', 'Hours']);
assert.deepEqual(bullets(view), ['uv']);

/* a format on a column still applies (the existing cell renderer) */
view = render({code: ['a-1', 'b-2'], hours: ['1', '2']},
  {code: {table: 't', format: 'code'}, hours: {table: 't'}});
assert.ok(view.querySelector('table td code'), 'the code format reaches the cells');

/* with no schema nothing is declared */
assert.equal(render({fields: ['a'], values: ['1']}, null).querySelectorAll('table').length, 0);

if (errors.length) throw new Error(errors.join('\n'));
