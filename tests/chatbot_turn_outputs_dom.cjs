/* The chat client finds the schema of every real turn output (fix-weih.19).
   tests/fixtures/chat_turn_outputs.json is asserted equal to what the real
   server code produces by tests/test_chat_turn_outputs_contract.py; here the
   same recorded command names and contexts go through the page's own lookups
   (tmResultsOf, tmResultSchema, tmCommandSchema) against a listing with the
   workflow's real keys. Each schema's properties are titled with their command's
   key, so a result resolving to a different command's schema is caught. */
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
const dom = new JSDOM(html, {
  runScripts: 'dangerously', url: 'http://127.0.0.1:39131/', virtualConsole,
  beforeParse(window) { window.fetch = () => new Promise(() => {}); },
});
const w = dom.window;

const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'chat_turn_outputs.json'), 'utf8'));

function listing(spec) {
  const out = {};
  Object.keys(spec.schemas).forEach((key) => {
    const properties = {};
    spec.schemas[key].output_properties.forEach((p) => { properties[p] = {type: 'string', title: key + ':' + p}; });
    out[key] = {kind: spec.schemas[key].kind, input: null,
      output: spec.schemas[key].output_properties.length ? {properties} : null};
  });
  return out;
}

let checked = 0;
let requests = 0;
for (const c of data.cases) {
  const spec = data.workflows[c.case.split(':')[0]];
  const schemas = listing(spec);
  for (const row of c.outputs) {
    const label = c.case + ' / ' + row.command_name;
    if (row.output_keys.length) {
      const output = {};
      row.output_keys.forEach((k) => { output[k] = 'v'; });
      const outputs = [{command_name: row.command_name, context: row.context, command_parameters: null,
        command_response: {success: true, response: 'ok', output}}];
      const results = w.tmResultsOf(outputs, schemas);
      assert.equal(results.length, 1, label);
      const schema = w.tmResultSchema(results[0], schemas);
      assert.ok(schema, 'no schema found for ' + label);
      assert.equal(schema, schemas[row.command_name].output, 'wrong schema for ' + label);
      row.output_keys.forEach((k) => {
        assert.equal(schema.properties[k].title, row.command_name + ':' + k, label);
      });
      checked++;
    }
    if (row.parameter_request_command) {
      const entry = w.tmCommandSchemaIn(schemas, row.parameter_request_command);
      assert.equal(entry, schemas[row.parameter_request_command], 'parameter request: ' + label);
      requests++;
    }
  }
}
assert.ok(checked >= 6, 'the fixture covers real results: ' + checked);
assert.ok(requests >= 2, 'the fixture covers failed validation: ' + requests);

/* A bare leaf owned by several contexts stays unresolved, whatever the record
   label says: a wrong schema mislabels fields, none renders plain. */
const todo = listing(data.workflows.todo);
assert.equal(w.tmCommandSchemaIn(todo, 'get_properties'), null);
const bare = [{command_name: 'get_properties', context: 'TodoList', command_parameters: null,
  command_response: {success: true, response: 'ok', output: {description: 'x'}}}];
assert.equal(w.tmResultSchema(w.tmResultsOf(bare, todo)[0], todo), null);
/* a bare global command is never qualified by the current context's label */
const global = {library_hours: {kind: 'execution', input: null, output: {properties: {a: {}}}},
  'Library/library_hours': {kind: 'execution', input: null, output: {properties: {b: {}}}}};
const hours = [{command_name: 'library_hours', context: 'Library', command_parameters: null,
  command_response: {success: true, response: 'ok', output: {a: 1}}}];
assert.equal(w.tmResultSchema(w.tmResultsOf(hours, global)[0], global), global.library_hours.output);

if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
process.exit(0);
