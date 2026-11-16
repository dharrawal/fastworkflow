/* The chat SPA forwards the parts of a split turn in the compose request. The page's own tmCardPayload,
   tmCardParts and tmArtifactsIn run against the assembled page; only window.fetch is replaced (it records the
   /compose_answer_card request). The parts are read from the final answer's artifacts (fw_sub_tasks), forwarded as
   they are, absent from the request when the turn has none, and never drawn as an artifact. */
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

const calls = [];
const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.alert = () => {};
    window.fetch = (url, opts) => {
      url = String(url);
      if (url.endsWith('/compose_answer_card')) {
        calls.push({url, body: JSON.parse(opts.body)});
        return Promise.resolve({ok: true, status: 200, json: () => Promise.resolve({card: null, reason: 'not_needed'})});
      }
      return new Promise(() => {});
    };
  },
});
const w = dom.window;
const tm = w.tm;

function plain(value) { return JSON.parse(JSON.stringify(value)); }
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function until(fn, what) {
  for (let i = 0; i < 200; i++) { if (fn()) return; await tick(); }
  throw Error('Timed out: ' + what);
}
function co(name, output, extra) {
  return Object.assign({
    command_name: name, context: 'Library', command_parameters: null,
    command_response: {success: true, response: 'ok', output},
  }, extra || {});
}
const ANSWER = '### 1. find the book\nDune is on shelf 4.\n\n### 2. check the fine\nThe fine is 12.';
const PARTS = [
  {number: 1, text: 'find the book', title: 'find the book', answered: true, start: 25, end: 45},
  {number: 2, text: 'check the fine', title: 'check the fine', answered: true, start: 70, end: 83},
  {number: 3, text: 'say hello', title: 'say hello', answered: false, start: null, end: null},
];
function finalOutput(artifacts) {
  return {command_name: '', command_parameters: null,
    command_response: {success: true, response: ANSWER, artifacts: artifacts || {}}};
}
function turn(outputs) {
  return {turn_key: 'k' + Math.random().toString(16).slice(2), status: 'completed', success: true,
    answer: ANSWER, command_outputs: outputs};
}
function results() {
  return [co('find_book', {total: 1}), co('get_fine', {amount: 12})];
}

async function main() {
  for (const fn of ['tmCardParts', 'tmCardPayload', 'tmArtifactsIn']) assert.equal(typeof w[fn], 'function', fn);
  assert.equal(w.TM_TURN_PARTS_KEY, 'fw_sub_tasks');
  tm.baseUrl = 'http://127.0.0.1:39131';
  tm.connected = true;
  const msg = w.tmBubble('agent', '...');
  msg.dataset.userMessage = 'two things';
  const payloadOf = (out) => w.tmCardPayload(msg, out, w.tmResultsOf(out.command_outputs, {}), {});

  /* present: forwarded as recorded, under "parts" */
  const withParts = turn([...results(), finalOutput({fw_sub_tasks: PARTS})]);
  assert.deepEqual(plain(payloadOf(withParts).parts), PARTS);

  /* absent: no "parts" key at all, and the rest of the request is the same */
  const without = turn([...results(), finalOutput({})]);
  const bare = payloadOf(without);
  assert.ok(!('parts' in bare), 'no parts key when the turn has none');
  const sent = payloadOf(withParts);
  delete sent.parts;
  delete sent.turn_key;
  delete bare.turn_key;
  assert.deepEqual(plain(sent), plain(bare), 'nothing else in the request depends on the parts');
  [null, undefined, [], 'x', 5, {}, {__fw_artifact_ref__: 'abc', size: 300000}].forEach((odd) =>
    assert.ok(!('parts' in payloadOf(turn([...results(), finalOutput({fw_sub_tasks: odd})]))), 'not sent: ' + JSON.stringify(odd)));
  assert.ok(!('parts' in payloadOf(turn([]))), 'a turn with no outputs');

  /* the real request of a painted turn carries them */
  calls.length = 0;
  const live = w.tmBubble('agent', '...');
  live.dataset.userMessage = 'two things';
  w.tmRenderTurn(live, turn([...results(), finalOutput({fw_sub_tasks: PARTS})]));
  await until(() => calls.length === 1, 'a compose request with parts');
  assert.deepEqual(plain(calls[0].body.parts), PARTS);
  calls.length = 0;
  const plainLive = w.tmBubble('agent', '...');
  plainLive.dataset.userMessage = 'two things';
  w.tmRenderTurn(plainLive, turn([...results(), finalOutput({})]));
  await until(() => calls.length === 1, 'a compose request without parts');
  assert.ok(!('parts' in calls[0].body));

  /* the parts are data, not an artifact the user is shown */
  const shown = w.tmArtifactsIn([finalOutput({fw_sub_tasks: PARTS, report: {a: 1}})]).map((a) => a.key);
  assert.deepEqual(plain(shown), ['report']);

  if (errors.length) throw new Error(errors.join('\n'));
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
