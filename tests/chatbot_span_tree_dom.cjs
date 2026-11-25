/* The debug page draws a turn by its stages (span contract 63, bd
 * fix-qjne.34: fw.split, fw.resolve, fw.run > fw.run.part > fw.step >
 * fw.command, fw.answer), through the real page against a real server. A span
 * node's feedback anchor is its own span id; the turn-wide record (what was
 * recorded, the execution ledger) sits on the Run stage. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], newTurn = process.argv[4];
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn, what) {
    for (let i = 0; i < 200; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + what + ' | detail=' +
      d.getElementById('detail').textContent.slice(0, 800) + ' | errors: ' + JSON.stringify(errors));
  }
  const rows = () => [...d.querySelectorAll('#detail .wfRow')]
    .map(r => r.getAttribute('aria-label').replace(/^Inspect /, ''));
  const row = label => [...d.querySelectorAll('#detail .wfRow')]
    .find(r => r.getAttribute('aria-label').startsWith('Inspect ' + label));
  const heading = () => (d.querySelector('#detail .levelHead h2') || {}).textContent || '';
  async function open(label) {
    const target = row(label);
    assert.ok(target, 'a row "' + label + '" among ' + JSON.stringify(rows()));
    target.click();
    await until(() => heading().startsWith(label), 'the level ' + label);
  }

  /* ---- a new turn: its stages, parts, steps and commands --------------- */
  w.selectTurn(newTurn);
  await until(() => rows().length && row('Split'), 'the new turn to open');
  assert.deepEqual(rows().map(label => label.split(' · ')[0]), ['Split', 'Resolve', 'Run', 'Answer'],
    'the new turn is drawn by its stages: ' + JSON.stringify(rows()));
  assert.ok(rows().some(label => label.startsWith('Answer')), 'the answer stage');
  await open('Answer');
  assert.ok(rows().some(label => label === 'LLM call · conversation summary'),
    'the summary call is labelled by its purpose: ' + JSON.stringify(rows()));
  w.selectTurn(newTurn);
  await until(() => row('Run'), 'the new turn again');
  await open('Run');
  assert.deepEqual(Array.from(w.feedbackAnchor(w.state.path[w.state.path.length - 1])), ['n-run'],
    'a stage node anchors on its own span');
  assert.ok(d.querySelector('#detail details.ledgerDisclosure'), 'the execution ledger sits on the Run stage');
  assert.ok(rows()[0].startsWith('Part 1 · find_person · ran'), 'the run part: ' + JSON.stringify(rows()));
  await open('Part 1');
  const partRows = rows().map(label => label.split(' · ')[0]);
  assert.deepEqual(partRows, ['Planning', 'Step 1', 'Question'], 'the part holds its plan, step and question: '
    + JSON.stringify(rows()));
  assert.ok(rows()[1].indexOf('find_person') !== -1, 'the step shows its command: ' + rows()[1]);
  await open('Step 1');
  assert.ok(rows().some(label => label.startsWith('Command · find_person')),
    'the step holds its command: ' + JSON.stringify(rows()));
  assert.ok(rows().some(label => label === 'LLM call · agent step'), 'and its agent call: ' + JSON.stringify(rows()));
  assert.deepEqual(Array.from(w.feedbackAnchor(w.state.path[w.state.path.length - 1])), ['n-step'],
    'a step node anchors on its own span');

  assert.deepEqual(errors, []);
  process.exit(0);
})().catch(e => {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
