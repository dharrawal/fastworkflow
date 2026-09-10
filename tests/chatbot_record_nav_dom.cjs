/* The four arrows of the record navigator, driven through the real page
   against a real server like the rail's own DOM test next door. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], turnKeys = process.argv.slice(4);
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  const detail = d.getElementById('detail');
  async function until(fn) {
    for (let i = 0; i < 150; i++) { if (fn()) return; await new Promise(r => setTimeout(r, 50)); }
    throw Error('Timed out: ' + detail.textContent + ' | errors: ' + JSON.stringify(errors));
  }

  const arrow = name => d.getElementById('recordNav' + name);
  const enabled = () => ['Up', 'Down', 'Prev', 'Next']
    .filter(name => !arrow(name).disabled).join(',');
  const press = name => { assert.ok(!arrow(name).disabled, name + ' is disabled'); arrow(name).click(); };
  const heading = () => (detail.querySelector('.levelHead h2') || {}).textContent || '';
  const crumbs = () => (detail.querySelector('nav.crumbs') || {}).textContent || '';

  d.getElementById('modeDebug').click();
  await until(() => d.querySelectorAll('#convList summary').length);

  // Nothing is selected yet, so the only move with a meaning is inward.
  assert.equal(d.getElementById('recordNav').className, 'visible');
  assert.equal(enabled(), 'Down');

  // Down the rail: date -> conversation -> the conversation's first turn.
  press('Down');
  await until(() => crumbs().includes('2026-09-08'));
  assert.equal(enabled(), 'Up,Down');   // one date, so no siblings either side
  press('Down');
  await until(() => detail.textContent.includes('3 turns'));
  press('Down');
  await until(() => heading().includes(turnKeys[0]));

  // First of three siblings: nothing before it, the next two after it.
  await until(() => !arrow('Down').disabled);   // the trace arrives over HTTP
  assert.equal(enabled(), 'Up,Down,Next');

  // Next walks across all three sibling turns.
  press('Next');
  await until(() => heading().includes(turnKeys[1]));
  assert.equal(enabled(), 'Up,Down,Prev,Next');

  press('Next');
  await until(() => heading().includes(turnKeys[2]));
  assert.equal(enabled(), 'Up,Down,Prev');   // last sibling: nowhere further on

  // Down reaches the trace components the right pane walks, and Up climbs back
  // out of them by the same route the crumbs describe.
  press('Down');
  await until(() => crumbs().includes('Planning'));
  assert.equal(heading(), 'Planning');
  assert.equal(enabled(), 'Up');   // a lone component with no children

  press('Up');
  await until(() => heading().includes(turnKeys[2]));
  press('Up');
  await until(() => detail.textContent.includes('3 turns'));
  assert.ok(!crumbs().includes(turnKeys[2]));

  // The arrows belong to the record hierarchy, so they go away with it.
  d.getElementById('modeTest').click();
  assert.equal(d.getElementById('recordNav').className, '');
  d.getElementById('modeDebug').click();
  await until(() => d.getElementById('recordNav').className === 'visible');

  // Switching modes leaves session and health requests in flight; let them land
  // before the window goes away, or they resolve against a torn-down document.
  await new Promise(r => setTimeout(r, 500));
  assert.deepEqual(errors, []);
  w.close();
  process.exit(0);
})().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
