const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3], eid = process.argv[4];
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });
(async () => {
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;
  async function until(fn) {
    for (let i=0; i<150; i++) { if (fn()) return; await new Promise(r=>setTimeout(r,50)); }
    throw Error('Timed out: '+ d.getElementById('detail').textContent);
  }
  const summaries = () => [...d.querySelectorAll('#convList summary')];
  const find = text => summaries().find(e => e.textContent.includes(text));
  const click = text => { const e=find(text); assert.ok(e, 'Missing node '+text); e.click(); };
  await until(()=>find('Tuning benchmark'));
  assert.equal(d.querySelector('#experimentsBtn'),null);
  assert.equal(d.querySelector('#benchmarksBtn'),null);
  assert.equal(d.querySelector('#fStatus'),null);
  assert.equal(d.querySelector('#rail nav.crumbs'),null);
  click('Tuning benchmark');
  await until(()=>d.getElementById('detail').textContent.includes('Review this benchmark'));
  click('Recorded experiment');
  await until(()=>d.getElementById('detail').textContent.includes('Analysis'));
  const exp = find('Recorded experiment').parentElement;
  assert.ok(exp.open);
  click('task'); // conversation node
  assert.ok(d.getElementById('detail').textContent.includes('1 turns'));
  assert.equal(w.benchmarkExperimentSource, eid);
  assert.equal(find('experiment-turn'), undefined);
  [...d.querySelectorAll('#detail button')].find(e=>e.textContent.includes('experiment-turn')).click();
  await until(()=>d.getElementById('detail').textContent.includes('Human feedback'));
  await until(()=>[...d.querySelectorAll('#detail .wfRow')].some(e=>e.textContent.includes('Planning')));
  [...d.querySelectorAll('#detail .wfRow')].find(e=>e.textContent.includes('Planning')).click();
  assert.equal(find('Planning'), undefined);
  assert.ok(find('task').parentElement.classList.contains('selected'));
  assert.ok(d.querySelector('#detail nav.crumbs').textContent.startsWith('Benchmarks'));
  const crumbs = d.querySelector('#detail nav.crumbs').textContent;
  for (const label of ['Tuning benchmark','Recorded experiment','task','experiment-turn','Planning']) assert.ok(crumbs.includes(label), crumbs);
  assert.ok(d.getElementById('detail').textContent.includes('Human feedback'));
  await w.refreshConvs();
  assert.ok(d.querySelector('#detail nav.crumbs').textContent.includes('Planning'));
  assert.equal(find('Planning'), undefined);
  // Return through the full right-pane breadcrumb, then leave a trace fetch pending.
  [...d.querySelectorAll('#detail nav.crumbs button')].find(e=>e.textContent==='task').click();
  [...d.querySelectorAll('#detail button')].find(e=>e.textContent.includes('experiment-turn')).click();
  click('ad-hoc conversations');
  click('2026-09-07');
  assert.ok(d.getElementById('detail').textContent.includes('UTC'));
  const day = find('2026-09-07').parentElement;
  assert.ok(day.textContent.includes('Conversation #1'));
  assert.ok(!day.textContent.includes('plain yesterday'));
  assert.ok(!day.textContent.includes('plain today'));
  await new Promise(r=>setTimeout(r,250));
  assert.ok(d.getElementById('detail').textContent.includes('2026-09-07'));
  assert.deepEqual(errors,[]);
  dom.window.close();
})().catch(e=>{ process.stderr.write(e.stack+'\n'); process.exit(1); });
