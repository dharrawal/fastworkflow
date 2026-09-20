/* Every finished run of a task, in a real DOM (`fix-9eg.3.2.2.1`/`.2`).
 *
 * The evidence is three tasks of one experiment, recorded as metadata only:
 *
 *   task        attempts 1 and 2 finished, attempt 3 still going
 *   empty_task  one run still going, nothing finished
 *   big_task    twenty-one finished runs, one more than may be summarized
 *
 * Three phases, chosen by argv[6]:
 *
 *   all        the rule's own button, the task's counts, and the members the
 *              SERVER resolved -- nothing here is assembled from the rows
 *   drift      a run finishes under the page; the explicit check says so
 *              while the membership on screen stays exactly as it was, and
 *              an explicit Refresh is what replaces it
 *   overlimit  a task above the bound is refused whole, with its counts
 *
 * Every request is real and so is every answer. */
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');
const url = process.argv[3];
const experiment = process.argv[4], task = process.argv[5];
const phase = process.argv[6] || 'all';
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', e => { if (e.type !== 'css-parsing') errors.push(e.message); });

(async () => {
  /* jsdom ships no `fetch`, so the page gets node's -- unwrapped, because
     nothing here is about timing. Every request and every answer is real. */
  const dom = await JSDOM.fromURL(url, {runScripts: 'dangerously',
    virtualConsole: console,
    beforeParse(window) {
      window.fetch = (path, options) => fetch(new URL(path, url), options);
    }});
  const w = dom.window, d = w.document;

  async function until(fn, what) {
    for (let i = 0; i < 300; i++) {
      const value = fn();
      if (value) return value;
      await new Promise(r => setTimeout(r, 50));
    }
    /* The page's own errors go in the message: a wait that ends in a timeout
       is usually a script that threw before it could do the thing, and the
       timeout alone says only that it never happened. */
    throw Error('Timed out waiting for ' + what + '. pageErrors=['
      + errors.join(' | ') + '] detail='
      + d.getElementById('detail').textContent.slice(0, 1500));
  }
  const wait = ms => new Promise(r => setTimeout(r, ms));
  const detail = () => d.getElementById('detail').textContent;
  const panel = () => d.querySelector('#detail [data-selected-runs]');
  const at = name => d.querySelector('#detail [data-selected-runs-' + name + ']');
  const textOf = name => { const n = at(name); return n ? n.textContent : ''; };
  const bodyText = () => textOf('body');
  const noticeText = () => textOf('notice');
  const members = () =>
    [...(at('body') ? at('body').querySelectorAll('[data-selected-member]') : [])]
      .map(node => node.getAttribute('data-selected-member'));

  d.getElementById('modeDebug').click();
  await until(() => w.session && w.hierarchyRoot, 'the session and the hierarchy');

  async function openRuns() {
    for (let attempt = 0; attempt < 8; attempt++) {
      w.taskView = 'runs';
      w.showExperimentTask(experiment, task, 'All finished runs');
      for (let i = 0; i < 20; i++) {
        if (panel() && at('all')) return;
        await wait(50);
      }
    }
    throw Error('the runs view never offered the panel. detail='
      + detail().slice(0, 1500));
  }
  await openRuns();

  if (phase === 'all') {
    /* ==============================================================
     * The rule is offered without anything being ticked
     * ============================================================== */
    assert.ok(at('all'), 'the whole-task button is on the panel');
    assert.ok(!at('all').disabled,
      'and is live with nothing selected, because it does not depend on the '
      + 'ticks: it asks the server for every finished run');
    assert.ok(at('submit').disabled,
      'while the explicit summarize button still needs a selection');
    assert.equal(bodyText(), '', 'nothing is summarized until somebody asks');
    assert.ok(!at('check') || at('check').hidden,
      'and there is nothing to check for changes against yet');

    /* ==============================================================
     * The members are the server's answer, not the page's reading of rows
     * ============================================================== */
    at('all').click();
    await until(() => bodyText().includes('run(s) asked for'), 'the summary');
    assert.deepEqual(members(), ['1', '2'],
      'the two finished runs are the members: ' + members().join(','));
    assert.ok(bodyText().includes('2 run(s) asked for · 2 in these counts'),
      'stated as counts over the resolved population: '
      + bodyText().slice(0, 400));

    /* ==============================================================
     * The task's own counts, beside what the summary is over
     * ============================================================== */
    const population = d.querySelector('#detail [data-task-population]');
    assert.ok(population, 'the task population line is shown');
    assert.ok(population.textContent.includes('3 run(s) recorded'),
      'every recorded run of the task: ' + population.textContent);
    assert.ok(population.textContent.includes('2 finished'),
      'the finished ones: ' + population.textContent);
    assert.ok(population.textContent.includes('1 still running'),
      'and the one that is not a member because it is not finished: '
      + population.textContent);
    assert.ok(population.textContent.includes('3 planned for this task'),
      'with the per-task plan that was actually recorded: '
      + population.textContent);

    /* ==============================================================
     * Checking is offered, and says nothing changed when nothing has
     * ============================================================== */
    assert.ok(at('check') && !at('check').hidden,
      'an all-finished result offers the explicit check');
    at('check').click();
    await until(() => noticeText().includes('Checked just now'), 'the check');
    assert.ok(noticeText().includes('membership below is unchanged'),
      'it answers about the runs, in the panel: ' + noticeText().slice(0, 400));
    assert.ok(noticeText().includes('not what they recorded'),
      'and does not claim the evidence behind them was re-read: '
      + noticeText().slice(0, 400));
    assert.deepEqual(members(), ['1', '2'],
      'and nothing on screen moved: ' + members().join(','));

    /* ==============================================================
     * Ticking a run is a different question, so the answer goes
     * ============================================================== */
    const box = await until(
      () => d.querySelector('#detail input[data-run-select="1"]'), 'a tick');
    box.checked = true;
    box.dispatchEvent(new w.Event('change', {bubbles: true}));
    assert.equal(bodyText(), '',
      'the whole-task answer is dropped rather than re-labelled with a '
      + 'selection it was not computed over');
    assert.equal(noticeText(), '', 'and so is the notice beside it');
    assert.ok(at('check').hidden,
      'with nothing left to check for changes against');
  }

  if (phase === 'drift') {
    at('all').click();
    await until(() => bodyText().includes('run(s) asked for'), 'the summary');
    assert.deepEqual(members(), ['1', '2'], 'the two finished runs');

    /* The driver finishes attempt 3 for real now, and answers when done. */
    process.stdout.write('READY-FOR-MUTATION\n');
    await new Promise(resolve => process.stdin.once('data', resolve));

    /* ==============================================================
     * Nothing notices on its own
     * ============================================================== */
    await wait(600);
    assert.equal(noticeText(), '',
      'no polling and no live addition: the page does not go looking');
    assert.deepEqual(members(), ['1', '2'],
      'and membership is untouched: ' + members().join(','));

    /* ==============================================================
     * The explicit check reports the drift and replaces nothing
     * ============================================================== */
    at('check').click();
    const notice = await until(
      () => noticeText().includes('The runs recorded for this task have changed')
        ? noticeText() : null,
      'the population notice');
    assert.ok(notice.includes('attempt 3'),
      'naming the run that became eligible: ' + notice.slice(0, 500));
    assert.ok(notice.includes('have finished'),
      'and saying what happened to it: ' + notice.slice(0, 500));
    assert.ok(notice.includes('Nothing below has been replaced'),
      'while the summary stands: ' + notice.slice(0, 500));
    assert.ok(!notice.includes('no longer records'),
      'and nothing claims a run this summary counted has changed: '
      + notice.slice(0, 500));
    assert.deepEqual(members(), ['1', '2'],
      'the members on screen are the ones this summary was computed over: '
      + members().join(','));
    assert.ok(bodyText().includes('2 run(s) asked for'),
      'and so are its counts: ' + bodyText().slice(0, 300));

    /* ==============================================================
     * Refresh is the only thing that changes membership
     * ============================================================== */
    at('refresh').click();
    await until(() => members().length === 3, 'the refreshed membership');
    assert.deepEqual(members(), ['1', '2', '3'],
      'the newly finished run is now a member: ' + members().join(','));
    assert.ok(bodyText().includes('3 run(s) asked for · 3 in these counts'),
      'counted over the population as it is now: ' + bodyText().slice(0, 400));
    const population = d.querySelector('#detail [data-task-population]');
    assert.ok(population.textContent.includes('0 still running'),
      'with nothing left unfinished: ' + population.textContent);
    assert.equal(noticeText(), '',
      'and the old notice is gone rather than standing over a new answer');

    /* Checking again against the refreshed baseline reports no change. */
    at('check').click();
    await until(() => noticeText().includes('Checked just now'), 'the check');
    assert.ok(noticeText().includes('membership below is unchanged'),
      'the refreshed summary is current: ' + noticeText().slice(0, 300));
  }

  if (phase === 'overlimit') {
    at('all').click();
    const refusal = await until(
      () => bodyText().includes('Not summarized') ? bodyText() : null,
      'the refusal');
    assert.ok(refusal.includes('21 finished runs'),
      'it says how many there are: ' + refusal.slice(0, 400));
    assert.ok(refusal.includes('at most 20'),
      'and what the bound is: ' + refusal.slice(0, 400));
    assert.ok(refusal.includes('nothing was sampled'),
      'and that no part of it was summarized: ' + refusal.slice(0, 400));
    assert.equal(members().length, 0, 'no member is shown');
    assert.ok(!d.querySelector('#detail [data-task-population]'),
      'and no figures at all, because none were computed');

    /* The fallback the refusal names still works. */
    for (const attempt of [1, 2]) {
      const box = await until(
        () => d.querySelector('#detail input[data-run-select="' + attempt + '"]'),
        'the tick for attempt ' + attempt);
      box.checked = true;
      box.dispatchEvent(new w.Event('change', {bubbles: true}));
    }
    await until(() => at('submit') && !at('submit').disabled, 'the button');
    at('submit').click();
    await until(() => bodyText().includes('2 run(s) asked for'),
      'the explicit summary of two of them');
    assert.ok(!d.querySelector('#detail [data-task-population]'),
      'an explicit selection did not ask about the rest of the task, so it '
      + 'is not told about it');
  }

  assert.deepEqual(errors, [], 'page errors: ' + errors.join(' | '));
  process.stdout.write('all-finished DOM checks passed (' + phase + ')\n');
  /* Exited rather than closed: an in-flight page fetch resolving after
     teardown fails inside the page's own callback, which is a harness
     artefact and not a finding. */
  process.exit(0);
})().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exit(1); });
