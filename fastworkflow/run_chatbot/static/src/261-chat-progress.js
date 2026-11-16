/* -- turn stage and elapsed time in the live panel [fix-kq49] -------------- */
/* Between sending a message and the first answer the runtime may run several model-backed stages
   (split, referents, match, draft, review) and show no command trace for any of them. This part
   adds, to the live turn's activity panel, the stage that is running and how long the turn has
   run. The stage comes from the stream's `progress` frames (a small payload: stage key, generic
   label, optional index/total); a server that sends none leaves just the timer. Frames of any
   other type fall through to tmApplyStreamFrame unchanged.

   The timer starts when the panel is made (the message was sent) and stops when the turn
   completes, fails, or waits for the user, when the source changes (tm.epoch), or when the panel
   leaves the page. Only the stage text is a live region; the timer is never announced.
   Restored (stored) turns never make a panel, so they show nothing. */

var TM_PROGRESS_TICK_MS = 1000;

function tmFormatElapsed(ms) {
  var total = Math.max(0, Math.floor(ms / 1000));
  var minutes = Math.floor(total / 60);
  var seconds = total % 60;
  if (minutes < 1) { return total + "s"; }
  return minutes + "m " + (seconds < 10 ? "0" : "") + seconds + "s";
}

/* The states tmStreamTurn/tmPollTurn set when the turn is over for now. "disconnected — recovering",
   "still working" and "already running — attaching" are not: the turn still runs. */
function tmProgressEndsAt(state) {
  return /^(completed|failed|waiting for your reply|refused|no live stream)/.test(state) ||
    state === "disconnected";
}

var tmActivityPanelBase = tmActivityPanel;
tmActivityPanel = function (bubbleMsg) {
  var activity = tmActivityPanelBase(bubbleMsg);
  var summary = activity.node.querySelector("summary");
  if (!summary) { return activity; }
  var stage = el("span", "actStage", "");
  stage.setAttribute("role", "status");
  stage.setAttribute("aria-live", "polite");
  var elapsed = el("span", "actElapsed", "· 0s");
  summary.appendChild(stage);
  summary.appendChild(elapsed);
  var started = Date.now();
  var epoch = tm.epoch;
  var timer = null;
  var stopped = false;

  function stop() {
    if (stopped) { return; }
    stopped = true;
    if (timer !== null) { clearInterval(timer); timer = null; }
    stage.textContent = "";
    elapsed.textContent = "· " + tmFormatElapsed(Date.now() - started);
  }
  function tick() {
    if (stopped) { return; }
    if (tm.epoch !== epoch || !activity.node.isConnected) { stop(); return; }
    elapsed.textContent = "· " + tmFormatElapsed(Date.now() - started);
  }
  timer = setInterval(tick, TM_PROGRESS_TICK_MS);

  var baseSetState = activity.setState;
  activity.setState = function (text) {
    baseSetState(text);
    if (tmProgressEndsAt(String(text))) { stop(); }
  };
  activity.progress = function (data) {
    if (stopped || !data || typeof data.label !== "string" || !data.label) { return; }
    stage.textContent = "· " + data.label + "…";
  };
  activity.stopProgress = stop;
  return activity;
};

var tmApplyStreamFrameBase = tmApplyStreamFrame;
tmApplyStreamFrame = function (frame, activity, sink) {
  if (frame && frame.type === "progress") {
    if (activity && typeof activity.progress === "function") { activity.progress(frame.data); }
    return;
  }
  tmApplyStreamFrameBase(frame, activity, sink);
};
