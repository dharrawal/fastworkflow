/* -- plan choices for a part that was not run [fix-1ma8.6] ---------------- */
/* Without an agent model (no-agent mode) a part no planning command matched comes back as an entry of the artifact
   fw_not_run with its `request` text and `available_plans`, [{name, description, closest}]: the planning commands of
   available here, the closest to the part first. This page shows them as buttons under the answer. A click
   starts a NEW turn: the part's request text with use_plan = the plan's name (the server runs that one part bound to
   that plan, asks for a missing argument as it does for any matched plan). Nothing here changes the answer text; the
   page leaves the plan list out of the rendered markdown (the buttons are the list) and the text itself stays as
   the server wrote it for the CLI and MCP.

   The buttons are a live form (tmRegisterLiveForm): sending any message disables them, and a turn with a later
   message after it paints them disabled, as the question and review forms are. A restored turn paints them from the
   stored record (tmRenderPlanChoices is called when the record loads). At most TM_PLAN_CAP buttons show; "show all"
   lists the rest. */

var TM_PLAN_CAP = 12;
var TM_PLAN_HEADINGS = ["The closest planning commands available here:", "Other planning commands available here:",
  "Planning commands available here:"];

/* The ad hoc entries of the turn's fw_not_run that can be offered as buttons. */
function tmPlanEntries(out) {
  var last = tmLastPromptArtifacts(out);
  var list = last && last.artifacts ? last.artifacts.fw_not_run : null;
  if (!Array.isArray(list)) { return []; }
  return list.filter(function (entry) {
    return tmIsPlainObject(entry) && !entry.command && !entry.open_step &&
      typeof entry.request === "string" && entry.request.trim() !== "" &&
      Array.isArray(entry.available_plans) && tmPlansOf(entry).length > 0;
  });
}

function tmPlansOf(entry) {
  return (entry.available_plans || []).filter(function (plan) {
    return tmIsPlainObject(plan) && typeof plan.name === "string" && plan.name !== "";
  });
}

/* The answer text without the lines that list the plans (the buttons list them). Only for a turn that gets buttons;
   a line that does not match leaves the text as it was. */
function tmPlansAnswer(out, answer) {
  var entries = tmPlanEntries(out);
  if (!entries.length) { return answer; }
  var names = {};
  entries.forEach(function (entry) {
    tmPlansOf(entry).forEach(function (plan) { names["- `" + plan.name + "`"] = true; });
  });
  var kept = String(answer).split("\n").filter(function (line) {
    var head = line.replace(/:.*$/, "");
    return !(names[head] || TM_PLAN_HEADINGS.indexOf(line) !== -1 ||
      /^and \d+ more; `what can i do` lists everything\.$/.test(line) ||
      /^The other planning commands available here are listed under part \d+\.$/.test(line));
  });
  return kept.join("\n").replace(/\n{3,}/g, "\n\n");
}

function tmPlanButton(plan) {
  var button = el("button", "planBtn" + (plan.closest ? " closest" : ""));
  button.type = "button";
  button.dataset.plan = plan.name;
  button.appendChild(el("span", "planName", plan.name));
  if (plan.closest) { button.appendChild(el("span", "planTag", "closest")); }
  if (plan.description) { button.appendChild(el("span", "planDesc", plan.description)); }
  return button;
}

function tmRenderPlanChoices(bubbleMsg, out) {
  Array.prototype.slice.call(bubbleMsg.querySelectorAll(".planChoices")).forEach(function (node) {
    node.parentNode.removeChild(node);
  });
  var entries = tmPlanEntries(out);
  if (!entries.length) { return null; }
  var root = el("div", "planChoices");
  var disabled = false;
  var buttons = [];
  var toggles = [];
  var split = entries.length > 1 || entries[0].sub_task !== null && entries[0].sub_task !== undefined;
  entries.forEach(function (entry) {
    var plans = tmPlansOf(entry);
    var group = el("div", "planGroup");
    if (entry.sub_task !== null && entry.sub_task !== undefined) { group.dataset.part = String(entry.sub_task); }
    group.appendChild(el("div", "planHead", split && entry.sub_task
      ? "Run part " + entry.sub_task + " with one of these plans" : "Run this with one of these plans"));
    var list = el("div", "planList");
    group.appendChild(list);
    plans.forEach(function (plan, index) {
      var button = tmPlanButton(plan);
      if (index >= TM_PLAN_CAP) { button.hidden = true; button.classList.add("planExtra"); }
      button.addEventListener("click", function () {
        if (disabled) { return; }
        tmSendChatText(entry.request, { usePlan: plan.name });
      });
      buttons.push(button);
      list.appendChild(button);
    });
    if (plans.length > TM_PLAN_CAP) {
      var more = el("button", "planMore", "show all (" + plans.length + ")");
      more.type = "button";
      more.setAttribute("aria-expanded", "false");
      more.addEventListener("click", function () {
        var open = more.getAttribute("aria-expanded") !== "true";
        Array.prototype.forEach.call(list.querySelectorAll(".planExtra"), function (node) { node.hidden = !open; });
        more.setAttribute("aria-expanded", open ? "true" : "false");
        more.textContent = open ? "show fewer" : "show all (" + plans.length + ")";
      });
      toggles.push(more);
      group.appendChild(more);
    }
    root.appendChild(group);
  });

  function setDisabled(flag) {
    disabled = !!flag;
    buttons.forEach(function (node) { node.disabled = disabled; });
    if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
    root.setAttribute("aria-disabled", disabled ? "true" : "false");
  }

  tmAppendBeforeMeta(bubbleMsg, root);
  var live = { element: root, setDisabled: setDisabled };
  tmRegisterLiveForm(live);
  setDisabled(!tmIsLiveTurn(bubbleMsg) || !tm.connected);
  /* A restored turn painted its text before the record arrived: repaint it without the plan list. */
  var bubble = bubbleMsg.querySelector(".bubble");
  if (bubbleMsg.dataset.restored === "1" && bubble && bubble.classList.contains("md")) {
    var answer = String(out.answer || "");
    var shown = tmPlansAnswer(out, answer);
    if (shown !== answer) { clear(bubble); bubble.classList.remove("md"); appendMarkdown(bubble, shown); }
  }
  return root;
}

/* -- the plan-choice question [fix-yu3l.5] -------------------------------- */
/* A turn somebody can answer does not end an unmatched part as not run: it asks which command or planning command is
   meant (question_kind "plan_choice", artifact plan_choice: {request, sub_task, choices: [{name, description, closest}],
   none_reply, decline, ambiguous, misunderstood}). The deterministic ("/") path asks the same question as a failed
   completed turn, not a suspended one: the question is on the last command output either way (bd fix-o1lg S7). The same plan buttons are shown, but a click here is the REPLY to the pending question (the
   plan's name as plain chat text), not a new turn with use_plan: the server binds the plan to the part that asked and
   carries the turn on from there. "None of these" and Cancel send the replies the data names. The root is not a
   .planChoices node, because tmRenderPlanChoices clears those on every paint of the turn. */

function tmPlanQuestionChoices(payload) {
  return tmIsPlainObject(payload) && Array.isArray(payload.choices)
    ? tmPlansOf({ available_plans: payload.choices }) : [];
}

function tmRenderPlanQuestion(bubbleMsg, out) {
  if (!out || bubbleMsg.dataset.restored === "1" || !tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  if (!last || last.artifacts.question_kind !== "plan_choice") { return null; }
  var entry = last.entry;
  var resp = entry && entry.command_response;
  if (!resp || resp.success !== false) { return null; }
  var payload = last.artifacts.plan_choice;
  var plans = tmPlanQuestionChoices(payload);
  /* Nothing to offer as buttons: the plain question stays, and the composer answers it. */
  if (!plans.length) { return null; }
  var key = "planQuestion:" + JSON.stringify(payload);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }

  var disabled = false;
  var root = el("div", "planQuestion");
  root.setAttribute("role", "group");
  var part = payload.sub_task !== null && payload.sub_task !== undefined ? payload.sub_task : null;
  if (part !== null) { root.dataset.part = String(part); }
  var subject = part !== null ? "part " + part : "this request";
  var head = el("div", "planHead", payload.misunderstood === true
    ? "No command matched " + subject + ". Is one of these meant?"
    : payload.ambiguous === true
      ? "More than one command fits " + subject + ". Which one is meant?"
      : "No plan matched " + subject + ". Which one should run it?");
  head.id = "planQuestionHead" + (tmFormSeq += 1);
  root.setAttribute("aria-labelledby", head.id);
  root.appendChild(head);
  if (typeof payload.request === "string" && payload.request.trim()) {
    root.appendChild(el("div", "planRequest", payload.request));
  }
  function send(text) {
    return function () { if (!disabled) { tmSendChatText(text); } };
  }
  var list = el("div", "planList");
  root.appendChild(list);
  plans.forEach(function (plan, index) {
    var button = tmPlanButton(plan);
    if (index >= TM_PLAN_CAP) { button.hidden = true; button.classList.add("planExtra"); }
    button.addEventListener("click", send(plan.name));
    list.appendChild(button);
  });
  if (plans.length > TM_PLAN_CAP) {
    var more = el("button", "planMore", "show all (" + plans.length + ")");
    more.type = "button";
    more.setAttribute("aria-expanded", "false");
    more.addEventListener("click", function () {
      var open = more.getAttribute("aria-expanded") !== "true";
      Array.prototype.forEach.call(list.querySelectorAll(".planExtra"), function (node) { node.hidden = !open; });
      more.setAttribute("aria-expanded", open ? "true" : "false");
      more.textContent = open ? "show fewer" : "show all (" + plans.length + ")";
    });
    root.appendChild(more);
  }
  var actions = el("div", "intentActions");
  var none = el("button", "planNone secondary", "None of these");
  none.type = "button";
  none.addEventListener("click", send(typeof payload.none_reply === "string" && payload.none_reply
    ? payload.none_reply : "none"));
  actions.appendChild(none);
  var cancel = el("button", "planCancel ghost", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", send(typeof payload.decline === "string" && payload.decline
    ? payload.decline : "cancel"));
  actions.appendChild(cancel);
  root.appendChild(actions);

  var rec = { key: key, element: root,
    setDisabled: function (flag) {
      disabled = !!flag;
      /* The "show all" toggle stays usable: it sends nothing. */
      Array.prototype.forEach.call(root.querySelectorAll(".planBtn, .planNone, .planCancel"), function (node) {
        node.disabled = disabled;
      });
      if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
      root.setAttribute("aria-disabled", disabled ? "true" : "false");
    },
    nodes: [root, tmDetailsFor(last.text, false)] };
  return tmMountPrompt(bubbleMsg, rec);
}

/* unshift, as the question form does: no other renderer claims this kind, and running first costs them nothing. */
TM_PROMPT_RENDERERS.unshift(tmRenderPlanQuestion);
