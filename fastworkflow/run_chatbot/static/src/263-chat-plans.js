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
