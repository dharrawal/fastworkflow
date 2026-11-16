/* -- review form [fix-weih.13] ------------------------------------------- */
/* A deterministic form for a sub-task or plan review (question_kind
   "review"), with no model call. The person edits goals and input values or
   skips steps; the form answers with ordinary chat text: "yes" (nothing
   changed), "no" (Cancel) or edits in the grammar review_engine.parse_edits
   reads ("change N: goal=...; change N: member=...; drop N"). Registered in
   TM_PROMPT_RENDERERS at load time (after 256-chat-forms.js); it only answers
   a prompt tagged "review", so it needs no particular position.

   What the parser forces on a composed reply (tests/test_review_reply_grammar.py
   runs every case of tests/fixtures/review_form_edits.json through the real
   parser; tmComposeReviewEdits is the pure function both sides share):
   - ";" and newlines end an edit, so in a value they become ", ".
   - a value is trimmed, and a pair of matching outer quotes is stripped, so a
     value wrapped in quotes is sent inside the other quote mark.
   - an input whose name is a SubTask field (goal, outcome, inputs, means,
     check, drafted_means) is sent as "inputs=<name>=<value>".
   - inside an inputs edit ", name=" / ", name:" / " and name=" starts another
     input, so such a value is sent in quotes (a value that is exactly
     "from sub-task N" too, which would otherwise be a reference).
   Things that cannot be said at all are refused on the field, visibly, and
   the primary button stays disabled until they are changed: the words
   none/null (they clear a value; the server refuses a quoted "none" too) and a
   value that needs quotes but holds both kinds. Dropping every step is refused
   by the form itself (the last unskipped step cannot be skipped).
   A step whose planning-command match was ambiguous (bd fix-1ma8.1) arrives with
   "alternatives": [{name, description}], the other plans that fit; the form shows
   a select on that step (kept as shown unless the person picks one) and composes
   "change N: means=planning:NAME", the same edit "step N: use NAME" reads, which
   re-drafts that step's inputs on the server.
   Since fix-2vyv the server reads three things the form used to refuse (they
   are now ordinary cases of the shared fixture): wording that looked like
   "accept" ("run it as shown") inside an edit's value, because a reply that
   holds a "change N:" / "drop N" clause is never an acceptance; an input named
   goal/outcome/means/check/desired_outcome beside another input edit of the same
   step; and a quoted value that would split beside another input edit. */

var TM_REVIEW_RESERVED = { goal: 1, outcome: 1, inputs: 1, means: 1, check: 1, drafted_means: 1 };
var TM_REVIEW_NAME = /^[A-Za-z_]\w*$/;
var TM_REVIEW_SPLIT = /,\s*(?:[A-Za-z_]\w*\s*[=:]|from sub-task\s+\d+)|\s+and\s+[A-Za-z_]\w*\s*[=:]/i;
var TM_REVIEW_FROM = /^from sub-task\s+\d+$/i;
var TM_REVIEW_GUARD = "At least one step must remain; click Cancel to decline the whole plan";
var TM_REVIEW_EMPTY = "Can't be empty — the original value is kept.";

function tmReviewSanitize(value) {
  var text = value === null || value === undefined ? "" : String(value);
  return text.replace(/\s*[;\r\n\u2028\u2029]+\s*/g, ", ").trim();
}

/* The text of an input value as it goes after "name=": quoted when the parser
   would otherwise strip, split or reinterpret it. */
function tmReviewInputText(text) {
  var first = text.charAt(0);
  var wrapped = text.length >= 2 && first === text.charAt(text.length - 1) &&
    (first === "\"" || first === "'");
  if (!(wrapped || TM_REVIEW_SPLIT.test(text) || TM_REVIEW_FROM.test(text))) { return text; }
  var mark = text.indexOf("\"") !== -1 ? "'" : "\"";
  return mark + text + mark;
}

/* Why a value cannot be sent ("" when it can). `text` is already sanitized. */
function tmReviewValueProblem(text, isGoal) {
  var low = text.toLowerCase();
  if (low === "none" || (!isGoal && low === "null")) {
    return "“" + low + "” removes a value here, so it cannot be sent as text. Type something else.";
  }
  return "";
}

/* edits: [{step, goal?, inputs?: {name: value}, skip?}] (1-based steps, only
   what the person changed). Returns [{step, field, message}]; empty when the
   composed reply says what the person meant. */
function tmReviewEditProblems(edits) {
  var out = [];
  (edits || []).forEach(function (edit) {
    if (edit.skip) { return; }
    if (edit.goal !== undefined) {
      var goal = tmReviewSanitize(edit.goal);
      var gm = goal ? tmReviewValueProblem(goal, true) : "";
      if (gm) { out.push({ step: edit.step, field: "goal", message: gm }); }
    }
    var inputs = tmIsPlainObject(edit.inputs) ? edit.inputs : {};
    var names = Object.keys(inputs).filter(function (n) { return tmReviewSanitize(inputs[n]) !== ""; });
    names.forEach(function (name) {
      var value = tmReviewSanitize(inputs[name]);
      var message = tmReviewValueProblem(value, false);
      var splits = TM_REVIEW_SPLIT.test(value);
      if (!message && splits && value.indexOf("\"") !== -1 && value.indexOf("'") !== -1) {
        message = "This value contains “, name=” and both kinds of quote, so it cannot " +
          "be sent safely. Reword it.";
      }
      if (message) { out.push({ step: edit.step, field: name, message: message }); }
    });
  });
  return out;
}

/* The exact reply text: "yes" when nothing was changed. Pure; it assumes
   tmReviewEditProblems(edits) is empty. */
function tmComposeReviewEdits(edits) {
  var ordered = (edits || []).slice().sort(function (a, b) { return a.step - b.step; });
  var clauses = [];
  ordered.forEach(function (edit) {
    var lead = "change " + edit.step + ": ";
    if (edit.skip) { clauses.push("drop " + edit.step); return; }
    var goal = edit.goal === undefined ? "" : tmReviewSanitize(edit.goal);
    if (goal) { clauses.push(lead + "goal=" + goal); }
    if (typeof edit.plan === "string" && edit.plan) { clauses.push(lead + "means=planning:" + edit.plan); }
    var inputs = tmIsPlainObject(edit.inputs) ? edit.inputs : {};
    Object.keys(inputs).forEach(function (name) {
      var value = tmReviewSanitize(inputs[name]);
      if (!value) { return; }
      var prefix = TM_REVIEW_RESERVED[name] === 1 ? "inputs=" + name + "=" : name + "=";
      clauses.push(lead + prefix + tmReviewInputText(value));
    });
  });
  return clauses.length ? clauses.join("; ") : "yes";
}

/* -- the payload ---------------------------------------------------------- */

function tmReviewMeansText(means) {
  if (!tmIsPlainObject(means)) { return ""; }
  if (means.kind === "planning" && typeof means.name === "string" && means.name) {
    return "uses " + means.name;
  }
  if (means.kind === "plain" && typeof means.name === "string" && means.name) {
    return "runs " + means.name;
  }
  return means.kind === "from_scratch" ? "built from scratch" : "";
}

/* The other plans that fit a step ([{name, description}], bd fix-1ma8.1), the well-formed ones; [] when none. */
function tmReviewAlternatives(item) {
  if (!tmIsPlainObject(item) || !Array.isArray(item.alternatives)) { return []; }
  return item.alternatives.filter(function (alt) {
    return tmIsPlainObject(alt) && typeof alt.name === "string" && alt.name !== "";
  });
}

/* The items when review has the shape of a sub_tasks or plan proposal; else
   null (no form, the plain prompt stays). */
function tmReviewItems(review) {
  if (!tmIsPlainObject(review)) { return null; }
  if (review.kind !== "sub_tasks" && review.kind !== "plan") { return null; }
  var items = review.items;
  if (!Array.isArray(items) || !items.length) { return null; }
  var ok = items.every(function (item) {
    if (!tmIsPlainObject(item) || typeof item.goal !== "string") { return false; }
    if (review.kind === "plan") {
      return item.steps === undefined || (Array.isArray(item.steps) &&
        item.steps.every(function (s) { return typeof s === "string"; }));
    }
    return item.inputs === undefined || (Array.isArray(item.inputs) &&
      item.inputs.every(function (inp) { return tmIsPlainObject(inp); }));
  });
  return ok ? items : null;
}

function tmReviewInputEditable(inp, seen) {
  return typeof inp.name === "string" && TM_REVIEW_NAME.test(inp.name) &&
    typeof inp.value === "string" && (inp.from_subtask === null || inp.from_subtask === undefined) &&
    !seen[inp.name];
}

/* -- the goal field: one line of text that wraps and grows [fix-weih.31] ----- */

/* The goal and the editable input values are growing one-line textareas (see
   tmGrowingLine in 256-chat-forms.js, which holds the sizing code). The old
   names stay as aliases. */
var TM_REVIEW_GOAL_MAX_LINES = TM_FIT_MAX_LINES;
function tmReviewOneLine(value) { return tmOneLine(value); }
function tmReviewFitHeight(scrollHeight, lineHeight, padding, border, maxLines) {
  return tmFitHeight(scrollHeight, lineHeight, padding, border, maxLines);
}
function tmReviewFitField(field) { return tmFitField(field); }
function tmReviewFitAll(root) { tmFitAll(root); }

/* -- the form ------------------------------------------------------------- */

function tmBuildReviewForm(review, items, text) {
  tmFormSeq += 1;
  var uid = "rv" + tmFormSeq;
  var isPlan = review.kind === "plan";
  var disabled = false;
  var steps = [];
  var root = el("div", "reviewForm");
  root.setAttribute("role", "group");
  var revision = typeof review.revision === "number" ? review.revision : 0;
  if (revision > 0) {
    var banner = el("div", "reviewRevision", "Updated plan (revision " + revision +
      ") — review and click Run as shown to confirm.");
    banner.setAttribute("role", "status");
    root.appendChild(banner);
  }
  var titleText = (typeof review.title === "string" && review.title.trim()
    ? review.title.trim() : (isPlan ? "Plan" : "Sub-tasks")) + " — check the steps before they run";
  var heading = el("div", "reviewTitle", titleText);
  heading.id = uid + "-t";
  root.setAttribute("aria-labelledby", heading.id);
  root.appendChild(heading);
  var list = el("div", "reviewSteps");
  root.appendChild(list);

  function addNote(parent, id) {
    var note = el("div", "reviewFieldNote");
    note.id = id;
    note.hidden = true;
    parent.appendChild(note);
    return note;
  }

  items.forEach(function (item, position) {
    var number = position + 1;
    var step = { index: number, inputs: [], origGoal: item.goal };
    var row = el("div", "reviewStep");
    row.setAttribute("data-step", String(number));
    step.row = row;
    if (review.changes_data && review.changes_data[position] === true) {
      var head = el("div", "reviewStepHead");
      head.appendChild(el("span", "reviewBadge", "Changes data"));
      row.appendChild(head);
    }

    /* The number belongs to the first label: one line, "1. Goal:". */
    var goalField = el("div", "reviewField reviewGoalField");
    var goalLabel = el("label", "reviewLabel reviewGoalLabel");
    goalLabel.appendChild(el("span", "reviewNum", number + "."));
    goalLabel.appendChild(document.createTextNode(" "));
    goalLabel.appendChild(el("span", "reviewLabelText", "Goal:"));
    goalLabel.setAttribute("for", uid + "-g" + number);
    goalField.appendChild(goalLabel);
    var goalInput = tmGrowingLine(el("textarea", "reviewInput reviewGoal"));
    goalInput.setAttribute("spellcheck", "true");
    goalInput.id = uid + "-g" + number;
    goalInput.value = tmReviewOneLine(item.goal);
    goalInput.setAttribute("data-field", "goal");
    goalField.appendChild(goalInput);
    step.goalInput = goalInput;
    step.goalNote = addNote(goalField, uid + "-gn" + number);
    goalInput.setAttribute("aria-describedby", step.goalNote.id);
    row.appendChild(goalField);

    if (isPlan) {
      if (Array.isArray(item.steps) && item.steps.length) {
        var plan = el("ul", "reviewPlanSteps");
        item.steps.forEach(function (s) { plan.appendChild(el("li", null, s)); });
        row.appendChild(plan);
      }
    } else {
      var seen = {};
      (Array.isArray(item.inputs) ? item.inputs : []).forEach(function (inp, k) {
        var field = el("div", "reviewField reviewValueField");
        var name = typeof inp.name === "string" ? inp.name : "";
        var label = el("label", "reviewLabel", name || "value");
        if (inp.drafted === true) { label.appendChild(el("span", "reviewDrafted", " (drafted)")); }
        label.appendChild(document.createTextNode(":"));
        var id = uid + "-v" + number + "-" + k;
        if (tmReviewInputEditable(inp, seen)) {
          seen[name] = true;
          label.setAttribute("for", id);
          field.setAttribute("data-input", name);
          field.appendChild(label);
          var input = tmGrowingLine(el("textarea", "reviewInput reviewValue"));
          input.id = id;
          input.value = tmOneLine(inp.value);
          var note = addNote(field, id + "-n");
          input.setAttribute("aria-describedby", note.id);
          field.appendChild(input);
          step.inputs.push({ name: name, orig: inp.value, input: input, note: note });
        } else {
          if (name) { field.setAttribute("data-input", name); }
          field.classList.add("readonly");
          field.appendChild(label);
          var shown = typeof inp.from_subtask === "number" ? "from step " + inp.from_subtask
            : (typeof inp.value === "string" ? inp.value : "");
          field.appendChild(el("span", "reviewFixed", shown));
        }
        row.appendChild(field);
      });
      var means = tmReviewMeansText(item.means);
      if (means) { row.appendChild(el("div", "reviewMeans", means)); }
      var alternatives = tmReviewAlternatives(item);
      if (alternatives.length) {
        var planField = el("div", "reviewField reviewPlanField");
        planField.setAttribute("data-field", "plan");
        var planLabel = el("label", "reviewLabel", "Other plans that fit:");
        planLabel.setAttribute("for", uid + "-p" + number);
        planField.appendChild(planLabel);
        var planSelect = el("select", "reviewInput reviewPlanSelect");
        planSelect.id = uid + "-p" + number;
        var keep = el("option", null, "Keep " + (tmIsPlainObject(item.means) && item.means.name ? item.means.name : "the plan shown"));
        keep.value = "";
        planSelect.appendChild(keep);
        alternatives.forEach(function (alt) {
          var option = el("option", null, alt.name + (typeof alt.description === "string" && alt.description ? " — " + alt.description : ""));
          option.value = alt.name;
          planSelect.appendChild(option);
        });
        planField.appendChild(planSelect);
        row.appendChild(planField);
        step.planSelect = planSelect;
      }
      if (typeof item.outcome === "string" && item.outcome.trim()) {
        row.appendChild(el("div", "reviewOutcome", item.outcome));
      }
    }

    var skipLabel = el("label", "reviewSkip");
    var skipBox = el("input");
    skipBox.type = "checkbox";
    skipBox.id = uid + "-s" + number;
    skipLabel.appendChild(skipBox);
    skipLabel.appendChild(el("span", null, "Skip this step"));
    row.appendChild(skipLabel);
    step.skipBox = skipBox;
    steps.push(step);
    list.appendChild(row);
  });

  var guard = el("div", "reviewGuard", TM_REVIEW_GUARD + ".");
  guard.hidden = true;
  root.appendChild(guard);
  var problemLine = el("div", "reviewFieldNote reviewProblemLine", "Fix the highlighted values first.");
  problemLine.hidden = true;
  root.appendChild(problemLine);
  var bar = el("div", "paramActions reviewActions");
  var primary = el("button", "paramBtn primary", "Run as shown");
  primary.type = "button";
  var cancel = el("button", "paramBtn ghost", "Cancel");
  cancel.type = "button";
  bar.appendChild(primary);
  bar.appendChild(cancel);
  root.appendChild(bar);

  /* Only what differs from the payload; an emptied field counts as unchanged. */
  function collect() {
    var edits = [];
    steps.forEach(function (s) {
      if (s.skipBox.checked) { edits.push({ step: s.index, skip: true }); return; }
      var edit = { step: s.index };
      var any = false;
      var goal = s.goalInput.value.trim();
      if (goal !== "" && goal !== s.origGoal.trim()) { edit.goal = goal; any = true; }
      if (s.planSelect && s.planSelect.value) { edit.plan = s.planSelect.value; any = true; }
      s.inputs.forEach(function (inp) {
        var value = inp.input.value.trim();
        if (value !== "" && value !== inp.orig.trim()) {
          if (!edit.inputs) { edit.inputs = {}; }
          edit.inputs[inp.name] = value;
          any = true;
        }
      });
      if (any) { edits.push(edit); }
    });
    return edits;
  }

  function setNote(node, input, message) {
    node.textContent = message;
    node.hidden = message === "";
    if (message) { input.setAttribute("aria-invalid", "true"); }
    else { input.removeAttribute("aria-invalid"); }
  }

  function refresh() {
    var edits = collect();
    var problems = tmReviewEditProblems(edits);
    var live = steps.filter(function (s) { return !s.skipBox.checked; });
    steps.forEach(function (s) {
      var skipped = s.skipBox.checked;
      var last = !skipped && live.length === 1;
      s.row.classList.toggle("skipped", skipped);
      s.skipBox.disabled = disabled || last;
      if (s.planSelect) { s.planSelect.disabled = disabled || skipped; }
      if (last) { s.skipBox.title = TM_REVIEW_GUARD; } else { s.skipBox.removeAttribute("title"); }
      var fieldOf = function (field, input, orig, note) {
        input.disabled = disabled || skipped;
        var message = "";
        problems.forEach(function (p) {
          if (p.step === s.index && p.field === field && !message) { message = p.message; }
        });
        if (!message && !skipped && input.value.trim() === "" && orig.trim() !== "") {
          message = TM_REVIEW_EMPTY;
        }
        setNote(note, input, message);
      };
      fieldOf("goal", s.goalInput, s.origGoal, s.goalNote);
      s.inputs.forEach(function (inp) { fieldOf(inp.name, inp.input, inp.orig, inp.note); });
    });
    guard.hidden = live.length !== 1;
    problemLine.hidden = problems.length === 0;
    primary.textContent = edits.length ? "Apply changes" : "Run as shown";
    primary.disabled = disabled || problems.length > 0;
    cancel.disabled = disabled;
    return { edits: edits, problems: problems };
  }

  function restoreIfEmpty(input, orig) {
    input.addEventListener("change", function () {
      if (input.value.trim() === "" && orig.trim() !== "") { input.value = tmOneLine(orig); }
      tmFitField(input);
      refresh();
    });
  }
  steps.forEach(function (s) {
    s.goalInput.addEventListener("input", refresh);
    restoreIfEmpty(s.goalInput, s.origGoal);
    s.skipBox.addEventListener("change", refresh);
    if (s.planSelect) { s.planSelect.addEventListener("change", refresh); }
    s.inputs.forEach(function (inp) {
      inp.input.addEventListener("input", refresh);
      restoreIfEmpty(inp.input, inp.orig);
    });
  });
  primary.addEventListener("click", function () {
    if (disabled) { return; }
    var state = refresh();
    if (state.problems.length) { return; }
    tmSendChatText(tmComposeReviewEdits(state.edits));
  });
  cancel.addEventListener("click", function () {
    if (!disabled) { tmSendChatText("no"); }
  });
  refresh();
  /* The fields are measured when the form is on the page (see refit), and again
     whenever the form's width changes or it goes from hidden to shown. */
  tmFitObserve(root);
  function refit() { tmFitAll(root); }

  var details = tmDetailsFor(text, false);
  details.classList.add("reviewDetails");
  return {
    element: root,
    focus: function () { steps[0].goalInput.focus(); },
    refit: refit,
    setDisabled: function (flag) {
      disabled = !!flag;
      if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
      root.setAttribute("aria-disabled", disabled ? "true" : "false");
      refresh();
    },
    nodes: [root, details]
  };
}

function tmRenderReviewForm(bubbleMsg, out) {
  if (!out || out.status !== "awaiting_user") { return null; }
  if (!tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  if (!last || last.artifacts.question_kind !== "review") { return null; }
  var review = last.artifacts.review;
  var items = tmReviewItems(review);
  if (!items) { return null; }
  /* The whole payload is the key: an edit the server applied comes back as a
     new revision with new items and gets a fresh form; an equal repaint
     keeps what was typed. */
  var key = "review:" + JSON.stringify(review);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) {
    var again = tmMountPrompt(bubbleMsg, prev);
    if (prev.refit) { prev.refit(); }
    return again;
  }
  var built = tmBuildReviewForm(review, items, last.text);
  var rec = { key: key, element: built.element, setDisabled: built.setDisabled, nodes: built.nodes,
    focus: built.focus, refit: built.refit };
  var mounted = tmMountPrompt(bubbleMsg, rec);
  if (mounted) {
    built.refit();
    built.focus();
    /* tmComposerState focuses the composer when the turn settles, after this paint. */
    setTimeout(function () {
      built.refit();
      var active = document.activeElement;
      if (!active || active === document.body || active === document.getElementById("chatInput")) {
        built.focus();
      }
    }, 0);
  }
  return mounted;
}

TM_PROMPT_RENDERERS.push(tmRenderReviewForm);
