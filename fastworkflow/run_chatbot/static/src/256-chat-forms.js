/* -- chat forms [fix-weih.5] -------------------------------------------- */
/* One generic form renderer, tmRenderForm, plus the two prompts of a human's
   "/" command that use it with no model call: a parameter form (missing or
   rejected values) and ambiguous-intent pills. The renderer knows nothing
   about "/" commands; later children reuse it for approvals, reviews and
   questions. Every string reaches the page through el()/textContent [R22].

   tmRenderForm(model, actions) -> {element, values(), setDisabled(flag),
   focusFirstAsked()}

   model   {title, fields: [{name, label, kind, required, help, placeholder,
            value, min, max, options: [{value, label}], state, allowOther,
            enforceRequired, forceRadio}]}
     kind     "text" | "longtext" | "password" | "integer" | "number" | "date"
              | "list" (one item per line) | "choice" | "multichoice" | "boolean"
     password is NOT a control: the form never collects a secret, because Run
              sends the reply as chat text that is echoed and recorded. The
              field is a labelled notice row (no input of any kind), reads as
              "", and when it is required every "primary" action is disabled
              and a role=status line says which parameter to supply another
              way.
     choice   radio buttons for 5 options or fewer (or forceRadio), else a select.
              A field that is not required starts with a "—" option (value "").
     state    "missing" | "invalid" | null: marks the field .asked, sets
              aria-invalid and says "Required" / "Not accepted — see Details".
     enforceRequired  with required, sets the HTML required attribute and makes
              a "primary" action refuse (and mark the field) while it is empty.
   actions [{label, kind: "primary" | "secondary" | "ghost", onClick(values)}]
   values() maps each field name to: a string (text kinds, password, date, a
   choice), a number or null (integer, number: null when blank), an array of
   strings (list, multichoice), true/false (boolean), null (a choice left at
   "—" or with nothing chosen), "" for a password (never a typed value).

   Forms and pill bars made for the live chat are remembered, and
   tmDisableLiveForms() (called by tmSend) disables them all.

   Prompt renderers. tmRenderTurn calls tmRenderPromptForms once; it tries
   each function of TM_PROMPT_RENDERERS in order and stops at the first that
   returns a node. A later part registers its own at load time with
   TM_PROMPT_RENDERERS.push(fn), or .unshift(fn) to run before the built-in
   ones. tmRenderPromptForms also runs for a restored turn once its stored
   record loads (bubbleMsg.dataset.restored === "1", out = the stored turn);
   a renderer that must not run there checks dataset.restored itself, as the
   two built-in ones do. tmIsLiveTurn only means no later message follows. A renderer is (bubbleMsg, out) -> node or null, and must return null
   quickly when its prompt is absent. It mounts through
   tmMountPrompt(bubbleMsg, rec) with rec = {key, element, setDisabled(flag),
   nodes: [nodes put in the bubble]} and reuses an earlier paint of the same
   prompt through tmReusePrompt(bubbleMsg, key) (pass the result back to
   tmMountPrompt). Helpers for renderers: tmIsLiveTurn, tmLastPromptArtifacts
   (-> {artifacts, text, entry} of the last command output), tmDetailsFor,
   tmSendChatText, tmFillComposer, tmRegisterLiveForm, tmDisableLiveForms,
   tmRenderForm. */

var TM_PROMPT_ARTIFACT_KEYS = ["question_kind", "approval", "review",
  "parameter_request", "intent_choices"];

var tmLiveForms = [];
var tmFormSeq = 0;
var TM_OTHER_TEXT = "Other…";

function tmFormIsEmpty(value) {
  if (value === null || value === undefined) { return true; }
  if (typeof value === "string") { return value.trim() === ""; }
  if (Array.isArray(value)) { return value.length === 0; }
  if (typeof value === "number") { return isNaN(value); }
  return false;
}

function tmFormOptionValue(opt) {
  return String(opt && opt.value !== undefined && opt.value !== null ? opt.value : "");
}

/* The label, then the value in muted text when the label says something else.
   A label that already starts "<value> — " is not followed by it again. */
function tmFormOptionParts(opt) {
  var value = tmFormOptionValue(opt);
  var label = opt && opt.label !== undefined && opt.label !== null ? String(opt.label) : "";
  if (!label || label === value) { return { text: value, muted: "" }; }
  if (label.indexOf(value + " — ") === 0) { return { text: label, muted: "" }; }
  return { text: label, muted: value };
}

function tmFormAddOptionText(node, opt) {
  var parts = tmFormOptionParts(opt);
  node.appendChild(el("span", "paramOptText", parts.text));
  if (parts.muted) {
    node.appendChild(document.createTextNode(" "));
    node.appendChild(el("span", "paramOptValue", parts.muted));
  }
}

/* -- one-line text that wraps and grows [fix-weih.31, fix-weih.33] --------- */

/* A free-text field is a <textarea rows=1 class="fitText"> so a long value wraps
   to the form's width and the field grows in height (capped, then it scrolls
   vertically). It still holds ONE line, exactly as the text input it replaced
   did: Enter inserts nothing, and a newline that arrives some other way (paste,
   drop, a value set from script) is removed the way an input's value
   sanitisation removes it, so what a form reads and sends is byte-identical.
   The parameter form, the question form and the review form all use these. */
var TM_FIT_MAX_LINES = 12;

/* What an <input type=text> would hold for this value (it strips CR and LF). */
function tmOneLine(value) {
  return value === null || value === undefined ? "" : String(value).replace(/[\r\n]/g, "");
}

/* The height to give a growing field, from measured numbers (pure, so it can be
   tested without layout). scrollHeight is the content height with padding, as the
   element reports it after its height was reset; border is the sum of the top
   and bottom borders. Returns null when nothing was measured (a hidden element
   measures 0): the caller leaves the field alone and measures again when it is
   shown. Past maxLines the field stops growing and scrolls vertically. */
function tmFitHeight(scrollHeight, lineHeight, padding, border, maxLines) {
  if (!(scrollHeight > 0)) { return null; }
  var cap = lineHeight * maxLines + padding;
  var capped = scrollHeight > cap;
  return { height: Math.ceil(Math.min(scrollHeight, cap) + border), scroll: capped };
}

/* Fits one field; false when it could not be measured (hidden, detached). */
function tmFitField(field) {
  if (!field || !field.isConnected) { return false; }
  var style = window.getComputedStyle ? window.getComputedStyle(field) : null;
  var num = function (v, fallback) { var n = parseFloat(v); return isNaN(n) ? fallback : n; };
  var fontSize = style ? num(style.fontSize, 13) : 13;
  /* browsers report a px line height; "normal" or a bare multiplier is turned into px here */
  var rawLine = style ? String(style.lineHeight) : "";
  var lineHeight = /px\s*$/.test(rawLine) ? num(rawLine, fontSize * 1.4)
    : (/^[\d.]+$/.test(rawLine) ? fontSize * num(rawLine, 1.4) : fontSize * 1.4);
  var padding = style ? num(style.paddingTop, 0) + num(style.paddingBottom, 0) : 0;
  var border = style ? num(style.borderTopWidth, 0) + num(style.borderBottomWidth, 0) : 0;
  field.style.height = "auto";
  var fit = tmFitHeight(field.scrollHeight, lineHeight, padding, border, TM_FIT_MAX_LINES);
  if (!fit) { field.style.height = ""; return false; }
  field.style.height = fit.height + "px";
  field.style.overflowY = fit.scroll ? "auto" : "hidden";
  return true;
}

/* Fits every growing field under root. A field that measured 0 is tried again
   when the form is resized or shown. */
function tmFitAll(root) {
  Array.prototype.forEach.call(root.querySelectorAll("textarea.fitText"), tmFitField);
}

/* A window resize re-measures every growing field on the page; a form whose own
   width changes (the results panel opening or closing) or that is shown after
   being hidden re-measures through tmFitObserve. */
window.addEventListener("resize", function () { tmFitAll(document); });

function tmFitObserve(root) {
  if (!window.ResizeObserver) { return; }
  var lastWidth = -1;
  new ResizeObserver(function () {
    var width = root.clientWidth;
    if (width !== lastWidth) { lastWidth = width; tmFitAll(root); }
  }).observe(root);
}

/* Makes a textarea a growing one-line field: Enter adds nothing, a pasted
   newline becomes a space, any other newline is removed, and the height follows
   the text. The caller's own "input" listeners still run (read the value after
   this one, which is registered first). */
function tmGrowingLine(field) {
  field.rows = 1;
  field.wrap = "soft";
  field.classList.add("fitText");
  field.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); }
  });
  field.addEventListener("paste", function (event) {
    var data = event.clipboardData && event.clipboardData.getData ? event.clipboardData.getData("text") : null;
    if (typeof data !== "string" || !/[\r\n]/.test(data) || typeof field.setRangeText !== "function") { return; }
    event.preventDefault();
    field.setRangeText(data.replace(/\s*[\r\n]+\s*/g, " "), field.selectionStart, field.selectionEnd, "end");
    field.dispatchEvent(new Event("input", { bubbles: true }));
  });
  field.addEventListener("input", function () {
    if (/[\r\n]/.test(field.value)) {
      var caret = tmOneLine(field.value.slice(0, field.selectionStart)).length;
      field.value = tmOneLine(field.value);
      if (typeof field.setSelectionRange === "function") { field.setSelectionRange(caret, caret); }
    }
    tmFitField(field);
  });
  return field;
}

/* A record per field: {field, wrap, controls, target, read(), clientInvalid}. */
function tmFormTextControl(field, ids) {
  var kind = field.kind;
  var multiline = kind === "longtext" || kind === "list";
  /* Free text is a growing one-line textarea; dates and numbers keep their
     native inputs (validation, pickers), longtext and list keep 4 rows. */
  var growing = kind === "text";
  var control = el(multiline || growing ? "textarea" : "input", "paramInput");
  control.id = ids.control;
  if (growing) {
    tmGrowingLine(control);
  } else if (multiline) {
    control.rows = 4;
  } else {
    control.type = kind === "date" ? "date"
      : (kind === "integer" || kind === "number" ? "number" : "text");
  }
  if (kind === "integer") { control.step = "1"; }
  if (kind === "number") { control.step = "any"; }
  if ((kind === "integer" || kind === "number") && field.min !== undefined && field.min !== null) {
    control.min = String(field.min);
  }
  if ((kind === "integer" || kind === "number") && field.max !== undefined && field.max !== null) {
    control.max = String(field.max);
  }
  if (field.placeholder) { control.placeholder = String(field.placeholder); }
  else if (kind === "list") { control.placeholder = "One item per line"; }
  var value = field.value;
  if (value !== null && value !== undefined) {
    if (kind === "list") {
      control.value = Array.isArray(value) ? value.join("\n") : String(value);
    } else if (typeof value !== "object") {
      control.value = growing ? tmOneLine(value) : String(value);
    }
  }
  var read;
  if (kind === "integer" || kind === "number") {
    read = function () {
      var raw = control.value.trim();
      if (raw === "") { return null; }
      var n = Number(raw);
      return isNaN(n) ? raw : n;
    };
  } else if (kind === "list") {
    read = function () {
      return control.value.split(/\r?\n/).map(function (line) { return line.trim(); })
        .filter(function (line) { return line !== ""; });
    };
  } else {
    read = function () { return control.value; };
  }
  return { controls: [control], target: control, aria: [control], read: read, group: null };
}

function tmFormBooleanControl(field, ids) {
  var box = el("input", "paramCheckbox");
  box.type = "checkbox";
  box.id = ids.control;
  box.checked = field.value === true || field.value === "true";
  return { controls: [box], target: box, aria: [box], group: null,
    read: function () { return box.checked; } };
}

function tmFormOtherInput(field, onType) {
  var text = el("input", "paramOther");
  text.type = "text";
  text.placeholder = TM_OTHER_TEXT;
  text.setAttribute("aria-label", (field.label || field.name) + " — other value");
  text.addEventListener("input", onType);
  return text;
}

function tmFormChoiceControl(field, ids) {
  var opts = Array.isArray(field.options) ? field.options : [];
  var required = field.required === true;
  var current = (field.value === null || field.value === undefined) ? "" : String(field.value);
  var matched = -1;
  opts.forEach(function (opt, i) { if (tmFormOptionValue(opt) === current) { matched = i; } });
  var otherSelected = field.allowOther === true && current !== "" && matched < 0;
  var useRadio = field.forceRadio === true || opts.length <= 5;
  var otherText = field.allowOther === true ? tmFormOtherInput(field, function () {
    if (useRadio) { otherRadio.checked = true; }
    else { select.selectedIndex = select.options.length - 1; }
  }) : null;
  var otherRadio = null;
  var select = null;
  if (otherText && otherSelected) { otherText.value = current; }

  if (useRadio) {
    var group = el("div", "paramChoices");
    group.id = ids.control;
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-labelledby", ids.label);
    var inputs = [];
    var addRadio = function (value, fill, checked) {
      var row = el("label", "paramOpt");
      var input = el("input");
      input.type = "radio";
      input.name = ids.control;
      input.value = value;
      if (required) { input.required = field.enforceRequired === true; }
      input.checked = checked;
      row.appendChild(input);
      fill(row);
      group.appendChild(row);
      inputs.push(input);
      return input;
    };
    if (!required) {
      addRadio("", function (row) { row.appendChild(el("span", "paramOptText", "—")); },
        current === "" && !otherSelected);
    }
    opts.forEach(function (opt, i) {
      addRadio(tmFormOptionValue(opt), function (row) { tmFormAddOptionText(row, opt); },
        i === matched);
    });
    if (otherText) {
      otherRadio = addRadio("\u0000other", function (row) {
        row.appendChild(el("span", "paramOptText", TM_OTHER_TEXT));
        row.appendChild(otherText);
      }, otherSelected);
    }
    return { controls: inputs.concat(otherText ? [otherText] : []), target: inputs[0] || otherText,
      aria: [group], group: group,
      read: function () {
        var i;
        for (i = 0; i < inputs.length; i++) {
          if (!inputs[i].checked) { continue; }
          if (inputs[i] === otherRadio) {
            var typed = otherText.value.trim();
            return typed === "" ? null : typed;
          }
          return inputs[i].value === "" ? null : inputs[i].value;
        }
        return null;
      } };
  }

  select = el("select", "paramSelect");
  select.id = ids.control;
  if (required) { select.required = field.enforceRequired === true; }
  var addOption = function (value, fill, selected, other) {
    var option = el("option");
    option.value = value;
    option.textContent = fill;
    if (other) { option.setAttribute("data-other", "1"); }
    option.selected = selected;
    select.appendChild(option);
  };
  if (!required) { addOption("", "—", current === "" && !otherSelected, false); }
  else if (current === "") { addOption("", "Select…", true, false); }
  opts.forEach(function (opt, i) {
    var parts = tmFormOptionParts(opt);
    addOption(tmFormOptionValue(opt), parts.muted ? parts.text + " (" + parts.muted + ")" : parts.text,
      i === matched, false);
  });
  if (otherText) { addOption("\u0000other", TM_OTHER_TEXT, otherSelected, true); }
  return { controls: [select].concat(otherText ? [otherText] : []), target: select,
    aria: [select], group: null,
    read: function () {
      var option = select.options[select.selectedIndex];
      if (!option) { return null; }
      if (option.getAttribute("data-other") === "1") {
        var typed = otherText.value.trim();
        return typed === "" ? null : typed;
      }
      return option.value === "" ? null : option.value;
    } };
}

function tmFormMultiControl(field, ids) {
  var opts = Array.isArray(field.options) ? field.options : [];
  var chosen = Array.isArray(field.value) ? field.value.map(String) : [];
  var group = el("div", "paramChoices");
  group.id = ids.control;
  group.setAttribute("role", "group");
  group.setAttribute("aria-labelledby", ids.label);
  var boxes = [];
  var known = {};
  opts.forEach(function (opt) {
    var value = tmFormOptionValue(opt);
    known[value] = true;
    var row = el("label", "paramOpt");
    var box = el("input");
    box.type = "checkbox";
    box.value = value;
    box.checked = chosen.indexOf(value) !== -1;
    row.appendChild(box);
    tmFormAddOptionText(row, opt);
    group.appendChild(row);
    boxes.push(box);
  });
  var otherBox = null;
  var otherText = null;
  if (field.allowOther === true) {
    var leftover = chosen.filter(function (v) { return !known[v]; });
    otherText = tmFormOtherInput(field, function () { otherBox.checked = true; });
    var otherRow = el("label", "paramOpt");
    otherBox = el("input");
    otherBox.type = "checkbox";
    otherBox.checked = leftover.length > 0;
    otherText.value = leftover.join(", ");
    otherRow.appendChild(otherBox);
    otherRow.appendChild(el("span", "paramOptText", TM_OTHER_TEXT));
    otherRow.appendChild(otherText);
    group.appendChild(otherRow);
  }
  var controls = boxes.slice();
  if (otherBox) { controls.push(otherBox, otherText); }
  return { controls: controls, target: controls[0] || null, aria: [group], group: group,
    read: function () {
      var picked = boxes.filter(function (b) { return b.checked; })
        .map(function (b) { return b.value; });
      if (otherBox && otherBox.checked && otherText.value.trim() !== "") {
        picked.push(otherText.value.trim());
      }
      return picked;
    } };
}

var TM_SECRET_NOTICE = "Not collected here: a value typed into this form is sent and " +
  "recorded as chat text. Supply it another way, or type the command yourself if you accept that.";

/* A password row: a label and a notice, no control. The group is named by the
   label and described by the notice. */
function tmFormSecretField(field, uid, index) {
  var name = String(field.name);
  var ids = { label: uid + "-l" + index, note: uid + "-n" + index };
  var wrap = el("div", "paramField kind-password paramSecret");
  wrap.setAttribute("data-field", name);
  wrap.setAttribute("role", "group");
  wrap.setAttribute("aria-labelledby", ids.label);
  wrap.setAttribute("aria-describedby", ids.note);
  var label = el("div", "paramLabel", field.label || name);
  label.id = ids.label;
  if (field.required === true) { label.appendChild(el("span", "paramReq", " *")); }
  wrap.appendChild(label);
  var note = el("div", "paramHelp paramSecretNote", TM_SECRET_NOTICE);
  note.id = ids.note;
  wrap.appendChild(note);
  var built = { controls: [], target: null, aria: [], group: null,
    read: function () { return ""; } };
  var record = { field: field, wrap: wrap, built: built, stateNode: null, secret: true,
    clientInvalid: false };
  record.mark = function () {};
  record.unmark = function () {};
  return record;
}

function tmFormField(field, uid, index) {
  var name = String(field.name);
  var kind = field.kind || "text";
  if (kind === "password") { return tmFormSecretField(field, uid, index); }
  var ids = { control: uid + "-f" + index, label: uid + "-l" + index,
    help: uid + "-h" + index, state: uid + "-s" + index };
  var built;
  if (kind === "boolean") { built = tmFormBooleanControl(field, ids); }
  else if (kind === "choice") { built = tmFormChoiceControl(field, ids); }
  else if (kind === "multichoice") { built = tmFormMultiControl(field, ids); }
  else { built = tmFormTextControl(field, ids); }

  var wrap = el("div", "paramField kind-" + kind);
  wrap.setAttribute("data-field", name);
  var title = field.label || name;
  var describedBy = [];
  if (kind === "boolean") {
    var checkRow = el("label", "paramCheck");
    checkRow.id = ids.label;
    checkRow.setAttribute("for", ids.control);
    checkRow.appendChild(built.controls[0]);
    checkRow.appendChild(el("span", "paramLabelText", title));
    wrap.appendChild(checkRow);
  } else if (built.group) {
    wrap.appendChild(el("div", "paramLabel", title)).id = ids.label;
    wrap.appendChild(built.group);
  } else {
    var label = el("label", "paramLabel", title);
    label.id = ids.label;
    label.setAttribute("for", ids.control);
    wrap.appendChild(label);
    built.controls.forEach(function (c) { wrap.appendChild(c); });
  }
  if (field.required === true) {
    var firstLabel = wrap.firstChild;
    firstLabel.appendChild(el("span", "paramReq", " *"));
  }
  if (field.help) {
    var help = el("div", "paramHelp", String(field.help));
    help.id = ids.help;
    wrap.appendChild(help);
    describedBy.push(ids.help);
  }
  var stateNode = el("div", "paramState");
  stateNode.id = ids.state;
  stateNode.hidden = true;
  wrap.appendChild(stateNode);
  describedBy.push(ids.state);

  built.aria.forEach(function (c) {
    c.setAttribute("aria-describedby", describedBy.join(" "));
    if (field.required === true) { c.setAttribute("aria-required", "true"); }
  });
  if (field.enforceRequired === true && field.required === true && kind !== "boolean") {
    built.controls.forEach(function (c) {
      if (c.type !== "checkbox" && c.className !== "paramOther") { c.required = true; }
    });
  }

  var record = { field: field, wrap: wrap, built: built, stateNode: stateNode,
    clientInvalid: false };
  record.mark = function (state) {
    var text = state === "missing" ? "Required"
      : (state === "invalid" ? "Not accepted — see Details" : "");
    wrap.classList.add("asked");
    built.aria.concat(built.group ? built.controls : []).forEach(function (c) {
      c.setAttribute("aria-invalid", "true");
    });
    stateNode.textContent = text;
    stateNode.hidden = false;
  };
  record.unmark = function () {
    wrap.classList.remove("asked");
    built.aria.concat(built.group ? built.controls : []).forEach(function (c) {
      c.removeAttribute("aria-invalid");
    });
    stateNode.textContent = "";
    stateNode.hidden = true;
  };
  if (field.state === "missing" || field.state === "invalid") { record.mark(field.state); }
  return record;
}

function tmRenderForm(model, actions) {
  var spec = model || {};
  var fields = Array.isArray(spec.fields) ? spec.fields : [];
  tmFormSeq += 1;
  var uid = "pf" + tmFormSeq;
  var root = el("div", "paramForm");
  root.setAttribute("role", "group");
  if (spec.title) {
    var heading = el("div", "paramTitle", String(spec.title));
    heading.id = uid + "-t";
    root.setAttribute("aria-labelledby", heading.id);
    root.appendChild(heading);
  }
  var body = el("div", "paramFields");
  var records = fields.map(function (field, index) {
    var record = tmFormField(field, uid, index);
    body.appendChild(record.wrap);
    return record;
  });
  root.appendChild(body);
  var disabled = false;
  var blocked = records.filter(function (r) { return r.secret && r.field.required === true; });

  function values() {
    var out = {};
    records.forEach(function (r) { out[String(r.field.name)] = r.built.read(); });
    return out;
  }

  /* Client-side required check; only fields that ask for it (enforceRequired). */
  function checkRequired() {
    var first = null;
    records.forEach(function (r) {
      if (r.clientInvalid) { r.clientInvalid = false; if (!r.field.state) { r.unmark(); } }
    });
    records.forEach(function (r) {
      var f = r.field;
      if (f.enforceRequired !== true || f.required !== true || f.kind === "boolean") { return; }
      if (!tmFormIsEmpty(r.built.read())) { return; }
      r.clientInvalid = true;
      r.mark("missing");
      if (!first) { first = r; }
    });
    if (first && first.built.target) { first.built.target.focus(); }
    return !first;
  }

  if (blocked.length) {
    var names = blocked.map(function (r) { return r.field.label || String(r.field.name); });
    var status = el("div", "paramStatus", "Run is unavailable: " + names.join(", ") +
      (names.length === 1 ? " is" : " are") + " required and not collected here. " +
      "Supply " + (names.length === 1 ? "it" : "them") + " another way.");
    status.setAttribute("role", "status");
    root.appendChild(status);
  }
  var bar = el("div", "paramActions");
  var primaries = [];
  (actions || []).forEach(function (action) {
    var kind = action.kind === "primary" || action.kind === "ghost" ? action.kind : "secondary";
    var button = el("button", "paramBtn " + kind + (kind === "primary" ? " primary" : ""),
      action.label);
    button.type = "button";
    button.addEventListener("click", function () {
      if (disabled) { return; }
      if (kind === "primary" && !checkRequired()) { return; }
      if (typeof action.onClick === "function") { action.onClick(values()); }
    });
    if (kind === "primary" && blocked.length) { button.disabled = true; primaries.push(button); }
    bar.appendChild(button);
  });
  root.appendChild(bar);
  tmFitObserve(root);

  function setDisabled(flag) {
    disabled = !!flag;
    Array.prototype.slice.call(root.querySelectorAll("input, select, textarea, button"))
      .forEach(function (node) { node.disabled = disabled; });
    primaries.forEach(function (node) { node.disabled = true; });
    if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
    root.setAttribute("aria-disabled", disabled ? "true" : "false");
  }

  function focusFirstAsked() {
    var i;
    for (i = 0; i < records.length; i++) {
      var target = records[i].built.target;
      if (records[i].wrap.classList.contains("asked") && target) {
        target.focus();
        return true;
      }
    }
    return false;
  }

  return { element: root, values: values, setDisabled: setDisabled,
    focusFirstAsked: focusFirstAsked };
}

/* -- live forms and the composer ----------------------------------------- */

function tmDisableLiveForms() {
  tmLiveForms.forEach(function (live) { live.setDisabled(true); });
  tmLiveForms = tmLiveForms.filter(function (live) {
    return live.element.parentNode !== null && live.element.ownerDocument.contains(live.element);
  });
}

function tmRegisterLiveForm(live) {
  if (tmLiveForms.indexOf(live) === -1) { tmLiveForms.push(live); }
}

/* A reply is ordinary chat text through the composer, so it is recorded
   verbatim. A draft already in the composer is put back afterwards. Refuses
   (returns false) while a turn runs or nothing is connected. */
function tmSendChatText(text, opts) {
  if (!tm.connected || tm.busy) { return false; }
  var input = document.getElementById("chatInput");
  var draft = input.value;
  input.value = text;
  /* opts.usePlan: the planning command to run this one message with (fw_not_run's plan buttons). */
  tm.nextUsePlan = opts && opts.usePlan ? String(opts.usePlan) : null;
  tmSend();
  tm.nextUsePlan = null;
  var sent = input.value === "";
  input.value = draft;
  tmSizeComposer();
  return sent;
}

function tmFillComposer(text) {
  var input = document.getElementById("chatInput");
  input.value = text;
  tmSizeComposer();
  if (!input.disabled) {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
}

/* The turn is live when no later message follows it in the log. */
function tmIsLiveTurn(bubbleMsg) {
  var next = bubbleMsg.nextElementSibling;
  while (next) {
    if (next.classList && next.classList.contains("chatMsg")) { return false; }
    next = next.nextElementSibling;
  }
  return true;
}

function tmLastPromptArtifacts(out) {
  var list = out && out.command_outputs;
  if (!Array.isArray(list) || !list.length) { return null; }
  var last = list[list.length - 1];
  var resp = last && last.command_response;
  if (!resp || typeof resp !== "object") { return null; }
  return { entry: last, artifacts: tmIsPlainObject(resp.artifacts) ? resp.artifacts : {},
    text: typeof resp.response === "string" ? resp.response : String(out.answer || "") };
}

/* Puts the prompt (form or pills) and the verbatim server text, folded into
   Details, where the answer text was. The prompt owns the bubble: the
   results link and panel may have been built before it, so they are taken
   away rather than left to fight over it. */
function tmMountPrompt(bubbleMsg, rec) {
  var bubble = bubbleMsg.querySelector(".bubble");
  if (!bubble) { return null; }
  clear(bubble);
  bubble.classList.remove("md");
  rec.nodes.forEach(function (node) { bubble.appendChild(node); });
  tmFitAll(bubble);
  tmClearResultsUi(bubbleMsg);
  Array.prototype.slice.call(bubbleMsg.children).forEach(function (node) {
    if (node.classList.contains("cmdOut")) { bubbleMsg.removeChild(node); }
  });
  bubbleMsg.classList.add("hasForm");
  bubbleMsg.tmPrompt = rec;
  tmRegisterLiveForm(rec);
  return rec.element;
}

function tmDetailsFor(text, open) {
  var details = el("details", "paramDetails");
  details.appendChild(el("summary", null, "Details"));
  details.appendChild(el("div", "paramDetailsBody", text));
  if (open) { details.open = true; }
  return details;
}

/* A second paint of the same prompt reuses the first one's element, so what
   was typed survives and nothing is doubled. */
function tmReusePrompt(bubbleMsg, key) {
  var prev = bubbleMsg.tmPrompt;
  return prev && prev.key === key ? prev : null;
}

/* -- command reply composition ------------------------------------------- */

/* What _extract_parameters_from_key_values binds: a value runs to the next
   " name=" of a field, its outer quotes go, and unquoted it loses a trailing
   "(note)" and trailing "," or ";". A value is therefore quoted when it
   holds "(" or ";", ends with ",", has a newline (made a space: the reply is
   one line), leading or trailing space, or is wrapped in quotes. Double
   quotes, or single quotes when the value holds a double quote. The literal
   text None/null is read as no value even in quotes, so quoting cannot help
   it. */
function tmCommandValue(value) {
  if (Array.isArray(value)) { return JSON.stringify(value); }
  if (typeof value === "boolean" || typeof value === "number") { return String(value); }
  var text = String(value);
  var flat = text.replace(/\s*[\r\n]+\s*/g, " ");
  var wrapped = flat.length >= 2 && flat.charAt(0) === flat.charAt(flat.length - 1) &&
    "\"'`".indexOf(flat.charAt(0)) !== -1;
  var quote = flat !== text || /[(;]/.test(flat) || /,$/.test(flat) || flat !== flat.trim() ||
    wrapped;
  if (!quote) { return flat; }
  var mark = flat.indexOf("\"") !== -1 ? "'" : "\"";
  return mark + flat + mark;
}

function tmComposeCommandReply(command, pairs) {
  var parts = (pairs || []).map(function (pair) {
    return pair.name + "=" + tmCommandValue(pair.value);
  });
  return "/" + tmLeafName(command) + (parts.length ? " " + parts.join(" ") : "");
}

/* -- parameter form (schema backed) -------------------------------------- */

function tmConfirmMeanings(prop) {
  var confirms = prop && prop.classifier_confirms;
  var meanings = confirms && confirms.meanings;
  return tmIsPlainObject(meanings) ? meanings : null;
}

/* The one definition of a secret parameter: a JSON-schema string with
   format "password" or writeOnly (what pydantic SecretStr produces). The
   parameter form, the approval card and the question form all use it. */
function tmIsSecretProperty(prop) {
  return !!prop && (prop.format === "password" || prop.writeOnly === true);
}

/* One form field for a schema property, or null when it has no form
   widget. `asked` fields lose their prefill. */
function tmParameterField(name, prop, required, state, given) {
  var field = { name: name, label: tmPrettyFieldTitle(name, prop), required: required,
    help: typeof prop.description === "string" ? prop.description : "", state: state };
  var ui = tmIsPlainObject(prop.ui) ? prop.ui : {};
  var prefill = (state || given === null || given === undefined) ? null : given;
  var type = prop.type;
  if (Array.isArray(prop.enum)) {
    var meanings = tmConfirmMeanings(prop);
    field.kind = "choice";
    field.required = required && prop.nullable !== true;
    field.forceRadio = ui.widget === "radio";
    field.options = prop.enum.map(function (v) {
      var value = String(v);
      var meaning = meanings && tmHas(meanings, value) ? meanings[value] : null;
      return { value: value, label: meaning ? value + " — " + meaning : value };
    });
    field.value = prefill === null ? "" : String(prefill);
  } else if (type === "boolean" && prop.nullable === true) {
    field.kind = "choice";
    field.required = false;
    field.options = [{ value: "true", label: "yes" }, { value: "false", label: "no" }];
    field.value = prefill === true || prefill === "true" ? "true"
      : (prefill === false || prefill === "false" ? "false" : "");
  } else if (type === "boolean") {
    field.kind = "boolean";
    field.value = prefill === true || prefill === "true";
  } else if (type === "integer" || type === "number") {
    field.kind = type;
    if (typeof prop.minimum === "number") { field.min = prop.minimum; }
    if (typeof prop.maximum === "number") { field.max = prop.maximum; }
    field.value = typeof prefill === "number" ? prefill : null;
  } else if (type === "array") {
    var items = tmIsPlainObject(prop.items) ? prop.items : null;
    var itemType = items && items.type;
    if (!(itemType === "string" || itemType === "integer" || itemType === "number" ||
          itemType === "boolean")) { return null; }
    field.kind = "list";
    field.value = Array.isArray(prefill) ? prefill : null;
  } else if (type === "string") {
    var writeOnly = tmIsSecretProperty(prop);
    field.kind = writeOnly ? "password" : (prop.format === "date" ? "date"
      : (ui.widget === "textarea" ? "longtext" : "text"));
    /* A secret is never prefilled, whatever was given or suggested. */
    field.value = (writeOnly || typeof prefill !== "string") ? "" : prefill;
    if (field.kind === "text" && Array.isArray(prop.examples) && prop.examples.length &&
        (typeof prop.examples[0] === "string" || typeof prop.examples[0] === "number")) {
      field.placeholder = String(prop.examples[0]);
    }
  } else {
    return null;
  }
  return field;
}

function tmParameterModel(req, input) {
  var missing = Array.isArray(req.missing) ? req.missing : [];
  var invalid = Array.isArray(req.invalid) ? req.invalid : [];
  var given = tmIsPlainObject(req.values) ? req.values : {};
  var required = Array.isArray(input.required) ? input.required : [];
  var fields = [];
  var ok = true;
  Object.keys(input.properties).forEach(function (name) {
    var prop = tmSchemaProperty(input, name);
    var state = missing.indexOf(name) !== -1 ? "missing"
      : (invalid.indexOf(name) !== -1 ? "invalid" : null);
    var field = prop ? tmParameterField(name, prop, required.indexOf(name) !== -1, state,
      tmHas(given, name) ? given[name] : null) : null;
    if (field) { fields.push(field); }
    else if (state) { ok = false; }
  });
  return ok && fields.length ? fields : null;
}

/* The pairs of a reply. A password parameter is never in it. */
function tmParameterPairs(fields, values) {
  var pairs = [];
  fields.forEach(function (f) {
    if (f.kind === "password") { return; }
    var value = values[f.name];
    if (tmFormIsEmpty(value)) { return; }
    pairs.push({ name: f.name, value: value });
  });
  return pairs;
}

function tmFocusSoon(form) {
  form.focusFirstAsked();
  /* tmComposerState focuses the composer when the turn settles, which is
     after this paint; take the focus back unless the person has moved it. */
  setTimeout(function () {
    var active = document.activeElement;
    var composer = document.getElementById("chatInput");
    if (!active || active === document.body || active === composer) { form.focusFirstAsked(); }
  }, 0);
}

function tmRenderParameterForm(bubbleMsg, out) {
  if (bubbleMsg.dataset.restored === "1") { return null; }
  if (!tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  var req = last && last.artifacts.parameter_request;
  if (!tmIsPlainObject(req) || typeof req.command !== "string" || !req.command) { return null; }
  /* A tagged prompt (an agent's planning refusal) is another part's; this form
     is for a human's failed "/" command, whose output has no question_kind. */
  if (tmHas(last.artifacts, "question_kind")) { return null; }
  var schema = tmCommandSchema(req.command);
  var input = schema && schema.input;
  if (!tmIsPlainObject(input) || !tmIsPlainObject(input.properties) ||
      !Object.keys(input.properties).length) { return null; }
  var key = "param:" + JSON.stringify(req);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }
  var fields = tmParameterModel(req, input);
  if (!fields) { return null; }
  var invalid = Array.isArray(req.invalid) ? req.invalid : [];
  var leaf = tmLeafName(req.command);
  var form = tmRenderForm({ title: "Parameters for " + leaf, fields: fields }, [
    { label: "Run", kind: "primary", onClick: function (values) {
      tmSendChatText(tmComposeCommandReply(leaf, tmParameterPairs(fields, values)));
    } },
    { label: "Edit as text", kind: "secondary", onClick: function (values) {
      /* A password is never copied into the visible composer. */
      tmFillComposer(tmComposeCommandReply(leaf, tmParameterPairs(fields, values)));
    } },
    { label: "Cancel", kind: "ghost", onClick: function () { tmSendChatText("abort"); } }
  ]);
  var rec = { key: key, element: form.element, setDisabled: form.setDisabled,
    nodes: [form.element, tmDetailsFor(last.text, invalid.length > 0)] };
  var mounted = tmMountPrompt(bubbleMsg, rec);
  if (mounted) { tmFocusSoon(form); }
  return mounted;
}

/* -- ambiguous intent pills ---------------------------------------------- */

function tmCommandTooltip(name) {
  var entry = tmCommandSchema(name);
  if (!entry) { return ""; }
  var parts = [entry.input, entry.output];
  var i;
  for (i = 0; i < parts.length; i++) {
    var p = parts[i];
    if (p && typeof p.description === "string" && p.description.trim()) {
      return p.description.trim();
    }
  }
  return "";
}

function tmRenderIntentChoices(bubbleMsg, out) {
  if (bubbleMsg.dataset.restored === "1") { return null; }
  if (!tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  var raw = last && last.artifacts.intent_choices;
  if (!Array.isArray(raw)) { return null; }
  var seen = {};
  var names = [];
  raw.forEach(function (cand) {
    if (typeof cand !== "string" || !cand.trim()) { return; }
    var leaf = tmLeafName(cand.trim());
    if (leaf && !seen[leaf]) { seen[leaf] = cand.trim(); names.push(cand.trim()); }
  });
  if (!names.length) { return null; }
  var key = "intent:" + JSON.stringify(raw);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }

  var root = el("div", "intentChoices");
  root.setAttribute("role", "group");
  var labelNode = el("div", "intentLabel", "Multiple commands matched — choose one:");
  labelNode.id = "intentLabel" + (tmFormSeq += 1);
  root.setAttribute("aria-labelledby", labelNode.id);
  root.appendChild(labelNode);
  var pills = el("div", "intentPills");
  var disabled = false;
  function add(parent, label, cls, text, tooltip) {
    var button = el("button", cls, label);
    button.type = "button";
    if (tooltip) { button.title = tooltip; }
    button.addEventListener("click", function () {
      if (!disabled) { tmSendChatText(text); }
    });
    parent.appendChild(button);
  }
  names.forEach(function (cand) {
    var leaf = tmLeafName(cand);
    add(pills, leaf, "intentPill", leaf, tmCommandTooltip(cand));
  });
  root.appendChild(pills);
  var more = el("div", "intentActions");
  add(more, "See all commands", "intentMore secondary", "what can i do?", "");
  add(more, "Cancel", "intentCancel ghost", "abort", "");
  root.appendChild(more);

  var rec = { key: key, element: root,
    setDisabled: function (flag) {
      disabled = !!flag;
      Array.prototype.slice.call(root.querySelectorAll("button")).forEach(function (b) {
        b.disabled = disabled;
      });
      if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
    },
    nodes: [root, tmDetailsFor(last.text, false)] };
  return tmMountPrompt(bubbleMsg, rec);
}

var TM_PROMPT_RENDERERS = [tmRenderParameterForm, tmRenderIntentChoices];

/* The one call tmRenderTurn makes. Anything it painted last time is gone
   (the bubble was cleared); a prompt is rebuilt, or reused when the paint
   carries the same request. */
function tmRenderPromptForms(bubbleMsg, out) {
  bubbleMsg.classList.remove("hasForm");
  var node = null;
  var i;
  for (i = 0; i < TM_PROMPT_RENDERERS.length && !node; i++) {
    node = TM_PROMPT_RENDERERS[i](bubbleMsg, out);
  }
  if (!node) { bubbleMsg.tmPrompt = null; }
  return node;
}
