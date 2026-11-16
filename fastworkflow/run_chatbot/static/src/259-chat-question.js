/* -- question form [fix-weih.11] ---------------------------------------- */
/* A turn that waits on the agent's question gets a form instead of a bare
   composer. Two kinds of question, told apart by artifacts.question_kind:

   "planning_parameter": a planning command's input was refused before any step
   ran. The form is built from that command's Input schema in GET
   /command_schemas (the same tmParameterField as the "/" parameter form),
   only for the fields asked about, with NO model call and whatever the
   "Rich answers" switch says.

   "agent": an open-ended ask_user. POST /compose_question_form turns the prose
   into fields grounded in the turn's data. The plain question stays visible
   and the composer stays usable while "Preparing a form…" shows; a null, failed
   or late answer changes nothing. Every option value and label from the server
   is shown as text only [R22].

   Path A's reply is NOT parsed in code. The server appends it to the original
   request and the parameter-extraction model reads the whole text, so each line
   is "<field name>: <value>" with the field name exactly as in the schema (the
   extractor is shown field names, never the pretty titles on screen). This was
   not exercised against a live model. Path B has no schema field behind a
   composed question, so it sends "<Label>: <value>" for the agent to read.

   Approval, review, unknown and untagged prompts never reach this part: the
   renderer returns null for them. Both kinds answer with plain chat text, one
   "<Label>: <value>" line per answered field. "Edit in composer" copies those
   lines into the composer (the form stays enabled) so the person can add what
   a small model left out.

   The composed form is cached per turn_key in tm.questionCache, stamped with
   the epoch it was made in, so a source switch (tm.epoch++) discards it
   without a change to the boot code. composeUnavailable is shared with the
   answer card: a request is not made once it is set, so a second notice cannot
   appear; only the answer that sets it shows one. */

var TM_QUESTION_TIMEOUT_MS = 25000;
var TM_QUESTION_MAX_CHARS = 20000;   /* the server refuses a longer question */
var TM_QUESTION_MAX_FIELDS = 12;
var tmQuestionFlights = {};
tm.questionCache = { epoch: tm.epoch, forms: {} };

function tmQuestionForms() {
  if (!tm.questionCache || tm.questionCache.epoch !== tm.epoch) {
    tm.questionCache = { epoch: tm.epoch, forms: {} };
  }
  return tm.questionCache.forms;
}

function tmQuestionText(value) {
  return (typeof value === "string" || typeof value === "number") ? String(value).trim() : "";
}

/* -- the reply ----------------------------------------------------------- */

function tmQuestionValueText(value) {
  if (typeof value === "boolean") { return value ? "Yes" : "No"; }
  if (Array.isArray(value)) {
    return value.map(function (v) { return String(v).trim(); })
      .filter(function (v) { return v !== ""; }).join(", ");
  }
  return String(value).replace(/\s*[\r\n]+\s*/g, " ").trim();
}

/* One "<wire>: <value>" line per answered field, in form order. The wire name
   is the field's wireLabel when it has one (path A: the schema field name),
   else its label (path B). */
function tmQuestionReply(fields, values) {
  var lines = [];
  fields.forEach(function (f) {
    var value = values[f.name];
    if (tmFormIsEmpty(value)) { return; }
    var text = tmQuestionValueText(value);
    var name = typeof f.wireLabel === "string" && f.wireLabel ? f.wireLabel : f.label;
    if (text) { lines.push(String(name).replace(/\s+/g, " ").trim() + ": " + text); }
  });
  return lines.join("\n");
}

function tmQuestionActions(fields) {
  return [
    { label: "Send answer", kind: "primary", onClick: function (values) {
      var reply = tmQuestionReply(fields, values);
      if (reply) { tmSendChatText(reply); }
    } },
    { label: "Edit in composer", kind: "secondary", onClick: function (values) {
      var reply = tmQuestionReply(fields, values);
      if (reply) { tmFillComposer(reply); return; }
      var input = document.getElementById("chatInput");
      if (input && !input.disabled) { input.focus(); }
    } }
  ];
}

/* Mounts the form with the question text kept visible above it. */
function tmQuestionMount(bubbleMsg, key, title, fields, text, focus) {
  var form = tmRenderForm({ title: title, fields: fields }, tmQuestionActions(fields));
  var question = el("div", "questionText", text);
  var rec = { key: key, element: form.element, setDisabled: form.setDisabled,
    nodes: [question, form.element] };
  var mounted = tmMountPrompt(bubbleMsg, rec);
  if (mounted && focus) { tmFocusSoon(form); }
  return mounted;
}

/* -- path A: planning_parameter ------------------------------------------- */

function tmQuestionAskedNames(req, input) {
  var missing = Array.isArray(req.missing) ? req.missing : [];
  var invalid = Array.isArray(req.invalid) ? req.invalid : [];
  var names = Object.keys(input.properties);
  var asked = [];
  var ok = true;
  [["missing", missing], ["invalid", invalid]].forEach(function (group) {
    names.forEach(function (name) {
      if (group[1].indexOf(name) !== -1 && !asked.some(function (a) { return a.name === name; })) {
        asked.push({ name: name, state: group[0] });
      }
    });
    group[1].forEach(function (name) {
      if (typeof name !== "string" || names.indexOf(name) === -1) { ok = false; }
    });
  });
  return ok && asked.length ? asked : null;
}

/* The fields of the asked names only, or null when one has no widget. A
   password is never asked for in chat text, so it has no form either. */
function tmQuestionPlanningFields(req, input) {
  var asked = tmQuestionAskedNames(req, input);
  if (!asked) { return null; }
  var fields = [];
  var i;
  for (i = 0; i < asked.length; i++) {
    var prop = tmSchemaProperty(input, asked[i].name);
    /* An asked field is one the call needs, so it is required here. */
    var field = prop ? tmParameterField(asked[i].name, prop, true, asked[i].state, null) : null;
    if (!field || field.kind === "password") { return null; }
    field.enforceRequired = true;
    field.wireLabel = asked[i].name;
    if (field.kind === "choice") { field.allowOther = false; }
    fields.push(field);
  }
  return fields;
}

/* The plain question as the prompt, with no form. It is mounted (not left to
   the bubble) so that the built-in "/" parameter form of 256, which matches
   any turn that carries a parameter_request, does not paint the whole Input
   schema and send a "/command name=value" line as the agent's answer. */
function tmQuestionPlain(bubbleMsg, text) {
  var key = "question:plain:" + text;
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }
  var node = el("div", "questionText", text);
  return tmMountPrompt(bubbleMsg, { key: key, element: node, setDisabled: function () {}, nodes: [node] });
}

/* A chosen plan (use_plan) asks for its parameters, not for a review: the intent is settled, so the form shows the
   plan's name and description (read-only) and one field per parameter of its Signature, the drafted values prefilled
   (parameter_request.parameters: name, required, value, state "missing" | "invalid" | null). Run sends one
   "<name>: <value>" line per filled field (the server reads exactly that); Cancel sends "cancel". No goal text exists
   here to edit. Null (the plain question stays) when a field has no widget. */
function tmQuestionPrefill(prop, value) {
  if (value === null || value === undefined || value === "") { return null; }
  if (prop.type === "integer" || prop.type === "number") {
    var n = Number(value);
    return isFinite(n) ? n : null;
  }
  if (prop.type === "array") { return null; }
  return value;
}

function tmQuestionPlanChoice(bubbleMsg, req, input) {
  var key = "question:P:" + JSON.stringify(req);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }
  var fields = [];
  var i;
  for (i = 0; i < req.parameters.length; i++) {
    var p = req.parameters[i];
    var prop = tmIsPlainObject(p) && typeof p.name === "string" ? tmSchemaProperty(input, p.name) : null;
    if (!prop) { return null; }
    var state = p.state === "missing" || p.state === "invalid" ? p.state : null;
    var field = tmParameterField(p.name, prop, p.required === true, state, state ? null : tmQuestionPrefill(prop, p.value));
    if (!field || field.kind === "password") { return null; }
    field.enforceRequired = p.required === true;
    field.wireLabel = p.name;
    if (field.kind === "choice") { field.allowOther = false; }
    fields.push(field);
  }
  var head = el("div", "planChosen");
  head.appendChild(el("div", "planChosenName", tmLeafName(req.command)));
  if (typeof req.description === "string" && req.description) {
    head.appendChild(el("div", "planChosenDesc", req.description));
  }
  var form = tmRenderForm({ title: "Parameters", fields: fields }, [
    { label: "Run", kind: "primary", onClick: function (values) {
      tmSendChatText(tmQuestionReply(fields, values) || "yes");
    } },
    { label: "Cancel", kind: "ghost", onClick: function () { tmSendChatText("cancel"); } }
  ]);
  var rec = { key: key, element: form.element, setDisabled: form.setDisabled, nodes: [head, form.element] };
  var mounted = tmMountPrompt(bubbleMsg, rec);
  if (mounted) { tmFocusSoon(form); }
  return mounted;
}

function tmQuestionRenderPlanning(bubbleMsg, last) {
  var req = last.artifacts.parameter_request;
  if (!tmIsPlainObject(req) || typeof req.command !== "string" || !req.command) {
    return tmQuestionPlain(bubbleMsg, last.text);
  }
  if (!tm.commandSchemas) {
    /* The listing is still loading (a restored turn paints early): paint
       again once it is here, unless something else has taken the bubble. */
    if (tm.commandSchemasPromise) {
      var epoch = tm.epoch;
      var late = bubbleMsg.tmOut;
      tm.commandSchemasPromise.then(function () {
        if (tm.epoch !== epoch || !tm.commandSchemas || bubbleMsg.tmPrompt || !late ||
            !document.body.contains(bubbleMsg) || !tmIsLiveTurn(bubbleMsg)) { return; }
        tmRenderPromptForms(bubbleMsg, late);
      });
    }
    return null;
  }
  var schema = tmCommandSchema(req.command);
  var input = schema && schema.input;
  if (!tmIsPlainObject(input) || !tmIsPlainObject(input.properties)) {
    return tmQuestionPlain(bubbleMsg, last.text);
  }
  if (req.confirm === true && Array.isArray(req.parameters) && req.parameters.length) {
    return tmQuestionPlanChoice(bubbleMsg, req, input) || tmQuestionPlain(bubbleMsg, last.text);
  }
  var key = "question:A:" + JSON.stringify(req);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }
  var fields = tmQuestionPlanningFields(req, input);
  if (!fields) { return tmQuestionPlain(bubbleMsg, last.text); }
  return tmQuestionMount(bubbleMsg, key, "More information for " + tmLeafName(req.command),
    fields, last.text, true);
}

/* -- path B: agent ask_user ----------------------------------------------- */

function tmQuestionOptions(raw) {
  var opts = [];
  var seen = {};
  (Array.isArray(raw) ? raw : []).forEach(function (o) {
    if (!tmIsPlainObject(o)) { return; }
    var value = tmQuestionText(o.value);
    if (!value || tmHas(seen, value)) { return; }
    seen[value] = true;
    opts.push({ value: value, label: tmQuestionText(o.label) || value });
  });
  return opts;
}

/* The composed fields as tmRenderForm fields. A field the server sent that
   cannot be shown (no label, or a choice with no options) is dropped, or
   becomes text, never an error. */
function tmQuestionComposedFields(form) {
  var raw = tmIsPlainObject(form) && Array.isArray(form.fields) ? form.fields : [];
  var fields = [];
  var used = {};
  raw.forEach(function (f, i) {
    if (fields.length >= TM_QUESTION_MAX_FIELDS || !tmIsPlainObject(f)) { return; }
    var label = tmQuestionText(f.label);
    if (!label) { return; }
    var name = tmQuestionText(f.name) || label;
    if (tmHas(used, name)) { name = name + "#" + i; }
    used[name] = true;
    var kind = tmQuestionText(f.kind);
    var out = { name: name, label: label, required: f.required === true, enforceRequired: true,
      help: tmQuestionText(f.help), value: null };
    var opts = tmQuestionOptions(f.options);
    if (kind === "boolean") {
      out.kind = "choice";
      out.options = [{ value: "Yes", label: "Yes" }, { value: "No", label: "No" }];
      out.value = "";
    } else if ((kind === "choice" || kind === "multichoice") && opts.length) {
      out.kind = kind;
      out.options = opts;
      out.allowOther = true;
      out.value = kind === "choice" ? "" : [];
    } else if (kind === "number" || kind === "date" || kind === "longtext") {
      out.kind = kind;
    } else {
      out.kind = "text";
    }
    fields.push(out);
  });
  return fields;
}

function tmQuestionWanted(bubbleMsg) {
  var box = document.getElementById("richAnswers");
  return !!(box && box.checked) && tm.connected === true && !tm.composeUnavailable &&
    bubbleMsg.dataset.deterministic !== "1";
}

function tmQuestionStale(bubbleMsg, status, epoch) {
  return tm.epoch !== epoch || !document.body.contains(bubbleMsg) || !bubbleMsg.contains(status) ||
    !tmIsLiveTurn(bubbleMsg) || !!bubbleMsg.tmPrompt;
}

function tmQuestionDrop(node) {
  if (node && node.parentNode) { node.parentNode.removeChild(node); }
}

/* Always resolves {form, reason}; never rejects. Aborts after 25 s. */
function tmQuestionFetch(payload) {
  return new Promise(function (resolve) {
    var done = false;
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timer = null;
    function finish(reply) {
      if (done) { return; }
      done = true;
      clearTimeout(timer);
      resolve(reply);
    }
    timer = setTimeout(function () {
      finish({ form: null, reason: "timeout" });
      if (controller) { controller.abort(); }
    }, TM_QUESTION_TIMEOUT_MS);
    tmFetch("/compose_question_form", payload, controller ? { signal: controller.signal } : undefined)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data || typeof data !== "object") { finish({ form: null, reason: "failed" }); return; }
        finish(tmIsPlainObject(data.form) ? { form: data.form, reason: "" }
          : { form: null, reason: String(data.reason || "failed") });
      })
      .catch(function () { finish({ form: null, reason: "failed" }); });
  });
}

function tmQuestionApply(bubbleMsg, out, form, text) {
  var fields = tmQuestionComposedFields(form);
  if (!fields.length) { return null; }
  var key = "question:B:" + (out.turn_key || "") + ":" + JSON.stringify(fields);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }
  return tmQuestionMount(bubbleMsg, key, "Your answer", fields, text, false);
}

function tmQuestionRequest(bubbleMsg, out, text, status, epoch) {
  var turnKey = out.turn_key ? String(out.turn_key) : "";
  tmWithCommandSchemas(function (schemas) {
    if (tm.epoch !== epoch || !bubbleMsg.contains(status)) { tmQuestionDrop(status); return; }
    var flightKey = turnKey ? epoch + ":" + turnKey : "";
    var flight = flightKey ? tmQuestionFlights[flightKey] : null;
    if (!flight) {
      var results = tmResultsOf(out.command_outputs, schemas);
      var cache = document.getElementById("useDspyCache");
      flight = tmQuestionFetch({
        question: text,
        user_message: bubbleMsg.dataset.userMessage || "",
        results: tmCardPayload(bubbleMsg, out, results, schemas).results,
        use_dspy_cache: !!(cache && cache.checked),
        turn_key: turnKey
      });
      if (flightKey) {
        tmQuestionFlights[flightKey] = flight;
        flight.then(function () { delete tmQuestionFlights[flightKey]; });
      }
    }
    flight.then(function (reply) {
      var kept = false;
      if (tm.epoch === epoch) {
        if (reply.reason === "no_model" && !tm.composeUnavailable) {
          tm.composeUnavailable = true;
          if (!tmQuestionStale(bubbleMsg, status, epoch)) {
            status.textContent = "Composed question forms are unavailable: the server has no LLM_RESPONSE_GEN model.";
            status.className = "questionStatus questionNotice";
            kept = true;
          }
        }
        if (reply.form) {
          if (turnKey) { tmQuestionForms()[turnKey] = reply.form; }
          if (!tmQuestionStale(bubbleMsg, status, epoch)) { tmQuestionApply(bubbleMsg, out, reply.form, text); }
        }
      }
      if (!kept) { tmQuestionDrop(status); }
    });
  });
}

function tmQuestionRenderAgent(bubbleMsg, out, last) {
  if (!tmQuestionWanted(bubbleMsg)) { return null; }
  var text = String(out.answer === null || out.answer === undefined ? last.text : out.answer);
  if (!text.trim() || text.length > TM_QUESTION_MAX_CHARS) { return null; }
  var turnKey = out.turn_key ? String(out.turn_key) : "";
  var forms = tmQuestionForms();
  if (turnKey && tmHas(forms, turnKey)) {
    return tmQuestionApply(bubbleMsg, out, forms[turnKey], text);
  }
  var bubble = bubbleMsg.querySelector(".bubble");
  if (!bubble) { return null; }
  var status = el("div", "questionStatus", "Preparing a form…");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  bubble.appendChild(status);
  tmQuestionRequest(bubbleMsg, out, text, status, tm.epoch);
  return null;
}

/* -- the renderer --------------------------------------------------------- */

/* Only a turn that waits, on an open ask_user entry, tagged "agent" or
   "planning_parameter"; every other prompt, and every other turn, is null. */
function tmRenderQuestionForm(bubbleMsg, out) {
  if (!out || out.status !== "awaiting_user" || !tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  if (!last) { return null; }
  var kind = last.artifacts.question_kind;
  if (kind !== "agent" && kind !== "planning_parameter") { return null; }
  var entry = last.entry;
  var resp = entry && entry.command_response;
  if (!resp || resp.success !== false ||
      !(entry.ask_user_entry === true || entry.command_name === "ask_user")) { return null; }
  bubbleMsg.tmOut = out;
  return kind === "planning_parameter" ? tmQuestionRenderPlanning(bubbleMsg, last)
    : tmQuestionRenderAgent(bubbleMsg, out, last);
}

/* unshift, not push: the built-in "/" parameter form (256) also matches any
   turn that carries a parameter_request, and would paint the whole Input
   schema for an agent turn's planning_parameter question. This renderer
   answers null for everything but its own two kinds, so running first costs
   the other prompts nothing. */
TM_PROMPT_RENDERERS.unshift(tmRenderQuestionForm);
