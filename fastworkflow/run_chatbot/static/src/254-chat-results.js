/* -- chat results view [fix-weih.4] ------------------------------------- */
/* Every command that declares Signature.Output returns it as
   command_outputs[i].command_response.output. This part turns those objects
   into a card (stat strip, tables, badges, collapsed upstream lookups, a JSON
   toggle) with no model call and no author hint. The turn shows it in a side
   panel behind a link under the answer (255); the answer is never folded. The
   data layer (tmNormalizeOutputObject, tmResultsOf,
   tmResultGroups, tmFormatValue, tmAutoTone) is pure; the renderers build DOM
   through el()/textContent only [R22], and agent prose goes through
   appendMarkdown. Schemas are optional help: with none, a result still
   renders. */

var TM_PAGING_PARAMETERS = ["start"];
var TM_RESULT_ROWS_SHOWN = 50;
var TM_RESULT_ROWS_MAX = 1000;
var TM_RESULT_CELL_CHARS = 120;
var TM_RESULT_MAX_DEPTH = 3;
var TM_RESULT_SHORT_TEXT = 32;
/* A small generic vocabulary only; every other word is neutral. Domain words belong to the application. */
var TM_TONES = {
  good: ["success", "successful", "ok", "done", "active", "enabled", "passed", "approved", "complete", "completed"],
  warning: ["warning", "pending", "waiting", "in_progress"],
  bad: ["error", "failed", "denied", "blocked", "expired", "rejected"]
};
var TM_AUTHOR_TONES = ["good", "warning", "bad", "neutral"];
var TM_NOT_JSON = {};
var tmResultsSeq = 0;

function tmHas(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj, key);
}
function tmIsPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function tmIsScalar(value) {
  return value === null || value === undefined || typeof value !== "object";
}

/* -- normalization (R1, R2) --------------------------------------------- */

function tmParseJsonValue(value) {
  if (typeof value !== "string") { return TM_NOT_JSON; }
  var text = value.trim();
  var first = text.charAt(0);
  if (first !== "{" && first !== "[") { return TM_NOT_JSON; }
  var parsed;
  try { parsed = JSON.parse(text); } catch (e) { return TM_NOT_JSON; }
  return (parsed !== null && typeof parsed === "object") ? parsed : TM_NOT_JSON;
}

function tmDecodeJsonValues(obj) {
  var out = {};
  Object.keys(obj).forEach(function (key) {
    var parsed = tmParseJsonValue(obj[key]);
    out[key] = parsed === TM_NOT_JSON ? obj[key] : parsed;
  });
  return out;
}

var TM_UNWRAP_LIMIT = 5;

/* Returns {output, unwrappedFrom, unwrappedPath}. String values holding JSON
   are replaced by what they hold; then, up to 5 times, a one-key output whose
   value is a non-empty plain object is replaced by that object (decoded
   again). The path lists the keys peeled off; unwrappedFrom is the first. This
   mirrors run_chatbot.composer.normalize_output_dict, so a ref the server writes
   resolves against this copy. Normalizing a normalized output changes
   nothing. */
function tmNormalizeOutputObject(rawOutput) {
  var out = tmDecodeJsonValues(rawOutput);
  var path = [];
  while (path.length < TM_UNWRAP_LIMIT) {
    var keys = Object.keys(out);
    if (keys.length !== 1) { break; }
    var inner = out[keys[0]];
    if (!tmIsPlainObject(inner) || Object.keys(inner).length === 0) { break; }
    path.push(keys[0]);
    out = tmDecodeJsonValues(inner);
  }
  return { output: out, unwrappedFrom: path.length ? path[0] : null, unwrappedPath: path };
}

function tmEligibleOutput(co) {
  if (!co || typeof co !== "object") { return null; }
  var resp = co.command_response;
  if (!resp || typeof resp !== "object" || resp.success === false) { return null; }
  if (co.ask_user_entry === true || co.command_name === "ask_user") { return null; }
  var output = resp.output;
  if (!tmIsPlainObject(output) || Object.keys(output).length === 0) { return null; }
  return output;
}

/* -- pages (R3) ---------------------------------------------------------- */

function tmPageKey(co, withPaging) {
  var params = co.command_parameters;
  var base;
  if (tmIsPlainObject(params)) {
    base = JSON.stringify(Object.keys(params).filter(function (k) {
      return withPaging || TM_PAGING_PARAMETERS.indexOf(k) < 0;
    }).sort().map(function (k) { return [k, params[k]]; }));
  } else if (params === null || params === undefined) {
    base = "[]";
  } else {
    base = JSON.stringify(params);
  }
  return "k:" + String(co.command_name) + "\u0000" + String(co.context || "") +
    "\u0000" + base;
}

/* Concatenate array fields without deduplicating (parallel columns stay
   row-aligned); a non-array field takes the last page's value. */
function tmJoinPages(outputs) {
  var joined = {};
  outputs.forEach(function (output) {
    Object.keys(output).forEach(function (key) {
      var value = output[key];
      if (Array.isArray(value) && Array.isArray(joined[key])) {
        joined[key] = joined[key].concat(value);
      } else {
        joined[key] = value;
      }
    });
  });
  return joined;
}

function tmCollectPages(commandOutputs) {
  var pages = [];
  (commandOutputs || []).forEach(function (co) {
    var raw = tmEligibleOutput(co);
    if (!raw) { return; }
    var entry = { co: co, raw: raw, key: tmPageKey(co, false), full: tmPageKey(co, true) };
    var i;
    for (i = 0; i < pages.length; i++) {
      if (pages[i].full === entry.full) { pages[i] = entry; return; }  /* a re-read */
    }
    pages.push(entry);
  });
  return pages;
}

/* The request part the recorded output ran for (CommandOutput.sub_task, 1-based): a positive integer, else null
   (the turn was not split, a record from before the field). Read from the last page, like command and context. */
function tmPartOf(co) {
  var n = co ? co.sub_task : null;
  return typeof n === "number" && isFinite(n) && Math.floor(n) === n && n >= 1 ? n : null;
}

function tmBuildResult(id, pages) {
  var last = pages[pages.length - 1];
  var normalized = pages.map(function (p) { return tmNormalizeOutputObject(p.raw); });
  return {
    id: id,
    command: String(last.co.command_name || ""),
    context: String(last.co.context || ""),
    parameters: last.co.command_parameters === undefined ? null : last.co.command_parameters,
    output: tmJoinPages(normalized.map(function (n) { return n.output; })),
    outputs: pages.map(function (p) { return p.raw; }),
    supporting: false,
    subTask: tmPartOf(last.co),
    unwrappedFrom: normalized[normalized.length - 1].unwrappedFrom,
    unwrappedPath: normalized[normalized.length - 1].unwrappedPath
  };
}

/* -- schema lookup and upstream collapse (R4) ---------------------------- */

/* A recorded command_name is already the schema listing's key: the context that
   defines the command qualifies it ("TodoList/get_properties"), a global or
   planning command is bare, and an inherited command run from a child context keeps
   its defining context's name. `context` is only the label of the current record
   ("Identity: Ann", "Library"), never a context key, so it is not used to build a
   key (it could qualify a bare global command with the wrong context). The name goes
   to tmCommandSchemaIn: exact key, else a unique leaf, else unresolved. */
function tmQualifiedName(schemas, command) {
  return command;
}

function tmSchemaView(root, node) {
  if (!node || !tmIsPlainObject(node.properties)) { return null; }
  return { properties: node.properties, "$defs": root ? root["$defs"] : undefined };
}

function tmResultSchema(result, schemas) {
  var entry = tmCommandSchemaIn(schemas, tmQualifiedName(schemas, result.command));
  var root = entry && entry.output;
  if (!tmIsPlainObject(root)) { return null; }
  var schema = root;
  var path = result.unwrappedPath || [];
  var i;
  for (i = 0; i < path.length && schema; i++) {
    schema = tmSchemaView(root, tmSchemaProperty(schema, path[i]));
  }
  return schema;
}

/* A recorded output is the framework's own plumbing (the plan and sub-task reviews) when the server says so:
   `framework: true` on the output, a fact core sets from the command's metadata. Such an output is no result,
   so tmResultsOf leaves it out of the card's data, its footer and the results panel. `framework: false` is
   authoritative too: an application command that happens to share a name with one of the framework's is shown.

   LEGACY FALLBACK, only for records without the key (turns stored before the server wrote it): the names of the
   files of fastworkflow/_framework_commands, bare (a workflow's own command of the same leaf is qualified).
   DELETE TM_LEGACY_FRAMEWORK_COMMANDS, and the fallback line at the end of tmIsFrameworkOutput, when no stored turn predates the
   `framework` key (a store older than that release is no longer readable). A test keeps the list equal to
   that folder until then. */
var TM_LEGACY_FRAMEWORK_COMMANDS = ["review_proposal", "review_subtasks"];

function tmIsFrameworkOutput(co) {
  if (!co || typeof co !== "object") { return false; }
  if (co.framework === true) { return true; }
  if (co.framework === false) { return false; }
  var name = co.command_name;  /* key absent (or null): a record written before the marker existed */
  return TM_LEGACY_FRAMEWORK_COMMANDS.indexOf(String(name === null || name === undefined ? "" : name)) >= 0;
}

var TM_FEED_MIN_CHARS = 8;

/* The text values (an identifier is one) a result's output holds as a scalar or as an item of a scalar list. */
function tmOutputTokens(output) {
  var tokens = {};
  Object.keys(output || {}).forEach(function (key) {
    var value = output[key];
    var items = Array.isArray(value) ? value : [value];
    items.forEach(function (item) {
      if (typeof item === "string" && item.length >= TM_FEED_MIN_CHARS) { tokens[item] = true; }
    });
  });
  return tokens;
}

/* Whether `later` was called with a value that `earlier` returned (a lookup's identifier passed on as an
   argument): the data says the earlier result only fed the later call. */
function tmFeedsLater(tokens, later) {
  var params = later.parameters;
  if (!tmIsPlainObject(params)) { return false; }
  return Object.keys(params).some(function (key) {
    var value = params[key];
    var items = Array.isArray(value) ? value : [value];
    return items.some(function (item) { return typeof item === "string" && tmHas(tokens, item); });
  });
}

/* A result is supporting when a later result of another command takes it as input: its schema says an
   Input field is available_from the earlier command, or the later call's arguments hold a value the
   earlier output returned. */
function tmMarkSupporting(results, schemas) {
  var upstream = results.map(function (r) {
    return tmUpstreamProducerCommands(schemas, tmQualifiedName(schemas, r.command));
  });
  results.forEach(function (earlier, i) {
    var tokens = tmOutputTokens(earlier.output);
    var j;
    for (j = i + 1; j < results.length; j++) {
      var later = results[j];
      if (later.command === earlier.command) { continue; }
      var up = upstream[j];
      if (up[earlier.command] === true || up[tmLeafName(earlier.command)] === true ||
          up[tmQualifiedName(schemas, earlier.command)] === true || tmFeedsLater(tokens, later)) {
        earlier.supporting = true;
        return;
      }
    }
  });
}

/* [{id, command, context, parameters, output, outputs, supporting, subTask, unwrappedFrom, unwrappedPath}] */
function tmResultsOf(commandOutputs, schemas) {
  var own = (commandOutputs || []).filter(function (co) {
    return !tmIsFrameworkOutput(co);
  });
  var pages = tmCollectPages(own);
  var order = [];
  var byKey = {};
  pages.forEach(function (page) {
    if (!tmHas(byKey, page.key)) { byKey[page.key] = []; order.push(page.key); }
    byKey[page.key].push(page);
  });
  var results = order.map(function (key, i) { return tmBuildResult("r" + (i + 1), byKey[key]); });
  tmMarkSupporting(results, schemas || {});
  return results;
}

/* {groups: [{command, results}], supporting: [result], selected: index}. The
   selected group holds the last primary result; with no primary result every
   result is primary. */
function tmResultGroups(results) {
  var primary = results.filter(function (r) { return !r.supporting; });
  var supporting = results.filter(function (r) { return r.supporting; });
  if (!primary.length) { primary = results; supporting = []; }
  var groups = [];
  var index = {};
  primary.forEach(function (r) {
    var key = "g:" + r.command;
    if (!tmHas(index, key)) { index[key] = groups.length; groups.push({ command: r.command, results: [] }); }
    groups[index[key]].results.push(r);
  });
  var selected = 0;
  if (primary.length) {
    selected = index["g:" + primary[primary.length - 1].command];
  }
  return { groups: groups, supporting: supporting, selected: selected };
}

/* -- values (R5) --------------------------------------------------------- */

function tmUiHint(prop) {
  return (prop && tmIsPlainObject(prop.ui)) ? prop.ui : {};
}

/* The author's ui.tones (value -> good/warning/bad/neutral) wins; keys match case-insensitively after trimming. */
function tmAuthorTone(val, prop) {
  var ui = tmUiHint(prop);
  var tones = ui.format === "badge" ? ui.tones : null;
  if (!tmIsPlainObject(tones)) { return null; }
  var want = String(val === null || val === undefined ? "" : val).trim().toLowerCase();
  var key;
  for (key in tones) {
    if (tmHas(tones, key) && key.trim().toLowerCase() === want &&
        TM_AUTHOR_TONES.indexOf(tones[key]) >= 0) { return tones[key]; }
  }
  return null;
}

function tmAutoTone(val, prop) {
  var author = tmAuthorTone(val, prop);
  if (author) { return author; }
  var key = String(val === null || val === undefined ? "" : val).trim().toLowerCase()
    .replace(/[\s-]+/g, "_");
  var tone;
  for (tone in TM_TONES) {
    if (tmHas(TM_TONES, tone) && TM_TONES[tone].indexOf(key) >= 0) { return tone; }
  }
  return "neutral";
}

function tmCompactJson(value) {
  var text;
  try { text = JSON.stringify(value); } catch (e) { text = undefined; }
  if (text === undefined) { text = String(value); }
  return text.length > TM_RESULT_CELL_CHARS
    ? text.slice(0, TM_RESULT_CELL_CHARS - 1) + "…" : text;
}

/* A string, or a Node for badge and code. A field's name never decides how it
   is drawn: a badge needs the `format: badge` hint. (`field` is kept for callers.) */
function tmFormatValue(val, prop, field) {
  if (val === null || val === undefined || val === "") { return "—"; }
  if (typeof val === "boolean") { return val ? "yes" : "no"; }
  if (typeof val === "object") { return tmCompactJson(val); }
  var format = tmUiHint(prop).format;
  if (format === "code") { return el("code", "rCode", String(val)); }
  var text = String(val);
  if (format === "badge") {
    return el("span", "rBadge tone-" + tmAutoTone(val, prop), text);
  }
  if (typeof val === "number") {
    if (format === "currency") {
      var code = tmUiHint(prop).currency;
      if (typeof code === "string" && /^[A-Za-z]{3}$/.test(code)) {
        try {
          return new Intl.NumberFormat(undefined, { style: "currency", currency: code.toUpperCase() }).format(val);
        } catch (e) { /* a code the browser rejects: the plain number below */ }
      }
      /* no code (or one the browser rejects): the browser's locale, two decimals, no symbol */
      return Number(val).toLocaleString(undefined,
        { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    if (format === "percent") { return Number(val).toLocaleString() + "%"; }
    return Number(val).toLocaleString();
  }
  return text;
}

function tmAppendValue(parent, value) {
  if (value && value.nodeType) { parent.appendChild(value); }
  else { parent.appendChild(document.createTextNode(String(value))); }
}

/* -- renderers ----------------------------------------------------------- */

/* items: [{label, valueNode, tone, title}] -> dl.cardStats */
function tmRenderStatStrip(items) {
  var dl = el("dl", "cardStats");
  items.forEach(function (item) {
    var box = el("div", "cardStat" + (item.tone ? " tone-" + item.tone : ""));
    var dt = el("dt", null, item.label);
    if (item.title) { dt.title = item.title; }
    var dd = el("dd");
    tmAppendValue(dd, item.valueNode);
    box.appendChild(dt);
    box.appendChild(dd);
    dl.appendChild(box);
  });
  return dl;
}

function tmFieldLabel(key, prop) {
  return { label: tmPrettyFieldTitle(key, prop), title: (prop && prop.description) || key };
}

function tmRenderKeyValues(obj, keys, schemaNode) {
  var dl = el("dl", "rKv");
  keys.forEach(function (key) {
    var prop = tmSchemaProperty(schemaNode, key);
    if (tmUiHint(prop).hidden === true) { return; }
    var names = tmFieldLabel(key, prop);
    var dt = el("dt", null, names.label);
    dt.title = names.title;
    var dd = el("dd");
    tmAppendValue(dd, tmFormatValue(obj[key], prop, key));
    dl.appendChild(dt);
    dl.appendChild(dd);
  });
  return dl;
}

function tmColumnIsNumeric(key, rows) {
  var seen = false;
  var i;
  for (i = 0; i < rows.length; i++) {
    var v = rows[i][key];
    if (v === null || v === undefined || v === "") { continue; }
    if (typeof v !== "number") { return false; }
    seen = true;
  }
  return seen;
}

function tmTableRow(columns, numeric, row) {
  var tr = el("tr");
  columns.forEach(function (col, c) {
    var value = row[col.key];
    var td = el("td", numeric[c] ? "num" : null);
    tmAppendValue(td, tmFormatValue(value, col.prop, col.key));
    if (value !== null && typeof value === "object") {
      try { td.title = JSON.stringify(value); } catch (e) { td.title = String(value); }
    }
    tr.appendChild(td);
  });
  return tr;
}

/* columns: [{key, label, title, prop, numeric}]; rows: [{<key>: value}].
   The first 50 rows render; the rest are added on request, up to 1000. */
function tmRenderRowsTable(columns, rows) {
  var wrap = el("div", "rTableWrap");
  var table = el("table", "rTable");
  var numeric = columns.map(function (col) {
    return col.numeric === true || (col.numeric === undefined && tmColumnIsNumeric(col.key, rows));
  });
  var headRow = el("tr");
  columns.forEach(function (col, c) {
    var th = el("th", numeric[c] ? "num" : null, col.label);
    th.setAttribute("scope", "col");
    if (col.title) { th.title = col.title; }
    headRow.appendChild(th);
  });
  var thead = el("thead");
  thead.appendChild(headRow);
  table.appendChild(thead);
  var tbody = el("tbody");
  var capped = rows.slice(0, TM_RESULT_ROWS_MAX);
  capped.slice(0, TM_RESULT_ROWS_SHOWN).forEach(function (row) {
    tbody.appendChild(tmTableRow(columns, numeric, row));
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  if (capped.length > TM_RESULT_ROWS_SHOWN) {
    var more = el("button", "rMore", rows.length > TM_RESULT_ROWS_MAX
      ? "Show first " + TM_RESULT_ROWS_MAX + " of " + rows.length + " rows"
      : "Show all " + rows.length + " rows");
    more.type = "button";
    more.addEventListener("click", function () {
      capped.slice(TM_RESULT_ROWS_SHOWN).forEach(function (row) {
        tbody.appendChild(tmTableRow(columns, numeric, row));
      });
      if (more.parentNode) { more.parentNode.removeChild(more); }
    });
    wrap.appendChild(more);
  }
  return wrap;
}

function tmShortScalar(value) {
  if (value === null || value === undefined) { return true; }
  if (typeof value === "string") {
    return value.length <= TM_RESULT_SHORT_TEXT && value.indexOf("\n") < 0;
  }
  return true;
}

function tmFieldKind(value) {
  if (tmIsScalar(value)) { return "scalar"; }
  if (tmIsPlainObject(value)) { return "object"; }
  if (value.every(tmIsScalar)) { return "scalarArray"; }
  if (value.length && value.every(tmIsPlainObject)) { return "objectArray"; }
  return "mixedArray";
}

function tmOutputFields(output, schema) {
  var fields = [];
  Object.keys(output).forEach(function (key) {
    var prop = tmSchemaProperty(schema, key);
    var ui = tmUiHint(prop);
    if (ui.hidden === true) { return; }
    var names = tmFieldLabel(key, prop);
    fields.push({
      key: key, value: output[key], prop: prop, ui: ui, kind: tmFieldKind(output[key]),
      label: names.label, title: names.title
    });
  });
  return fields;
}

/* Decide which scalars go in the horizontal stat strip (R6). */
function tmSplitScalars(fields) {
  var scalars = fields.filter(function (f) { return f.kind === "scalar"; });
  var heroes = scalars.filter(function (f) { return f.ui.hero === true; });
  var stat = [];
  if (heroes.length) {
    stat = heroes;
  } else {
    var short = scalars.filter(function (f) { return tmShortScalar(f.value); });
    var structured = fields.length - scalars.length;
    if (short.length >= 1 && short.length <= 4 && (structured >= 1 || fields.length <= 4)) {
      stat = short;
    }
  }
  return {
    stat: stat,
    kv: scalars.filter(function (f) { return stat.indexOf(f) < 0; })
  };
}

/* NOT used by the results panel: the panel draws parallel lists as a table only through a declared
   ui.table group (fix-sush step 3; a field name never decides). The answer card (258-chat-card.js
   tmCardIsPage) and the composer still use this rule until fix-sush step 4.
   Mirrors observation_offloading.facts.output_page (fixture page_shape_cases.json
   is read by both): an output is a page when its `total` is an integer (not a
   boolean or a string) and every list field other than `total` has the same
   length. Only a page has parallel columns, one row per index. Residual: two
   unrelated lists that happen to share a length in an output that also has an
   integer `total` are, by the server's own definition, a page. */
function tmIsPageOutput(output) {
  if (!tmIsPlainObject(output)) { return false; }
  var total = output.total;
  if (typeof total !== "number" || !isFinite(total) || Math.floor(total) !== total) { return false; }
  var length = -1;
  var same = true;
  Object.keys(output).forEach(function (key) {
    if (key === "total" || !Array.isArray(output[key])) { return; }
    if (length < 0) { length = output[key].length; }
    else if (output[key].length !== length) { same = false; }
  });
  return same && length >= 0;
}

/* The columns an author declared (ui.table, a group name on each column field of Signature.Output). A
   declared group is the only way parallel lists become a table in this panel: every member list present in this output has one
   length of at least 1. Members come from the schema (field order), hidden ones included, because a hidden
   member still counts for alignment but is not drawn. A member that is missing, null or not a list of
   scalars, or lists of different lengths, make no table: the members render as bullets, never trimmed or
   padded. The chat never infers columns from equal lengths alone. Returns {tables, claimed}: tables are
   arrays of visible fields; claimed holds every key a declaration names, drawn or not. */
function tmDeclaredTables(fields, output, schema) {
  var claimed = {};
  var props = schema && tmIsPlainObject(schema.properties) ? schema.properties : null;
  if (!props || !tmIsPlainObject(output)) { return { tables: [], claimed: claimed }; }
  var groups = {};
  var order = [];
  Object.keys(props).forEach(function (key) {
    var ui = tmUiHint(tmSchemaProperty(schema, key));
    if (typeof ui.table !== "string" || !ui.table) { return; }
    if (!tmHas(groups, ui.table)) { groups[ui.table] = []; order.push(ui.table); }
    groups[ui.table].push(key);
    claimed[key] = true;
  });
  var tables = [];
  order.forEach(function (name) {
    var keys = groups[name];
    var length = -1;
    var ok = keys.length >= 2 && keys.every(function (key) {
      var value = output[key];
      if (!tmHas(output, key) || !Array.isArray(value) || !value.every(tmIsScalar)) { return false; }
      if (length < 0) { length = value.length; }
      return value.length === length;
    });
    if (!ok || length < 1) { return; }
    var columns = fields.filter(function (f) { return keys.indexOf(f.key) >= 0; });
    if (columns.length >= 2) { tables.push(columns); }
  });
  return { tables: tables, claimed: claimed };
}

function tmSection(label, title) {
  var box = el("div", "rSection");
  var head = el("div", "rLabel", label);
  if (title) { head.title = title; }
  box.appendChild(head);
  return box;
}

function tmItemSchema(schema, prop) {
  var items = prop && prop.items ? tmSchemaNode(schema, prop.items) : null;
  return tmSchemaView(schema, items);
}

function tmObjectArrayTable(rows, itemSchema) {
  var keys = [];
  rows.forEach(function (row) {
    Object.keys(row).forEach(function (k) { if (keys.indexOf(k) < 0) { keys.push(k); } });
  });
  var columns = [];
  keys.forEach(function (key) {
    var prop = tmSchemaProperty(itemSchema, key);
    if (tmUiHint(prop).hidden === true) { return; }
    var names = tmFieldLabel(key, prop);
    columns.push({ key: key, label: names.label, title: names.title, prop: prop });
  });
  return tmRenderRowsTable(columns, rows);
}

function tmAlignedTable(group) {
  var columns = group.map(function (f) {
    return { key: f.key, label: f.label, title: f.title, prop: f.prop };
  });
  var rows = [];
  var i;
  for (i = 0; i < group[0].value.length; i++) {
    var row = {};
    group.forEach(function (f) { row[f.key] = f.value[i]; });
    rows.push(row);
  }
  return tmRenderRowsTable(columns, rows);
}

function tmBulletList(values, field) {
  if (!values.length) { return el("div", "rNone", "none"); }
  var ul = el("ul", "rList");
  values.forEach(function (v) {
    var li = el("li");
    tmAppendValue(li, tmFormatValue(v, field.prop, field.key));
    ul.appendChild(li);
  });
  return ul;
}

/* aligned: the tables of this output, each an array of fields (a table is drawn at its first column). */
function tmRenderField(field, schema, depth, aligned) {
  if (field.kind === "scalarArray") {
    var i;
    for (i = 0; i < aligned.length; i++) {
      if (aligned[i].indexOf(field) >= 0) {
        return aligned[i][0] === field ? tmAlignedTable(aligned[i]) : null;
      }
    }
  }
  var section = tmSection(field.label, field.title);
  if (field.kind === "objectArray") {
    section.appendChild(tmObjectArrayTable(field.value, tmItemSchema(schema, field.prop)));
  } else if (field.kind === "object") {
    section.appendChild(tmRenderOutputView(field.value, tmSchemaView(schema, field.prop), depth + 1));
  } else {
    section.appendChild(tmBulletList(field.value, field));
  }
  return section;
}

/* One output object as a card (R6). outputSchema may be null. */
function tmRenderOutputView(output, outputSchema, depth) {
  var level = depth || 0;
  var view = el("div", "rView");
  if (!tmIsPlainObject(output) || level > TM_RESULT_MAX_DEPTH) {
    view.appendChild(el("pre", "json", pretty(output)));
    return view;
  }
  var fields = tmOutputFields(output, outputSchema);
  if (!fields.length) {
    view.appendChild(el("div", "rNone", "none"));
    return view;
  }
  var split = tmSplitScalars(fields);
  if (split.stat.length) {
    view.appendChild(tmRenderStatStrip(split.stat.map(function (f) {
      var node = tmFormatValue(f.value, f.prop, f.key);
      var tone = (node && node.nodeType && node.className.indexOf("rBadge") === 0)
        ? tmAutoTone(f.value, f.prop) : null;
      return { label: f.label, title: f.title, valueNode: node, tone: tone };
    })));
  }
  if (split.kv.length) {
    view.appendChild(tmRenderKeyValues(output, split.kv.map(function (f) { return f.key; }),
      outputSchema));
  }
  var declared = tmDeclaredTables(fields, output, outputSchema);
  var aligned = declared.tables.slice();
  fields.forEach(function (f) {
    if (f.kind === "scalar") { return; }
    var node = tmRenderField(f, outputSchema, level, aligned);
    if (node) { view.appendChild(node); }
  });
  return view;
}

/* -- groups and the card ------------------------------------------------- */

function tmFlatResult(result) {
  return Object.keys(result.output).every(function (k) { return tmIsScalar(result.output[k]); });
}

function tmGroupTable(group, schemas) {
  var first = group.results[0];
  var schema = tmResultSchema(first, schemas);
  var contexts = {};
  group.results.forEach(function (r) { contexts[r.context] = true; });
  var keys = [];
  group.results.forEach(function (r) {
    Object.keys(r.output).forEach(function (k) { if (keys.indexOf(k) < 0) { keys.push(k); } });
  });
  var columns = [];
  if (Object.keys(contexts).length > 1) {
    columns.push({ key: "__record", label: "Record", title: "context", prop: null });
  }
  keys.forEach(function (key) {
    var prop = tmSchemaProperty(schema, key);
    if (tmUiHint(prop).hidden === true) { return; }
    var names = tmFieldLabel(key, prop);
    columns.push({ key: key, label: names.label, title: names.title, prop: prop });
  });
  var rows = group.results.map(function (r) {
    var row = {};
    Object.keys(r.output).forEach(function (k) { row[k] = r.output[k]; });
    row.__record = r.context || r.id;
    return row;
  });
  return tmRenderRowsTable(columns, rows);
}

function tmResultLabel(result) {
  return tmLeafName(result.command) + " · " + result.id;
}

function tmRenderGroupView(group, schemas) {
  if (group.results.length === 1) {
    var only = group.results[0];
    return tmRenderOutputView(only.output, tmResultSchema(only, schemas), 0);
  }
  if (group.results.every(tmFlatResult)) {
    var view = el("div", "rView");
    view.appendChild(tmGroupTable(group, schemas));
    return view;
  }
  var stack = el("div", "rView");
  group.results.forEach(function (r) {
    var section = tmSection(tmResultLabel(r), r.context || null);
    section.appendChild(tmRenderOutputView(r.output, tmResultSchema(r, schemas), 0));
    stack.appendChild(section);
  });
  return stack;
}

function tmGroupRawJson(group) {
  var raws = [];
  group.results.forEach(function (r) { r.outputs.forEach(function (o) { raws.push(o); }); });
  return pretty(raws.length === 1 ? raws[0] : raws);
}

/* A group: a bar holding the JSON toggle, over the stage it swaps. */
function tmRenderGroupPanel(group, schemas) {
  tmResultsSeq += 1;
  var panel = el("div", "rPanel");
  var bar = el("div", "rBar");
  var toggle = el("button", "rJson", "JSON");
  toggle.type = "button";
  toggle.setAttribute("aria-pressed", "false");
  var stage = el("div", "rStage");
  stage.id = "rStage" + tmResultsSeq;
  toggle.setAttribute("aria-controls", stage.id);
  var view = tmRenderGroupView(group, schemas);
  var raw = el("pre", "json");
  raw.hidden = true;
  stage.appendChild(view);
  stage.appendChild(raw);
  toggle.addEventListener("click", function () {
    var on = toggle.getAttribute("aria-pressed") !== "true";
    toggle.setAttribute("aria-pressed", on ? "true" : "false");
    if (on && !raw.textContent) { raw.textContent = tmGroupRawJson(group); }
    raw.hidden = !on;
    view.hidden = on;
  });
  bar.appendChild(toggle);
  panel.appendChild(bar);
  panel.appendChild(stage);
  return panel;
}

function tmGroupChipLabel(group) {
  return tmLeafName(group.command) + (group.results.length > 1 ? " ×" + group.results.length : "");
}

function tmRenderGroups(plan, schemas) {
  var box = el("div", "rGroups");
  var head = el("div", "rHead");
  var panels = [];
  var chips = [];
  function select(index) {
    panels.forEach(function (p, i) { p.hidden = i !== index; });
    chips.forEach(function (c, i) { c.setAttribute("aria-pressed", i === index ? "true" : "false"); });
  }
  plan.groups.forEach(function (group, i) {
    panels.push(tmRenderGroupPanel(group, schemas));
    if (plan.groups.length < 2) {
      head.appendChild(el("span", "rTitle", tmGroupChipLabel(group)));
      return;
    }
    var chip = el("button", "rChip", tmGroupChipLabel(group));
    chip.type = "button";
    chip.addEventListener("click", function () { select(i); });
    chips.push(chip);
    head.appendChild(chip);
  });
  box.appendChild(head);
  panels.forEach(function (p) { box.appendChild(p); });
  select(plan.selected);
  return box;
}

function tmRenderSupporting(results, schemas) {
  var details = el("details", "supportingResults");
  details.appendChild(el("summary", null, "Supporting lookups (" + results.length + ")"));
  results.forEach(function (r) {
    var section = tmSection(tmResultLabel(r), r.context || null);
    section.appendChild(tmRenderOutputView(r.output, tmResultSchema(r, schemas), 0));
    details.appendChild(section);
  });
  return details;
}

function tmBuildResultsNode(commandOutputs, schemas) {
  var results = tmResultsOf(commandOutputs, schemas);
  if (!results.length) { return null; }
  var plan = tmResultGroups(results);
  var root = el("div", "results");
  root.setAttribute("role", "group");
  root.setAttribute("aria-label", "Results of this turn");
  root.dataset.resultCount = String(results.length);
  root.appendChild(tmRenderGroups(plan, schemas));
  if (plan.supporting.length) { root.appendChild(tmRenderSupporting(plan.supporting, schemas)); }
  return root;
}

/* Paint (or repaint) the results of one turn into the panel behind the link
   under its answer (255). Renders at once with whatever schemas are loaded;
   when the listing is still in flight it is painted again, once, if the
   listing arrives non-empty. A repaint replaces the same panel and link, so a
   second paint of the turn never stacks, and a panel that was open stays open. */
function tmRenderResults(bubbleMsg, out) {
  tmDropResultsUi(bubbleMsg);
  var outputs = out && out.command_outputs;
  var node = tmBuildResultsNode(outputs, tm.commandSchemas || {});
  if (!node) {
    bubbleMsg.dataset.resultsOpen = "";
    return null;
  }
  tmMountResults(bubbleMsg, node);
  if (tm.commandSchemas === null && tm.commandSchemasPromise) {
    tmWithCommandSchemas(function (loaded) {
      if (!bubbleMsg.contains(node) || !Object.keys(loaded).length) { return; }
      var fresh = tmBuildResultsNode(outputs, loaded);
      if (fresh) { node.parentNode.replaceChild(fresh, node); }
    });
  }
  return node;
}

/* -- the toggle the composed card keeps for the whole answer ------------- */

function tmAnswerToggle(rest) {
  var button = el("button", "answerToggle", "Show full answer");
  button.type = "button";
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-controls", rest.id);
  button.addEventListener("click", function () {
    var open = button.getAttribute("aria-expanded") !== "true";
    button.setAttribute("aria-expanded", open ? "true" : "false");
    button.textContent = open ? "Hide full answer" : "Show full answer";
    rest.hidden = !open;
  });
  return button;
}
