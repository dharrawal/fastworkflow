/* -- composed answer card [fix-weih.10] --------------------------------- */
/* On a multi-result agent turn, or a long prose answer, the page composes a
   card by itself: POST /compose_answer_card asks a small model for a short
   lead plus blocks (stats, tables, records, quotes) and shows them in the
   bubble. While the request is out the bubble keeps the whole plain answer and
   a small "Composing answer..." status line; any failure (no model, error,
   timeout, a stale answer) removes the status and leaves the plain answer as it
   was. The model only chooses: every VALUE on the card is read here, from this
   page's own copy of the data, through a ref like "r1.status". Model text is
   limited to the lead, block titles, stat labels and quotes, and all of it
   enters the DOM through textContent [R22]. Calls stay bounded: a live turn
   composes once, a card is cached by turn_key (a repaint or a reopen never asks
   again), a stored turn composes only once its record is loaded, and a
   request in flight is shared per turn and dropped when the bubble or the
   source changes. The "Rich answers" checkbox off means no composition.

   A card is either flat (lead + blocks) or, from a server that sends `sections`
   [fix-weih.25], a lead and sections: each a heading, a one-sentence verdict
   and up to 4 blocks. A response without `sections` renders as a flat card, and
   a server that predates sections is never asked for them. Two things are
   written by this page, never by the model: a table shows 8 rows and then a
   "+M more" control that opens the results panel, and the card ends with a
   footer line (a button opening the panel) for every primary result group that
   no block of the card shows, so nothing a turn returned is unreachable.

   What is sent as each result's `output` is the NORMALIZED output from
   tmResultsOf (JSON strings decoded, one-key wrappers peeled up to 5 times).
   run_chatbot.composer.normalize_output_dict does the same steps, so normalizing it
   again changes nothing and a ref the server returns resolves against the same
   object here (tmResolveCardRef).

   Composition with the answer and the results panel (255) [fix-weih.36]: a drawn
   card REPLACES the text answer. The bubble becomes [div.answerCard, div.answerRest
   (the WHOLE answer, hidden, kept in the DOM), button.answerToggle]; the toggle
   ("Show full text" / "Show summary", aria-expanded, aria-controls) swaps the two
   and its state is kept per turn in bubbleMsg.dataset.cardText. Until a card is
   drawn, and whenever none is (skip, failure, notice), the bubble holds the text
   as painted. The deterministic .results node is never touched: it stays in the
   results panel behind the link under the answer.

   A skip is never silent unless it is by design [fix-weih.34]. Where the
   "Composing answer..." line sits, a short muted notice (span.cardNotice, the
   reason code in data-reason and the title) says why no card came: the answer is
   too long (no request is made), the composer model is not configured, the
   composition failed or timed out, or the page could not draw the card. Silent by
   design, because the user chose it or nothing is lost: "Rich answers" off; a "/"
   turn; a page not connected to a server; a turn ending in a question (the form
   asks); a record still loading (the paint after it decides); one short result
   or a short answer (the deterministic card is the answer; the server's
   not_needed likewise); a bubble or source that changed while the request was out.
   More than 60 results is no skip: the first 60 are sent, primary first, and the
   footer lists any group the card does not show. A notice is dropped by every
   repaint and by a card, and is shown once per turn; a restored turn makes the same
   decision again when its record loads (the too-long and not-configured notices
   cost no request). */

var TM_CARD_TIMEOUT_MS = 35000;
var TM_CARD_LONG_ANSWER = 400;
var TM_CARD_MAX_ANSWER = 150000;  /* the server refuses a longer answer (run_chatbot.composer.MAX_ANSWER_CHARS) */
var TM_CARD_MAX_RESULTS = 60;     /* ... and more results */
var TM_CARD_LIST_SHOWN = 5;       /* items of a list-valued field a record shows before "+N more" */
var TM_CARD_ROWS_SHOWN = 8;       /* rows of a table on the card; the rest are in the results panel (run_chatbot.composer.TABLE_ROWS_SHOWN: the server drops an "all shown" claim over more rows than this) */
var TM_CARD_MAX_SECTIONS = 24;    /* the caps the server applies (run_chatbot.composer.MAX_SECTIONS: one per part of the request, at least 5, at most this) */
var TM_CARD_MAX_SECTION_BLOCKS = 4;
var TM_CARD_MAX_BLOCKS = 48;      /* 10 up to 5 sections, 2 per section after, at most this (run_chatbot.composer.MAX_CARD_BLOCKS_CEILING) */
var TM_CARD_HEADING_CHARS = 60;
var TM_CARD_VERDICT_CHARS = 200;
var TM_CARD_DESCRIPTION_CHARS = 100;
var TM_CARD_MAX_STATS = 6;
var TM_CARD_FOOTER_LINES = 3;     /* footer lines before they collapse into one "N more results" line */
var TM_CARD_NOT_FOUND = {};
var TM_CARD_STAT_TONES = ["good", "warning", "bad"];
var TM_CARD_QUOTE_TONES = ["success", "warning", "danger", "info"];
var tmCardSeq = 0;
var tmCardSeqId = 0;
var tmCardFlights = {};
var tmCardTried = {};   /* epoch:turn_key already asked once; a failure is not retried */
var tmCardOutcome = {}; /* epoch:turn_key -> the reason code of its last skip, "" when nothing is to be said */

/* -- refs: pure ---------------------------------------------------------- */

function tmCardFindResult(id, results) {
  var i;
  for (i = 0; i < results.length; i++) {
    if (results[i].id === id) { return results[i]; }
  }
  return null;
}

/* The value a ref like "r1.a.b" points at, or TM_CARD_NOT_FOUND. The first
   segment is a result id; the rest are own keys of plain objects. A list index
   never resolves. */
function tmResolveCardRef(ref, results) {
  if (typeof ref !== "string" || !ref) { return TM_CARD_NOT_FOUND; }
  var segs = ref.split(".");
  var result = tmCardFindResult(segs[0], results || []);
  if (!result) { return TM_CARD_NOT_FOUND; }
  var value = result.output;
  var i;
  for (i = 1; i < segs.length; i++) {
    if (!segs[i] || !tmIsPlainObject(value) || !tmHas(value, segs[i])) {
      return TM_CARD_NOT_FOUND;
    }
    value = value[segs[i]];
  }
  return value;
}

/* {parent, prop, view}: the schema view holding the field a ref ends at, that
   field's property node, and the view of the object it is (all may be null). */
function tmCardSchemaAt(ref, results, schemas) {
  var segs = ref.split(".");
  var result = tmCardFindResult(segs[0], results);
  var view = result ? tmResultSchema(result, schemas) : null;
  var parent = null;
  var prop = null;
  var i;
  for (i = 1; i < segs.length; i++) {
    parent = view;
    prop = tmSchemaProperty(parent, segs[i]);
    view = tmSchemaView(parent, prop);
  }
  return { parent: parent, prop: prop, view: view };
}

function tmCardText(value, max) {
  if (typeof value !== "string") { return ""; }
  var text = value.trim();
  return text.length > max ? text.slice(0, max) : text;
}

/* The same normalization the server applies to a quote and to the answer. */
function tmCardClean(text) {
  return String(text).replace(/[*_`]/g, "").split(/\s+/).filter(function (w) { return w !== ""; })
    .join(" ").toLowerCase();
}

function tmCardQuoteGrounded(quote, answer) {
  var cleaned = tmCardClean(quote);
  var whole = tmCardClean(answer);
  return cleaned.length >= 20 && cleaned.length <= 600 &&
    cleaned.length <= 0.8 * whole.length && whole.indexOf(cleaned) >= 0;
}

function tmCardScalarArrayKeys(obj) {
  return Object.keys(obj).filter(function (k) {
    return Array.isArray(obj[k]) && obj[k].every(tmIsScalar);
  });
}

/* A page as the server defines it (tmIsPageOutput: an integer total and every list field of one
   length) that also holds two or more non-empty scalar lists, the columns of a combined table.
   Two unrelated lists of one length without a total are not a page. The server's mirror is
   run_chatbot.composer._is_page_shape; tests/fixtures/page_shape_cases.json pins both. */
function tmCardIsPage(value) {
  if (!tmIsPageOutput(value)) { return false; }
  var keys = tmCardScalarArrayKeys(value);
  return keys.length >= 2 && value[keys[0]].length >= 1;
}

/* The wanted fields that exist; when fewer than `least` of them do, all of them
   (a table of one column is not a table; a record keeps even one). */
function tmCardKeepFields(wanted, keys, least) {
  var kept = Array.isArray(wanted) ? wanted.filter(function (f) {
    return typeof f === "string" && keys.indexOf(f) >= 0;
  }) : [];
  return kept.length >= (least || 1) ? kept : keys;
}

/* -- blocks -------------------------------------------------------------- */

function tmCardTitled(kind, block, content) {
  var box = el("div", "cardBlock " + kind);
  var title = tmCardText(block.title, 120);
  if (title) { box.appendChild(el("div", "cardBlockTitle", title)); }
  box.appendChild(content);
  return box;
}

/* ctx: {bubbleMsg, refs}. refs collects the result id of every block drawn. */
function tmCardNoteRef(ctx, ref) {
  if (ctx && ctx.refs && typeof ref === "string") { ctx.refs[ref.split(".")[0]] = true; }
}

function tmCardStatsNode(block, results, schemas, ctx) {
  var items = [];
  (Array.isArray(block.stats) ? block.stats : []).forEach(function (stat) {
    if (items.length >= TM_CARD_MAX_STATS || !stat || typeof stat !== "object") { return; }
    var value = tmResolveCardRef(stat.ref, results);
    if (value === TM_CARD_NOT_FOUND || !tmIsScalar(value)) { return; }
    var at = tmCardSchemaAt(stat.ref, results, schemas);
    if (tmUiHint(at.prop).hidden === true) { return; }
    var segs = stat.ref.split(".");
    var key = segs[segs.length - 1];
    tmCardNoteRef(ctx, stat.ref);
    var node = tmFormatValue(value, at.prop, key);
    var tone = TM_CARD_STAT_TONES.indexOf(stat.tone) >= 0 ? stat.tone
      : ((node && node.nodeType && node.className.indexOf("rBadge") === 0) ? tmAutoTone(value, at.prop) : null);
    items.push({
      label: tmCardText(stat.label, 60) || tmPrettyFieldTitle(key, at.prop),
      title: (at.prop && at.prop.description) || key,
      valueNode: node,
      tone: tone
    });
  });
  return items.length ? tmCardTitled("stats", block, tmRenderStatStrip(items)) : null;
}

function tmCardPageRows(value, keys) {
  var rows = [];
  var n = value[keys[0]].length;
  var i;
  for (i = 0; i < n; i++) {
    var row = {};
    keys.forEach(function (k) { row[k] = value[k][i]; });
    rows.push(row);
  }
  return rows;
}

/* A button that opens the results panel of the turn; null when the turn has none. */
function tmCardOpenButton(bubbleMsg, label, className) {
  if (!bubbleMsg || !bubbleMsg.querySelector(".resultsPanel")) { return null; }
  var button = el("button", className, label);
  button.type = "button";
  button.addEventListener("click", function () { tmSetResultsOpen(bubbleMsg, true, "panel"); });
  return button;
}

function tmCardTableNode(block, results, schemas, ctx) {
  var value = tmResolveCardRef(block.ref, results);
  if (value === TM_CARD_NOT_FOUND) { return null; }
  var at = tmCardSchemaAt(block.ref, results, schemas);
  var keys, rows, itemSchema;
  if (tmCardIsPage(value)) {
    keys = tmCardScalarArrayKeys(value);
    rows = tmCardPageRows(value, keys);
    itemSchema = at.view;
  } else if (Array.isArray(value) && value.length && value.every(tmIsPlainObject)) {
    keys = [];
    value.forEach(function (row) {
      Object.keys(row).forEach(function (k) { if (keys.indexOf(k) < 0) { keys.push(k); } });
    });
    rows = value;
    itemSchema = tmItemSchema(at.parent, at.prop);
  } else {
    return null;
  }
  var columns = [];
  tmCardKeepFields(block.fields, keys, 2).forEach(function (key) {
    var prop = tmSchemaProperty(itemSchema, key);
    if (tmUiHint(prop).hidden === true) { return; }
    var names = tmFieldLabel(key, prop);
    columns.push({ key: key, label: names.label, title: names.title, prop: prop });
  });
  if (!columns.length) { return null; }
  tmCardNoteRef(ctx, block.ref);
  var holder = el("div", "cardTableHolder");
  holder.appendChild(tmRenderRowsTable(columns, rows.slice(0, TM_CARD_ROWS_SHOWN)));
  if (rows.length > TM_CARD_ROWS_SHOWN) {
    var label = "+" + (rows.length - TM_CARD_ROWS_SHOWN) + " more — open in results";
    var open = tmCardOpenButton(ctx && ctx.bubbleMsg, label, "cardMore rMore");
    holder.appendChild(open || el("div", "cardMore rMore", "+" + (rows.length - TM_CARD_ROWS_SHOWN) + " more"));
  }
  return tmCardTitled("table", block, holder);
}

/* A list a record shows, as text and never raw JSON: a list of scalars as its first
   TM_CARD_LIST_SHOWN items joined by commas, then "+N more"; a list holding objects (or other
   lists) as a count. An empty list is blank (the page draws a dash). */
function tmCardListText(list) {
  if (!list.length) { return ""; }
  if (!list.every(function (item) { return item === null || typeof item !== "object"; })) {
    return list.length + (list.length === 1 ? " item" : " items");
  }
  var text = list.slice(0, TM_CARD_LIST_SHOWN).map(function (item) {
    return item === null || item === undefined ? "—" : String(item);
  }).join(", ");
  return list.length > TM_CARD_LIST_SHOWN ? text + ", +" + (list.length - TM_CARD_LIST_SHOWN) + " more" : text;
}

function tmCardRecordNode(block, results, schemas, ctx) {
  var value = tmResolveCardRef(block.ref, results);
  if (value === TM_CARD_NOT_FOUND || !tmIsPlainObject(value) || !Object.keys(value).length) {
    return null;
  }
  var at = tmCardSchemaAt(block.ref, results, schemas);
  var shown = {};
  Object.keys(value).forEach(function (k) {
    shown[k] = Array.isArray(value[k]) ? tmCardListText(value[k]) : value[k];
  });
  var dl = tmRenderKeyValues(shown, tmCardKeepFields(block.fields, Object.keys(value)), at.view);
  if (!dl.firstChild) { return null; }
  tmCardNoteRef(ctx, block.ref);
  return tmCardTitled("record", block, dl);
}

function tmCardQuoteNode(block, answer) {
  var quote = typeof block.quote === "string" ? block.quote.trim() : "";
  if (!quote || !tmCardQuoteGrounded(quote, answer)) { return null; }
  var tone = TM_CARD_QUOTE_TONES.indexOf(block.tone) >= 0 ? block.tone : "info";
  var node = el("blockquote", "cardQuote tone-" + tone, quote);
  return tmCardTitled("quote", block, node);
}

/* One block, or null when it cannot be drawn from this page's data. */
function tmCardBlockNode(block, results, schemas, answer, ctx) {
  if (!block || typeof block !== "object") { return null; }
  if (block.kind === "stats") { return tmCardStatsNode(block, results, schemas, ctx); }
  if (block.kind === "table") { return tmCardTableNode(block, results, schemas, ctx); }
  if (block.kind === "record") { return tmCardRecordNode(block, results, schemas, ctx); }
  if (block.kind === "quote") { return tmCardQuoteNode(block, answer); }
  return null;
}

/* -- the trigger --------------------------------------------------------- */

function tmCardAnswerText(out) {
  return String(out.answer === null || out.answer === undefined ? "" : out.answer);
}

function tmCardLastAwaiting(commandOutputs) {
  var last = commandOutputs && commandOutputs.length ? commandOutputs[commandOutputs.length - 1] : null;
  var artifacts = last && last.command_response && last.command_response.artifacts;
  return !!artifacts && artifacts.awaiting_user === true;
}

/* The conditions that do not depend on the results. */
function tmCardWanted(bubbleMsg, out) {
  var box = document.getElementById("richAnswers");
  return !!(box && box.checked) && tm.connected === true &&
    out.status !== "awaiting_user" && !tmCardLastAwaiting(out.command_outputs) &&
    bubbleMsg.dataset.deterministic !== "1";
}

/* Synthesis is worth a model call only across several primary results or for
   a long answer; one short result keeps its instant deterministic card. */
function tmCardNeeded(results, out) {
  var primary = results.filter(function (r) { return !r.supporting; });
  if (!primary.length) { primary = results; }
  return primary.length > 1 || tmCardAnswerText(out).length > TM_CARD_LONG_ANSWER;
}

function tmCardStale(bubbleMsg, token, epoch) {
  return tm.epoch !== epoch || bubbleMsg.dataset.cardPaint !== token ||
    !document.body.contains(bubbleMsg);
}

/* -- where the status line sits -------------------------------------------- */

function tmCardHost(bubbleMsg) {
  var linkRow = bubbleMsg.querySelector(".resultsLinkRow");
  if (linkRow) { return linkRow; }
  var row = bubbleMsg.querySelector(".cardStatusRow");
  if (!row) {
    row = el("div", "cardStatusRow");
    var bubble = bubbleMsg.querySelector(".bubble");
    bubbleMsg.insertBefore(row, bubble ? bubble.nextSibling : null);
  }
  return row;
}

function tmCardDrop(node) {
  if (node && node.parentNode) { node.parentNode.removeChild(node); }
}

/* The status leaves; the row it sat in goes with it once it is empty. */
function tmCardDropFrom(node) {
  var host = node && node.parentNode;
  tmCardDrop(node);
  if (host && host.className === "cardStatusRow" && !host.firstChild) { tmCardDrop(host); }
}

/* What an earlier paint of this turn left behind. */
function tmCardTidy(bubbleMsg) {
  tmCardDrop(bubbleMsg.querySelector(".cardStatusRow"));
  tmCardDropNotice(bubbleMsg);
}

/* The notice's text for a reason code; null when there is nothing to say (the composer found no card needed). */
function tmCardNoticeText(reason) {
  if (reason === "not_needed" || !reason) { return null; }
  if (reason === "too_long") { return "Summary card skipped: the answer is too long to summarise."; }
  if (reason === "no_model") { return "Summary card unavailable (composer model not configured)."; }
  if (reason === "timeout") { return "Summary card could not be composed in time."; }
  return "Summary card could not be composed.";
}

function tmCardDropNotice(bubbleMsg) {
  Array.prototype.slice.call(bubbleMsg.querySelectorAll(".cardNotice")).forEach(tmCardDropFrom);
}

/* One notice per bubble, where the status line sits; never beside a card. */
function tmCardNotice(bubbleMsg, reason) {
  tmCardDropNotice(bubbleMsg);
  var text = tmCardNoticeText(reason);
  if (!text || bubbleMsg.querySelector(".answerCard")) { return; }
  var node = el("span", "cardNotice", text);
  node.setAttribute("data-reason", reason);
  node.title = "Why: " + reason;
  tmCardHost(bubbleMsg).appendChild(node);
}

function tmCardStatus(bubbleMsg) {
  var node = el("span", "cardStatus", "Composing answer…");
  node.setAttribute("role", "status");
  node.setAttribute("aria-live", "polite");
  node.tabIndex = -1;   /* can take focus from the button that was just pressed */
  tmCardHost(bubbleMsg).appendChild(node);
  return node;
}

/* Focus fell to the page because the node that held it left: put it on the
   turn's answer toggle, else its results link, else the bubble itself. */
function tmCardRefocus(bubbleMsg) {
  var target = bubbleMsg.querySelector("button.answerToggle") ||
    bubbleMsg.querySelector("button.resultsLink") || bubbleMsg.querySelector(".bubble");
  if (!target) { return; }
  if (target.tabIndex < 0 && !target.getAttribute("tabindex")) { target.tabIndex = -1; }
  target.focus();
}

/* -- applying a card ----------------------------------------------------- */

function tmCardMakeRest(answer, open) {
  tmResultsSeq += 1;
  var rest = el("div", "answerRest cardRest");
  rest.id = "answerRest" + tmResultsSeq;
  rest.hidden = !open;
  appendMarkdown(rest, answer);
  return rest;
}

/* The one quiet switch between the composed answer and the full text, under
   both. Its state is the turn's own (bubbleMsg.dataset.cardText, "full" or
   "" for the summary), so a repaint, a cached card and a results-panel open or
   close all keep it. The text node stays in the DOM, hidden, while the card shows. */
function tmCardTextToggle(bubbleMsg, box, rest) {
  var button = el("button", "answerToggle cardTextToggle", "Show full text");
  button.type = "button";
  button.setAttribute("aria-controls", box.id + " " + rest.id);
  function paint() {
    var full = bubbleMsg.dataset.cardText === "full";
    button.setAttribute("aria-expanded", full ? "true" : "false");
    button.textContent = full ? "Show summary" : "Show full text";
    box.hidden = full;
    rest.hidden = !full;
  }
  button.addEventListener("click", function () {
    var following = tmFollowingLog();
    bubbleMsg.dataset.cardText = bubbleMsg.dataset.cardText === "full" ? "" : "full";
    paint();
    tmScrollIfFollowing(following);
  });
  paint();
  return button;
}

function tmCardBlockNodes(blocks, results, schemas, answer, ctx, cap) {
  var nodes = [];
  (Array.isArray(blocks) ? blocks : []).forEach(function (block) {
    if (nodes.length >= cap) { return; }
    var node = null;
    try { node = tmCardBlockNode(block, results, schemas, answer, ctx); } catch (e) { node = null; }
    if (node) { nodes.push(node); }
  });
  return nodes;
}

/* div.cardSection nodes for a v2 card; [] when none survives (the flat blocks are then used). The caps
   mirror the server's ceilings: 8 sections, 4 blocks each, 16 in all. Text only through el() [R22]. */
function tmCardSectionNodes(sections, results, schemas, answer, ctx) {
  var out = [];
  var total = 0;
  (Array.isArray(sections) ? sections : []).forEach(function (section) {
    if (out.length >= TM_CARD_MAX_SECTIONS || !section || typeof section !== "object") { return; }
    var room = Math.min(TM_CARD_MAX_SECTION_BLOCKS, TM_CARD_MAX_BLOCKS - total);
    var nodes = room > 0 ? tmCardBlockNodes(section.blocks, results, schemas, answer, ctx, room) : [];
    var verdict = tmCardText(section.verdict, TM_CARD_VERDICT_CHARS);
    if (!verdict && !nodes.length) { return; }
    total += nodes.length;
    var box = el("div", "cardSection");
    var heading = tmCardText(section.heading, TM_CARD_HEADING_CHARS);
    if (heading) { box.appendChild(el("h3", "cardSectionHeading", heading)); }
    if (verdict) { box.appendChild(el("p", "cardVerdict", verdict)); }
    nodes.forEach(function (node) { box.appendChild(node); });
    out.push(box);
  });
  return out;
}

/* A footer line's name for a group: its command as the panel's chip names it, and the record it ran for
   (the result's context, the entity label the panel shows over its fields) when there are one or two. */
function tmCardGroupLabel(group) {
  var label = tmPrettyFieldTitle(tmLeafName(group.command), null);
  var seen = {};
  var contexts = [];
  group.results.forEach(function (r) {
    var context = String(r.context || "").trim();
    if (context && context.toLowerCase() !== "global" && !tmHas(seen, context)) {
      seen[context] = true;
      contexts.push(context);
    }
  });
  var named = contexts.join(", ");
  return named && contexts.length <= 2 && named.length <= 60 ? label + " · " + named : label;
}

/* The footer lists what the card does not show, so nothing a turn returned is unreachable, and it is
   never a wall: up to TM_CARD_FOOTER_LINES primary result groups with no block on the card get a line of
   their own (a button opening the results panel); more than that is ONE line, "N more results", N being
   the results of those groups. Written here from the results' own labels (the command and the record
   it ran for, as the results panel shows them) and counts, never by the model. null when every group is
   shown or the turn has no results panel. */
function tmCardFooter(bubbleMsg, results, ctx) {
  if (!bubbleMsg.querySelector(".resultsPanel")) { return null; }
  var missing = tmResultGroups(results).groups.filter(function (group) {
    return !group.results.some(function (r) { return ctx.refs[r.id] === true; });
  });
  if (!missing.length) { return null; }
  var labels = [];
  if (missing.length <= TM_CARD_FOOTER_LINES) {
    missing.forEach(function (group) {
      var n = group.results.length;
      labels.push(tmCardGroupLabel(group) + " (" + n +
        (n === 1 ? " result" : " results") + ") — open in results");
    });
  } else {
    var total = 0;
    missing.forEach(function (group) { total += group.results.length; });
    labels.push(total + " more " + (total === 1 ? "result" : "results") + " — open in results");
  }
  var footer = el("div", "cardFooter");
  labels.forEach(function (label) {
    var button = tmCardOpenButton(bubbleMsg, label, "cardFooterLink");
    if (!button) { return; }
    var line = el("div", "cardFooterLine");
    line.appendChild(button);
    footer.appendChild(line);
  });
  return footer.firstChild ? footer : null;
}

function tmCardApply(bubbleMsg, out, card, results, schemas) {
  var bubble = bubbleMsg.querySelector(".bubble");
  if (!bubble || bubble.querySelector(".answerCard") || !card || typeof card !== "object") {
    return false;
  }
  var answer = tmCardAnswerText(out);
  var ctx = { bubbleMsg: bubbleMsg, refs: {} };
  var nodes = tmCardSectionNodes(card.sections, results, schemas, answer, ctx);
  if (!nodes.length) {
    ctx.refs = {};
    nodes = tmCardBlockNodes(card.blocks, results, schemas, answer, ctx, Infinity);
  }
  var lead = tmCardText(card.lead, 600);
  if (!lead && !nodes.length) { return false; }
  /* A lead alone would replace the plain answer with no data: with a results panel to open, the plain
     answer stays as it is (the server does not send such a card; this is the page's own guard). */
  if (!nodes.length && bubbleMsg.querySelector(".resultsPanel")) { return false; }

  /* Keep the reader where they were: the log keeps following only if it was
     following, and focus that sat in the replaced text moves to the bubble. */
  var following = tmFollowingLog();
  var hadFocus = bubble.contains(document.activeElement);
  var box = el("div", "answerCard");
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", "Summary of this turn");
  if (lead) { box.appendChild(el("p", "cardLead", lead)); }
  nodes.forEach(function (node) { box.appendChild(node); });
  var footer = tmCardFooter(bubbleMsg, results, ctx);
  if (footer) { box.appendChild(footer); }

  /* The card REPLACES the text answer: the text stays in the DOM, hidden, behind the toggle. */
  tmCardSeqId += 1;
  box.id = "answerCard" + tmCardSeqId;
  clear(bubble);
  bubble.classList.remove("md");
  bubble.appendChild(box);
  var rest = tmCardMakeRest(answer, false);
  var toggle = tmCardTextToggle(bubbleMsg, box, rest);
  bubble.appendChild(rest);
  bubble.appendChild(toggle);
  if (hadFocus) {   /* the node that held focus is gone: the bubble takes it, nothing is announced */
    if (!bubble.getAttribute("tabindex")) { bubble.tabIndex = -1; }
    bubble.focus();
  }
  bubbleMsg.classList.add("hasCard");
  tmCardDropNotice(bubbleMsg);
  tmScrollIfFollowing(following);
  return true;
}

/* -- the request --------------------------------------------------------- */

/* The schema's ui hint of a field as one short string: "hidden", or "hero" and the format joined by
   commas; "" when it has none. The server leaves a hidden key out of what the model sees. */
function tmCardHint(prop) {
  var ui = tmUiHint(prop);
  if (ui.hidden === true) { return "hidden"; }
  var words = [];
  if (ui.hero === true) { words.push("hero"); }
  if (typeof ui.format === "string" && ui.format) { words.push(ui.format); }
  return words.join(",");
}

/* Primary results first, so a turn over the limit sheds supporting lookups, not what the user asked for. */
function tmCardSendOrder(results) {
  return results.filter(function (r) { return !r.supporting; })
    .concat(results.filter(function (r) { return r.supporting; }));
}

/* The results the request carries, at most TM_CARD_MAX_RESULTS, in send order. Over the limit the rule is
   fair across the request's parts (a result's subTask): first the first primary result of every part, then the
   rest in send order, so a part with primary results never loses all of them while another part keeps
   supporting ones. Results with no part number, or a turn with none, are cut exactly as before (the first N). */
function tmCardSelect(results) {
  var ordered = tmCardSendOrder(results);
  if (ordered.length <= TM_CARD_MAX_RESULTS) { return ordered; }
  var keep = {};
  var count = 0;
  var seen = {};
  ordered.forEach(function (r, i) {
    if (r.supporting || r.subTask === null || r.subTask === undefined || tmHas(seen, r.subTask)) { return; }
    seen[r.subTask] = true;
    if (count < TM_CARD_MAX_RESULTS) { keep[i] = true; count++; }
  });
  ordered.forEach(function (r, i) {
    if (count < TM_CARD_MAX_RESULTS && !tmHas(keep, i)) { keep[i] = true; count++; }
  });
  return ordered.filter(function (r, i) { return tmHas(keep, i); });
}

/* The parts of a split turn as the core recorded them, in the final answer's artifacts under fw_sub_tasks
   (core: workflow_execution_context.TURN_PARTS_ARTIFACT): a list, or null when the turn has none (an unsplit turn, an
   older record, a value the store offloaded to a reference). Forwarded as it is; the server checks it against the
   answer and reads the answer's headings when it does not fit. Nothing is drawn from it here. */
function tmCardParts(out) {
  var list = out.command_outputs;
  var last = list && list.length ? list[list.length - 1] : null;
  var artifacts = last && last.command_response && last.command_response.artifacts;
  var parts = artifacts ? artifacts.fw_sub_tasks : null;
  return Array.isArray(parts) && parts.length ? parts : null;
}

function tmCardPayload(bubbleMsg, out, results, schemas) {
  var cache = document.getElementById("useDspyCache");
  var parts = tmCardParts(out);
  var payload = {
    user_message: bubbleMsg.dataset.userMessage || "",
    answer: tmCardAnswerText(out),
    status: out.status || "completed",
    use_dspy_cache: !!(cache && cache.checked),
    turn_key: out.turn_key ? String(out.turn_key) : "",   /* files the compose span under this turn's trace [fix-weih.28] */
    results: tmCardSelect(results).map(function (r) {
      var schema = tmResultSchema(r, schemas);
      var titles = {};
      var descriptions = {};
      var hints = {};
      Object.keys(r.output).forEach(function (key) {
        var prop = tmSchemaProperty(schema, key);
        titles[key] = tmPrettyFieldTitle(key, prop);
        if (prop && typeof prop.description === "string" && prop.description.trim()) {
          descriptions[key] = prop.description.trim().slice(0, TM_CARD_DESCRIPTION_CHARS);
        }
        var hint = tmCardHint(prop);
        if (hint) { hints[key] = hint; }
      });
      var entry = {
        id: r.id, command: r.command, context: r.context,
        parameters: tmIsPlainObject(r.parameters) ? r.parameters : {},
        output: r.output, titles: titles, descriptions: descriptions, hints: hints,
        supporting: r.supporting === true
      };
      if (typeof r.subTask === "number") { entry.sub_task = r.subTask; }  /* the request part; absent when unknown */
      return entry;
    })
  };
  if (parts) { payload.parts = parts; }   /* absent, not null, when the turn has none */
  return payload;
}

/* Always resolves {card, reason}; never rejects. Aborts after 35 s. */
function tmCardFetch(payload) {
  return new Promise(function (resolve) {
    var done = false;
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    function finish(reply) {
      if (done) { return; }
      done = true;
      clearTimeout(timer);
      resolve(reply);
    }
    var timer = setTimeout(function () {
      finish({ card: null, reason: "timeout" });
      if (controller) { controller.abort(); }
    }, TM_CARD_TIMEOUT_MS);
    tmFetch("/compose_answer_card", payload, controller ? { signal: controller.signal } : undefined)
      .then(function (r) {
        if (r.ok) { return r.json(); }
        finish({ card: null, reason: "http_" + String(r.status) });
        return null;
      })
      .then(function (data) {
        if (!data || typeof data !== "object") { finish({ card: null, reason: "failed" }); return; }
        finish(tmIsPlainObject(data.card) ? { card: data.card, reason: "" }
          : { card: null, reason: String(data.reason || "failed") });
      })
      .catch(function () { finish({ card: null, reason: "failed" }); });
  });
}

function tmCardRequest(bubbleMsg, out, results, schemas, token, epoch) {
  if (tmCardAnswerText(out).length > TM_CARD_MAX_ANSWER) { tmCardNotice(bubbleMsg, "too_long"); return; }
  var key = out.turn_key ? epoch + ":" + out.turn_key : "";
  var flight = key ? tmCardFlights[key] : null;
  if (!flight) {
    flight = tmCardFetch(tmCardPayload(bubbleMsg, out, results, schemas));
    if (key) {
      tmCardFlights[key] = flight;
      flight.then(function () { delete tmCardFlights[key]; });
    }
  }
  var status = tmCardStatus(bubbleMsg);
  flight.then(function (reply) {
    var heldFocus = document.activeElement === status;
    var skip = "";
    if (tm.epoch === epoch) {
      if (reply.reason === "no_model" && !tm.composeUnavailable) {
        tm.composeUnavailable = true;
      }
      if (reply.card) {
        if (out.turn_key) { tm.cardCache[out.turn_key] = reply.card; }
        if (!tmCardStale(bubbleMsg, token, epoch) &&
            !tmCardApply(bubbleMsg, out, reply.card, results, schemas)) {
          skip = "card_refused";   /* nothing on it the page could draw */
        }
      } else {
        skip = reply.reason || "failed";
      }
      if (key) { tmCardOutcome[key] = skip; }
    }
    tmCardDropFrom(status);
    if (skip && !tmCardStale(bubbleMsg, token, epoch)) { tmCardNotice(bubbleMsg, skip); }
    var active = document.activeElement;
    if (heldFocus && (!active || active === document.body || active === status) &&
        document.body.contains(bubbleMsg)) {
      tmCardRefocus(bubbleMsg);
    }
  });
}

function tmCardStart(bubbleMsg, out, schemas, token, epoch) {
  if (tmCardStale(bubbleMsg, token, epoch) || !tmCardWanted(bubbleMsg, out)) { return; }
  var results = tmResultsOf(out.command_outputs, schemas);
  if (!tmCardNeeded(results, out)) { return; }
  if (out.turn_key && tm.cardCache && tmHas(tm.cardCache, String(out.turn_key))) {
    if (!tmCardApply(bubbleMsg, out, tm.cardCache[out.turn_key], results, schemas)) {
      tmCardNotice(bubbleMsg, "card_refused");
    }
    return;
  }
  if (tm.composeUnavailable) { tmCardNotice(bubbleMsg, "no_model"); return; }
  if (tmCardAnswerText(out).length > TM_CARD_MAX_ANSWER) { tmCardNotice(bubbleMsg, "too_long"); return; }
  var key = out.turn_key ? epoch + ":" + out.turn_key : "";
  if (key && tmCardTried[key] && !tmCardFlights[key]) {
    tmCardNotice(bubbleMsg, tmCardOutcome[key] || "");   /* the earlier outcome, shown again after a repaint */
    return;
  }
  if (key) { tmCardTried[key] = true; }
  tmCardRequest(bubbleMsg, out, results, schemas, token, epoch);
}

function tmComposeCardNow(bubbleMsg, out) {
  tmCardTidy(bubbleMsg);
  tmCardSeq += 1;
  var token = String(tmCardSeq);
  bubbleMsg.dataset.cardPaint = token;
  if (!out || bubbleMsg.dataset.pendingRecord === "1" || !tmCardWanted(bubbleMsg, out)) { return; }
  var epoch = tm.epoch;
  tmWithCommandSchemas(function (schemas) {
    try { tmCardStart(bubbleMsg, out, schemas, token, epoch); } catch (e) { /* plain view stays */ }
  });
}

/* Called after the deterministic paint of a turn. It never throws. */
function tmComposeCard(bubbleMsg, out) {
  try { tmComposeCardNow(bubbleMsg, out); } catch (e) { /* plain view stays */ }
}
