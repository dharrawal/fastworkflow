/* -- chat approval card [fix-weih.12] ------------------------------------- */
/* An execution approval (question_kind "approval") as a card with the bound
   parameters and three buttons, with no model call. ActionReplyPolicy accepts
   a yes (yes, true, ok, agreed, accept or go), always or no, each as the
   whole reply with case and a trailing . or ! ignored ("yes please" costs a
   round), so the buttons send the words yes, always and no, kept below.
   The turn waits when out.status is "awaiting_user" (the agent asked) or when
   the last output says artifacts.awaiting_user === true (a "/" command the
   user typed: its turn status is "completed"). Registered after the built-in
   prompt renderers: no other renderer answers an approval. */

var TM_APPROVAL_REPLIES = { approve: "yes", always: "always", decline: "no" };
var TM_APPROVAL_MASK = "••••";

/* The approval token a host-surface server delivers with an approval prompt
   (fix-rkr1). It is held in this variable only: never rendered, never put in
   chat text, never written to localStorage, and not kept on the turn's output.
   It goes back once, as approval_token on the reply that answers the prompt,
   and is dropped on any other message, on a new conversation and on a source
   (epoch) change. A server on the channel surface delivers none, so nothing
   here ever fires for it. */
var tmApprovalTokenHeld = null;

/* Take the token off the turn's output (out.approval, live only) and keep it. Any output without one
   drops the token held before it: that prompt is over. */
function tmApprovalTokenRemember(out) {
  tmApprovalTokenHeld = null;
  if (!tmIsPlainObject(out) || !tmIsPlainObject(out.approval)) { return; }
  var token = out.approval.token;
  delete out.approval;
  if (typeof token === "string" && token) {
    tmApprovalTokenHeld = { token: token, epoch: tm.epoch };
  }
}

function tmApprovalTokenClear() { tmApprovalTokenHeld = null; }

/* The token for the message about to be sent: only for one of the three exact
   replies, and only in the epoch it was delivered in. Always releases it. */
function tmApprovalTokenFor(text) {
  var held = tmApprovalTokenHeld;
  tmApprovalTokenHeld = null;
  if (!held || held.epoch !== tm.epoch) { return null; }
  var replies = Object.keys(TM_APPROVAL_REPLIES).map(function (k) { return TM_APPROVAL_REPLIES[k]; });
  return replies.indexOf(text) === -1 ? null : held.token;
}

function tmApprovalIsWaiting(out, artifacts) {
  return !!out && (out.status === "awaiting_user" || artifacts.awaiting_user === true);
}

function tmApprovalPayload(raw) {
  if (!tmIsPlainObject(raw) || typeof raw.command !== "string" || !raw.command) { return null; }
  return raw;
}

/* The bound parameters with every password or writeOnly input masked, so a
   secret never reaches the page in clear text. */
function tmApprovalParameters(approval, input) {
  var params = tmIsPlainObject(approval.parameters) ? approval.parameters : {};
  var shown = {};
  Object.keys(params).forEach(function (key) {
    var prop = tmSchemaProperty(input, key);
    shown[key] = tmIsSecretProperty(prop) ? TM_APPROVAL_MASK : params[key];
  });
  return shown;
}

/* Focus Approve unless the person is already typing: a draft in the composer
   is never taken from them. tmComposerState focuses the composer when the
   turn settles, after this paint, so the check also runs once the stack
   unwinds. */
function tmApprovalFocus(button) {
  function wanted() {
    var composer = document.getElementById("chatInput");
    var active = document.activeElement;
    if (composer && composer.value !== "") { return false; }
    return !active || active === document.body || active === composer;
  }
  if (wanted()) { button.focus(); }
  setTimeout(function () {
    if (button.disabled || !button.ownerDocument.contains(button)) { return; }
    if (wanted()) { button.focus(); }
  }, 0);
}

function tmRenderApprovalCard(bubbleMsg, out) {
  if (!tmIsLiveTurn(bubbleMsg)) { return null; }
  var last = tmLastPromptArtifacts(out);
  if (!last || last.artifacts.question_kind !== "approval") { return null; }
  if (!tmApprovalIsWaiting(out, last.artifacts)) { return null; }
  var approval = tmApprovalPayload(last.artifacts.approval);
  var key = "approval:" + JSON.stringify(approval) + ":" + (approval ? "" : last.text);
  var prev = tmReusePrompt(bubbleMsg, key);
  if (prev) { return tmMountPrompt(bubbleMsg, prev); }

  var root = el("div", "approvalCard");
  root.setAttribute("role", "group");
  var leaf = approval ? tmLeafName(approval.command) : "";
  if (approval) {
    var title = el("div", "approvalTitle",
      (approval.origin === "user" ? "Run " : "Approve ") + leaf + "?");
    title.id = "approvalTitle" + (tmFormSeq += 1);
    root.setAttribute("aria-labelledby", title.id);
    root.appendChild(title);
    var schema = tmCommandSchema(approval.command);
    var shown = tmApprovalParameters(approval, schema && schema.input);
    if (Object.keys(shown).length) {
      root.appendChild(tmRenderKeyValues(shown, Object.keys(shown),
        schema && tmIsPlainObject(schema.input) ? schema.input : null));
    }
  } else {
    root.setAttribute("aria-label", "Approval needed");
    root.appendChild(el("div", "approvalText", last.text));
  }

  var disabled = false;
  var buttons = [];
  var bar = el("div", "paramActions approvalActions");
  function add(label, cls, text) {
    var button = el("button", "paramBtn " + cls + (cls === "primary" ? " primary" : ""), label);
    button.type = "button";
    button.addEventListener("click", function () {
      if (!disabled) { tmSendChatText(text); }
    });
    bar.appendChild(button);
    buttons.push(button);
    return button;
  }
  var approve = add("Approve", "primary", TM_APPROVAL_REPLIES.approve);
  if (!approval || approval.can_always !== false) {
    add(approval ? "Always allow " + leaf : "Always allow", "secondary", TM_APPROVAL_REPLIES.always);
  }
  add("Decline", "ghost", TM_APPROVAL_REPLIES.decline);
  root.appendChild(bar);

  var nodes = [root];
  if (approval) { nodes.push(tmDetailsFor(last.text, false)); }
  var rec = { key: key, element: root, nodes: nodes,
    setDisabled: function (flag) {
      disabled = !!flag;
      buttons.forEach(function (b) { b.disabled = disabled; });
      if (disabled) { root.classList.add("disabled"); } else { root.classList.remove("disabled"); }
    } };
  var mounted = tmMountPrompt(bubbleMsg, rec);
  if (mounted) { tmApprovalFocus(approve); }
  return mounted;
}

TM_PROMPT_RENDERERS.push(tmRenderApprovalCard);
