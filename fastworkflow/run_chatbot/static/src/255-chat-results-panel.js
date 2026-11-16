/* -- results side panel [fix-weih.24] ----------------------------------- */
/* The answer bubble shows the whole answer and nothing else. The structured
   results (254) sit behind one link under it ("Show results (N)"), in a panel
   to the right of the answer that is as tall as the answer, or below it on a
   narrow window. Closed by default. The panel and its link are siblings of the
   bubble, so a repaint of the turn drops and rebuilds both (tmDropResultsUi),
   and the open state survives it in bubbleMsg.dataset.resultsOpen. Everything
   enters the DOM through el()/textContent [R22]. */

var tmResultsPanelSeq = 0;

function tmResultsPanelIsOpen(bubbleMsg) {
  var panel = bubbleMsg.querySelector(".resultsPanel");
  return !!panel && !panel.hidden;
}

/* focus: "panel" moves it to the close control, "link" returns it to the link,
   anything else leaves it where it is (a repaint must not take focus). */
function tmSetResultsOpen(bubbleMsg, open, focus) {
  var panel = bubbleMsg.querySelector(".resultsPanel");
  var link = bubbleMsg.querySelector(".resultsLink");
  if (!panel || !link) { return; }
  /* The artifact panel opens in the same place, so only one of them shows. */
  if (open) { tmCloseArtifactPanel(bubbleMsg); }
  panel.hidden = !open;
  link.setAttribute("aria-expanded", open ? "true" : "false");
  link.textContent = (open ? "Hide results (" : "Show results (") + link.dataset.count + ")";
  bubbleMsg.classList.toggle("resultsOpen", open);
  bubbleMsg.dataset.resultsOpen = open ? "1" : "";
  if (open && focus === "panel") {
    var close = panel.querySelector(".resultsClose");
    if (close) { close.focus(); }
  } else if (!open && focus === "link") {
    link.focus();
  }
}

function tmCloseResultsPanel(bubbleMsg) {
  if (tmResultsPanelIsOpen(bubbleMsg)) { tmSetResultsOpen(bubbleMsg, false, ""); }
}

/* Takes the link row and the panel away; the open state is kept for the next
   mount. */
function tmDropResultsUi(bubbleMsg) {
  Array.prototype.slice.call(
    bubbleMsg.querySelectorAll(".resultsPanel, .resultsLinkRow")
  ).forEach(function (node) { node.parentNode.removeChild(node); });
  bubbleMsg.classList.remove("hasResults", "resultsOpen");
  tmFooterTidy(bubbleMsg);
}

/* A turn that is a prompt owns its bubble and shows no results at all. */
function tmClearResultsUi(bubbleMsg) {
  tmDropResultsUi(bubbleMsg);
  bubbleMsg.dataset.resultsOpen = "";
}

function tmMountResults(bubbleMsg, node) {
  tmResultsPanelSeq += 1;
  var count = node.dataset.resultCount || "0";
  var panel = el("div", "resultsPanel");
  panel.id = "resultsPanel" + tmResultsPanelSeq;
  panel.hidden = true;
  panel.setAttribute("role", "region");
  panel.setAttribute("aria-label", "Results");
  var head = el("div", "resultsPanelHead");
  head.appendChild(el("div", "resultsPanelTitle", "Results"));
  var close = el("button", "resultsClose", "×");
  close.type = "button";
  close.setAttribute("aria-label", "Close results");
  close.title = "Close";
  head.appendChild(close);
  panel.appendChild(head);
  var body = el("div", "resultsPanelBody");
  body.appendChild(node);
  panel.appendChild(body);

  var link = el("button", "resultsLink", "Show results (" + count + ")");
  link.type = "button";
  link.dataset.count = count;
  link.setAttribute("aria-expanded", "false");
  link.setAttribute("aria-controls", panel.id);
  var row = el("div", "resultsLinkRow");
  row.appendChild(link);

  link.addEventListener("click", function () {
    tmSetResultsOpen(bubbleMsg, panel.hidden, "panel");
  });
  close.addEventListener("click", function () {
    tmSetResultsOpen(bubbleMsg, false, "link");
  });
  panel.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") { return; }
    tmSetResultsOpen(bubbleMsg, false, "link");
    event.preventDefault();
  });

  /* The panel directly follows the bubble: the page's grid puts it beside the
     bubble on the same row. The link row goes in the left slot of the turn
     footer, below, full width [fix-weih.32]. */
  var bubble = bubbleMsg.querySelector(".bubble");
  if (bubble && bubble.parentNode === bubbleMsg) {
    bubbleMsg.insertBefore(panel, bubble.nextSibling);
  } else {
    bubbleMsg.appendChild(panel);
  }
  tmFooterSlot(bubbleMsg, "left").appendChild(row);
  bubbleMsg.classList.add("hasResults");
  if (bubbleMsg.dataset.resultsOpen === "1") { tmSetResultsOpen(bubbleMsg, true, ""); }
}
