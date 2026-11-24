// The rail and its filter, the same on every page. Data: window.IDIOMS_NAV from nav.js.
(function () {
  var nav = window.IDIOMS_NAV;
  var rail = document.querySelector("aside.rail");
  if (!nav || !rail) return;
  var current = document.body.getAttribute("data-page");
  var KEY = "design-idioms-filter";

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (name) { node.setAttribute(name, attrs[name]); });
    if (text != null) node.textContent = text;
    return node;
  }

  rail.appendChild(el("p", { "class": "rail-brand" }, "fastWorkflow"));
  var title = rail.appendChild(el("p", { "class": "title" }));
  title.appendChild(el("a", { href: "index.html" }, "Design idioms"));
  var input = rail.appendChild(el("input", {
    id: "q", type: "search", placeholder: "Filter", "aria-label": "Filter idioms"
  }));
  // index.html has its own count and empty note in the main column; other pages show them here.
  var shown = document.getElementById("shown") ||
    rail.appendChild(el("p", { id: "shown", "class": "rail-note", hidden: "" }));
  var empty = document.getElementById("empty") ||
    rail.appendChild(el("p", { id: "empty", "class": "rail-note", hidden: "" }, "Nothing matches that filter."));
  var index = rail.appendChild(el("nav", { "class": "index", "aria-label": "Idioms" }));

  var entries = [];
  var active = null;
  nav.groups.forEach(function (group) {
    var pages = nav.pages.filter(function (page) { return page.group === group.key; });
    var details = index.appendChild(el("details", { "class": group.key, open: "" }));
    details.appendChild(el("summary", null, group.counted ? group.label + " · " + pages.length : group.label));
    pages.forEach(function (page) {
      var link = details.appendChild(el("a", { href: page.id + ".html" }, page.title));
      if (page.id === current) { link.classList.add("active"); active = link; }
      entries.push({
        page: page, link: link, details: details,
        haystack: (page.title + " " + page.text).toLowerCase(),
        row: document.querySelector('.toc li[data-id="' + page.id + '"]')
      });
    });
  });
  var sections = Array.prototype.slice.call(document.querySelectorAll("main section"))
    .filter(function (section) { return section.querySelector(".toc"); });

  function apply() {
    var needle = input.value.trim().toLowerCase();
    var hits = entries.filter(function (entry) {
      var hit = !needle || entry.haystack.indexOf(needle) !== -1;
      entry.link.hidden = !hit;
      if (entry.row) entry.row.hidden = !hit;
      return hit;
    });
    index.querySelectorAll("details").forEach(function (details) {
      details.hidden = !details.querySelector("a:not([hidden])");
    });
    sections.forEach(function (section) {
      section.hidden = !section.querySelector(".toc li:not([hidden])");
    });
    empty.hidden = hits.length !== 0;
    shown.hidden = !needle;
    shown.textContent = hits.length + " shown";
    try { sessionStorage.setItem(KEY, input.value); } catch (e) { /* storage unavailable */ }
    return hits;
  }

  input.addEventListener("input", apply);
  input.addEventListener("keydown", function (event) {
    if (event.key !== "Enter") return;
    var hits = apply();
    if (input.value.trim() && hits.length === 1) location.href = hits[0].link.getAttribute("href");
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "/" && document.activeElement !== input) {
      event.preventDefault();
      input.focus();
    }
  });

  try { input.value = sessionStorage.getItem(KEY) || ""; } catch (e) { /* storage unavailable */ }
  apply();
  if (active) {
    var box = rail.getBoundingClientRect();
    var at = active.getBoundingClientRect();
    if (at.top < box.top || at.bottom > box.bottom) {
      rail.scrollTop += at.top - box.top - rail.clientHeight / 3;
    }
  }
})();
