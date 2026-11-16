/* -- command schemas [fix-weih.3] --------------------------------------- */
/* One cached copy of GET /command_schemas per connection, shared by the
   results view and the parameter forms. The listing maps a qualified command
   name to {kind, input, output}, each a JSON Schema or null. Loading never
   throws and never shows a notice: a missing or unreadable listing settles to
   {} and the chat simply renders without schema help. The helpers below are
   pure; they only read what they are given. */
function tmLoadCommandSchemas() {
  var epoch = tm.epoch;
  tm.commandSchemas = null;
  var promise = tmFetch("/command_schemas", null, { method: "GET" })
    .then(function (r) {
      if (!r.ok) { return {}; }
      return r.json();
    })
    .catch(function () { return {}; })
    .then(function (data) {
      var listing = (data && typeof data === "object" && !Array.isArray(data)) ? data : {};
      if (tm.epoch === epoch) { tm.commandSchemas = listing; }
      return listing;
    });
  tm.commandSchemasPromise = promise;
  return promise;
}

function tmWithCommandSchemas(fn) {
  if (tm.commandSchemas) {
    fn(tm.commandSchemas);
    return;
  }
  if (tm.commandSchemasPromise) {
    var queuedEpoch = tm.epoch;
    tm.commandSchemasPromise.then(function () {
      fn(tm.epoch === queuedEpoch ? (tm.commandSchemas || {}) : {});
    });
    return;
  }
  fn({});
}

function tmLeafName(name) {
  var parts = String(name === null || name === undefined ? "" : name).split("/");
  return parts[parts.length - 1];
}

function tmCommandSchema(name) {
  return tmCommandSchemaIn(tm.commandSchemas, name);
}

function tmCommandSchemaIn(schemas, name) {
  if (!schemas || typeof schemas !== "object" || name === null || name === undefined) {
    return null;
  }
  var key = String(name);
  if (Object.prototype.hasOwnProperty.call(schemas, key)) { return schemas[key]; }
  var leaf = tmLeafName(key);
  var found = null;
  var count = 0;
  Object.keys(schemas).forEach(function (candidate) {
    if (tmLeafName(candidate) === leaf) {
      found = schemas[candidate];
      count++;
    }
  });
  return count === 1 ? found : null;
}

var TM_SCHEMA_KEEP = ["title", "description", "default", "examples", "ui",
  "available_from", "classifier_confirms"];

/* The outer node (the property) wins: its title, description and the rest
   describe the field, while a $defs target carries the class's own. The inner
   node only supplies what the outer lacks. */
function tmSchemaFill(target, outer) {
  TM_SCHEMA_KEEP.forEach(function (key) {
    if (Object.prototype.hasOwnProperty.call(outer, key)) {
      target[key] = outer[key];
    }
  });
}

function tmSchemaCopy(node) {
  var copy = {};
  Object.keys(node).forEach(function (key) { copy[key] = node[key]; });
  return copy;
}

function tmSchemaNode(root, node, depth) {
  if (!node || typeof node !== "object" || Array.isArray(node)) { return node || null; }
  var level = depth || 0;
  if (level > 8) { return node; }
  var ref = typeof node["$ref"] === "string" ? node["$ref"] : "";
  var prefix = "#/$defs/";
  if (ref.indexOf(prefix) === 0) {
    var defs = root && root["$defs"];
    var name = ref.slice(prefix.length);
    if (defs && Object.prototype.hasOwnProperty.call(defs, name) &&
        defs[name] && typeof defs[name] === "object") {
      var resolved = tmSchemaCopy(defs[name]);
      tmSchemaFill(resolved, node);
      return tmSchemaNode(root, resolved, level + 1);
    }
    return node;
  }
  var branches = Array.isArray(node.anyOf) ? node.anyOf : (Array.isArray(node.oneOf) ? node.oneOf : null);
  if (branches) {
    var real = branches.filter(function (b) { return !(b && b.type === "null"); });
    if (real.length === 1 && real.length < branches.length) {
      var inner = tmSchemaNode(root, real[0], level + 1);
      var out = inner && typeof inner === "object" ? tmSchemaCopy(inner) : {};
      tmSchemaFill(out, node);
      out.nullable = true;
      return out;
    }
    return node;
  }
  if (Array.isArray(node.type) && node.type.indexOf("null") !== -1) {
    var types = node.type.filter(function (t) { return t !== "null"; });
    var flat = tmSchemaCopy(node);
    flat.type = types.length === 1 ? types[0] : types;
    flat.nullable = true;
    return flat;
  }
  return node;
}

function tmSchemaProperty(schema, field) {
  var props = schema && schema.properties;
  if (!props || !Object.prototype.hasOwnProperty.call(props, field)) { return null; }
  return tmSchemaNode(schema, props[field]) || null;
}

var TM_ACRONYMS = {
  id: "ID", ids: "IDs", uid: "UID", uids: "UIDs", url: "URL", urls: "URLs",
  uri: "URI", api: "API", utc: "UTC", ip: "IP", json: "JSON", sql: "SQL",
  html: "HTML", sku: "SKU"
};

function tmTitleWords(field) {
  return String(field).split("_").filter(function (w) { return w !== ""; });
}

function tmPrettyFieldTitle(field, prop) {
  var name = String(field === null || field === undefined ? "" : field);
  var words = tmTitleWords(name);
  if (prop && prop.title) {
    var mechanical = words.map(function (w) {
      return w.charAt(0).toUpperCase() + w.slice(1);
    }).join(" ");
    if (String(prop.title) !== mechanical) { return String(prop.title); }
  }
  return words.map(function (w) {
    var lower = w.toLowerCase();
    if (Object.prototype.hasOwnProperty.call(TM_ACRONYMS, lower)) { return TM_ACRONYMS[lower]; }
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  }).join(" ");
}

/* The upstream commands that feed `commandName`: each Input field's
   available_from entry is "[Context/]cmd[.field]". Returns a map whose keys
   are both the qualified and the leaf command name, each true. */
function tmUpstreamProducerCommands(schemas, commandName) {
  var found = {};
  var entry = tmCommandSchemaIn(schemas, commandName);
  var input = entry && entry.input;
  if (!input || !input.properties) { return found; }
  Object.keys(input.properties).forEach(function (field) {
    var prop = tmSchemaNode(input, input.properties[field]);
    var edges = prop && prop.available_from;
    if (!Array.isArray(edges)) { return; }
    edges.forEach(function (edge) {
      if (typeof edge !== "string") { return; }
      var command = edge.split(".")[0].replace(/^\/+/, "");
      if (!command) { return; }
      found[command] = true;
      found[tmLeafName(command)] = true;
    });
  });
  return found;
}
