/* The chat SPA loads GET /command_schemas once per connection and exposes
   pure helpers over it (fix-weih.3). The page's own tmLoadCommandSchemas,
   tmWithCommandSchemas, tmCommandSchema, tmSchemaNode, tmSchemaProperty,
   tmPrettyFieldTitle and tmUpstreamProducerCommands are exercised. Only the
   network edge (window.fetch) is replaced, so each response can be scripted. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require(process.argv[2] + '/node_modules/jsdom');

const srcDir = process.argv[3];
const names = fs.readdirSync(srcDir).filter((name) => !name.startsWith('.')).sort();
const html = names.map((name) => fs.readFileSync(path.join(srcDir, name))).join('');
const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (err) => {
  if (err.type !== 'css-parsing') errors.push(String(err.message || err));
});
virtualConsole.on('error', (...args) => errors.push('console.error: ' + args.join(' ')));

const BASE = 'http://127.0.0.1:9';
const calls = [];
let script = null; /* the next scripted response for BASE urls */

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  url: 'http://127.0.0.1:39131/',
  virtualConsole,
  beforeParse(window) {
    window.fetch = (url, opts) => {
      if (String(url).indexOf(BASE) !== 0) return new Promise(() => {});
      calls.push({url: String(url), opts});
      return script(url, opts);
    };
  },
});
const w = dom.window;

function respond(status, body) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => (typeof body === 'string'
      ? Promise.reject(new SyntaxError('Unexpected token'))
      : Promise.resolve(body)),
  });
}
function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return {promise, resolve};
}
/* jsdom objects come from another realm; compare by value through JSON. */
function same(actual, expected) {
  assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected);
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

for (const fn of ['tmLoadCommandSchemas', 'tmWithCommandSchemas', 'tmCommandSchema',
  'tmSchemaNode', 'tmSchemaProperty', 'tmPrettyFieldTitle', 'tmUpstreamProducerCommands']) {
  assert.equal(typeof w[fn], 'function', fn);
}
const tm = w.tm;
assert.equal(tm.commandSchemas, null);
assert.equal(tm.commandSchemasPromise, null);

const LISTING = {
  'Library/find_book': {kind: 'execution', input: {properties: {}}, output: null},
  get_order_details: {
    kind: 'execution',
    input: {
      properties: {
        order_id: {type: 'string', title: 'Order Id', available_from: ['get_user_details.order_ids']},
      },
    },
    output: null,
  },
  get_user_details: {kind: 'execution', input: null, output: null},
  'A/show': {kind: 'execution', input: null, output: null},
  'B/show': {kind: 'execution', input: null, output: null},
};

async function main() {
  tm.baseUrl = BASE;
  tm.token = 't';

  /* 1. request shape and population */
  const gate = deferred();
  script = () => gate.promise;
  const loading = w.tmLoadCommandSchemas();
  assert.equal(tm.commandSchemas, null);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, BASE + '/command_schemas');
  assert.equal(calls[0].opts.method, 'GET');
  assert.equal(calls[0].opts.headers.Authorization, 'Bearer t');
  assert.equal(calls[0].opts.body, undefined);

  /* 4a. in flight: the callback waits for the load */
  let seen = null;
  w.tmWithCommandSchemas((s) => { seen = s; });
  assert.equal(seen, null, 'callback must not run while the load is in flight');
  gate.resolve(await respond(200, LISTING));
  await loading;
  await tick();
  same(seen, LISTING);
  same(tm.commandSchemas, LISTING);

  /* 4b. settled: synchronous */
  let sync = null;
  w.tmWithCommandSchemas((s) => { sync = s; });
  same(sync, LISTING);

  /* 2. name resolution */
  const entry = LISTING['Library/find_book'];
  assert.equal(w.tmCommandSchema('Library/find_book'), entry);
  assert.equal(w.tmCommandSchema('find_book'), entry);
  assert.equal(w.tmCommandSchema('/find_book'), entry);
  assert.equal(w.tmCommandSchema('show'), null, 'two contexts share the leaf');
  assert.equal(w.tmCommandSchema('A/show'), LISTING['A/show'], 'exact key still wins');
  assert.equal(w.tmCommandSchema('nope'), null);

  /* 3. failures settle to {} without errors */
  const failures = [
    () => respond(404, {detail: 'nope'}),
    () => Promise.reject(new Error('network down')),
    () => respond(200, 'not json'),
    () => respond(200, [1, 2]),
  ];
  for (const make of failures) {
    script = make;
    await w.tmLoadCommandSchemas();
    same(tm.commandSchemas, {});
  }

  /* 4c. stale epoch drops a late response; the real source switch resets */
  const late = deferred();
  script = () => late.promise;
  const stale = w.tmLoadCommandSchemas();
  let queued = 'unset';
  w.tmWithCommandSchemas((s) => { queued = s; });
  w.resetSourceScopedState();
  assert.equal(tm.commandSchemas, null);
  assert.equal(tm.commandSchemasPromise, null);
  tm.baseUrl = BASE;
  tm.token = 't';
  late.resolve(await respond(200, LISTING));
  await stale;
  await tick();
  assert.equal(tm.commandSchemas, null);
  same(queued, {});
  let idle = null;
  w.tmWithCommandSchemas((s) => { idle = s; });
  same(idle, {});

  /* the real connect path requests the listing after /initialize succeeds */
  calls.length = 0;
  script = (url) => {
    if (String(url).endsWith('/initialize')) {
      return respond(200, {access_token: 'jwt'});
    }
    if (String(url).endsWith('/command_schemas')) return respond(200, LISTING);
    return new Promise(() => {});
  };
  w.tmConnect();
  for (let i = 0; i < 20 && !tm.commandSchemas; i++) await tick();
  const urls = calls.map((c) => c.url);
  assert.deepEqual(urls.slice(0, 2), [BASE + '/initialize', BASE + '/command_schemas']);
  assert.equal(calls[1].opts.method, 'GET');
  same(tm.commandSchemas, LISTING);
  assert.equal(tm.connected, true);

  /* 5. $ref, Optional unwrapping, nullable */
  const root = {
    properties: {
      flag: {
        anyOf: [{type: 'boolean'}, {type: 'null'}],
        default: null,
        title: 'Flag',
        description: 'Whether to flag it',
        ui: {widget: 'radio'},
      },
      plain: {type: 'boolean', title: 'Plain'},
      kind: {'$ref': '#/$defs/Kind', description: 'Which kind'},
      maybe_kind: {anyOf: [{'$ref': '#/$defs/Kind'}, {type: 'null'}], default: null},
      note: {type: ['string', 'null']},
    },
    '$defs': {Kind: {type: 'string', enum: ['a', 'b'], title: 'Kind'}},
  };
  const flag = w.tmSchemaProperty(root, 'flag');
  assert.equal(flag.type, 'boolean');
  assert.equal(flag.nullable, true);
  assert.equal(flag.default, null);
  assert.ok(Object.prototype.hasOwnProperty.call(flag, 'default'));
  assert.deepEqual(JSON.parse(JSON.stringify(flag.ui)), {widget: 'radio'});
  assert.equal(flag.description, 'Whether to flag it');
  const plain = w.tmSchemaProperty(root, 'plain');
  assert.equal(plain.type, 'boolean');
  assert.ok(!plain.nullable);
  const kind = w.tmSchemaProperty(root, 'kind');
  assert.deepEqual(JSON.parse(JSON.stringify(kind.enum)), ['a', 'b']);
  assert.equal(kind.description, 'Which kind');
  assert.ok(!kind.nullable);
  const maybe = w.tmSchemaProperty(root, 'maybe_kind');
  assert.deepEqual(JSON.parse(JSON.stringify(maybe.enum)), ['a', 'b']);
  assert.equal(maybe.nullable, true);
  const note = w.tmSchemaProperty(root, 'note');
  assert.equal(note.type, 'string');
  assert.equal(note.nullable, true);
  assert.equal(w.tmSchemaProperty(root, 'missing'), null);
  assert.equal(w.tmSchemaProperty(null, 'flag'), null);
  /* the source schema is never mutated */
  assert.ok(root.properties.flag.anyOf && !('nullable' in root.properties.flag));

  /* Optional[Model]: the field's own title and description beat the class's */
  const addr = {
    properties: {
      shipping: {
        anyOf: [{'$ref': '#/$defs/Address'}, {type: 'null'}],
        title: 'Shipping Address',
        description: 'Where it goes',
        default: null,
      },
    },
    '$defs': {Address: {
      type: 'object', title: 'Address', description: 'A postal address',
      properties: {city: {type: 'string'}},
    }},
  };
  const ship = w.tmSchemaProperty(addr, 'shipping');
  assert.equal(ship.title, 'Shipping Address');
  assert.equal(ship.description, 'Where it goes');
  assert.equal(ship.nullable, true);
  assert.equal(ship.type, 'object');
  assert.ok(ship.properties && ship.properties.city);

  /* 6. field titles */
  assert.equal(w.tmPrettyFieldTitle('order_id', {title: 'Order Id'}), 'Order ID');
  assert.equal(w.tmPrettyFieldTitle('book_uids', {title: 'Book Uids'}), 'Book UIDs');
  assert.equal(w.tmPrettyFieldTitle('total_paid', {title: 'Total Paid ($)'}), 'Total Paid ($)');
  assert.equal(w.tmPrettyFieldTitle('api_url', null), 'API URL');
  assert.equal(w.tmPrettyFieldTitle('first_name'), 'First Name');

  /* 7. upstream producers */
  tm.commandSchemas = null;
  const up = w.tmUpstreamProducerCommands(LISTING, 'get_order_details');
  assert.equal(up.get_user_details, true);
  assert.equal(Object.keys(up).length, 1);
  const none = w.tmUpstreamProducerCommands(LISTING, 'find_book');
  assert.equal(Object.keys(none).length, 0);
  /* it reads its argument, not the cache, even when the cache differs */
  tm.commandSchemas = {get_order_details: {input: {properties: {
    x: {available_from: ['other_cmd.y']},
  }}}};
  const own = w.tmUpstreamProducerCommands(LISTING, 'get_order_details');
  assert.deepEqual(Object.keys(own), ['get_user_details']);
  assert.equal(w.tmCommandSchemaIn(LISTING, 'find_book'), LISTING['Library/find_book']);
  assert.equal(w.tmCommandSchemaIn(LISTING, 'show'), null);
  assert.equal(w.tmCommandSchemaIn(null, 'show'), null);
  tm.commandSchemas = {
    'Shop/get_order_details': {input: {properties: {
      order_id: {available_from: ['Shop/get_user_details.order_ids', '/lookup']},
    }}},
  };
  const qual = w.tmUpstreamProducerCommands(tm.commandSchemas, 'get_order_details');
  assert.deepEqual(Object.keys(qual).sort(), ['Shop/get_user_details', 'get_user_details', 'lookup']);

  if (errors.length) throw new Error(errors.join('\n'));
}

main().then(() => process.exit(0), (err) => {
  console.error(err && err.stack || err);
  process.exit(1);
});
