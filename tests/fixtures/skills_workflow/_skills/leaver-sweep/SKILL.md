---
name: leaver-sweep
description: One employee is leaving. Enumerate what they can reach and propose fixes.
level: task
goal: "{identity_query}'s reachable accounts are enumerated and remediations are proposed."
slots:
  - name: identity_query
    required: true
    on_repeat: find_identity with query=*, then offer the named matches
    description: Leaver name, login, or email
uses:
  - inspect-thing
---

# Leaver sweep

1. inspect-thing `entity_type=identity` `query={identity_query}`
2. Take the FIRST account on that portrait: `open_account_by_uid`, then `open_portrait`
3. `propose_fix` on the identity
