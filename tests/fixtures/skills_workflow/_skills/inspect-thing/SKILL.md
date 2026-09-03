---
name: inspect-thing
description: Look up a named entity, open it, and return its portrait.
level: atomic
slots:
  - name: entity_type
    required: true
    on_repeat: browse the catalogue, then offer the kinds that returned rows
    description: identity, account, or permission
  - name: query
    required: true
    on_repeat: find_identity with query=*, then offer the named matches
    description: Name fragment, login, or uid
---

# Inspect thing

Walk to the matching opener, then `open_portrait`.
