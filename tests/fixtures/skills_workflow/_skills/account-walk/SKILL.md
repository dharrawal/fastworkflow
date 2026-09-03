---
name: account-walk
description: Walk one identity down to the first account on its portrait.
level: task
goal: The first account of {identity_query} has been opened and portrayed.
slots:
  - name: identity_query
    required: true
    on_repeat: find_identity with query=*, then offer the named matches
    description: Identity name, login, or email
uses:
  - account-portrait
---

# Account walk

1. `list_accounts` on the identity
2. account-portrait `account_uid={account_uid}`
