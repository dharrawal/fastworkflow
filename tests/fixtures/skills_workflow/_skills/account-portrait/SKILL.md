---
name: account-portrait
description: Open one account by uid and return its portrait.
level: task
goal: Account {account_uid} has been opened and portrayed.
slots:
  - name: account_uid
    required: false
    description: The account uid a prior step produced
uses:
  - inspect-thing
---

# Account portrait

1. inspect-thing `entity_type=account` `query={account_uid}`
2. `open_portrait`
