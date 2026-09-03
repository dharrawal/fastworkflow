---
name: offboarding-batch
description: Several people are leaving at once. Sweep each of them in turn.
level: composite
goal: Every leaver named in {identity_queries} has been swept.
slots:
  - name: identity_queries
    required: true
    on_repeat: find_identity with query=*, then offer the named matches
    description: The leavers the request names
    list: true
uses:
  - leaver-sweep
---

# Offboarding batch

1. for each {leaver} in {identity_queries}: leaver-sweep `identity_query={leaver}`
