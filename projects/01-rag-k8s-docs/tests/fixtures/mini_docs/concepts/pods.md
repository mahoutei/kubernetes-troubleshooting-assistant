---
title: Pods
draft: false
---

# Pods

A Pod is the smallest deployable unit.

## Lifecycle

{{< note >}}
Pods are ephemeral.
{{< /note >}}

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: demo

  labels:
    app: demo
```

Pods can be evicted.
