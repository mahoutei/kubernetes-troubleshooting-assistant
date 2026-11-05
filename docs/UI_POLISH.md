# Interface polish and validation

Verified locally on 2026-11-05 using [Hallmark](https://github.com/Nutlope/hallmark/tree/de4999e0cd7a7a41df0b4e6903fa02fe9f1492a8), in app-screen polish mode. The [current status](STATUS.md) remains authoritative for release readiness.

The dark slate/mint palette, system fonts, two-column workflow, evidence fields, endpoints, and safety statements remain. The interface now uses role-based color and spacing tokens, a fixed application type scale, stronger field borders, keyboard focus rings and a skip link, native upload controls, visible request/error states, and separated citation links. Reduced-motion and JavaScript-disabled behavior are explicit. No new frontend framework or runtime dependency was added.

The finish review caught clipped demo titles, crowded citations, and a missing header beta label. All three were corrected. The demo selector retains a full row; the beta label follows the title rather than appearing as a tracked uppercase eyebrow. Recorded/live labels use sentence case, and evidence labels use slashes instead of middle dots. Safety and limitation statements retain their wording.

## Verification

| Check | Result |
|---|---|
| Hallmark static + Chromium rendered sweep | Before: 17 floor failures, 12 reflex findings, 4 warnings. After: 0 failures, 0 reflex findings, 0 warnings. One note retains original safety-copy punctuation. |
| Preservation audit | 10/10 fields, 3/3 headings, 1/1 recorded script hooks, and 1/1 figures retained. Browser checks also exercise the original demo, diagnosis, upload, and Q&A handlers. |
| Shared CPU workflow | 23 tooling + 131 RAG + 54 fine-tuning tests pass; all lint and 60 gold-question checks pass. Existing Starlette/httpx deprecation warning remains. |
| Real recorded-demo browser workflow | All three incidents, optional recorded GitHub evidence, desktop/mobile screenshots, and walkthrough recording pass without JavaScript errors. |
| Browser interactions with simulated provider replies | Keyboard focus, UTF-8 and upload-size errors, busy/disabled state, successful response, provider error recovery, safe text rendering, citation targets, and Q&A pass. |
| Responsive layout | No horizontal page overflow or off-screen form controls at 375, 390, 800, 1440, and 1920 CSS pixels. JavaScript-disabled guidance verified separately. |

The audit covers the local application and static HTML at 1280, 1440, and 1920 desktop widths plus 375 mobile. Local before/after captures and comparison pages are under `.hallmark/` (ignored by Git); they are not served as application assets. The only audit copy note is the preserved dash in the original read-only safety statement. A passing visual audit is not a claim of complete accessibility certification.

## Preview and reproduce

- [Desktop screenshot](media/troubleshooting-demo.png)
- [Mobile screenshot](media/troubleshooting-mobile.png)
- [Recorded walkthrough](media/troubleshooting-demo.webm)

With the application running, regenerate those three assets from the repository root:

```bash
uv run --no-project --with playwright python scripts/record_troubleshooting.py --base-url http://127.0.0.1:8000
```

Playwright and its Chromium browser are optional recording tools, outside the shared CPU environment. Install Chromium with Playwright before the first capture. The recording uses actual authored demonstration responses; simulated-provider test replies are excluded from the published media.

Live hosted-provider quality, public deployment, and Colab GPU execution remain pending as listed in current status. This polish does not establish new model-quality or benchmark results.
