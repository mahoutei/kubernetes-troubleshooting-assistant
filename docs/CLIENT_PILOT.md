# Kubernetes investigation pilot

This is a pilot protocol, not evidence of time savings or a production service guarantee. See [current status](STATUS.md) for release limits and the [application guide](../projects/01-rag-k8s-docs/README.md) for setup.

## Client and decision

Target users are on-call engineers in small Kubernetes teams and cloud consultancies. The proposed benefit is reducing the effort needed to collect evidence and choose an appropriate next diagnostic check. The buyer would normally own platform reliability or engineering operations. Validate that need through discovery before promising a deliverable or quoting a pilot.

The assistant accepts redacted manifests, logs and events; it supplies cited hypotheses, missing-information requests and suggested checks. It does not connect to clusters, execute commands, or establish a root cause automatically. The standalone fine-tuning notebook remains experimental and is not required for this pilot.

## Before a client pilot

1. Finish real-provider release checks and human semantic review described in [the release guide](beta-release.md). A local demo or schema-valid answer alone does not meet this gate.
2. Agree on the incident classes, allowed data, participant consent, hosted-provider processing, evidence retention, fixed attempt budget and review rubric. Use synthetic development cases initially. Remove secrets and identifying infrastructure details; keep actual client observations and response artifacts outside the public repository.
3. Freeze the application revision, documentation/index identity, model/provider, configuration, case versions and scoring rubric. Save response JSON including usage, timings and citations in private artifacts. Record errors and model unavailability as failed attempts; do not silently substitute a demo.
4. Assign at least two participants to development cases. The generated two-participant worksheet alternates conditions and reverses case order. Nobody sees the same case twice; each case appears in both arms across different people. More participants and balanced experience are needed before drawing strong conclusions. The ten held-out evaluation cases remain outside this worksheet and must not be used for prompt tuning.

## Run the worksheet

From the repository root, without models, datasets or API calls:

```bash
uv run --frozen python scripts/pilot_study.py init --output /tmp/k8s-pilot/observations.jsonl
# Complete the worksheet with actual observations and optional human reviews.
uv run --frozen python scripts/pilot_study.py summarize --input /tmp/k8s-pilot/observations.jsonl --output /tmp/k8s-pilot/report.json
```

Both commands refuse to overwrite existing files. The initial worksheet contains null observations and cannot be summarized until attempts have been recorded. Initialization uses only the ten existing development cases, producing twenty unfilled rows; these are assignments, not measured results. Independent client cases may use their own stable case IDs.

For each attempt:

- **Manual:** use the incident evidence and official Kubernetes documentation, with no AI assistance. **Assistant:** use the same allowed information plus live assistant output. Keep the task and time budget equal.
- Start timing when the participant opens the evidence. Stop at submission or the agreed budget. Record positive `elapsed_seconds` for every attempt, including errors, abandoned and timed-out attempts; `completed` means a submitted investigation, not a correct answer.
- Submit observed facts, ranked hypotheses, missing evidence, citations and next checks. Store that submission under `artifact_reference`, a private stable path or identity. Use pseudonymous participant IDs, unique positive attempt orders, and notes for interruptions or deviations.
- Set `evidence_mode` to `live` for actual hosted assistant output. Mark any recorded/fake rehearsal accurately; the report excludes those assisted attempts from its comparisons. Do not mix rehearsal participants with the later pilot without recording prior exposure.
- Have a reviewer independently assess the submission against the frozen case rubric, ideally without knowing its condition. Fill `review` with `diagnosis_correct` and `citation_support` booleans, a nonnegative integer `unsupported_claims`, and a nonempty `reviewer` identity. Leave the entire review null while pending. Save the justification in private notes/artifacts.

Example review shape (format only, not an observed result):

```json
{"diagnosis_correct": true, "citation_support": true, "unsupported_claims": 0, "reviewer": "reviewer-id"}
```

A reviewed acceptable outcome requires completion, a correct investigation, supported citations and zero unsupported claims. The quality denominator is reviewed attempts, with unreviewed counts shown separately. Review incomplete attempts as well to avoid scoring only successful submissions.

## Interpret and deliver

The report shows median attempt duration and reviewed acceptable outcomes per arm, then compares median durations for cases observed in both arms. Positive `manual_minus_assistant_seconds` means the assisted median was shorter; negative values are retained. Durations include failures, so a shorter failed attempt is not evidence of greater productivity. Inspect quality and completion alongside timing. An empty arm, missing review or lack of common cases yields null metrics rather than an invented gain.

The observations are human-entered and not independently verified by the script. Participant skill, order, prior exposure, case selection, reviewer subjectivity and small samples can confound differences. Treat comparisons as descriptive; do not claim causal improvement, financial ROI, annual savings or production MTTR reduction from this pilot. Existing evaluation tools separately record model latency and token usage; provider prices and engineering labor costs are not assumed here.

A client-facing pilot delivery should include the frozen build, reproducible setup, redacted per-case outcomes, review coverage, time/quality comparisons, representative failures, observed operating requirements and a recommendation to continue, narrow or stop. Agree acceptance targets before collecting results; do not choose them after seeing the scores. Public case studies require permission and must distinguish local synthetic cases from client observations.
