# Direct action adapter: reproducible contract test

The active direct-continuous adapter preserves original object IDs: one view per original sequence. A 32-decision input produces 32 adapter items, not 768 identity-augmented items. Historical releases and trained weights have not been rewritten.

With Python and PyTorch installed, run from the repository root:

```sh
python -B -m unittest discover -s tests -p test_direct_action_adapter_synthetic.py -v
```

All four tests use explicitly synthetic, in-memory records and exercise the actual `scripts/convert_terra_direct_continuous.py` API. No saved demonstrations, private checkpoints, simulator, teacher, trained model, robot, or network service is used. The test review JSON is synthetic too: a matching digest is not authentication of a real reviewer.

Coverage: original-ID/feature preservation, variable-length original boundaries, parent/initial-state split leakage rejection, and review fingerprint tamper rejection. This is an adapter contract test, not evidence of learned performance or end-to-end autonomous model improvement.

## Scope and dependencies

The nine imported local dependency files already in this repository were byte-matched against the tested source snapshot. PyTorch is the external dependency.

Raw journal conversion (`convert_case`, `convert_public`, visual-evidence validation) additionally requires the original collection scripts and recorded artifacts, including private checkpoints. Those are deliberately not included in this minimal publication. The legacy nonidentity `remap_visible` helper lazily requires the historical training module and is not part of the active single-view contract. Consequently, this minimal publication does not claim to reproduce raw private demonstration validation or historical 24-view experiments.

The legacy helper names and record fields remain, but callers that assumed 24 views must now use the adapter's actual `count`. A split assignment or successful synthetic test does not grant training eligibility.
