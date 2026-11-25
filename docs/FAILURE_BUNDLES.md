# Layered offline FailureBundle

`maple_vision.failure_bundle` implements the chapter 15 archive for actual offline
run bundles. Failed, stopped, budget-exhausted and successfully completed runs
use the same format. It does not control an official client or establish real
game accuracy, account safety, or 24-hour wall-clock acceptance.

The default verification claim is deliberately specific: source integrity,
actual applied actions, deterministic simulator states, exact capture RGB, and
faithful projection of the recorded evidence. Perception and policy reproduction
require a separate, actual reinference call.

## API

```python
from pathlib import Path

from maple_vision.failure_bundle import (
    export_failure_bundle,
    verify_failure_bundle,
)

exported = export_failure_bundle(
    Path("runs/example"),
    Path("failures/F-000001"),
    include_video=False,
)
checked = verify_failure_bundle(Path("failures/F-000001"))

# Actually re-executes the original pixel perception and policy on all N+1 frames.
reproduced = verify_failure_bundle(
    Path("failures/F-000001"),
    reinfer=True,
)
```

Both functions accept `reinfer=False`, `detector=None` and
`terrain_source="fixture"`. Export additionally accepts `include_video=False`.
A supplied detector or nondefault terrain mode requires `reinfer=True`; merely
passing a detector cannot manufacture a reproduction claim. For ONNX runs,
supply the matching local `OnnxDetector` and terrain mode. The existing
`verify_perception` checks model hash, full manifest and inference options and
actually reruns the supervisor, policy and optional structured runtime.

The returned dictionary includes `verified`, `steps_replayed`,
`frames_reconstructed`, `action_replay_verified`,
`deterministic_state_reproduced`, `rgb_reproduced`,
`layer_projection_verified`, `perception_reproduced`, `policy_reproduced`,
`action_replay`, `reinference`, `video`, source identity and recorded outcome.
`reinference` is `None` by default. It contains the actual verifier result only
for the current explicit reinference call. An export-time reinference result is
not serialized as an assertion that later verification has rerun the model.

Source-run schemas 1 and 2 are supported, provided their explicit versioned
terminal perception evidence is present. Older action-only archives lacking a
real terminal observation remain usable with `lab.verify_run`; they must be
regenerated before layered export. The exporter never synthesizes that missing
observation or substitutes the preceding frame.

## Files and alignment

The archive has a strict schema-1 `offline-failure-bundle` manifest. Its complete
file map covers the exact `source-run/` copy, layers, PNGs and optional video.
The source manifest hash and schema, frame counts, simulation timeline, layer
row counts and verification semantics are bound separately.

| File | Rows / content | Meaning |
|---|---|---|
| `source-run/` | Exact original manifest and every listed source file | Self-contained input for action replay and optional reinference |
| `ground_truth.jsonl` | N+1 | Reconstructed complete simulator saves, state hashes, clean-render annotations, clean and captured RGB hashes |
| `observations.jsonl` | N+1 | Original raw observation at each actual capture, including termination |
| `world_states.jsonl` | N+1 | Recorded temporal belief for schema 2; explicit unavailable/null for legacy schema 1 |
| `task_trace.jsonl` | N+1 | Recorded decision, whether policy was evaluated, structured decision/state/interruption when present, and real termination cause |
| `action_trace.jsonl` | N | Requested action alongside actual execution, dispatch, receipt and resulting state hash |
| `verification.jsonl` | N | Replayed execution result/state hash and separately named recorded visual-effect result |
| `supervisor.jsonl` | N+1 | Actual recorded guard decision; null at an unevaluated budget terminal |
| `frames.jsonl` | N+1 | Simulation time, capture identity, exact RGB hash and PNG reference |
| `frames/000000.png`, etc. | N+1 | Lossless 320×192 RGB capture images, including injected faults |
| `versions.lock` | JSON | Recorded runtime/model/content identities and source manifest identity |
| `config.yaml` | JSON, a YAML 1.2 subset | Exact run, perception and structured-runtime configuration |
| `report.md` | Markdown | Recorded outcome and explicit verification boundaries |
| `manifest.json` | Strict JSON | Full inventory and source/layer/timeline binding |
| `replay.mp4` | Optional | Lossy display of the N+1 capture frames |
| `video-info.json` | Optional | Container timing, file hash, codec fingerprint, source RGB timeline hash and display-only claim |
| `video_frames.jsonl` | Optional, N+1 | Source RGB reference, playback PTS and recorded decoded-display RGB hash |

All frame-aligned rows have `schema_version`, `frame_index`, actual simulator
`sequence` and `timestamp_ms`, a `terminal` boolean, and a source-reference hash.
The source reference identifies the original trace row or terminal perception
record. Action-aligned rows identify the **before-action** frame. The action's
result appears at the next frame; no fabricated terminal action is appended.

Simulation time and capture time are distinct. For the duplicate-frame fault,
simulation step 20 occurs at 1000 ms while its repeated capture can still be
sequence 19 at 950 ms. The archive preserves both. It also preserves black
dropout captures separately from clean simulator annotations. Those annotations
are synthetic render truth and are not claims that the faulted image contains
visible objects.

The original task engine is never copied into this module. The exporter loads
the original saved simulator and embedded map, static sprite, animation and
class-package content through their existing loaders. Schema-4 class saves are
restored by `Simulator.restore`. It steps only `execution.applied_action`, then
checks real acceptance, reason, events and resulting state. In a freeze-input
run, a requested attack can coexist with an actually applied idle action;
ground truth follows the idle.

## Terminal evidence is not normalized away

At budget exhaustion there is a real final observation/belief, but no extra
policy evaluation. `decision=None`, `policy_evaluated=False`, and the structured
terminal interruption remain explicit. A supervisor stop similarly retains the
stop guard and interruption without inventing a policy decision.

Policy termination retains the actual terminal decision. One intentional dual
outcome is supported: the policy claims `completed`, but the independent mission
oracle rejects completion. In that case the archive preserves the completed
policy decision and `summary.status="failed"` with
`reason="independent_goal_check_failed"`. It permits this override only after
actual `verify_run` independently returns `independent_goal_check=False`.
The exporter does not edit the terminal decision to hide the failure.

## Verification and tampering boundary

Export first captures bounded private source bytes, checks the exact source
inventory, requires real terminal evidence, and calls `verify_run`. Verification
of an existing FailureBundle also operates on a private bounded copy and leaves
the supplied archive unchanged. Original source paths can be removed before
verification.

The verifier checks all file hashes, replays the embedded source, independently
reconstructs each frame and state, and reprojects the layers. Changing a derived
ground-truth value, belief, task result, observation, action, verification flag,
supervisor result, configuration or report and then repairing all outer hashes
still fails comparison with the verified source-run projection. Re-encoding a
PNG losslessly is allowed only when its exact RGB pixels still match replay.
Duplicate JSON keys, unsupported schemas, missing or extra files, unsafe paths,
symlinks, special files, mismatched capture identities and false terminal-policy
claims are rejected.

Default action replay does **not** prove that an originally recorded observation
or policy choice was computed correctly. For example, a coherently rewritten
source observation can pass action replay while failing actual reinference.
The report never labels the former as perception reproduction. Checksums and
reprojection provide integrity and consistency, not a signature attesting who
originally created a fully re-authored source bundle.

The package retains all embedded content provenance, including self-authored
synthetic animation metadata. The supplied geometric animation example is a
self-drawn synthetic fixture, not licensed official-game art. Its inclusion in
a replay does not change that scope.

## Optional video

Creation uses the installed local OpenCV FFmpeg `mp4v` encoder. It makes no
network or package-install request. A missing encoder/dependency is an explicit
error and leaves no completed output or fake video. Default exports need no
OpenCV dependency.

The source is independently confirmed to start at simulation time zero, have
320×192 frames, and advance by exactly 50 ms on every applied action. Only then
is 20 fps valid. Presentation timestamps are `frame_index × 50 ms`, including
the final frame; its display duration is also 50 ms. A repeated capture still
occupies the next simulation playback slot. Nonuniform or relabelled source
timelines are rejected.

Video checks include the actual ISO-BMFF file structure, a single local video
track, codec/dimensions, movie and media duration, `stts` sample intervals,
`stsz` sample count, chunk-to-sample mappings and bounded nonoverlapping media
offsets. External media references, composition-offset or fragmented formats
are outside this generated replay contract. Container timing is not inferred
from an FPS hint alone.

When the current OpenCV build fingerprint matches the recorded build, decoding
also checks all N+1 frame dimensions, PTS and exact **decoded display** RGB
hashes. The transport whitelist permits local file/pipe decoding only. A missing
decoder or different build is reported as `not_run_decoder_unavailable` or
`not_run_decoder_mismatch`; it does not invalidate independently reproduced
authoritative RGB/state evidence. Container, timeline and file hash checks
remain required.

`video.semantic_reproduction` is always false in this version. MP4 hashes and
display decoding checks are not exact equality with the pre-encoding source
RGB, and they do not establish semantic authenticity of a completely replaced
video whose display metadata and hashes are all re-authored. The authoritative
semantic evidence is the exact simulator/RGB reconstruction. No approximate
pixel threshold is presented as exact agreement. Cross-encoder re-encoding
equivalence is not implemented.

## Bounds and publication

| Resource | Limit |
|---|---:|
| Captured frames | 10,001, including terminal |
| Source files / all archive files | 12,000 / 25,000 |
| Source bytes / total archive bytes | 256 MiB / 512 MiB |
| Any individual file / optional MP4 | 128 MiB |
| Manifest / standalone JSON | 4 MiB / 8 MiB |
| JSONL row / replay PNG file | 512 KiB |
| Raw observation detections | 512 |
| Relative path length / depth | 240 characters / 5 components |

These are archive limits, not extensions to the stricter limits of map, asset,
animation, runtime or class-package loaders. Existing large action-only runs
may exceed them and remain usable with their original verifier.

All file counts and stat sizes are checked before payload hashing or decoding.
Reads are bounded again during private capture, JSONL rows are bounded before
parsing, and replay PNG dimensions are checked before decoding source images.
Derived writes enforce line, file and total output budgets.

Publication uses a private staging directory and an exclusively reserved new
destination. The manifest is the final commit marker: a partial publication
cannot validate as a bundle. Existing outputs are never overwritten, sources
are not modified, and failed exports clean up their own staged output. Output
inside the source tree is rejected.

## Focused validation

```bash
PYTHONPATH=src python -m pytest tests/test_failure_bundle.py -q
python -m ruff check src/maple_vision/failure_bundle.py tests/test_failure_bundle.py
python -m mypy src/maple_vision/failure_bundle.py --follow-imports=silent
```

The suite exercises schema-1/2 N+1 terminals, successful and false-completion
runs, actual-applied versus requested freeze-input behavior, all visual fault
families, real reinference, read-only operation, deleted original paths,
external map/animation/schema-4 class content, rehashed layered tampering,
resource/path/output boundaries, real MP4 generation, and independent MP4
timing/chunk-offset corruption. Native video cases require the local optional
video dependency; dependency-unavailable behavior has a separate regression.
