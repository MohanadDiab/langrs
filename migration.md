# Rex-Omni Migration Notes for LangRS

## Goal

Integrate Rex-Omni directly inside LangRS so users do not depend on the upstream `Rex-Omni` repository for core detection workflows.

The target is not strict byte-for-byte parity.  
The target is:

- Rex-Omni works reliably inside LangRS
- It fits LangRS architecture and UX
- It supports LangRS pipeline features end-to-end

---

## Current Integration Map

### Upstream-style components (vendored)

LangRS includes a vendored Rex-Omni implementation under:

- `langrs/rex_omni/wrapper.py`
- `langrs/rex_omni/parser.py`
- `langrs/rex_omni/tasks.py`
- `langrs/rex_omni/utils.py`
- `langrs/rex_omni/__init__.py`

These map to upstream `Rex-Omni/rex_omni/*` and preserve the same core detection flow:

1. Build prompt + multimodal chat message
2. Run backend (`transformers` or `vllm`)
3. Parse structured tokenized output into predictions
4. Convert bins (`0..999`) to pixel-space coordinates

### LangRS adapter/integration layer

Rex-Omni is integrated into LangRS through:

- `langrs/models/detection/rex_omni.py` (adapter implementing `DetectionModel`)
- `langrs/models/__init__.py` (registry registration as `"rex_omni"`)
- `langrs/models/factory.py` (factory creation path)
- `langrs/core/builder.py` (default detection model)
- `langrs/core/pipeline.py` (`detect_objects` orchestration with tiling/NMS/export)

---

## What Works Well Today

### 1) Core detection is wired correctly

`LangRS.detect_objects()` -> `RexOmniDetector.detect()` -> `RexOmniWrapper.inference(task="detection")` -> parser -> LangRS box list.

### 2) It is aligned with LangRS architecture

- Uses `ModelRegistry` and `ModelFactory`
- Conforms to `DetectionModel` interface
- Supports lazy model loading (`load_weights`)
- Participates in normal pipeline + output manager + visualization + geospatial export

### 3) Rex-Omni-specific behavior is handled

- Prompt categories are translated into Rex-Omni-compatible category lists
- Rex output token format is parsed into absolute pixel coordinates
- Both backends are represented in wrapper (`transformers`, `vllm`)

### 4) LangRS features remain available on Rex-Omni path

- Sliding-window tiling controls (`auto`/`always`/`never`)
- `max_tiles` protection and dynamic window adjustment
- Optional cross-tile NMS
- Standard LangRS visualization/output paths

---

## Intentional / Acceptable Differences vs Upstream

These do not block integration goals and can remain as-is unless product decisions change.

- LangRS adapter returns plain bounding boxes (no class labels/scores in pipeline state)
- DINO-style thresholds exist in method signature for compatibility but are not used by Rex-Omni decoding
- Some vendored files include cleanup changes (logging/docstrings/error-print suppression) without changing core detection semantics
- LangRS defaults may differ slightly from wrapper defaults (for example generation knobs)

---

## Integration Risks to Track

### 1) Dependency/runtime sensitivity

- `qwen-vl-utils` is required by wrapper inference paths
- Transformers path depends on environment support for configured attention implementation
- `vllm` availability is platform-dependent (especially on Windows)

### 2) Prompt/category UX contract

Current adapter expects comma-separated categories in `text_prompt` for best results.  
If user prompt style becomes natural language paragraphs, detection quality and predictability may drop.

### 3) Device behavior expectations

Passing `device` into LangRS does not always imply strict backend placement unless backend-specific args are aligned (for example `device_map` path in transformers mode).

---

## Feature Coverage Status (LangRS perspective)

- Detection backend integration: **Covered**
- Tiling + large-image handling: **Covered**
- Cross-tile aggregation + optional NMS: **Covered**
- Segmentation handoff compatibility (boxes into SAM flow): **Covered**
- Geospatial output compatibility: **Covered**
- Batch-oriented advanced Rex tasks (OCR/keypoint/gui/etc.) inside LangRS pipeline API: **Not primary target currently**

Note: Vendored Rex wrapper contains those task capabilities, but LangRS pipeline currently consumes Rex mainly as a box detector.

---

## Practical Validation Checklist

Use this checklist whenever Rex-Omni integration changes:

1. **Model load**
   - `LangRS(detection_model="rex_omni")` initializes and lazy-loads correctly
2. **Small image detection**
   - Full-image mode returns boxes without tiling issues
3. **Large image detection**
   - Auto-tiling path executes and returns aggregated global coordinates
4. **Post-processing**
   - Optional NMS toggles correctly
5. **Pipeline continuity**
   - `run_full_pipeline()` completes detection -> outlier filter -> segmentation
6. **Output artifacts**
   - Detection visualizations and expected output files are produced
7. **Environment matrix**
   - Verify at least one stable transformers environment and one optional vLLM environment

---

## Companion Test Plan (Execution Ready)

This section is the operational companion to the checklist above. Use it as a quick runbook.

### A) Environment Smoke Tests

1. **Import and package wiring**
   - `python -c "from langrs import LangRS; print('ok')"`
2. **Rex detector registration path**
   - `python -c "from langrs.models import ModelRegistry; print(ModelRegistry.list_detection_models())"`
3. **Rex wrapper import path**
   - `python -c "from langrs.rex_omni import RexOmniWrapper; print('wrapper ok')"`

Expected result: imports succeed and `"rex_omni"` is listed.

### B) Functional Detection Scenarios

Use one small RGB image and one large image (or a GeoTIFF for geospatial checks).

1. **Small image / no tiling pressure**
   - Build `LangRS(output_path='output', detection_model='rex_omni')`
   - Call `load_image()` + `detect_objects(text_prompt='building, road')`
   - Confirm non-empty (or at least valid-format) list of 4-float boxes
2. **Large image / auto tiling**
   - Set `config.detection.tiling_mode='auto'`
   - Use large input, run detection
   - Confirm boxes are in global image coordinates (not tile-local)
3. **Force single-call mode**
   - Set `config.detection.tiling_mode='never'`
   - Confirm one-pass behavior still returns valid boxes
4. **Force tiled mode**
   - Set `config.detection.tiling_mode='always'`
   - Confirm tiled path works without coordinate corruption

### C) Post-processing and Pipeline Continuity

1. **NMS off/on parity**
   - Run once with `apply_nms=False`, once with `apply_nms=True`
   - Confirm both execute; compare box count/shape validity
2. **Full pipeline continuity**
   - Call `run_full_pipeline(image_source=..., text_prompt='building, road')`
   - Confirm detection -> outlier filtering -> segmentation completes
3. **Output artifact validation**
   - Confirm expected output images/files are generated in output run directory

### D) Optional Backend Matrix

1. **Transformers backend**
   - Validate baseline run in the primary supported environment
2. **vLLM backend (optional)**
   - Validate only where vLLM is supported by platform/runtime

### E) Failure-Mode Checks

1. **Missing category prompt**
   - Use empty/invalid prompt and confirm clear actionable error
2. **Missing runtime dependency**
   - Ensure dependency error messages are explicit (for example `qwen-vl-utils`)
3. **Device/backend mismatch**
   - Validate error surface is understandable when runtime cannot honor requested setup

---

## Verification Notes (Current Read-Through)

After reviewing upstream-like `Rex-Omni/rex_omni/*` and embedded `langrs/rex_omni/*` plus adapter integration:

- Core detection call flow is consistent with migration intent
- Vendored implementation preserves essential Rex prompt/generation/parsing behavior
- Adapter integration into LangRS is architecture-aligned and pipeline-compatible
- Remaining differences are mostly adapter policy/defaults, not fundamental integration blockers

---

## Implementation Kickoff Plan

When we begin implementation work, proceed in this order:

1. **Lock acceptance criteria**
   - "Rex works inside LangRS" + "all LangRS pipeline features remain functional"
2. **Add/expand tests first**
   - Detection adapter tests (prompt parsing, output box normalization, error paths)
   - Pipeline integration tests (tiling modes, NMS toggle, full pipeline continuity)
3. **Stabilize defaults**
   - Make generation/config defaults explicit and documented for LangRS users
4. **Improve diagnostics**
   - Ensure dependency/backend/device errors are clear and actionable
5. **Run matrix and sign off**
   - Execute companion test plan and record pass/fail with notes

Implementation readiness: **Ready to start**.

---

## Recommended Direction

Given the current objective, the integration is in a good state:

- Keep Rex-Omni vendored in `langrs/rex_omni`
- Continue using adapter boundary in `langrs/models/detection/rex_omni.py`
- Focus future work on stability, defaults, and tests around LangRS workflows instead of strict upstream lockstep

If needed later, we can add a dedicated "Rex integration test matrix" doc and CI subset specifically for:

- backend selection
- category parsing contract
- tiling/coordinate correctness
- dependency failure messages

