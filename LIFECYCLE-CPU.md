# Parent-scheduled bounded CPU qualification

The parent allocated a small synthetic test slot after source-only commit
`faa47f71a833a3a3f5fa28ce9d787a338b71be48`. The new check module is
`tests/lumen_studio/test_image_lifecycle.py`; it reuses the existing random
tiny Cosmos runtime/cache fixture from `test_pg09_training.py`. No pretrained
weights are loaded. The owned-loader test substitutes that tiny fixture for
the actual local model loader and injects a partial-load failure; it does not
qualify real native weight loading or its environment lock.

Six checks passed:

1. Actual routed bank updates and unchanged frozen base; exact disk split resume
   including metrics, full game state, EMA, private RNG and coverage; registered
   `open_session` resume/EMA inference load; inference cannot train.
2. Exact zero bypass, finite fractional responses, deterministic observation,
   unchanged global RNG/modes/full training packet, and checksum/EMA/cohort
   rejection before mutation with a live session remaining usable.
3. Render/reference/signed/out-of-range/non-CPU capability refusal before the
   model factory is invoked.
4. Borrowed runtime resident adapter/mix/hooks and checkpoint settings survive
   training, observation, close and repeated close.
5. Injected setup and update failures restore owned adapter/hooks/checkpoint
   settings and close the session as required.
6. Complete owned-factory construction/training/cleanup through a synthetic
   runtime loader, plus cleanup after a partial synthetic loading failure.

Core is installed 0.7.0. All 39 Python/JSON files were compared byte-for-byte
against local milestone `fd89e1c780ab5210cd1be98ec3b70977e6349379`, and the
released PG0.9.1 source fingerprint was verified as
`04624298310cd4b28d5bf687a1eac8951efd9439fb0310709ba5b9703208ceab`.
The exact milestone dependency-direction parser found only its existing frozen
`pg09_moss_campaign -> moss_sliders.lora.Slider` exception; no new edge.

The product's small pure-Python wheel is built using existing setuptools,
extracted into an isolated task-local directory and placed on the isolated
interpreter's path. No dependency installation occurs and the existing
environment is unchanged. Tests import product code from that extracted wheel,
resolve its installed backend entrypoint, and run from `/tmp` with no core source
checkout on the import path. This establishes wheel contents/registration and
bounded behavior, not clean dependency resolution or native environment parity.

The resource monitor launches one subprocess tree at a time, samples resident
memory and host `MemAvailable` every 0.1 seconds, and kills the job above 4 GiB
RSS, below 16 GiB host availability, or after 180 seconds. Final test workloads
use one Torch/OMP/MKL/OpenBLAS thread, HF/Transformers offline mode, empty CUDA
visibility, and assert CUDA remains uninitialized. Peak RSS stayed below 0.9 GiB;
host available memory stayed above 44 GiB. Exact per-run numbers and commands
are in the sibling task `../evidence/*.json` resource receipts.

The initial run passed the behavior assertions but failed all six final thread
assertions: historical `tests/lumen_studio/conftest.py` unconditionally sets
two Torch threads during import. The new module now overrides this with an
autouse fixture before any fixture/model/update work. The original failed
log/JUnit/resource receipts remain as `conformance-initial-thread-setting.*`.
No product workaround or core change was needed.

Executed runner forms, with absolute task-root paths in each saved receipt:

```text
python3.14 evidence/run_guarded.py evidence/dependencies <existing-env>/bin/python -I <task>/evidence/dependencies.py
python3.14 evidence/run_guarded.py evidence/conformance <existing-env>/bin/python -I <task>/evidence/run_conformance.py
python3.14 evidence/run_guarded.py evidence/product-build <existing-env>/bin/python -I <task>/evidence/build_product.py
python3.14 evidence/run_guarded.py evidence/wheel-conformance <existing-env>/bin/python -I <task>/evidence/run_wheel_conformance.py
python3.14 evidence/run_guarded.py evidence/core-source-and-direction <existing-env>/bin/python -I <task>/evidence/check_core_guard.py
```

The core-native admission callback gap is confirmed by source and the preserved
non-CPU refusal; no GPU execution was attempted. These bounded checks found no
new core defect. Compact inference, real native loading/sampler/decode parity,
HNDL matched-network parity, full memory/quality/convergence and shared clean-main
integration/publication remain separate work. Existing PR5 source files and
historical pins remain untouched. No parent-thread message was sent in this slot.
