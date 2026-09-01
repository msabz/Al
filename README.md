# RSNN Lab V2 Android

Android on-device lab for **Open-Growth RSNN V2**.

- QAT training with FP32 master state and fake INT8 forward quantization.
- Full integer-only INT8 inference path: INT8 weights/input/membrane/spikes with INT32 accumulators.
- No in-app synthetic equation generator. Training accepts only app-exported `algebra.linear_2d` files produced from pinned official Google DeepMind `mathematics_dataset` commit `427f45075f84b8b9774950196ad63867ca20ffb3`.
- Expert Model Control screen for architecture, training, growth, quantization and runtime parameters.
- Live Model Core screen for spikes, membrane activity, active/dormant/protected counts, growth/pruning events, INT8 saturation and gradient state.
- External model/weights import by content signature rather than extension.
- Google Drive Model Vault via Android Storage Access Framework: choose a Drive folder and back up/restore versioned checkpoints.
- No teacher/student distillation code.

Generate DeepMind files:

```bash
python scripts/export_deepmind_linear2d.py --split train --count 50000 --out deepmind_train.dmd
python scripts/export_deepmind_linear2d.py --split interpolate --count 5000 --out deepmind_validation.dmd
```
