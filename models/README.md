# Model bundles

No model weights are distributed in this Git repository.

- [Laya](https://huggingface.co/convaiinnovations/laya) is downloaded as an ignored Hugging Face cache under `models/models--convaiinnovations--laya/`. Preserve its `refs/`, `snapshots/`, and content-addressed cache directories together.
- [LayaCrime.v0](https://huggingface.co/leloss/layacrime) is the public binary CSA checkpoint downloaded into the ignored `models/fine-tuned/layacrime-public/` directory. Additional user-defined exports remain part of the broader LayaCrime platform.

Run `scripts/setup.ps1` on Windows or `scripts/setup_environment.sh` on Linux to acquire both repositories. Set `HF_TOKEN` only when access to a configured repository requires authentication.

Place additional custom exports in `models/fine-tuned/<model-name>/`. Directory names are user-defined. A reloadable Laya export must contain:

```text
model.safetensors
rl_agent_config.json
encoder/config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

Exports created by this project also contain `training_report.json` and `data_manifest.json`. Keep both: they are not required for inference, but they preserve metrics, training history, dataset fingerprint, split provenance, and the information shown in the Fine-Tuning Console. The training prompt is stored in both `data_manifest.json` and `rl_agent_config.json`; selecting the checkpoint in Individual Test or Benchmark loads that prompt automatically. `checkpoint_latest/` is useful only while retaining intermediate training state and is omitted from Hugging Face uploads.

Set `LAYA_MODEL_BUNDLE` when the base Hugging Face cache lives elsewhere. Set `LAYA_FINE_TUNED_MODELS_DIR` when custom exports are not under `models/fine-tuned/`.