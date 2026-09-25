# Local English checkpoint

This directory is supported as an optional single-English-model override through `LAYA_MODEL_PATH`. The default project configuration instead resolves all three checkpoints from the copied `models/models--convaiinnovations--laya` Hugging Face cache bundle. Runtime downloads are disabled.

Required layout:

```text
models/english/
|-- rl_agent_config.json
|-- model.safetensors
`-- tokenizer/
    |-- tokenizer.json
    |-- tokenizer_config.json
    `-- other tokenizer files supplied with the checkpoint
```

The checkpoint is published separately on Hugging Face at `convaiinnovations/laya`. Download the English root checkpoint while preserving the directory structure above. Use `LAYA_MODEL_PATH` when storing the bundle elsewhere.