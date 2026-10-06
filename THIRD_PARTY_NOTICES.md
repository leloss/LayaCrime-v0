# Third-party notices

This project depends on third-party software distributed under its own license terms. The dependency declarations in `pyproject.toml` are authoritative; transitive dependencies may add further licenses.

| Project | Use | License | Source |
| --- | --- | --- | --- |
| Laya | Model runtime and training | Apache-2.0 | https://github.com/NandhaKishorM/laya |
| llama.cpp | Local GGUF inference runtime | MIT | https://github.com/ggml-org/llama.cpp |
| FastAPI | HTTP application framework | MIT | https://github.com/fastapi/fastapi |
| OpenAI Python | Azure OpenAI client | Apache-2.0 | https://github.com/openai/openai-python |
| Pydantic | Request and response validation | MIT | https://github.com/pydantic/pydantic |
| python-dotenv | Local environment loading | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| Uvicorn | ASGI server | BSD-3-Clause | https://github.com/encode/uvicorn |
| NumPy | Training numerics | BSD-3-Clause | https://github.com/numpy/numpy |
| safetensors | Model serialization | Apache-2.0 | https://github.com/huggingface/safetensors |
| scikit-learn | Academic baseline implementations | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| spaCy | Tartu baseline preprocessing | MIT | https://github.com/explosion/spaCy |
| PyTorch | Model training | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Transformers | Model components | Apache-2.0 | https://github.com/huggingface/transformers |
| NewsMTSC / NewsSentiment | Academic target-sentiment comparison | MIT | https://github.com/fhamborg/NewsMTSC |
| University of Tartu adverse-media project | Academic article-level comparison | MIT | https://github.com/kristjanr/ut-ml-adverse-media |

The Laya and llama.cpp source checkouts created by the setup scripts retain their upstream license files. Neither checkout is distributed by this repository. Model weights are also not distributed; review their upstream terms before obtaining or redistributing them.

The two public dataset directories are release-scoped bundles whose manifests identify their license and `cleared-for-public-release` status. No other local dataset is distributed. The repository's Apache-2.0 license does not change the terms of third-party software, model checkpoints, or upstream datasets acquired separately by the reproduction commands.
