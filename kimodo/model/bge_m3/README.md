# BGE-M3

This package mirrors the existing `llm2vec` package structure:

- `bge_m3.py` owns the tokenizer/model and dense encoding operation.
- `bge_m3_wrapper.py` adapts it to Kimodo's text-encoder interface.

`BGEM3Encoder` returns one 1024-dimensional conditioning token per input text
and records model provenance for training and evaluation.

The stable public import is:

```python
from kimodo.model import BGEM3Encoder
```
