# Segment any Text

Rust port of [Segment any Text (SaT)](https://arxiv.org/abs/2406.16678) model inference code (original code in [wtpsplit](https://github.com/segment-any-text/wtpsplit))

## Example

```bash
# --release tag is optional
cargo run --release --example split-file -- -i examples/test.txt
```
## Current status

- Only implements ONNX CPU runtime
