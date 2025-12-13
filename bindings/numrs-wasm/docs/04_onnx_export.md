# 04. ONNX Export (WASM)

## Export to JSON
WASM binding exports a JSON representation of the ONNX proto.

```javascript
import { OnnxModelWrapper } from 'numrs-wasm';

const dummy = new Tensor(...);
const json = OnnxModelWrapper.export_model_to_json(model, dummy);

// Save 'json' string to file or send to server
```
