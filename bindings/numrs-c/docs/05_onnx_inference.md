# 05. ONNX Inference (C)

> **Note**: Loading ONNX models directly via C ABI is under development.

The standard pattern is to define the architecture natively using `NumRsSequential` which maps 1:1 to the graph you might otherwise load.
