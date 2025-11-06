import torch
from ann_model import Net
import numpy as np

dummy_input = torch.randn(1, 1, 200, 200)
model = Net()
model.load('best_model.pth')
torch.onnx.export(model, dummy_input, "best_model.onnx", opset_version=11)

# model_compile -m model.onnx -t apu -f Onnx --input_nodes "input_1" --input_shapes "1,1,200,200" --output_nodes "output"
