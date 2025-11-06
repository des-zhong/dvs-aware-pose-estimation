# DVS-aware Pose Estimation

Short README for this repository. Use this as a starting point to run, train, and deploy the models included in the workspace.

## Repository layout (key files)
- [data_loader_slice.py](data_loader_slice.py) — data loader utilities.
- [snn_model_cnn.py](snn_model_cnn.py) — spiking neural network model definition.
- [train_bp_cnn.py](train_bp_cnn.py) — training script (backprop).
- [GPU_inference.py](GPU_inference.py) — GPU inference helper.
- [wrap_load_save.py](wrap_load_save.py) — helper wrappers for load/save operations.
- [snn_tutorial.md](snn_tutorial.md) — tutorial and helper functions such as [`snn_tutorial.get_lr_list`](snn_tutorial.md) and [`snn_tutorial.spike_net_forward`](snn_tutorial.md).
- [ann_model/](ann_model/) — saved ANN models and conversion tools:
  - [ann_model/ann_model.py](ann_model/ann_model.py)
  - [ann_model/ann_inference.py](ann_model/ann_inference.py)
  - [ann_model/best_model.pth](ann_model/best_model.pth)
  - [ann_model/best_model.onnx](ann_model/best_model.onnx)
- [custom_op_in_pytorch/](custom_op_in_pytorch/) —Lynxi api

## Requirements
Install dependencies listed in [requirements.txt](requirements.txt):
```sh
git clone git@github.com:des-zhong/dvs-aware-pose-estimation.git
cd dvs-aware-pose-estimation
```
Packages:

```sh
pip install -r requirements.txt
```

## Quickstart

1. Prepare dataset and update dataset path in [data_loader_slice.py](data_loader_slice.py).
2. Train an SCNN:
```sh
python train_bp_cnn.py
```
3. Run GPU inference (checkpoint in exp/decay01):
```sh
python GPU_inference.py
```
4. Compile to Lynxi KA200 fomat (checkpoint in exp/mapping_output)

   First convert pth file to onnx:

   ```
   python convert2onnx.py
   ```

   Then compile onnx file to Lynxi format

   ```sh
   model_compile -m model.onnx -t apu -f Onnx --input_nodes "input_1" --input_shapes "1,1,200,200" --output_nodes "output"
   ```
## Custom ops
Build and register custom operators in [custom_op_in_pytorch/](custom_op_in_pytorch/). See:
- [`custom_op_myLif.reset_with_decay`](custom_op_in_pytorch/custom_op_myLif.cpp)
- [`custom_op_myLif.cmp_and_fire`](custom_op_in_pytorch/custom_op_myLif.cpp)

