from onnxconverter_common import auto_mixed_precision
import onnx
import torch

model = onnx.load("outputs/deimv2_dinov3_s_coco/best_stg2.pth")
# Assuming x is the input to the model
# dynamic_axes = {
#         'images': {0: 'N', },
#         'orig_target_sizes': {0: 'N'}
#     }
feed_dict = {'input': torch.rand(32, 3, (640, 640))}
model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)
onnx.save(model_fp16, "path/to/model_fp16.onnx")
