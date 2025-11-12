
import onnx
import onnx_tf.backend
import tensorflow as tf

ONNX_FILE_PATH = "ppo_cargame.onnx"
MODEL_PREFIX = "tf_model"

onnx_model = onnx.load(ONNX_FILE_PATH)
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

print('Converting ONNX to TF...')
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(MODEL_PREFIX)
