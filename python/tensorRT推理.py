import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit


def post_process(outputs, input_data):
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    [height, width, _] = input_data.shape
    scale_x = width / 640
    scale_y = height / 640
    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]

        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    if len(result_boxes) != 0:
        for index in result_boxes:
            box = boxes[index]
            x = round(box[0] * scale_x)
            y = round(box[1] * scale_y)
            x_plus_w = round((box[0] + box[2]) * scale_x)
            y_plus_h = round((box[1] + box[3]) * scale_y)
            print(x, y, x_plus_w, y_plus_h)

            overlay = input_data.copy()
            output = input_data.copy()
            alpha = 0.2
            cv2.rectangle(overlay, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), thickness=-1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            cv2.rectangle(output, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 3)
        cv2.imwrite('result.jpg', output)


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        pass
        # if severity >= trt.ILogger.Severity.WARNING:
        #     print(f"[{severity}] {msg}")


logger = MyLogger()
engine_file = (r'../best.plan')
# builder = trt.Builder(logger)
#
# network = builder.create_network()
# parser = trt.OnnxParser(network, logger)
# success = parser.parse_from_file(r'F:\code\yolov11-master\python\runs\detect\train\weights\best.onnx')
# if not success:
#     print("Failed to parse the ONNX model.")
#     for idx in range(parser.num_errors):
#         print(parser.get_error(idx))
#
# config = builder.create_builder_config()
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# config.set_flag(trt.BuilderFlag.FP16)
# serialized_engine = builder.build_serialized_network(network, config)
#
#
# with open(engine_file, "wb") as f:
#     f.write(serialized_engine)

runtime = trt.Runtime(logger)
with open(engine_file, "rb") as f:
    model_data = f.read()
engine = runtime.deserialize_cuda_engine(model_data)

context = engine.create_execution_context()

input_data = cv2.imread(r"F:\dataset\mycat\images\train\DJI_20250112193303_0064_D_HZH.MP4_130.jpg")
model_input = cv2.dnn.blobFromImage(input_data, scalefactor=1 / 255, size=(640, 640), swapRB=True)
input_size = cuda.pagelocked_empty(tuple(context.get_tensor_shape(engine.get_tensor_name(0))), dtype=np.float32)

input_mem = cuda.mem_alloc(input_size.nbytes)
cuda.memcpy_htod(input_mem, model_input)

stream = cuda.Stream()

output_data = cuda.pagelocked_empty(tuple(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
output_mem = cuda.mem_alloc(output_data.nbytes)

bindings = [int(input_mem)] + [int(output_mem)]
for i in range(engine.num_io_tensors):
    context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

context.execute_async_v3(stream.handle)
stream.synchronize()

cuda.memcpy_dtoh_async(output_data, output_mem, stream)  # 从 GPU 拷贝输出数据到主机内存

post_process(output_data, input_data)
