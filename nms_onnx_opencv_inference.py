#import cv2
import os

import numpy as np
import torch
import onnx
import onnxruntime
import onnx.helper as helper
from onnx import TensorProto
import pickle
import PIL
from PIL import Image
import io
import glob
import time

def nms_onnx_module():
    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    # Create one input (ValueInfoProto)
    boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6, 4])  # 输入为[]时，代表输入的是0D的张量，也就是标量
    # Create one output (ValueInfoProto)
    scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 6])
    max_output_boxes_per_class = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [2])
    iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.7])
    score_threshold = helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.0])
    # Create one output (ValueInfoProto)
    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [2, 3])
    # Create a node (NodeProto) - This is based on Pad-11
    node_def = helper.make_node(
        'NonMaxSuppression',  # node name
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],  # inputs
        # inputs=['boxes', 'scores'], # inputs
        outputs=['selected_indices'],
        # outputs
        center_point_box=0
    )
    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        'test-model',
        [boxes, scores],
        [selected_indices],
        [max_output_boxes_per_class, iou_threshold, score_threshold]
    )
    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name='zyx')
    model_def.opset_import[0].version = 12
    try:
        onnx.checker.check_model(model_def)
    except onnx.checker.ValidationError as e:
        print("model is invalid: %s" % (e))
    else:
        onnx.save(model_def, "nms_11_new.onnx")
        print('The model is:\n{}'.format(model_def))

def add_nms():
    detect_path = r'F:\temp_data\20220811_best_pretrained_20220830_best.onnx'
    onnx_model = onnx.load(detect_path)

    # no_used
    # boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1,36288,4])
    # scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1,1,36288])#[1,2,36288]

    # used
    max_output_boxes_per_class = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [20])
    iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.45])
    score_threshold = helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.25])

    selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [20, 3])

    node_def = helper.make_node(
        'NonMaxSuppression',  # node name
        # inputs=['box', 'score', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold']
        inputs=['847', '851', '854', '853', '852'],  # inputs
        outputs=['selected_indices'],#'selected_indices'
        center_point_box=0)

    onnx_model.graph.node.append(node_def)
    del onnx_model.graph.output[1]
    del onnx_model.graph.output[1]
    del onnx_model.graph.output[1]
    del onnx_model.graph.output[1]
    del onnx_model.graph.output[1]

    onnx_model.graph.output.append(selected_indices)

    model_def = onnx.helper.make_model(onnx_model.graph, producer_name='qdm')
    model_def.opset_import[0].version = 12
    onnx.save(model_def, r'F:\temp_data\modelXXX.onnx')

def onnx_inference(onnx_path, img_path):
    onnx_model = onnxruntime.InferenceSession(onnx_path)
    print(onnx_model.get_inputs()[0].name, onnx_model.get_inputs()[0].type)

    image = cv2.imread(img_path)/255.0
    img = image[:,:,::-1].transpose(2, 0, 1)[None]
    img_ = torch.from_numpy(img.astype("float32"))

    inputs = {onnx_model.get_inputs()[0].name: img_.cpu().numpy()}

    outs = onnx_model.run(None, inputs)
    print(outs)
    for i in outs:
        print(i.shape)
    print(len(outs))
    print(outs[0])

def opencv_inference(onnx_path, img_path):
    net = cv2.dnn.readNet(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print(net)

    #image = cv2.imread(img_path)
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 576), swapRB=True, crop=False)#INPUT_WIDTH, INPUT_HEIGHT

    #print(blob.type)
    #print(blob.dtype)
    #video = np.zeros

def infer_STNet():
    onnx_path = r'C:\Users\1\Desktop\tmp\test_StNet_onnx\STNet.onnx'
    time1 = time.time()
    onnx_model = onnxruntime.InferenceSession(onnx_path)
    time2 = time.time()
    print("cost_time:", time2 - time1)
    print(onnx_model.get_inputs()[0].name, onnx_model.get_inputs()[0].type, onnx_model.get_inputs()[0].shape)

    # pkl_path = r'C:\Users\1\Desktop\tmp\tmp_20230427\test_data\pos_000007.pkl'
    # pkl_path = r'C:\Users\1\Desktop\tmp\tmp_20230427\test_data\nonsmoke2021_1_13_2_51_11_t1.pkl'
    all_time = 0
    pkl_list = glob.glob(os.path.join(r'C:\Users\1\Desktop\tmp\test_StNet_onnx\test_data' + '\\*.pkl'))

    batch_input_array = np.zeros([5, 7, 15, 256, 256])

    for k, pkl_path in enumerate(pkl_list[:5]):
        print(pkl_path)
        f = open(pkl_path, 'rb')
        input_data = pickle.loads(f.read())
        f.close()

        list_data = input_data[2]
        per_list_data = list_data[0]

        # origal_array = np.zeros([17, 256, 256, 3])
        origal_array = np.zeros([17, 3, 256, 256])

        for i, io_data in enumerate(list_data):
            image = Image.open(io.BytesIO(io_data))
            array_ = (np.array(image) / 255.0)
            # print(array_.shape)
            array = (array_ - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            array = array.transpose(2, 0, 1)
            origal_array[i, :, :, :] = array
            if (i == 14):
                origal_array[-1, :, :, :] = array
                origal_array[-2, :, :, :] = array

        # block_set = np.zeros([7, 5, 256, 256, 3])
        block_set = np.zeros([7, 5, 3, 256, 256])

        for j in range(7):
            block_set[j, :, :, :, :] = origal_array[j * 2:j * 2 + 5, :, :, :]

        input_array = block_set.reshape([1, 7, 15, 256, 256])

        print(batch_input_array[k, :, :, :, :].shape)

        batch_input_array[k, :, :, :, :] = input_array[0, :, :, :, :]

        # input_array = np.full([2, 7, 15, 256, 256], 0.5, dtype="float32", )

    img_ = torch.from_numpy(batch_input_array.astype("float32"))

    inputs = {onnx_model.get_inputs()[0].name: img_.cpu().numpy()}

    start_time = time.time()
    outs = onnx_model.run(None, inputs)
    end_time = time.time()
    cost_time = end_time - start_time
    all_time += cost_time
    print(outs)
    # for i in outs:
    #     print(i.shape)
    # print(len(outs))
    # print(outs[0])
    print(all_time)
    # print(all_time / 20)



if __name__ == '__main__':

