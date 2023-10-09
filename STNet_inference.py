import onnxruntime
import numpy as np
import pickle
from PIL import Image
import io
import glob
import time
import os
import cv2

def infer_opencv_cpu():
    onnx_path = r'STNet.onnx'
    time1 = time.time()
    onnx_model = cv2.dnn.readNet(onnx_path)
    onnx_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    onnx_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    time2 = time.time()
    print("cost_time:", time2 - time1)

    pkl_list = glob.glob(os.path.join(r'C:\Users\1\Desktop\test_data' + '\\*.pkl'))

    print(pkl_list)

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

    img_ = batch_input_array.astype("float32")

    onnx_model.setInput(img_)

    for i in range(10):
        outputs = onnx_model.forward()
        print("==========")

    print(outputs.shape)

    print(outputs)


def infer_STNet_gpu():
    onnx_path = 'STNet.onnx'
    time1 = time.time()
    onnx_model = onnxruntime.InferenceSession(onnx_path, providers=[
        'CPUExecutionProvider'])  # ['TensorrtExecutionProvider', 'CUDAExecutionProvider',CPUExecutionProvider]
    time2 = time.time()
    print("cost_time:", time2 - time1)
    print(onnx_model.get_inputs()[0].name, onnx_model.get_inputs()[0].type, onnx_model.get_inputs()[0].shape)

    all_time = 0
    pkl_list = glob.glob(os.path.join('test_data\pkl' + '\\*.pkl'))

    for k, pkl_path in enumerate(pkl_list):
        print(pkl_path)
        f = open(pkl_path, 'rb')
        input_data = pickle.loads(f.read())
        f.close()

        list_data = input_data[2]
        per_list_data = list_data[0]

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

        block_set = np.zeros([7, 5, 3, 256, 256])

        for j in range(7):
            block_set[j, :, :, :, :] = origal_array[j * 2:j * 2 + 5, :, :, :]

        input_array = block_set.reshape([1, 7, 15, 256, 256])

        inputs = {onnx_model.get_inputs()[0].name: input_array.astype("float32")}

        start_time = time.time()

        outs = onnx_model.run(None, inputs)

        end_time = time.time()

        cost_time = end_time - start_time

        all_time += cost_time

        print(outs)

        print(cost_time)

    print("==========")
    print(all_time)


if __name__ == "__main__":
    infer_opencv_cpu()






