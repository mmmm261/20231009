import torch
import onnxruntime
#from dd_1 import VGG16


def run():
    weights = r'F:\temp_data\test_vgg16.pt'
    model = torch.load(weights, map_location='cpu')
    model.eval()
    im = torch.zeros([1, 3, 224, 224])
    onnx_path = r'F:\temp_data\test_vgg16.onnx'

    torch.onnx.export(model, im, onnx_path,verbose=False,opset_version = 12,do_constant_folding = False,input_names=['images'],output_names=['output'],
                    dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                    'output': {0: 'batch', 1: 'classes'}})  # shape(1,25200,85)

    onnx_model = onnxruntime.InferenceSession(onnx_path)
    print(onnx_model.get_inputs()[0].name)
    inputs = {onnx_model.get_inputs()[0].name: im.cpu().numpy()}
    outs = onnx_model.run(None, inputs)
    print(outs[0])


# -*-coding: utf-8 -*-

import os, sys

sys.path.append(os.getcwd())
import onnxruntime
import onnx


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores, boxes

def dd_0(onnx_path):
    model = onnx.load(onnx_path)
    #print(model.graph)
    for i, j in enumerate(model.graph.node):
        print(i)
        print(j)
        if i == 3 :
            break

    inputs = model.graph.input  #
    print(inputs)
    #inputs[0].type.tensor_type.shape.dim[0].dim_value = 1
    #onnx.save(model, 'YOLOv5x_change_shape.onnx')

if __name__ == '__main__':
    #run()
    onnx_weight_path = r'C:\Users\1\Desktop\tmp\test_StNet_onnx\STNet.onnx'
    dd_0(onnx_weight_path)
    #print(mol.get_input_name())
