from osgeo import gdal
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import time
import random
from torch.nn import Linear, ReLU, Sequential
import cv2
from matplotlib import pyplot as plt

random.seed(52)
class test_Dataset(data.Dataset):
    def __init__(self, tif_path, train_mode):
        self.input_list = []
        self.label_list = []

        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        NRGB_img = gdal.Open(tif_path)

        im_width = NRGB_img.RasterXSize
        im_height = NRGB_img.RasterYSize
        N_band = NRGB_img.GetRasterBand(1)
        N_band_data = N_band.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
        R_band = NRGB_img.GetRasterBand(2)
        R_band_data = R_band.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
        #print(np.max(N_band_data), np.min(N_band_data))
        ndvi = (N_band_data - R_band_data + 0.0001) / (N_band_data + R_band_data + 0.0001).astype(np.float32)
        #print(np.max(ndvi), np.min(ndvi))
        R_band_data = R_band_data / 255.0
        N_band_data = N_band_data / 255.0

        idxs = [i for i in range(im_width)]
        #random.shuffle(idxs)

        if train_mode:
            for i in range(im_width):
                x1 = N_band_data[:, idxs[i]].reshape(-1,1)
                x2 = R_band_data[:, idxs[i]].reshape(-1,1)
                x = np.concatenate((x1, x2), axis = 1)
                y = ndvi[:, idxs[i]]
                self.input_list.append(x)
                self.label_list.append(y)
        else:
            for i in range(800):
                x1 = N_band_data[:, idxs[6000+i]].reshape(-1,1)
                x2 = R_band_data[:, idxs[6000+i]].reshape(-1,1)
                x = np.concatenate((x1, x2), axis = 1)
                y = ndvi[:, idxs[6000+i]]
                self.input_list.append(x)
                self.label_list.append(y)

    def __getitem__(self, item):
        input = self.input_list[item]
        label = self.label_list[item]
        input_tensor = torch.from_numpy(input).to(torch.float32)
        label_tensor = torch.from_numpy(label).to(torch.float32)
        return(input_tensor, label_tensor)

    def __len__(self):
        return len(self.label_list)

    def __str__(self):
        return("datasetLength is {}".format(len(self.label_list)))

class regression_dataset(data.Dataset):
    def __init__(self):
        self.x_list = []
        for i in range(256):
            for j in range(256):
                ndvi = (i - j + 0.001) / (i + j + 0.001)
                x = np.array([i/255, j/255, ndvi], dtype = np.float32)
                self.x_list.append(x)
        random.shuffle(self.x_list)
    def __getitem__(self, item):
        x_idx = self.x_list[item][:2]
        ndvi = np.array(self.x_list[item][-1])
        x_idx_tensor = torch.from_numpy(x_idx).to(torch.float32)
        ndvi_tensor = torch.from_numpy(ndvi).to(torch.float32)
        return x_idx_tensor, ndvi_tensor
    def __len__(self):
        return len(self.x_list)
    def get_demo(self):
        x_idx = self.x_list[0][:2]
        ndvi = np.array(self.x_list[0][-1])
        x_idx_tensor = torch.from_numpy(x_idx).to(torch.float32)
        ndvi_tensor = torch.from_numpy(ndvi).to(torch.float32)
        print(x_idx_tensor, ndvi_tensor)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.input_layer = Sequential(
            Linear(2, 8, bias=False),
            ReLU()
        )
        self.hidden_layer1 = Sequential(
            Linear(8, 16, bias=False),
            ReLU()
        )
        self.hidden_layer2 = Sequential(
            Linear(16, 16, bias=False),
            ReLU()
        )
        self.hidden_layer3 = Sequential(
            Linear(16, 8, bias=False),
            ReLU()
        )
        self.out_layer = Sequential(
            Linear(8, 1, bias=False),
        )
    def forward(self, x):
        x1 = self.input_layer(x)
        #print("x1.shape", x1.shape)
        x2 = self.hidden_layer1(x1)
        #print("x2.shape", x2.shape)
        x3 = self.hidden_layer2(x2)
        #print("x3.shape", x3.shape)
        x4 = self.hidden_layer3(x3)
        #print("x4.shape", x4.shape)
        y = self.out_layer(x1)
        #print("y range:",torch.max(y), torch.min(y))
        output = y.squeeze()
        return output


def train(train_dataset, model, optimizer_fn, loss_fn, lr_fn, epoch_nums):
    for epoch in range(epoch_nums):
        print("epoch_idx:", epoch)
        start_time = time.time()
        loss = 0
        for input, label in train_dataset:
            #print(input.shape)
            output = model(input)
            #print(output.shape,label.shape)
            #print("label range:", torch.max(label), torch.min(label))
            loss = loss_fn(output, label)
            #print(loss)
            optimizer_fn.zero_grad()
            loss.backward()
            optimizer_fn.step()
        lr_fn.step()
        print("train_loss:", loss)
        end_time = time.time()
        print("costTime:", end_time - start_time)
        # if epoch % 10 == 0:
        #     model.eval()
        #     for input_, label_ in test_dataset:
        #         #print(input_.shape)
        #         output_ = model(input_)
        #         #print(output_.shape, label_.shape)
        #         test_loss = torch.nn.MSELoss()(output_, label_)
        #     print("test_loss:", test_loss)
        #     model.train()

    torch.save(model.state_dict(), "dd_SGD_standart1.pth")

def run():
    tif_path = r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\tmp_ori_nrgb\GF2_PMS2__L1A0000607681-MSS2.tif'
    train_dataset = test_Dataset(tif_path, True)
    test_dataset = test_Dataset(tif_path, False)
    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = Model()
    epoch_nums = 20
    loss_fn = torch.nn.MSELoss()
    optimizer_fn = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    lr_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fn, T_max=10,
                                                                    eta_min=0.0005)  # ([0: T_max]= max_lr->0; [T_max: 2*T_max]= 0->max_lr;)
    train(train_dataloader, test_dataloader, model, optimizer_fn, loss_fn, lr_fn, epoch_nums)

def test():
    model = Model()
    model.load_state_dict(torch.load("dd_SGD_standart1.pth"))
    test_tif_path = r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\tmp_ori_nrgb\GF2_PMS2__L1A0000607681-MSS2.tif'
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    NRGB_img = gdal.Open(test_tif_path)

    im_width = NRGB_img.RasterXSize
    im_height = NRGB_img.RasterYSize
    N_band = NRGB_img.GetRasterBand(1)
    N_band_data = N_band.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
    R_band = NRGB_img.GetRasterBand(2)
    R_band_data = R_band.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
    R_band_data = R_band_data / 255.0
    N_band_data = N_band_data / 255.0
    output = np.zeros([im_height, im_width])
    for i in range(im_width):
        x1 = N_band_data[:, i].reshape(-1, 1)
        x2 = R_band_data[:, i].reshape(-1, 1)
        x = np.concatenate((x1, x2), axis=1)
        x_tensor = torch.from_numpy(x).to(torch.float32)
        #print(x_tensor.shape)
        y = model(x_tensor)
        #print(y.shape)
        #print(output[i,:].shape)
        output[:, i] = y.detach().numpy()
    print(np.max(output), np.min(output))

    # output[output > 1] = 1
    # output[output < -1] = -1
    output = (output + 1) / 2
    output[output > 1] = 1
    output[output < 0] = 0
    print(np.max(output), np.min(output))
    output = (output * 255.0).astype(np.uint8)
    print(np.max(output), np.min(output))
    equ = cv2.equalizeHist(output)
    cv2.imwrite(r"E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\GF2_PMS2__L1A0000607681-MSS2_result4.png", equ)

def run_regression():
    train_dataset = regression_dataset()
    train_dataset.get_demo()
    train_dataloader = data.DataLoader(train_dataset, batch_size=1280, shuffle=True)
    model = Model()
    epoch_nums = 50
    loss_fn = torch.nn.MSELoss()
    optimizer_fn = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer_fn = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fn, T_max=20,
                                                                    eta_min=0.000001)  # ([0: T_max]= max_lr->0; [T_max: 2*T_max]= 0->max_lr;)
    train(train_dataloader, model, optimizer_fn, loss_fn, lr_fn, epoch_nums)

def draw_func():
    data = np.linspace(1, 255, 10)
    x, y = np.meshgrid(data, data)
    z = (y - x) / (y + x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    plt.title("function")
    plt.show()
if __name__ == '__main__':
    test()
    #draw_func()

