# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:45:16 2018

@author: Administrator
"""

from osgeo import gdal
import os
import pickle
import numpy as np
import cv2

def readTif(fileName):

	driver = gdal.GetDriverByName('GTiff')
	driver.Register()
 
	dataset = gdal.Open(fileName)
	if dataset == None:
		print(fileName+ "掩膜失败，文件无法打开")

	im_width = dataset.RasterXSize #栅格矩阵的列数
	im_height = dataset.RasterYSize #栅格矩阵的行数

	im_bands = dataset.RasterCount #波段数
	im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
	im_proj = dataset.GetProjection()#获取投影信息


	if im_bands == 1:
		band = dataset.GetRasterBand(1)
		im_data = band.ReadAsArray(0,0,im_width,im_height) #获取数据
		cdata = im_data.astype(np.uint8)
		print(cdata.shape)
		# merge_img = cv2.merge([cdata,cdata,cdata])
		# cv2.imwrite('D:\\', merge_img)

	if im_bands == 3:
		for i in range(im_bands):
			band = dataset.GetRasterBand(i + 1)
			band_data = band.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)
			print(band_data[2000:2100, 2000:2100])

	if im_bands == 4:
		img = np.zeros((im_height, im_width, 4), dtype=np.uint16)

		for i in range(im_bands):
			band = dataset.GetRasterBand(i + 1)
			band_data = band.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)
			print(band_data[2000:2100, 2000:2100])
			img[:, :, i] = band_data
			img[:, :, i] = cv2.convertScaleAbs(band_data, alpha=(255.0/65535.0))
		print(img.shape)

		#rgb_img = img[:, :, :3]
		#bgr_img = rgb_img[:, :, ::-1]
		#cv2.imwrite(r'C:\Users\1\Desktop\out_2.png', bgr_img)

if __name__ == '__main__':
	file_path = r'E:\pycode\datasets\cgwx_20220920\JL1KF01C_200101850_001_L5D_PSH_204029\JL1KF01C_200101850_001_L5D_PSH.tif'
	#r'E:\pycode\datasets\cgwx_20220920\JL1GF02B_PMS2_20220711092631_200092286_102_0016_001_L3C_PSH_204043\JL1GF02B_PMS2_20220711092631_200092286_102_0016_001_L3C_PSH.tif'
	readTif(file_path)