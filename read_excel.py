#读取xlsx文件，按照条件筛选，另存为新的xlsx文件。

import pandas as pd
import xlrd
import openpyxl

input = xlrd.open_workbook('C:/Users/1/Desktop/jdwz.xlsx')
input.sheets()
input_data = input.sheet_by_index(0)
nrows = input_data.nrows
ncols = input_data.ncols
print(nrows, ncols)


def writeToExcel(file_path, new_list):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '吉林军队文职'
    for r in range(len(new_list)):
        for c in range(len(new_list[0])):
            ws.cell(r + 1, c + 1).value = new_list[r][c]
            # excel中的行和列是从1开始计数的，所以需要+1
    wb.save(file_path)  # 注意，写入后一定要保存
    print("成功写入文件: " + file_path + " !")

jilin = []
for i in range (nrows):
    cell = input_data.row_values(i)[-2]
    if i < 6 :
        jilin.append(input_data.row_values(i))
    if cell == '吉林长春':
        jilin.append(input_data.row_values(i))
print(len(jilin))

writeToExcel('C:/Users/1/Desktop/jl.xlsx',jilin)