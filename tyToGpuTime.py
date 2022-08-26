import torch
import time
import sys
import numpy as np
import xlsxwriter as xw

gpu = torch.device("cuda")
def gpuTimeTest():
    torch.cuda.init()
    x = np.random.random(size=(640, 6720))
    start = time.time()
    torch.from_numpy(x).to(gpu)
    torch.cuda.synchronize()
    gputime = ((time.time() - start)/10)*1000
    return gputime

if __name__ == "__main__":
    workbook = xw.Workbook('tyTimeListGPUTemp.xlsx')
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    for i in range(5000):
        gputime = gpuTimeTest()
        row = 'A' + str(i)
        insertdataA = [str(gputime)]
        worksheet1.write_row(row, insertdataA)
    workbook.close()