import xlrd
import xlwt
import matplotlib.pyplot as plt

workBook = xlrd.open_workbook('timelist.xlsx')
sheet1_content1 = workBook.sheet_by_index(0)
x = [i for i in range(1,1000)]
y = sheet1_content1.col_values(0)

plt.plot(x, y)
plt.yticks([ i * 0.0005 for i in range(0,5)])
plt.show()

