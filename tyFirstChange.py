import glob
import string
txt_list = glob.glob('tytraining1/label_2/*.txt')
def show_category(txt_list):
    category_list= []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ') # 去掉前后多余的字符并把其分开
                    category_list.append(labeldata[0]) # 只要第一个字段，即类别
        except IOError as ioerr:
            print('File error:'+str(ioerr))
    print(set(category_list)) # 输出集合
def merge(line):
    each_line=''
    for i in range(len(line)):
        if i!= (len(line)-1):
            each_line=each_line+line[i]+' '
        else:
            each_line=each_line+line[i] # 最后一条字段后面不加空格
    each_line=each_line+'\n'
    return (each_line)
print('before modify categories are:\n')
show_category(txt_list)
for item in txt_list:
    new_txt=[]
    try:
        with open(item, 'r') as r_tdf:
            for each_line in r_tdf:
                labeldata = each_line.strip().split(' ')
                if labeldata[0] in ['Truck','Van','Tram']: # 合并汽车类
                    labeldata[0] = labeldata[0].replace(labeldata[0],'Car')
                if labeldata[0] == 'Person_sitting': # 合并行人类
                    labeldata[0] = labeldata[0].replace(labeldata[0],'Pedestrian')
                if labeldata[0] == 'DontCare': # 忽略Dontcare类
                    continue
                if labeldata[0] == 'Misc': # 忽略Misc类
                    continue
                new_txt.append(merge(labeldata)) # 重新写入新的txt文件
        with open(item,'w+') as w_tdf: # w+是打开原文件将内容删除，另写新内容进去
            for temp in new_txt:
                w_tdf.write(temp)
    except IOError as ioerr:
        print('File error:'+str(ioerr))
print('\nafter modify categories are:\n')
show_category(txt_list)
