import glob
import xml.etree.ElementTree as ET
# xml里面的类名
class_names = ['Car', 'Cyclist', 'Pedestrian']
# xml文件路径
path = './Annotations/'
# 转换一个xml文件为txt
def single_xml_to_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 保存的txt文件路径
    txt_file = xml_file.split('.')[0]+'.'+xml_file.split('.')[1]+'.txt'
    with open(txt_file, 'w') as txt_file:
        for member in root.findall('object'):
            #filename = root.find('filename').text
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            class_name = member[0].text
            # 类名对应的index
            class_num = class_names.index(class_name)

            box_x_min = int(member[1][0].text) # 左上角横坐标
            box_y_min = int(member[1][1].text) # 左上角纵坐标
            box_x_max = int(member[1][2].text) # 右下角横坐标
            box_y_max = int(member[1][3].text) # 右下角纵坐标
            print(box_x_max,box_x_min,box_y_max,box_y_min)
            # 转成相对位置和宽高
            x_center = float(box_x_min + box_x_max) / (2 * picture_width)
            y_center = float(box_y_min + box_y_max) / (2 * picture_height)
            width = float(box_x_max - box_x_min) /  picture_width
            height = float(box_y_max - box_y_min) /  picture_height
            print(class_num, x_center, y_center, width, height)
            txt_file.write(str(class_num) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
# 转换文件夹下的所有xml文件为txt
def dir_xml_to_txt(path):
    for xml_file in glob.glob(path + '*.xml'):
        single_xml_to_txt(xml_file)
dir_xml_to_txt(path)
