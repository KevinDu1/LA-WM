import pickle
import numpy as np

# 读取 .pkl 文件
def load_pkl_to_numpy(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)  # 读取 pkl 文件
        
    split_data = [[] for _ in range(10)]
    
    for row in data:
        for i in range(len(row)):  # 仅遍历 row 内已有的列
            split_data[i].append(row[i])  # 按列存储数据
            
    return split_data

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # 递归展开嵌套列表
            flat_list.extend(flatten_list(item))
        elif isinstance(item, (int, float)) and item <= 5:  # 只保留数值
            # if item == 5:
            #     item = 3 #pedetrain成立
            flat_list.append(item)
    return flat_list
    
file_path = "iou_a.pkl"  # 替换为你的 .pkl 文件路径
split_data = load_pkl_to_numpy(file_path)

# 打印数据类型和形状
# print(split_data[0])2
mingzi = ['Road','Landmark','Vehicle','Pedestrian']
# print(len(flatten_list(split_data[0])))
for i in range(1,5):
    print(f'{mingzi[i-1]} is {np.mean(flatten_list(split_data[i]))}')#Road Landmark Vehicle Pedestrian
light = []
for i in range(5,8):
    light.extend(flatten_list(split_data[i]))
print(f'traffic light is {np.mean(light)}')