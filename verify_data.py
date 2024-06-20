import numpy as np
import os 

# 文件路径
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path_features = os.path.join(script_directory, 'features.npy')
file_path_labels = os.path.join(script_directory, 'labels.npy')

# 读取 .npy 文件
data_features = np.load(file_path_features)
data_labels = np.load(file_path_labels)

# 打印数据
print("data_features",data_features)
print("data_labels",data_labels)
