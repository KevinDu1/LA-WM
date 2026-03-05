import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
import os

# 加载 pkl 文件
with open("all_results_0410_duibi_dongzhu.pkl", "rb") as f:
    loaded_results = pickle.load(f)

# 如果 loaded_results 不是 defaultdict，则转换为 defaultdict(list)
if not isinstance(loaded_results, defaultdict):
    loaded_results = defaultdict(list, loaded_results)

# print(loaded_results)
# # 遍历字典，将列表转换为 numpy 数组
numpy_results = {key: np.array(value) for key, value in loaded_results.items()}

# 打印 numpy 数组的形状（用于检查）
for key, array in numpy_results.items():
    print(f"{key}: shape {array.shape}")
dataframes = {}
for key, value in loaded_results.items():
    array = np.array(value)
    df = pd.DataFrame(array)
    dataframes[key] = df

# print(dataframes)
# output_dir = "csv_outputs"
# os.makedirs(output_dir, exist_ok=True)

# # 将每个 DataFrame 保存为 CSV 文件
# for key, df in dataframes.items():
#     csv_path = os.path.join(output_dir, f"{key}.csv")
#     df.to_csv(csv_path, index=False)

# print("所有 DataFrame 已保存为 CSV 文件。")

# 计算每项的平均值
mean_values = {}
for key, value in loaded_results.items():
    array = np.array(value)
    df = pd.DataFrame(array)
    mean_values[key] = df.mean().values  # 保存为 ndarray

# 转换为 DataFrame 展示
mean_df = pd.DataFrame.from_dict(mean_values, orient="index")
mean_df.columns = [f"mean_col_{i}" for i in range(mean_df.shape[1])]

print(mean_df)

# 计算碰撞率
# prob_results = {}

# for key, df in dataframes.items():
#     if 'obj' in key:
#         # 计算每列中 >=1 的比例（布尔矩阵求均值）
#         prob = (df >= 1).mean()
#         prob_results[key] = prob

# # 整理成汇总 DataFrame
# prob_df = pd.DataFrame(prob_results).T  # 行为key，列为原始列名

# # 可视化或进一步使用
# print(prob_df)