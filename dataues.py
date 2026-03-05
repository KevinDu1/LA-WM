import os
import shutil

# 定义源和目标文件夹
trainA_dir = "trainA"
trainB_dir = "traincala"

# 确保目标文件夹存在
os.makedirs(trainB_dir, exist_ok=True)

# 遍历 trainA 目录中的所有子文件夹
for folder in os.listdir(trainA_dir):
    folder_path = os.path.join(trainA_dir, folder)
    
    # 确保是文件夹
    if os.path.isdir(folder_path):
        image_folder = os.path.join(folder_path, "image")
        
        # 检查 image 文件夹是否存在
        if os.path.exists(image_folder):
            for idx, file in enumerate(os.listdir(image_folder)):
                file_path = os.path.join(image_folder, file)
                
                # 确保是文件
                if os.path.isfile(file_path):
                    # 生成新的文件名
                    file_extension = os.path.splitext(file)[1]  # 获取文件扩展名
                    new_filename = f"{folder}_{idx}{file_extension}"  # 新的命名规则
                    
                    # 目标文件路径
                    dest_path = os.path.join(trainB_dir, new_filename)
                    
                    # 复制文件
                    shutil.copy(file_path, dest_path)

print("所有图片已成功重命名并复制到 trainB 文件夹！")
