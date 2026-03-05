import numpy as np
import torch

from scipy.ndimage import binary_dilation

from scipy.spatial.distance import cdist
from scipy.stats import zscore
import scipy.ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian
import cv2


def iou_sum(render_dict, imagine, input_data, upsample_bev_factor=2):
    iou_BEV_now = []
    for i in range(input_data['image'].shape[0]):
        birdview_label = input_data['birdview_label'][i, 5]
        birdview_label = torch.rot90(birdview_label, k=1, dims=[1, 2])

        # Check if the bev predictions are in the outputs
        if 'bev_segmentation_1' in render_dict:
            bev_prediction = render_dict['bev_segmentation_1'][i, -1]
            # Rotate prediction
            bev_prediction = torch.rot90(bev_prediction, k=1, dims=[1, 2])
            bev_prediction = torch.argmax(bev_prediction, dim=0, keepdim=True)

            iou_BEV_now.append(iou(birdview_label, bev_prediction))

    iou_BEV_imagine = []
    for i in range(input_data['image'].shape[0]):
        iou_BEV_imagine_log = []
        for j in range(6):
            birdview_label = input_data['birdview_label'][i, 6+j]
            birdview_label = torch.rot90(birdview_label, k=1, dims=[1, 2])

            # Check if the bev predictions are in the outputs
            if 'bev_segmentation_1' in render_dict:
                bev_prediction = imagine['bev_segmentation_1'][i, j]
                # Rotate prediction
                bev_prediction = torch.rot90(bev_prediction, k=1, dims=[1, 2])
                bev_prediction = torch.argmax(bev_prediction, dim=0, keepdim=True)

                iou_BEV_imagine_log.append(iou(birdview_label, bev_prediction))
        iou_BEV_imagine.append(iou_BEV_imagine_log)
    return iou_BEV_now, iou_BEV_imagine




BIRDVIEW_COLOURS = np.array([[255, 255, 255],          # Background
                             [225, 225, 225],       # Road
                             [160, 160, 160],      # Lane marking
                             [0, 83, 138],        # Vehicle
                             [127, 255, 212],      # Pedestrian
                             [50, 205, 50],        # Green light
                             [255, 215, 0],      # Yellow light
                             [220, 20, 60],        # Red light and stop sign
                             [0, 110, 160],
                             [60, 185, 50],  # [147, 255, 232],
                             [220, 20, 60]
                             ], dtype=np.uint8)

def squeeze_bev(pred):
    priority_list = [0, 1, 2, 5, 6, 7, 4, 3]
    # 初始化输出为全 0
    compressed = np.zeros_like(pred[0], dtype=np.uint8)

    # 按照优先级逐层更新压缩结果
    for channel in priority_list:
        compressed = np.where(pred[channel] >= 0.2, channel, compressed)

    compressed = BIRDVIEW_COLOURS[compressed]
    np.transpose(compressed, (1, 2, 0))
    return compressed

def squeeze_bev_label(pred):
    priority_list = [0, 1, 2, 5, 6, 7, 4, 3]
    # 初始化输出为全 0
    compressed = np.zeros_like(pred[0], dtype=np.uint8)

    # 按照优先级逐层更新压缩结果
    for channel in priority_list:
        compressed = np.where(pred[channel] >= 0.2, channel, compressed)

    return compressed

def find_peaks_and_valleys_centers(matrix, sigma=1, threshold_rel=0.5, min_distance=1):
    # Apply Gaussian smoothing
    smoothed_matrix = gaussian(matrix, sigma=sigma)

    # Find peaks
    peaks = peak_local_max(smoothed_matrix, threshold_rel=threshold_rel, min_distance=min_distance)

    return peaks


def calculate_min_distances(centers1, centers2):
    # 计算两个中心点列表之间的距离矩阵
    distances = cdist(centers1, centers2, metric='euclidean')

    # 找到第一个列表中每个点与第二个列表中点的最小距离
    min_distances = distances.min(axis=1)

    return min_distances


def remove_outliers(distances, method='zscore', threshold=3):
    if method == 'zscore':
        z_scores = zscore(distances)
        filtered_distances = distances[np.abs(z_scores) < threshold]
    elif method == 'iqr':
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        filtered_distances = distances[(distances >= Q1 - 1.5 * IQR) & (distances <= Q3 + 1.5 * IQR)]
    else:
        raise ValueError("Unsupported method. Use 'zscore' or 'iqr'.")

    return filtered_distances


def calculate_min_distances(centers1, centers2):
    # 计算两个中心点列表之间的距离矩阵
    distances = cdist(centers1, centers2, metric='euclidean')

    # 找到第一个列表中每个点与第二个列表中点的最小距离
    min_distances = distances.min(axis=1)

    return min_distances


def get_neighbors(x, y, matrix):
    neighbors = []
    rows, cols = len(matrix), len(matrix[0])
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and matrix[nx][ny] == 1:
            neighbors.append((nx, ny))
    return neighbors


def bfs(matrix, x, y, visited):
    queue = [(x, y)]
    visited.add((x, y))
    points = [(x, y)]

    while queue:
        cx, cy = queue.pop(0)
        for nx, ny in get_neighbors(cx, cy, matrix):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
                points.append((nx, ny))

    return points


def find_clusters(matrix):
    clusters = []
    visited = set()

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1 and (i, j) not in visited:
                cluster = bfs(matrix, i, j, visited)
                clusters.append(cluster)

    return clusters


def calculate_center(cluster):
    x_coords = [x for x, y in cluster]
    y_coords = [y for x, y in cluster]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)


def find_cluster_centers(matrix, i):
    clusters = find_clusters(matrix)
    if i == 3:
        clusters_new = []
        for clu in clusters:
            if len(clu) < 50:
                continue
            clusters_new.append(clu)
        centers = [calculate_center(cluster) for cluster in clusters_new]
    else:
        centers = [calculate_center(cluster) for cluster in clusters]
    return centers


def calculate_iou(matrix1, matrix2):
    """
    计算两个二值矩阵的交并比（IoU）

    参数:
    matrix1 -- 第一个二值矩阵
    matrix2 -- 第二个二值矩阵

    返回:
    iou -- 交并比（IoU）
    """

    # 确保两个矩阵大小相同
    if matrix1.shape != matrix2.shape:
        raise ValueError("输入的两个矩阵大小必须相同")

    # 计算交集
    intersection = np.logical_and(matrix1, matrix2)
    intersection_sum = np.sum(intersection) + 1

    # 计算并集
    union = np.logical_or(matrix1, matrix2)
    union_sum = np.sum(union) + 1

    # 计算IoU
    iou = intersection_sum / union_sum
    return iou


def expand_matrix(matrix, iterations=1):
    """
    扩展二值矩阵中1的部分

    参数:
    matrix -- 输入的二值矩阵
    iterations -- 膨胀的迭代次数，默认为1

    返回:
    expanded_matrix -- 扩展后的二值矩阵
    """

    # 使用binary_dilation进行膨胀操作
    expanded_matrix = binary_dilation(matrix, iterations=iterations)

    return expanded_matrix


def iou(tensor1, tensor2, iterations=1):
    # print(tensor2.shape)
    tensor1 = tensor1.view(1,192,192)
    tensor2 = tensor2.view(1,192,192)
    # matrix_3 8*192*192 0-1
    iou_bev = [0] * 10
    matrix_1 = tensor1.cpu().numpy() # 真值 1*192*192 1-8
    matrix_2 = tensor2.cpu().numpy() # mile 1*192*192 1-8
    matrix_1 = np.squeeze(matrix_1)  # 真值 192*192 1-8
    matrix_2 = np.squeeze(matrix_2)  # mile 192*192 1-8

    instance_1 = np.zeros(matrix_1.shape, dtype=bool)
    instance_1[(matrix_1 == 3) | (matrix_1 == 4)] = True  # 筛选出来行人和车辆
    # instance_display_1 = BIRDVIEW_COLOURS[matrix_1]

    instance_2 = np.zeros(matrix_2.shape, dtype=bool)
    instance_2[(matrix_2 == 3) | (matrix_2 == 4)] = True  # 筛选出来行人和车辆
    # instance_2 = BIRDVIEW_COLOURS[matrix_2]


    # instance_1_display = BIRDVIEW_COLOURS[matrix_1].astype(np.uint8)
    # cv2.imwrite('truth.png', instance_1_display)
    # instance_2_display = BIRDVIEW_COLOURS[matrix_2].astype(np.uint8)
    # cv2.imwrite('predict.png', instance_2_display)

    # instance_1 = instance_1[0]
    # 每帧图里有几个实例
    instance_label_1, _ = scipy.ndimage.label(instance_1[None].astype(np.int64))
    instance_label_1 = instance_label_1[0]  # 192*192 0-实例最大索引

    instance_label_2, _ = scipy.ndimage.label(instance_2[None].astype(np.int64))
    instance_label_2 = instance_label_2[0]  # 192*192 0-实例最大索引

    # 生成一个包含行和列索引的三维数组，表示每个位置的行号和列号
    xy = np.indices(instance_1.shape)
    x_1 = xy[0]
    y_1 = xy[1]


    # 初始化
    num_instance_truth = instance_label_1.max()
    num_instance_2 = instance_label_2.max()
    center_1 = []
    center_2 = []
    center_1_v = []
    center_1_p = []
    center_2_v = []
    center_2_p = []


    # 真值获取实例中点
    for num in range(1, num_instance_truth + 1):
        mask = (instance_label_1 == num)
        xc = x_1[mask].mean().round()
        yc = y_1[mask].mean().round()
        center_1.append((xc, yc))

    for center_i_1 in center_1:
        x = int(center_i_1[0])
        y = int(center_i_1[1])
        n_class = matrix_1[x][y]
        if n_class == 3:
            center_1_v.append((x, y))
        elif n_class == 4:
            center_1_p.append((x, y))


    # 两种方法寻找中心点

    # # mile自己可以根据自身的center预测实现
    # center_2_temp = find_peaks_and_valleys_centers(center)
    # for center_i in center_2_temp:
    #     n_class = matrix_2[center_i[0]][center_i[1]]
    #     if n_class == 3:
    #         center_2_v.append((center_i[0], center_i[1]))
    #     elif n_class == 4:
    #         center_2_p.append((center_i[0], center_i[1]))
    #     else:
    #         continue

    # mile 采用色块中心点
    for num in range(1, num_instance_2 + 1):
        mask_2 = (instance_label_2 == num)
        xc = x_1[mask_2].mean().round()
        yc = y_1[mask_2].mean().round()
        center_2.append((xc, yc))

    for center_i_2 in center_2:
        x = int(center_i_2[0])
        y = int(center_i_2[1])
        n_class = matrix_2[x][y]
        if n_class == 3:
            center_2_v.append((x, y))
        elif n_class == 4:
            center_2_p.append((x, y))

    # 判断标准
    # 都不为零 直接计算即可
    # 每检测出来按照距离为10
    # 多检测了按照5
    # 都没有为空
    if len(center_2_v) != 0 and len(center_1_v) != 0:
        min_distances_v = calculate_min_distances(center_2_v, center_1_v)  # 以center_2_v为基础遍历找距离最近的点，min_distances_v长度与enter_2_v相同
    elif len(center_2_v) == 0 and len(center_1_v) != 0:
        min_distances_v = np.array([5]*len(center_1_v))
    elif len(center_2_v) != 0 and len(center_1_v) == 0:
        min_distances_v = np.array([1]*len(center_2_v))
    elif len(center_2_v) == 0 and len(center_1_v) == 0:
        min_distances_v = np.array([])

    if len(center_2_p) != 0 and len(center_1_p) != 0:
        min_distances_p = calculate_min_distances(center_2_p, center_1_p)
    elif len(center_2_p) == 0 and len(center_1_p) != 0:
        min_distances_p = np.array([5] * len(center_1_p))
    elif len(center_2_p) != 0 and len(center_1_p) == 0:
        min_distances_p = np.array([1] * len(center_2_p))
    elif len(center_2_p) == 0 and len(center_1_p) == 0:
        min_distances_p = np.array([])

    # # 以前的判断模式
    # if len(center_3_p) != 0 and len(center_1_p) != 0:
    #     min_distances_3_p = calculate_min_distances(center_3_p, center_1_p)
    # else:
    #     min_distances_3_p = np.array([0.01])


    for i in range(8):
        temp_1 = np.zeros((192, 192))
        temp_2 = np.zeros((192, 192))
        indices_1 = np.where(matrix_1 == i)
        indices_2 = np.where(matrix_2 == i)
        temp_1[indices_1] = 1  # 192*192 0 1 筛选元素
        temp_2[indices_2] = 1  # 192*192 0 1 筛选元素
        # if i == 9:
        #     iou_bev.append(np.mean(min_distances_3_v))
        #     print('vehicle_LSS', np.mean(min_distances_3_v))
        #     continue
        # if i == 10:
        #     iou_bev.append(np.mean(min_distances_3_p))
        #     print('pedestrian_LSS', np.mean(min_distances_3_p))
        #     continue

        # 行人和车辆层结果
        if i == 3:
            temp_1 = expand_matrix(temp_1, iterations)
            temp_2 = expand_matrix(temp_2, iterations)
            iou_bev[i] = min_distances_v.tolist()
            iou_bev[8] = calculate_iou(temp_1, temp_2)
            # print('vehicle_mile', np.mean(min_distances_v),'vehicle_LSS', np.mean(min_distances_3_v),)
            continue
        if i == 4:
            temp_1 = expand_matrix(temp_1, iterations)
            temp_2 = expand_matrix(temp_2, iterations)
            iou_bev[i] = min_distances_p.tolist()
            iou_bev[9] = calculate_iou(temp_1, temp_2)
            # print('pedestrian', np.mean(min_distances_p),'pedestrian_LSS', np.mean(min_distances_3_p))
            continue

        # 红绿灯识别成功率
        if i == 5 or i == 6 or i == 7:
            yes_2 = 0

            # 判断标准
            # 都没有是1
            # 正值或者预测一方有一方没有是0
            # 都有正常计算
            if len(indices_1[0]) == 0 and len(indices_2[0]) != 0:
                per_2 = 0  # 比例
            elif len(indices_1[0]) == 0 and len(indices_2[0]) == 0:
                per_2 = 1
            elif len(indices_1[0]) != 0 and len(indices_2[0]) == 0:
                per_2 = 0
            else:
                # 寻找停止线
                centers_1 = find_cluster_centers(temp_1, i)
                centers_2 = find_cluster_centers(temp_2, i)
                min_distances_2 = calculate_min_distances(centers_2, centers_1)
                for dist in min_distances_2:
                    if dist < 70:
                        yes_2 += 1
                per_2 = yes_2 / len(min_distances_2)

            iou_bev[i] = (per_2)

            # if i == 5:
            #     print('green', [per_2, per_3])
            # if i == 6:
            #     print('yellow', [per_2, per_3])
            # if i == 7:
            #     print('red', [per_2, per_3])
            continue

        temp_1 = expand_matrix(temp_1, iterations)
        temp_2 = expand_matrix(temp_2, iterations)
        iou_bev[i] = calculate_iou(temp_1, temp_2)
    return iou_bev

