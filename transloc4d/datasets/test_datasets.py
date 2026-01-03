import torch
import numpy as np
import os
import pickle
import numpy as np
import torch.utils.data as data
import MinkowskiEngine as ME

from .augmentation import ValSetTransform
from .pc_loader import Rad4DPointCloudLoader
import cv2
#radar
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def pq_to_mat(xyzqxqyqzqw):
    trans = np.array(xyzqxqyqzqw[0:3])
    quat = np.array(xyzqxqyqzqw[3:7])  # xyzw
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T

#SNAIL4DPR投影RGB
def pro_point_to_rgb(points_radar, img_path):
    Vis = False

    # === 相机参数（以 SUV 为例，其他平台请替换）===
    extr = [0.0695283917427731, -0.008381612991474873, -0.17223038663727022,
            0.019635536507920586, 0.7097335839994078, -0.7039714840647372, -0.017799861603211602]
    K = np.array([[266.52, 0, 345.01],
                  [0, 266.80, 189.99],
                  [0, 0, 1]])
    dist = np.array([-0.0567891, 0.032141, 0.000169392, -0.000430803, -0.0128658])

    # === 32Lidar到Oculii Radar的变换矩阵 ===
    xt32_T_oculii = np.array([
        [-0.0197501325711932, 0.999429818150974, 0.0273856103418523, 0],
        [-0.999756831101593, -0.0200104940487212, 0.00926598195627003, -0.07],
        [0.0098086977351427, -0.027195946430554, 0.999581997333877, -0.115],
        [0, 0, 0, 1]
    ])

    T_lc = pq_to_mat(extr)
    T_cl = np.linalg.inv(T_lc)  # 从相机坐标系到激光雷达坐标系的逆变换矩阵


    # 1. 加载雷达点云 (N, 5)，包括XYZ + 2个特征
    points_xyz = points_radar[:, :3]  # XYZ坐标
    features = points_radar[:, 3:]   # 额外的两个通道特征

    # 2. 读入图片，确保图片加载成功
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 3. 雷达到激光雷达坐标系变换，齐次坐标，变换XYZ，feature不变
    pc_hom_radar = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))  # (N,4)
    points_lidar = (xt32_T_oculii @ pc_hom_radar.T).T[:, :3]  # (N,3)

    # 4. 激光雷达到相机坐标系变换
    pc_hom_lidar = np.hstack((points_lidar, np.ones((points_lidar.shape[0], 1))))  # (N,4)
    points_cam_xyz = (T_cl @ pc_hom_lidar.T).T[:, :3]  # (N,3)

    if Vis:
        # 5. 只保留Z>0的点（相机前方）
        valid_mask = points_cam_xyz[:, 2] > 0
        points_cam_xyz = points_cam_xyz[valid_mask]
        features = features[valid_mask]

        # 6. 投影到图像平面 (N,2)
        pts2d, _ = cv2.projectPoints(points_cam_xyz, np.zeros(3), np.zeros(3), K, dist)
        pts2d = pts2d.reshape(-1, 2)

        # 7. 过滤无效点（包含NaN或Inf）
        valid_mask = np.isfinite(pts2d).all(axis=1)
        pts2d = pts2d[valid_mask]
        points_cam_xyz = points_cam_xyz[valid_mask]
        features = features[valid_mask]

        # 8. 转为整数像素坐标
        pts2d = pts2d.astype(int)

        # 9. 只保留图像内的点
        inside_mask = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
        pts2d = pts2d[inside_mask]
        points_cam_xyz = points_cam_xyz[inside_mask]
        features = features[inside_mask]

        # 10. 拼接5通道数据 (N,5)
        points_cam_5d = np.hstack((points_cam_xyz, features))

        # 11. 可视化：这里仅用XYZ投影位置 + 按深度着色
        depths = points_cam_xyz[:, 2]
        colors = (plt.cm.jet((depths - depths.min()) / (depths.max() - depths.min()))[:, :3] * 255).astype(np.uint8)

        img_vis = img.copy()
        for (x, y), c in zip(pts2d, colors):
            cv2.circle(img_vis, (x, y), 1, (int(c[2]), int(c[1]), int(c[0])), -1)

        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        # 6. 投影到图像平面 (N,2)
        pts2d, _ = cv2.projectPoints(points_cam_xyz, np.zeros(3), np.zeros(3), K, dist)
        pts2d = pts2d.reshape(-1, 2)

        # 8. 转为整数像素坐标
        # pts2d = pts2d.astype(int)

        # 10. 拼接5通道数据 (N,5)
        points_cam_5d = np.hstack((points_cam_xyz, features))


    return points_cam_5d, pts2d


#SEU4DPR投影RGB
def pro_point_to_rgb2(points_radar, img_path):
    visualize = False
    # === 新相机参数 ===
    K = np.array([[427.66601418, 0., 363.61438561],
                  [0., 295.59938648, 189.11192141],
                  [0., 0., 1.]])
    dist = np.array([-0.0567891, 0.032141, 0.000169392, -0.000430803, -0.0128658])

    # === 雷达到相机的外参 ===
    T_cr = np.array([[-0.04933482, -0.99872783, -0.01043048, -0.32431708],
                     [-0.03792112,  0.01230869, -0.99920492, -0.06773846],
                     [ 0.99806215, -0.04890005, -0.03848013,  0.06541855],
                     [0., 0., 0., 1.]])

    # === 数据准备 ===
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img.shape[:2]

    points_xyz = points_radar[:, :3]
    features = points_radar[:, 3:]

    # === 1. 雷达点转为相机坐标系 ===
    pc_hom_radar = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))  # (N,4)
    points_cam_xyz = (T_cr @ pc_hom_radar.T).T[:, :3]  # (N,3)

    # === 2. 投影到图像平面 ===
    pts2d, _ = cv2.projectPoints(points_cam_xyz, np.zeros(3), np.zeros(3), K, dist)
    pts2d = pts2d.reshape(-1, 2).astype(int)

    # === 3. 可视化模式（深度着色） ===
    if visualize:
        # Z > 0 筛选
        valid_mask = points_cam_xyz[:, 2] > 0
        pts2d = pts2d[valid_mask]
        points_cam_xyz = points_cam_xyz[valid_mask]
        features = features[valid_mask]

        # 图像范围内
        inside_mask = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
        pts2d = pts2d[inside_mask]
        points_cam_xyz = points_cam_xyz[inside_mask]
        features = features[inside_mask]

        # 深度着色
        depths = points_cam_xyz[:, 2]
        colors = (plt.cm.jet((depths - depths.min()) / (depths.ptp()))[:, :3] * 255).astype(np.uint8)

        img_vis = img.copy()
        for (x, y), c in zip(pts2d, colors):
            cv2.circle(img_vis, (x, y), 1, (int(c[2]), int(c[1]), int(c[0])), -1)

        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title("Projected Radar Points")
        plt.axis("off")
        plt.show()

    # === 4. 返回5通道和投影点 ===
    points_cam_5d = np.hstack((points_cam_xyz, features))
    return points_cam_5d, pts2d


class WholeDataset(data.Dataset):
    def __init__(
        self,
        database_pickle,
        query_pickle,
        image_size,
        dataset,
        input_transform = ValSetTransform(aug_mode=1),
        set_transform=None,
        split=None,
        test_split=None,
        mode="test",
    ):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.dataset = dataset
        self.pc_loader = Rad4DPointCloudLoader()
        self.database = self.load_pickle(database_pickle)
        self.queries = self.load_pickle(query_pickle)
        self.construct_quries(self.queries)
        self.construct_database(self.database)
        self.wholedataset_pc_file = self.queries_pc_file + self.database_pc_file
        self.wholedataset_image_file = self.queries_image_file + self.database_image_file
        self.len_q = len(self.queries_pc_file)
        self.len_db = len(self.database_pc_file)
        self.input_transform = input_transform
        # pc_loader must be set in the inheriting class
        self.set_transform = set_transform
        self.mode = mode

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def construct_quries(self, df_query):
        self.queries = {}
        self.queries_id = []
        self.queries_pc_file = []
        self.queries_image_file = []
        self.queries_nonnegatives = []
        for anchor_ndx in range(len(df_query)):
            non_negatives = df_query[anchor_ndx]["positives"]
            if len(non_negatives) == 0:
                continue

            # Extract timestamp from the filename
            scan_filename = df_query[anchor_ndx]["file"]
            assert os.path.isfile(scan_filename), "point cloud file {} is found".format(scan_filename)

            scan_image_filename = df_query[anchor_ndx]["image_file"]

            # Sort ascending order
            non_negatives = np.sort(non_negatives)

            self.queries_id.append(anchor_ndx)
            self.queries_pc_file.append(scan_filename)
            self.queries_image_file.append(scan_image_filename)
            # self.queries_positives.append(positives)
            self.queries_nonnegatives.append(non_negatives)

        print(f"==> Queries: {len(self.queries_nonnegatives)} valid queries")

    def construct_database(self, df_db):
        self.database_id = []
        self.database_pc_file = []
        self.database_image_file = []

        for idx in range(len(df_db)):
            self.database_id.append(idx)
            scan_filename = df_db[idx]["file"]
            self.database_pc_file.append(scan_filename)
            scan_image_filename = df_db[idx]["image_file"]
            self.database_image_file.append(scan_image_filename)

        print(f"==> Database: {len(self.database_pc_file)} references")

    def __len__(self):
        return len(self.wholedataset_pc_file)

    def process_image(self, img, size=(224, 224)):
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to fixed size
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Standard normalization (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC → CHW, and convert to Tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # C×H×W
        return img

    def normalize_xyz(self, points):
        # 获取 x, y, z 的最小值和最大值
        min_vals = points[:, :3].min(axis=0)  # 仅针对 x, y, z 获取最小值
        max_vals = points[:, :3].max(axis=0)  # 仅针对 x, y, z 获取最大值

        # 对 x, y, z 进行归一化
        points[:, :3] = 2 * (points[:, :3] - min_vals) / (max_vals - min_vals) - 1

        return points


    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = self.wholedataset_pc_file[ndx]
        image_pathname = self.wholedataset_image_file[ndx]
        img = self.process_image(cv2.imread(image_pathname), self.image_size)

        # 2. 加载原始点云（未增强）
        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)  # (4096, 3)

        # 3. 用原始点云投影，提取图像特征（关键！）
        if self.dataset == 'SNAIL4DPR':
            query_pc, uv = pro_point_to_rgb(query_pc, image_pathname)  #将点云投影至RGB空间
        else:
            query_pc, uv = pro_point_to_rgb2(query_pc, image_pathname)  # 将点云投影至RGB空间

        uv = torch.tensor(uv, dtype=torch.long)
        H, W = img.shape[1:]
        uv[:, 0].clamp_(0, W - 1)
        uv[:, 1].clamp_(0, H - 1)
        rgb_feats = img[:, uv[:, 1], uv[:, 0]]  # shape: (3, N)
        rgb_feats = rgb_feats.permute(1, 0)  # shape: (N, 3)

        # 4. 点云归一化
        query_pc = self.normalize_xyz(query_pc)  # (N, 3)
        query_pc = torch.tensor(query_pc, device=img.device, dtype=torch.float)

        # 5. 融合
        fused_feats = torch.cat([query_pc, rgb_feats], dim=1)  # shape: (N, 6)

        # 6. 数据增强
        if self.input_transform is not None:
            fused_feats = self.input_transform(fused_feats)

        return fused_feats, img, ndx

    # def __getitem__(self, ndx):
    #     # Load point cloud and apply transform
    #     file_pathname = self.wholedataset_pc_file[ndx]
    #     image_pathname = self.wholedataset_image_file[ndx]
    #     img = self.process_image(cv2.imread(image_pathname), self.image_size)
    #     query_pc = self.pc_loader(file_pathname)
    #     query_pc = torch.tensor(query_pc, dtype=torch.float)  # (4096, 3)
    #     # 投影至RGB空间
    #     query_pc, uv = pro_point_to_rgb(query_pc, image_pathname)
    #     query_pc = self.normalize_xyz(query_pc)  # 归一化XYZ
    #     query_pc = torch.tensor(query_pc, device=img.device, dtype=torch.float)
    #     uv = torch.tensor(uv, device=img.device)
    #
    #     #验证集follow训练，也先不增强
    #     if self.input_transform is not None:
    #         query_pc = self.input_transform(query_pc)
    #
    #     return query_pc, uv, img, ndx

    def get_path(self, ndx):
        return self.wholedataset_pc_file[ndx]

    def get_non_negatives(self, ndx):
        return self.queries_nonnegatives[ndx]

def resample_tensor(input_tensor, output_shape=4096):
    input_tensor = input_tensor.transpose(0, 1).unsqueeze(0)

    output_tensor = torch.nn.functional.interpolate(
        input_tensor, size=output_shape, mode="nearest"
    )

    output_tensor = output_tensor.squeeze(0).transpose(0, 1).contiguous()

    return output_tensor

def test_collate_fn(
    dataset,
    quantizer,
    batch_split_size=None,
    input_representation="RVI",
    scancontext_input=False,
):
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        images = [e[1] for e in data_list]



        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.transform(clouds)
            clouds = clouds.split(lens)

        clouds_org_coords = clouds
        coords_quant = ([quantizer(e)[0]
             for e in clouds] if quantizer is not None else clouds)
        coords = [e[:, :3] for e in coords_quant]
        if input_representation == "RV":
            feats = [e[:, 3:4] for e in coords_quant]
        elif input_representation == "RI":
            feats = [e[:, 4:] for e in coords_quant]
        else:
            feats = [e[:, 3:] for e in coords_quant]

        if batch_split_size is None or batch_split_size == 0:
            c = (ME.utils.batched_coordinates(coords)
                if not scancontext_input
                else ME.utils.batched_coordinates(coords, dtype=torch.float32))
            # Assign a dummy feature equal to 1 to each point
            if input_representation == "R":
                feats = torch.ones((c.shape[0], 1), dtype=torch.float32)
            else:
                feats = torch.cat(feats, 0)
            batch = {
                "coords": c,
                "features": feats,
                "batch": torch.stack(clouds_org_coords),
                "image": torch.stack(images),
            }

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp_org = clouds_org_coords[i: i + batch_split_size]
                temp = coords[i: i + batch_split_size]
                temp_imgs = images[i: i + batch_split_size]  # 同步切图像
                c = (ME.utils.batched_coordinates(temp)
                    if not scancontext_input
                    else ME.utils.batched_coordinates(temp, dtype=torch.float32))
                if input_representation == "R":
                    f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                else:
                    f = torch.cat(feats[i: i + batch_split_size], 0)
                minibatch = {"coords": c, "features": f,
                             "batch": torch.stack(temp_org),
                             "image": torch.stack(temp_imgs),
                             }
                batch.append(minibatch)

        return batch

    return collate_fn
