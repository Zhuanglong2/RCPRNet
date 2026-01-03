import os
import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import test_collate_fn


def get_predictions(dataset, embeddings):
    gt = []
    for i in range(dataset.len_q):
        # 获取每个查询的真实匹配项（positive samples）
        positives = dataset.get_non_negatives(i)
        gt.append(positives)

    # 获取查询和数据库特征的距离
    qFeat = embeddings[: dataset.len_q].cpu().numpy().astype("float32")
    dbFeat = embeddings[dataset.len_q:].cpu().numpy().astype("float32")

    print("==> Building faiss index")
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dis, predictions = faiss_index.search(qFeat, 20)

    return predictions, gt


def evaluate_4drad_dataset(model, device, dataset, params, return_matches=False):
    model.eval()
    quantizer = params.model_params.quantizer
    val_collate_fn = test_collate_fn(
        dataset,
        quantizer,
        params.batch_split_size,
        params.model_params.input_representation,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=params.val_batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    embeddings_dataset = torch.empty((len(dataset), 256))

    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(dataloader, desc="==> Computing embeddings")):
            embeddings_l = []
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                ys = model(minibatch)
                embeddings_l.append(ys["global"])
                del ys

            embeddings = torch.cat(embeddings_l, dim=0)
            del embeddings_l
            embeddings_dataset[
            iteration
            * params.val_batch_size: (iteration + 1)
                                     * params.val_batch_size
            ] = embeddings.detach()
            torch.cuda.empty_cache()  # 防止稀疏张量导致的 GPU 内存占用过高

    predictions, gt = get_predictions(dataset, embeddings_dataset)  # 获取预测结果和真实值（gt）
    recalls = compute_recall_all(predictions, gt)  # 计算召回率
    print("==> Evaluation completed!")

    if return_matches:
        # `predictions` 是 `(num_queries, k)`，取 top-1 作为匹配的 database 索引
        matches = predictions
        return recalls, matches.tolist(), gt  # 转换为 Python 列表，便于可视化

    return recalls

def compute_recall(predictions, gt, n_values=[1, 5, 10, 20]):
    numQ = 0
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            continue
        else:
            numQ += 1
        for i, n in enumerate(n_values):
            # 如果在前 N 名内，则认为是正确匹配
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # 构造返回的字典
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
    return all_recalls

def compute_recall_all(predictions, gt, n_values=range(1,21)):
    numQ = 0
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        if len(gt[qIx]) == 0:
            continue
        else:
            numQ += 1
        for i, n in enumerate(n_values):
            # 如果在前 N 名内，则认为是正确匹配
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # 构造返回的字典
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
    return all_recalls

def save_recall_results(model_name, dataset_name, recall_metrics, result_dir):
    # 创建结果保存目录（如果不存在）
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 构建文件名和文件路径
    filename = f"{model_name}.txt"
    filepath = os.path.join(result_dir, filename)

    # 准备需要写入文件的内容
    recall_results_str = f"Dataset: {dataset_name}, Recall@1: {recall_metrics[1]:.4f}, Recall@5: {recall_metrics[5]:.4f}, Recall@10: {recall_metrics[10]:.4f}\n"

    # 检查文件是否存在并追加结果，如果文件不存在则创建并写入
    with open(filepath, 'a') as file:
        file.write(recall_results_str)
    print(f"==> Results saved to {filepath}")
