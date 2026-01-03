import os
import argparse
import pickle
import torch
from os.path import isfile

from transloc4d.misc import TrainingParams
from transloc4d.datasets import WholeDataset
from transloc4d.models import get_model
from transloc4d import save_recall_results
from transloc4d.eval_funcs import evaluate_4drad_dataset


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Note:
# Snail-Night 对应 snail-tv
# Snail-Rain 对应 snail-lrn

config_path = {"snail-lrn": '/home/long/PycharmProjects/RCPR/config/train/snail-n.txt',
               "snail-tv": '/home/long/PycharmProjects/RCPR/config/train/snail-n.txt',
               "seu-p":'/home/long/PycharmProjects/RCPR/config/train/seu-p.txt',
               "seu-af":'/home/long/PycharmProjects/RCPR/config/train/seu-p.txt'}

database_path = {"snail-lrn": '/media/myssd/Datasets/RCPR-Dataset/snail-lrn_test_evaluation_database_rc_25.pickle',
                 "snail-tv": '/media/myssd/Datasets/RCPR-Dataset/snail-tv_test_evaluation_database_rc_25.pickle',
                 "seu-p":'/media/myssd/Datasets/RCPR-Dataset/seu-p_test_evaluation_database_rc_25.pickle',
                 "seu-af":'/media/myssd/Datasets/RCPR-Dataset/seu-af_test_evaluation_database_rc_25.pickle'}

query_path = {"snail-lrn": '/media/myssd/Datasets/RCPR-Dataset/snail-lrn_test_evaluation_query_rc_25.pickle',
              "snail-tv": '/media/myssd/Datasets/RCPR-Dataset/snail-tv_test_evaluation_query_rc_25.pickle',
              "seu-p":'/media/myssd/Datasets/RCPR-Dataset/seu-p_test_evaluation_query_rc_25.pickle',
              "seu-af": '/media/myssd/Datasets/RCPR-Dataset/seu-af_test_evaluation_query_rc_25.pickle'}

test = 'snail-tv' #对应query_path
savepath = 'SNAIL4DPR' # SEU4DPR  SNAIL4DPR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")
    parser.add_argument("--database_pickle",
                        default= database_path[test],
                        required=False, help="Path to the database pickle file")
    parser.add_argument("--query_pickle",
                        default= query_path[test],
                        required=False, help="Path to the query pickle file")
    parser.add_argument("--config", default= config_path[test],
                        help="Path to the configuration file")
    parser.add_argument("--model_config",
                        default='/home/long/PycharmProjects/RCPRNet/config/model/Test.txt',
                        help="Path to the model-specific configuration file")
    parser.add_argument("--weights", default='/home/long/PycharmProjects/RCPRNet/weights/train-on-snail/RCPRNet/model_best.pth',
                        required=False, help="Path to the trained model weights")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_config is None:
        model_folder = os.path.dirname(args.weights)
        model_config_path = os.path.join(model_folder, "model_config.txt")
        assert isfile(
            model_config_path), f"Model configuration file not found at {model_folder}, please provide the path to the model configuration file"
        args.model_config = model_config_path

    print(f"==> Loaded model parameters from {args.model_config}")

    params = TrainingParams(args.config, args.model_config, debug=False)
    model = get_model(params, device, args.weights)

    database_sets = load_pickle(args.database_pickle)
    query_sets = load_pickle(args.query_pickle)
    test_set = WholeDataset(args.database_pickle, args.query_pickle, image_size=params.image_size, dataset=params.dataset)

    # 运行评估
    recalls, matches, gt = evaluate_4drad_dataset(model, device, test_set, params, return_matches=True)

    print(f"Recall@1: {recalls[1]:.4f}, Recall@5: {recalls[5]:.4f}, Recall@10: {recalls[10]:.4f}")

    model_name = params.model_name
    result_dir = os.path.dirname(args.weights)
    database_name = os.path.basename(args.database_pickle).split('.')[0]
    query_name = os.path.basename(args.query_pickle).split('.')[0]

    dataset_name = f"{database_name}_{query_name}"
    save_recall_results(model_name, dataset_name, recalls, result_dir)

