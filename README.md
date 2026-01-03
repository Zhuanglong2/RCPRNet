# RCPRNet: Robust 4D Radar-Camera Place Recognition
This repository is primarily referenced by [TransLoc4D](https://github.com/phatli/TransLoc4D), and we sincerely thank the authors for their valuable contributions to the community.

### Environment Setting
1. Install PyTorch:
   Ensure that you install the correct version of PyTorch that is compatible with your CUDA version to leverage GPU acceleration. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the command tailored to your environment, or use a general installation command like:
   ```
   pip install torch torchvision torchaudio
   ```
   
2. Clone and install MinkowskiEngine:
   ```
   git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
   cd MinkowskiEngine
   python setup.py install --force_cuda --blas=openblas
   ```

3. Install other required Python packages:
   ```
   cd {Project Folder}
   pip install -r requirements.txt
   pip install -e .
   ```

## Datasets
Based on the [Snail-Radar](https://snail-radar.github.io/) dataset and our self-collected data, we construct two new datasets, namely SNAIL4DPR and SEU4DPR, which can be found [here](https://pan.baidu.com/s/1zTDnDV8vjj3NWkeIAJRqpw?pwd=1234).

## Model Weights
We have released our trained model weights, which are available [here](https://pan.baidu.com/s/1zTDnDV8vjj3NWkeIAJRqpw?pwd=1234).

## Usage
To prepare datasets for training and evaluation, run the following scripts:
- For training set generation, modify path and config in `scripts/generate_trainset.py` and then run: 
  ```
  python scripts/generate_trainset.py 
  ```
  - `base_path`: Path to base_path.
  - `dataset_name`: Path to dataset_name.

- For test set generation, modify path and config in `scripts/generate_testsets.py`:
  ```
  python scripts/generate_testsets.py
  ```
  - `base_path`: Path to base_path.

### Running Evaluation and Training
#### Evaluation
To evaluate the model, execute the following command:
```
python scripts/eval.py
```
- `test`: Path to the sequence.
- `savepath`: Path to savepath.
- `--model_config`: Path to the model-specific configuration file.
- `--weights`: Path to the trained model weights.

#### Training
To train the model, run the following command:
```
python scripts/train.py
```
- `--config`: Path to the training configuration file.
- `--model_config`: Path to the model-specific configuration file.
- `--your path/model_best.pth`

Training result can then be found in `weights`.

## Citation
If you find this work useful, please cite our paper.

## Acknowledgement
Our code is based on [TransLoc4D](https://github.com/phatli/TransLoc4D), [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2), [PPT-Net](https://github.com/fpthink/PPT-Net) and [PTC-Net](https://github.com/LeegoChen/PTC-Net).
