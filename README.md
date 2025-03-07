# trt_pose

<img src="https://user-images.githubusercontent.com/4212806/67125332-71a64580-f1a9-11e9-8ee1-e759a38de215.gif" height=256/>

trt_pose is aimed at enabling real-time pose estimation on NVIDIA Jetson.  You may find it useful for other NVIDIA platforms as well.  Currently the project includes

- Pre-trained models for human pose estimation capable of running in real time on Jetson Nano.  This makes it easy to detect features like ``left_eye``, ``left_elbow``, ``right_ankle``, etc.

- Training scripts to train on any keypoint task data in [MSCOCO](https://cocodataset.org/#home) format.  This means you can experiment with training trt_pose for keypoint detection tasks other than human pose.

To get started, follow the instructions below.  If you run into any issues please [let us know](../../issues).

## Getting Started

To get started with trt_pose, follow these steps.

### Step 1 - Install Dependencies

1. Install PyTorch and Torchvision in a new terminal window. To do this on NVIDIA Jetson, we recommend following [this guide](https://forums.developer.nvidia.com/t/72048) for Nvidia Jetson Nano JetPack 4.

    First, check if PyTorch and Torchvision are already installed in the home directory of the Jetson Nano.

    ```bash
    # Verify installation
    python3 -c "import torch, torchvision; print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)"
    ```

    If not, install PyTorch and Torchvision using the following commands:

    ```bash
    wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev libomp-dev
    pip3 install 'Cython<3'
    pip3 install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl

    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
    git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision   # PyTorch v1.10 - torchvision v0.11.1
    cd torchvision
    export BUILD_VERSION=0.11.1 # PyTorch v1.10 - torchvision v0.11.1
    python3 setup.py install --user
    cd ../  # attempting to load torchvision from build dir will result in import error

    # Verify installation
    python3 -c "import torch, torchvision; print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__)"
    ```

2. Check if the [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) is already installed in the `Documents` directory. If not, install it using the following commands:

    ```bash
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo python3 setup.py install --plugins
    ```

3. Install other miscellaneous packages

    ```bash
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib
    ```

### Step 2 - Install trt_pose

Check if the `trt_pose` repository is already installed in the `Documents` directory. If not, install it using the following commands:

```python
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

### Step 3 - Run the example notebook

There are a couple of human pose estimation models pre-trained on the MSCOCO dataset. The throughput in FPS is shown for each platform

| Model | Jetson Nano | Weights |
|-------|-------------|---------|
| resnet18_baseline_att_224x224_A | 22 | [download (81MB)](./tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth) |
| densenet121_baseline_att_256x256_B | 12 | [download (84MB)](./tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160.pth) |

Open and follow the [live_demo.ipynb](tasks/human_pose/live_demo.ipynb)  notebook (VS Code Jupyter extension v2023.9.1102792234)

> You may need to modify the notebook, depending on which model you use
