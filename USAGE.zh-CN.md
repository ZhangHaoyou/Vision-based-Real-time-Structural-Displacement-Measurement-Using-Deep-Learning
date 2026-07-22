**[English](USAGE.md) | [简体中文](USAGE.zh-CN.md)**

# 使用指南

本指南介绍项目背景、完整的仓库结构、数据集、环境配置，以及如何运行每一个脚本——包括用于复现论文中验证结果的脚本。

如果您只想使用已包含的训练模型进行推理，请参见 [README.zh-CN.md](README.zh-CN.md) 中的“快速开始”部分。

## 1. 背景

结构平动位移是评估结构损伤和动力响应的关键指标，但在破坏性试验或现场监测中，接触式传感器（如 LVDT）往往难以安装。本项目提供了一种基于视觉的替代方案：在结构上安装棋盘格标靶，通过视频对其进行跟踪，并将其运动换算为位移。

该框架分为两个阶段：

1. **改进的 YOLO 检测器（Y-BWA-W）** —— 在 YOLOv5 检测器的 CSPDarknet 主干网络中加入了本文提出的**黑白注意力（Black-White Attention, BWA）**模块（由对比度增强模块 `CEB` 和角点注意力模块 `CAB` 组成，见 `nets/CSPdarknet.py`），以更好地定位棋盘格标靶；同时采用**加权非极大值抑制（WNMS）**（`utils/utils_bbox.py` 中的 `non_max_suppression(..., wnms=True)`）以稳定连续帧间的检测框。
2. **Deep SORT 跟踪器**（`nets/deep_sort.py`、`nets/sort/`）—— 结合卡尔曼滤波运动预测与学习到的外观特征描述子，在短时遮挡下保持目标身份的连续性。

面内（水平方向）位移直接由跟踪得到的检测框位置读取；面外（深度方向）位移则通过 `depth_measurement.py` 中基于相似三角形原理、依据标靶表观尺寸变化进行估计。

论文中的验证部分共使用了三个试验，均可通过本仓库复现：

- 室外**振动台试验**（动力试验，面内）—— `validation.py: validate_EQ1()`
- **静力往复剪力墙试验**（面内）—— `validation.py: validate_PCW2()`
- 使用正面和斜角相机的**小型振动平台深度试验**（面外）—— `depth_measurement.py`

## 2. 仓库结构

```
validation/
├── main.py                  # 实时 YOLO + Deep SORT 演示程序（摄像头或视频文件，命令行）
├── predict.py                # 单张图片预测 / 批量预测 / FPS 测试 / 热力图 / ONNX 导出等模式
├── predict_video.py          # 跟踪用户在录制视频中选定的单个目标（见 5.8 节）
├── predict_yolo.py           # predict.py 的早期/替代版本
├── validation.py              # Validator 类以及面内验证试验（论文第 5.1 节）
├── depth_measurement.py       # Depth_Estimator 类以及面外验证试验（论文第 5.2 节）
├── target.py                  # 用于斜角相机的独立透视校正程序
├── plot_template.py           # 复现论文绘图风格的公共 matplotlib 工具
├── test.py, util.py           # 若干小型辅助脚本（不属于主流程）
├── yolo.py                     # YOLO 封装类：模型加载、推理、热力图、ONNX 导出
│
├── nets/
│   ├── CSPdarknet.py           # 主干网络 + BWA 注意力模块（CEB、CAB，以及 SE/ECA/CBAM/CA 等基线模块）
│   ├── yolo.py, yolo_training.py  # YOLO 检测头/主体结构及损失函数
│   ├── ConvNext.py, Swin_transformer.py  # yolo.py 中 `backbone` 选项支持的其它主干网络
│   ├── deep_sort.py            # DeepSort 类（跟踪器 + 外观特征提取器）
│   ├── sort/                    # 卡尔曼滤波、匈牙利匹配、IoU/外观度量
│   └── deep/                    # 外观 ReID 网络（`model.py`）及其训练脚本（`train_deepsort.py`）
│
├── utils/
│   ├── utils.py, dataloader.py, augmentation.py   # 配置、数据加载、数据增强
│   ├── utils_bbox.py            # 检测框解码 + （加权）非极大值抑制
│   ├── utils_fit.py, utils_map.py  # 训练循环 / mAP 评估工具
│   └── correct_perspective.py   # 棋盘格角点检测 + 透视校正
│
├── utils_ds/                    # Deep SORT 配套工具：绘图、日志、YAML 配置解析
├── utils_coco/                  # COCO 格式 mAP 评估工具
│
├── model_data/
│   ├── best_epoch_weights.pth   # 训练好的 Y-BWA-W YOLO 检测器权重
│   ├── ckpt_target_epoch50.t7   # 训练好的 Deep SORT 外观（ReID）权重
│   ├── yolov5_s.pth             # ImageNet 预训练主干网络权重（仅用于训练）
│   ├── target_classes.txt, coco_classes.txt, yolo_anchors.txt  # 类别/先验框定义文件
│   ├── deep_sort.yaml           # Deep SORT 超参数配置（供 main.py / predict_video.py 使用）
│   └── 2007_train.txt / 2007_val.txt / 2007_test.txt  # VOC 风格标注索引文件（详见“数据集”一节）
│
├── experimental_data/            # 原始录像与传感器日志（未纳入 git 版本控制 —— 详见“数据集”一节）
│   ├── front_camera/, side_camera/   # 视频录像（动力、静力、深度试验）
│   └── raw_data/                     # 传感器日志（.bsi / .txt）
│
├── logs/                          # 脚本输出：位移曲线、检测位置等（未纳入 git 版本控制）
└── detections/                    # 脚本输出：裁剪/标注后的检测图像（未纳入 git 版本控制）
```

`experimental_data/`、`logs/` 以及 `detections/` 已通过 `.gitignore` 排除在版本控制之外（详见下方“数据集”一节）—— 它们可通过运行脚本重新生成，或从 Zenodo 归档中恢复。

## 3. 数据集

Y-BWA-W 检测器基于一个专门构建的棋盘格标靶图像数据库进行训练，详见论文第 3 节：

- **2,132 张标注高分辨率图像**：其中 800 张来自动力（振动台）试验，1,332 张来自静力试验，分辨率范围从 1920×1080 到 6000×4000 不等，涵盖了不同的标靶尺度、朝向、位置，以及光照变化、部分遮挡、运动模糊、复杂背景等具有挑战性的场景。
- **标注流程**：先通过模板匹配（快速归一化互相关）自动生成初始检测框，再人工检查并修正。
- **数据增强**：包括垂直/水平翻转、随机 90° 旋转、平移/缩放/旋转、透视变换、色调-饱和度-明度调整、亮度/对比度随机调整，以及添加阴影、日光耀斑、高斯噪声、ISO 噪声和高斯模糊，各类增强以随机组合方式施加。原始图像与增强后的图像均被裁剪为统一的 1920×1080 分辨率，最终共得到 **10,593 张裁剪图像**。
- **训练/验证集划分**：1,492 张图像（70%，经过增强）用于训练，640 张图像（30%，未经增强）用于验证。
- **标注格式**：与 [bubbliiiing/yolov5-pytorch](https://github.com/bubbliiiing/yolov5-pytorch) 相同的 VOC 风格格式 —— 每行对应一张图像，格式为 `image_path xmin,ymin,xmax,ymax,class_id ...`。本仓库中，索引文件位于 `model_data/2007_train.txt`、`2007_val.txt` 和 `2007_test.txt`，类别列表位于 `model_data/target_classes.txt`，先验框定义位于 `model_data/yolo_anchors.txt`。

**数据获取方式**：本仓库已包含训练好的权重（`model_data/best_epoch_weights.pth`、`model_data/ckpt_target_epoch50.t7`）以及标注索引文件，因此直接运行推理或验证脚本无需额外下载数据集。

上述 2,132 张原始标注图像，以及原始试验录像/传感器日志（`experimental_data/`），因体积过大而未纳入本 git 仓库，而是以 **CC-BY 4.0** 协议开放发布在 Zenodo：

> Zhang, H., Cheng, X., Li, Y., Guan, H. (2026). *Vision-based Real-time Structural Displacement Measurement Using Deep Learning — Dataset* [Data set]. Zenodo. [doi.org/10.5281/zenodo.21400638](https://doi.org/10.5281/zenodo.21400638)
>
> 概念 DOI（始终指向最新版本）：[doi.org/10.5281/zenodo.21400637](https://doi.org/10.5281/zenodo.21400637)

共提供两个压缩包：

| 压缩包 | 大小 | 内容 | 解压位置 |
|---|---|---|---|
| `VOCdevkit.zip` | 24.2 GB | 完整的 2,132 张标注数据集（训练+验证），采用 `model_data/2007_train.txt` / `2007_val.txt` / `2007_test.txt` 所要求的 VOC 目录结构 | 仓库根目录，使 `VOCdevkit/VOC2007/JPEGImages/...` 与 `model_data/`、`nets/` 等目录并列 |
| `experimental_data.zip` | 19.6 GB | 三个验证试验的原始相机录像 + 传感器日志 | 仓库根目录（创建/替换 `experimental_data/`） |

```bash
# 在仓库根目录下执行
curl -L -o VOCdevkit.zip "https://zenodo.org/records/21400638/files/VOCdevkit.zip?download=1"
curl -L -o experimental_data.zip "https://zenodo.org/records/21400638/files/experimental_data.zip?download=1"

unzip VOCdevkit.zip -d .
unzip experimental_data.zip -d .
```

## 4. 环境配置

以下脚本是在名为 `yolo` 的 conda 环境中开发和运行的（Python 3.8，PyTorch 2.4.1 + CUDA 11.8）：

```bash
conda create -n yolo python=3.8
conda activate yolo
pip install -r requirements.txt
```

`requirements.txt` 中固定了与 CPU/CUDA 无关的通用依赖包版本；PyTorch 本身则指定为面向 CUDA 11.8 的 `torch==2.4.1+cu118`。如果您没有对应的 GPU/驱动，请改为从 [pytorch.org](https://pytorch.org) 安装 CPU 版本或其它 CUDA 版本，并将 `yolo.py` 中 `_defaults` 里的 `"cuda"` 设为 `False`（或在运行 `main.py` 时传入 `--device cpu`）。

若要使用 `predict_video.py` 的 Deep SORT 选项，还需要 `PyYAML` 和 `easydict`（供 `utils_ds/parser.py` 解析 `deep_sort.yaml` 配置文件使用）；这两个包未列在 `requirements.txt` 中，如遇到 `ModuleNotFoundError`，请自行 `pip install pyyaml easydict` 安装。

以下命令均假设已激活该环境，且当前目录为 `validation/`：

```bash
conda activate yolo
```

## 5. 运行脚本

### 5.1 `predict.py` —— 单图预测 / 批量预测 / 摄像头 / FPS / 热力图 / ONNX 导出

该脚本是通用推理的统一入口。打开文件后，修改文件末尾附近的 `mode` 变量即可切换模式（每种模式各自的参数就写在该变量下方）：

| `mode` | 功能 |
|---|---|
| `"predict"` | 提示输入图片路径，显示检测结果 |
| `"video"` | 对摄像头或视频文件进行检测 |
| `"fps"` | 在单张图片上测试推理速度 |
| `"dir_predict"` | 批量检测 `dir_origin_path` 中的所有图像，将标注结果保存到 `dir_save_path` |
| `"heatmap"` | 为输入图像生成 Eigen-CAM 注意力热力图（对应论文图 9） |
| `"export_onnx"` | 将模型导出为 ONNX 格式 |

```bash
python predict.py
```

### 5.2 `main.py` —— 实时检测 + Deep SORT 跟踪

一个命令行演示程序，对视频文件或摄像头运行完整的 Y-BWA-W + Deep SORT 流程，并可保存标注后的视频以及逐帧检测框日志。

```bash
python main.py --input_path path/to/video.mp4 --save_path video/ --save_txt logs/deepSORT/
# 或使用实时摄像头：
python main.py --camera 0
```

主要参数：`--device`（`cuda`/`cpu`）、`--display`（显示实时窗口）、`--config_deepsort`（指向 `model_data/deep_sort.yaml`，用于设置 Deep SORT 的距离阈值、最大存活帧数等）。

### 5.3 `validation.py` —— 面内验证（论文第 5.1 节）

`Validator` 类负责帧序列排序、检测框中心点计算、异常值剔除，以及与传感器实测位移之间的 RMSE 计算。两个针对具体试验的函数用于复现论文中的图表：

- `validate_EQ1()` —— 动力振动台试验与 LVDT 实测值对比（对应图 14），RMSE ≈ 2.05 mm
- `validate_PCW2()` —— 静力往复剪力墙试验（对应图 15），RMSE ≈ 0.11 mm
- `compare()` —— 两个验证数据集之间的相关系数 / R²

这些脚本针对具体试验硬编码了路径与裁剪区域（例如 `validate_PCW2()` 中被注释掉的多个 `crop = (...)` 备选项）。若要复现某个具体结果，请在文件末尾 `if __name__ == '__main__':` 代码块中取消相应函数调用的注释，然后运行：

```bash
python validation.py
```

输出结果（检测到的位置、位移曲线、对比图）保存在 `logs/disp/` 和 `logs/detected_disp/` 目录下。

### 5.4 `depth_measurement.py` —— 面外（深度方向）验证（论文第 5.2 节）

`Depth_Estimator` 继承自 `Validator`，实现了基于相似三角形原理（对应论文公式 26）、依据标靶表观尺寸变化进行深度估计的方法。文件末尾的 `main()` 函数可复现正面相机与斜角相机的对比结果（对应图 17b）：

```bash
python depth_measurement.py
```

在使用斜角相机时，会先通过 `utils/correct_perspective.py` 对画面进行透视校正（调用 `read_video` 时传入 `perspective=True`）。

### 5.5 `target.py` / `utils/correct_perspective.py` —— 透视校正

通过边缘检测 + 霍夫直线检测出棋盘格标靶的四个角点，再计算并应用 `cv2` 透视变换矩阵，将斜角相机画面校正为等效正视画面。可独立运行以进行检查：

```bash
python target.py
```

### 5.6 `plot_template.py` —— 通用绘图工具

一个 matplotlib 封装函数（`plot_template(...)`），被 `validation.py` 和 `depth_measurement.py` 调用，用于生成与论文一致的折线图风格（Times New Roman 字体、刻度朝内、图例/坐标范围可自定义）。可直接运行查看示例效果：

```bash
python plot_template.py
```

### 5.7 `test.py`、`util.py` —— 其它辅助脚本

`test.py` 是一个简短脚本，用于扫描 `logs/positions.txt`，找出检测失败（`None`）的帧，并将其序号记录到 `numbers.txt`。`util.py` 中包含项目早期阶段使用的一些独立辅助函数（视频转帧、基于 `pytesseract` 的位置文字识别等）；两者均不属于主要的检测/跟踪/验证流程。

### 5.8 `predict_video.py` —— 跟踪用户自选的单个目标（通用视频工具）

一个面向普通用户、可跟踪自己录制视频中任意目标的通用脚本，无需修改代码。使用流程：

1. 用鼠标在首帧上**移动一个精确 960×960（模型输入尺寸，来自 `yolo.input_shape`）的裁剪框**，左键点击确定位置——这样每一帧都会以原始分辨率裁剪后再送入检测器，避免整帧缩放导致小目标精度下降。
2. 在该裁剪范围内，**拖拽一个矩形**选定您关心的目标（ROI）。
3. 可选择是否将 ROI 以外的区域像素置零。
4. 可选填入目标的真实宽度（cm），脚本会据此换算出像素与毫米的比例，输出真实位移。
5. 可选启用 **Deep SORT**（`--use-deepsort` 或按提示输入 `y`），使目标在短暂遮挡或暂时移出/移回 ROI 时仍能保持身份连续；未启用时，则每一帧独立判断“完全落在 ROI 内、置信度最高”的检测框。

```bash
python predict_video.py "./videos/your_video.mov"
python predict_video.py --use-deepsort --step 2 --save-all-frames
```

输出保存在 `./detections/frames/<视频名>/`（带检测框的标注帧）、`./detections/videos/<视频名>_detected.mp4`（完整标注视频）以及 `./detections/motion/<视频名>/`（`motion.csv` 逐帧像素/毫米位移、`config.txt` 记录裁剪框与 ROI 等参数、位移曲线图）。

需要注意：若某一帧未检测到目标，该帧**不会**在 `motion.csv` 中生成任何记录（既不插值也不填充空值），因此 `frame` 序号可能出现跳跃；标注视频仍会包含该帧（标注“target not detected”），但不会计入位移曲线。

## 6. 重新训练或适配新目标

本仓库已包含论文中使用的训练权重，因此无需训练即可进行推理或复现上述验证结果。如果您想将该框架应用到不同的目标或数据集：

- **Deep SORT 外观模型** 可使用 `nets/deep/train_deepsort.py`，在 Market1501 风格的裁剪目标图像文件夹上重新训练。
- **Y-BWA-W 检测器本身的训练脚本未包含在本文件夹中**（本仓库聚焦于推理/验证）。如果您希望重新训练或微调检测器，欢迎联系我（见“联系方式”）——我很乐意分享训练流程，并讨论如何将其适配到其它类型的目标，正如论文未来工作部分所提到的那样。

## 7. 关键结果（摘自论文）

| 模型 | AP (%) | 精确率 (%) | 召回率 (%) |
|---|---|---|---|
| YOLO（基线模型） | 93.20 | 97.31 | 92.71 |
| Y-SE / Y-CBAM / Y-CA / Y-ECA | 93.05–93.25 | 97.19–97.38 | 90.99–91.35 |
| **Y-BWA（本文提出）** | **93.33** | 97.44 | 91.73 |

| 验证试验 | RMSE | 备注 |
|---|---|---|
| 动力振动台试验（面内） | 2.05 mm | 小于 1 个像素对应的物理位移（4.305 mm） |
| 静力剪力墙试验（面内） | 0.11 mm | 小于 1 个像素对应的物理位移（1.8 mm） |
| 深度试验，正面相机（面外） | 6.98 mm | 约为 2 个像素级别 |
| 深度试验，斜角相机（面外） | 7.50 mm | 约为 2 个像素级别 |

实时处理速度：在 NVIDIA RTX 4090 上约为 35 FPS。

## 8. 参考文献

本实现基于以下两个开源代码库构建：

- Bubbliiiing，[**yolov5-pytorch**](https://github.com/bubbliiiing/yolov5-pytorch) —— 本项目 `yolo.py`/`predict.py`/`nets/` 所基于的 YOLOv5（PyTorch）检测器实现，并在此基础上加入了 BWA 注意力模块与 WNMS。
- ZQPei，[**deep_sort_pytorch**](https://github.com/ZQPei/deep_sort_pytorch) —— 本项目 `nets/deep_sort.py`、`nets/sort/`、`nets/deep/` 所基于的 Deep SORT 跟踪器实现。

以及以下方法/文献（完整引用信息见已接收的论文正文）：

- J. Redmon, S. Divvala, R. Girshick, A. Farhadi, *You Only Look Once: Unified, Real-Time Object Detection*, CVPR, 2016.
- J. Redmon, A. Farhadi, *YOLO9000: Better, Faster, Stronger*, CVPR, 2017.
- J. Redmon, A. Farhadi, *YOLOv3: An Incremental Improvement*, arXiv:1804.02767, 2018.
- C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, I.-H. Yeh, *CSPNet: A New Backbone that can Enhance Learning Capability of CNN*, CVPRW, 2020.
- K. He, X. Zhang, S. Ren, J. Sun, *Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition*, IEEE TPAMI, 2015.
- K. Wang, J. H. Liew, Y. Zou, D. Zhou, J. Feng, *PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment*, ICCV, 2019.
- J. Hu, L. Shen, G. Sun, *Squeeze-and-Excitation Networks*, CVPR, 2018.（SE 注意力基线模块）
- Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, Q. Hu, *ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks*, CVPR, 2020.（ECA 注意力基线模块）
- S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, *CBAM: Convolutional Block Attention Module*, ECCV, 2018.（CBAM 注意力基线模块）
- Q. Hou, D. Zhou, J. Feng, *Coordinate Attention for Efficient Mobile Network Design*, CVPR, 2021.（CA 注意力基线模块）
- M. B. Muhammad, M. Yeasin, *Eigen-CAM: Class Activation Map using Principal Components*, IJCNN, 2020.（用于生成可解释性热力图）
- J. P. Lewis, *Fast Normalized Cross-Correlation*, Vision Interface, 1995.（用于数据集初始标注的模板匹配方法）
- H. Zhang, N. Wang, *On the Stability of Video Detection and Tracking*, arXiv:1611.06467, 2017.（WNMS 抖动缓解思路的参考依据）
- N. Wojke, A. Bewley, D. Paulus, *Simple Online and Realtime Tracking with a Deep Association Metric*, ICIP, 2017.（Deep SORT）
- R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, J. Basic Eng., 1960.（卡尔曼滤波）
- P. C. Mahalanobis, *On Tests and Measures of Group Divergence*, J. Asiat. Soc. Bengal, 1930.（马氏距离）
- H. W. Kuhn, *The Hungarian Method for the Assignment Problem*, Naval Research Logistics Quarterly, 1955.（数据关联匹配算法）
- A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS, 2019.

## 9. 引用

```bibtex
@article{zhang2026vision,
  title   = {Vision-based Real-time Measurement of Structural Translational Motion Using Deep Learning and Object Tracking Methods},
  author  = {Zhang, Haoyou and Cheng, Xiaowei and Li, Yi and Guan, Hong},
  journal = {Applied Mathematical Modelling},
  year    = {2026},
  doi     = {10.1016/j.apm.2026.117187}
}
```

如果您使用了本数据集，请同时引用：

```bibtex
@dataset{zhang2026dataset,
  title     = {Vision-based Real-time Structural Displacement Measurement Using Deep Learning --- Dataset},
  author    = {Zhang, Haoyou and Cheng, Xiaowei and Li, Yi and Guan, Hong},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21400638}
}
```

## 10. 联系方式

Haoyou Zhang（张浩宇）—— **haoyou.zhang@marquette.edu**

无论是代码问题、如何将该方法应用到您自己的目标/试验，还是单纯想交流基于视觉的结构健康监测技术，都欢迎随时联系 —— 我很乐意讨论这项工作。
