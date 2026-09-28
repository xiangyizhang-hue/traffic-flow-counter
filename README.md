# 高速车流量计数器

中国移动“梧桐·鸿鹄”人工智能方向实训中的课程实践整理版。项目覆盖 VOC 标注转换、YOLO 训练/验证、视频推理、目标跟踪、跨线计数和 Gradio 展示。

## 整理与改进

原作业把 YOLOv5、DeepSORT、模型权重、虚拟环境和业务脚本放在同一目录，难以复现，也不适合直接上传。本仓库保留实践目标并重构为：

- 通过 `ultralytics` 包调用 YOLO，不复制第三方框架源码和权重。
- 使用 ByteTrack 的持久化轨迹 ID，按目标中心点“从线的一侧移动到另一侧”计数。
- 加入迟滞区，减少目标在线附近抖动造成的重复计数。
- 每次视频处理创建独立计数器，避免原全局变量跨任务累积。
- 把标注转换和跨线逻辑拆成可单元测试的模块。

## 安装与运行

```bash
python -m venv .venv
pip install -r requirements.txt
python app.py input.mp4 --output runs/result.mp4 --model yolo11n.pt
python webui.py
```

第一次使用官方模型名称时，Ultralytics 可能联网下载权重；也可以把 `--model` 指向本地权重。

## 数据与训练

```bash
python voc_to_yolo.py dataset/annotations dataset/labels --classes person,car,motorcycle,bus,truck
python train.py dataset.yaml --model yolo11n.pt --epochs 50
python train.py dataset.yaml --model runs/detect/train/weights/best.pt --validate-only
```

仓库不包含课程视频、个人数据、训练集和大模型权重。`dataset.example.yaml` 展示所需目录结构。

## 测试

```bash
python -m unittest discover -s tests -v
```

单元测试不加载深度学习模型，覆盖真实跨线、初次出现在线下不误计数，以及 VOC 坐标归一化。

## 项目边界

计数准确率受摄像机角度、遮挡、检测模型和跟踪 ID 稳定性影响；当前实现采用单条水平线，不处理复杂路口、多摄像头重识别或正式交通统计标定。该项目定位为实训基础实践，不是生产级交通系统。
