# 人工智能实训流程笔记

本项目对应中国移动“梧桐·鸿鹄”人工智能方向实训中的基础实践。目标是走通一条完整但可控的计算机视觉流程：整理数据、转换标注、训练与验证目标检测模型、在视频中保持目标 ID、判断目标是否真正跨线，最后通过网页界面展示结果。

## 1. 数据整理

训练集通常分为 `train`、`val` 两部分。训练集用于更新模型参数，验证集只用于评估，不能把同一张图片同时放入两部分，否则指标会虚高。每张图片应有同名标注文件，并检查类别名称、空标注、越界框和重复图片。

建议目录结构：

```text
dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

仓库中的 `dataset.example.yaml` 是配置示例。真实数据集、课程视频和个人资料不进入 Git 仓库。

## 2. VOC 标注转换为 YOLO 格式

Pascal VOC 使用 XML 保存像素坐标 `(xmin, ymin, xmax, ymax)`；YOLO 每行使用：

```text
class_id center_x center_y width height
```

后四项都除以图像宽高，归一化到 0～1。`voc_to_yolo.py` 会裁剪越界坐标、跳过空框，并按给定类别顺序生成类别编号：

```bash
python voc_to_yolo.py dataset/annotations dataset/labels \
  --classes person,car,motorcycle,bus,truck
```

转换后应抽查若干图片，将框画回原图确认类别和坐标无误。只看脚本无报错并不能证明标注正确。

## 3. 训练与验证

`train.py` 通过 Ultralytics 的公开接口加载模型。基础训练示例：

```bash
python train.py dataset.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

训练时重点关注：

- `box_loss`、`cls_loss` 是否总体下降；
- Precision 表示预测为目标的结果中有多少是正确的；
- Recall 表示真实目标中有多少被检出；
- mAP 是综合不同置信度阈值后的检测性能指标；
- 训练指标很好而验证指标明显较差，可能发生过拟合。

训练结束后用最佳权重单独验证：

```bash
python train.py dataset.yaml \
  --model runs/detect/train/weights/best.pt --validate-only
```

本仓库不提交 `.pt` 权重。首次使用官方模型名可能联网下载权重。

## 4. 视频检测、跟踪与计数

目标检测只回答“这一帧中哪里有车或人”。流量计数还需要知道相邻帧中的目标是不是同一个，因此 `app.py` 使用 ByteTrack 生成持续的 `track_id`。

对每个轨迹，程序记录目标框中心点相对计数线的位置：

1. 中心点在线上方，状态记为 `above`；在线下方，记为 `below`。
2. 在线附近设置迟滞区，中心点位于迟滞区时不切换状态，避免抖动。
3. 只有历史状态从 `above` 变为 `below`，或从 `below` 变为 `above`，才算真正跨线。
4. 同一个轨迹在同一方向只计数一次。
5. 每次处理视频都创建新计数器，计数不会带到下一次任务。

运行示例：

```bash
python app.py input.mp4 --output runs/result.mp4 \
  --model yolo11n.pt --line-position 0.65 --confidence 0.35
```

结果包括带检测框和计数线的 MP4，以及同名 JSON 汇总。准确率仍会受到遮挡、摄像机视角、置信度阈值和跟踪 ID 切换影响。

## 5. Gradio 展示

`webui.py` 把视频上传、参数输入、处理按钮、结果视频和统计信息组合成网页界面：

```bash
python webui.py
```

前端只负责收集输入和展示输出，真正的检测、跟踪和计数仍由 `process_video` 完成。这样命令行和网页入口复用同一套业务逻辑。

## 6. 测试与项目边界

```bash
python -m unittest discover -s tests -v
```

单元测试不下载模型，重点验证可确定的核心逻辑：真正跨线才计数、目标初次出现在线下不误计数，以及 VOC 坐标转换正确。模型在具体道路上的识别效果需要另行使用实际视频验证。

当前项目是实训基础实践：支持单条水平线和常见人车类别，不声称解决复杂路口、多摄像头重识别、速度测量或生产级交通统计标定。
