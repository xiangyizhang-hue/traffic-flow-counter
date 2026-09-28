from pathlib import Path
from uuid import uuid4

from app import process_video


def run_video(video_path: str, line_position: float, confidence: float, model_name: str):
    if not video_path:
        raise ValueError("请先上传视频")
    output = Path("runs") / f"result-{uuid4().hex[:8]}.mp4"
    summary = process_video(video_path, str(output), model_name, line_position, confidence)
    return str(output), summary


def build_demo():
    import gradio as gr

    with gr.Blocks(title="高速车流量计数器") as demo:
        gr.Markdown("# 高速车流量计数器\n上传视频后，系统使用 YOLO 检测、ByteTrack 跟踪并按跨线事件计数。")
        with gr.Row():
            source = gr.Video(label="输入视频")
            result = gr.Video(label="处理结果")
        with gr.Row():
            line_position = gr.Slider(0.1, 0.9, value=0.65, step=0.01, label="计数线位置")
            confidence = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="检测置信度")
        model_name = gr.Textbox(value="yolo11n.pt", label="模型路径或名称")
        run_button = gr.Button("开始处理", variant="primary")
        summary = gr.JSON(label="统计结果")
        run_button.click(run_video, [source, line_position, confidence, model_name], [result, summary])
    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
