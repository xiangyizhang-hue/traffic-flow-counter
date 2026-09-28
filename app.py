import argparse
import json
from pathlib import Path

from counter import LineCounter


def process_video(source: str, output: str, model_name: str = "yolo11n.pt",
                  line_position: float = 0.65, confidence: float = 0.35) -> dict:
    if not 0.05 <= line_position <= 0.95:
        raise ValueError("line_position must be between 0.05 and 0.95")

    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    line_y = int(height * line_position)
    counter = LineCounter(line_y=line_y, hysteresis=max(3, height * 0.005))
    model = YOLO(model_name)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create output video: {output}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence, verbose=False)[0]
            annotated = result.plot()
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                for track_id, class_id, (x1, y1, x2, y2) in zip(ids, classes, boxes):
                    center_y = (y1 + y2) / 2.0
                    counter.update(track_id, class_id, center_y, frame_index)

            cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 255), 2)
            cv2.putText(annotated, f"Person: {counter.counts['person']}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Vehicle: {counter.counts['vehicle']}", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 0), 2)
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    summary = {**counter.counts, "frames": frame_index, "output": str(output_path)}
    output_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack traffic flow counter")
    parser.add_argument("source", help="input video path")
    parser.add_argument("--output", default="runs/output.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--line-position", type=float, default=0.65)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    print(json.dumps(process_video(args.source, args.output, args.model,
                                   args.line_position, args.confidence), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
