import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, ...]:
    xmin, ymin, xmax, ymax = box
    xmin, xmax = sorted((max(0.0, xmin), min(float(width), xmax)))
    ymin, ymax = sorted((max(0.0, ymin), min(float(height), ymax)))
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("invalid or empty bounding box")
    return ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height,
            (xmax - xmin) / width, (ymax - ymin) / height)


def convert_file(xml_path: Path, output_path: Path, classes: list[str]) -> int:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"missing <size> in {xml_path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size in {xml_path}")

    labels = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in classes:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        raw = tuple(float(bbox.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
        try:
            x, y, w, h = convert_box(raw, width, height)
        except ValueError:
            continue
        labels.append(f"{classes.index(name)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(labels), encoding="utf-8")
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML labels to YOLO TXT")
    parser.add_argument("annotations", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--classes", default="person,car,motorcycle,bus,truck")
    args = parser.parse_args()
    classes = [item.strip().lower() for item in args.classes.split(",") if item.strip()]
    total = 0
    for xml_path in sorted(args.annotations.glob("*.xml")):
        total += convert_file(xml_path, args.labels / f"{xml_path.stem}.txt", classes)
    print(f"converted {total} objects from {args.annotations}")


if __name__ == "__main__":
    main()
