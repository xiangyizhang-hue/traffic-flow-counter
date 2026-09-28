import tempfile
import unittest
from pathlib import Path

from counter import LineCounter
from voc_to_yolo import convert_box, convert_file


class CounterTests(unittest.TestCase):
    def test_counts_real_downward_crossing_once(self):
        counter = LineCounter(100, hysteresis=5)
        self.assertIsNone(counter.update(7, 2, 80, 0))
        self.assertEqual(counter.update(7, 2, 106, 1), "down")
        self.assertIsNone(counter.update(7, 2, 130, 2))
        self.assertEqual(counter.counts["vehicle"], 1)
        self.assertEqual(counter.counts["down"], 1)

    def test_does_not_count_object_first_seen_below_line(self):
        counter = LineCounter(100, hysteresis=5)
        counter.update(9, 0, 130, 0)
        counter.update(9, 0, 140, 1)
        self.assertEqual(counter.counts["person"], 0)

    def test_voc_conversion(self):
        self.assertEqual(convert_box((10, 20, 30, 60), 100, 100), (0.2, 0.4, 0.2, 0.4))
        xml = """<annotation><size><width>100</width><height>100</height></size>
        <object><name>car</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>
        </annotation>"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.xml"
            target = Path(tmp) / "sample.txt"
            source.write_text(xml, encoding="utf-8")
            self.assertEqual(convert_file(source, target, ["person", "car"]), 1)
            self.assertTrue(target.read_text(encoding="utf-8").startswith("1 0.200000"))


if __name__ == "__main__":
    unittest.main()
