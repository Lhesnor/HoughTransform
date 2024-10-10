import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from houghtransform.hough_detector import HoughLineDetector

class TestHoughLineDetector(unittest.TestCase):
    
    def setUp(self):
        self.test_image = 'src/houghtransform/img/two_lines.png'

    @patch('cv2.imread')
    def test_read_image_success(self, mock_imread):
        mock_imread.return_value = np.ones((400, 400, 3), dtype=np.uint8)
        detector = HoughLineDetector(self.test_image)
        self.assertIsNotNone(detector.image)
        self.assertEqual(detector.image.shape, (400, 400, 3))

    @patch('cv2.imread')
    def test_read_image_failure(self, mock_imread):
        mock_imread.return_value = None
        with self.assertRaises(ValueError):
            HoughLineDetector(self.test_image)

    def test_resize_image(self):
        detector = HoughLineDetector(self.test_image)
        detector.image = np.ones((800, 800, 3), dtype=np.uint8)
        resized_image = detector.resize_image(400, 400)
        self.assertEqual(resized_image.shape, (400, 400, 3))

    def test_to_grayscale(self):
        detector = HoughLineDetector(self.test_image)
        detector.resized_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        grayscale_image = detector.to_grayscale()
        self.assertEqual(grayscale_image.shape, (400, 400))
        self.assertEqual(grayscale_image[0, 0], 255)

    def test_sobel_edge_detection(self):
        detector = HoughLineDetector(self.test_image)
        detector.grayscale_image = np.zeros((400, 400), dtype=np.uint8)
        edges = detector.sobel_edge_detection()
        self.assertEqual(edges.shape, (400, 400))

    def test_hough_transform(self):
        detector = HoughLineDetector(self.test_image)
        detector.edges = np.zeros((400, 400), dtype=np.uint8)
        lines = detector.hough_transform(100)
        self.assertIsInstance(lines, list)

    def test_adaptive_hough_threshold(self):
        detector = HoughLineDetector(self.test_image)
        detector.edges = np.zeros((400, 400), dtype=np.uint8)
        lines = detector.adaptive_hough_threshold()
        self.assertIsInstance(lines, list)

    @patch('cv2.line')
    def test_draw_lines(self, mock_line):
        detector = HoughLineDetector(self.test_image)
        lines = [(100, np.pi / 4), (200, np.pi / 6)]
        detector.resized_image = np.zeros((400, 400, 3), dtype=np.uint8)
        detector.draw_lines(lines)
        self.assertEqual(mock_line.call_count, 2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_image = sys.argv.pop()  
    else:
        test_image = 'src/houghtransform/img/two_lines.png'
    
    TestHoughLineDetector.test_image = test_image
    unittest.main()
