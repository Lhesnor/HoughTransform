import numpy as np
import cv2
import math


class HoughLineDetector:
    """
    A class for detecting lines in an image using Hough Transform.

    Attributes:
        image_path (str): The path to the image file.
        target_min_lines (int): Minimum number of lines to detect.
        target_max_lines (int): Maximum number of lines to detect.
        rho_res (float): Distance resolution of the accumulator in pixels.
        theta_res (float): Angle resolution of the accumulator in radians.
        image (ndarray): The loaded image.
        resized_image (ndarray): The resized image.
        grayscale_image (ndarray): The grayscale version of the image.
        edges (ndarray): The result of edge detection.
    """

    def __init__(
        self,
        image_path,
        target_min_lines=5,
        target_max_lines=20,
        rho_res=1,
        theta_res=np.pi / 180,
    ):
        """
        Initializes the HoughLineDetector with the given parameters.

        Args:
            image_path (str): Path to the image.
            target_min_lines (int): Minimum lines to detect.
            target_max_lines (int): Maximum lines to detect.
            rho_res (float): Distance resolution in pixels.
            theta_res (float): Angle resolution in radians.
        """
        self.image_path = image_path
        self.target_min_lines = target_min_lines
        self.target_max_lines = target_max_lines
        self.rho_res = rho_res
        self.theta_res = theta_res
        self.image = self.read_image()
        self.resized_image = self.resize_image(400, 400)
        self.grayscale_image = self.to_grayscale()
        self.edges = self.sobel_edge_detection()

    def read_image(self):
        """
        Reads an image from the specified file path.

        Raises:
            ValueError: If the image could not be loaded.

        Returns:
            ndarray: The loaded image.
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError(f"Image at {self.image_path} could not be loaded.")
        return img

    def resize_image(self, new_width, new_height):
        """
        Resizes the image to the specified dimensions.

        Args:
            new_width (int): The target width of the image.
            new_height (int): The target height of the image.

        Returns:
            ndarray: The resized image.
        """
        height, width = self.image.shape[:2]
        resized = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        x_ratio = width / new_width
        y_ratio = height / new_height

        for i in range(new_height):
            for j in range(new_width):
                x = int(j * x_ratio)
                y = int(i * y_ratio)
                resized[i, j] = self.image[y, x]

        return resized

    def to_grayscale(self):
        """
        Converts the resized image to grayscale.

        Returns:
            ndarray: The grayscale image.
        """
        height, width = self.resized_image.shape[:2]
        grayscale = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                r, g, b = self.resized_image[i, j]
                grayscale[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)

        return grayscale

    def sobel_edge_detection(self):
        """
        Detects edges in the grayscale image using the Sobel operator.

        Returns:
            ndarray: The edges of the image.
        """
        height, width = self.grayscale_image.shape
        edges = np.zeros((height, width), dtype=np.uint8)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                gx = np.sum(
                    sobel_x * self.grayscale_image[i - 1 : i + 2, j - 1 : j + 2]
                )
                gy = np.sum(
                    sobel_y * self.grayscale_image[i - 1 : i + 2, j - 1 : j + 2]
                )
                gradient = math.sqrt(gx**2 + gy**2)
                edges[i, j] = min(255, int(gradient))

        return edges

    def hough_transform(self, threshold):
        """
        Performs the Hough Transform to detect lines in the edge-detected image.

        Args:
            threshold (int): The threshold to use in the Hough Transform.

        Returns:
            list: A list of detected lines represented as (rho, theta).
        """
        height, width = self.edges.shape
        diag_len = int(np.sqrt(height**2 + width**2))
        rhos = np.arange(-diag_len, diag_len, self.rho_res)
        thetas = np.arange(0, np.pi, self.theta_res)
        accumulator = np.zeros((2 * diag_len, len(thetas)), dtype=np.int32)

        for y in range(height):
            for x in range(width):
                if self.edges[y, x] > 0:
                    for theta_index in range(len(thetas)):
                        theta = thetas[theta_index]
                        rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                        accumulator[rho, theta_index] += 1

        lines = []
        for rho_index in range(accumulator.shape[0]):
            for theta_index in range(accumulator.shape[1]):
                if accumulator[rho_index, theta_index] >= threshold:
                    rho = rhos[rho_index]
                    theta = thetas[theta_index]
                    lines.append((rho, theta))

        return lines

    def adaptive_hough_threshold(self):
        """
        Adapts the Hough Transform threshold to find an optimal number of lines.

        Returns:
            list: A list of lines detected with the adapted threshold.
        """
        low, high = 10, 1000
        best_lines = []

        while low <= high:
            mid = (low + high) // 2
            lines = self.hough_transform(mid)
            num_lines = len(lines)

            if self.target_min_lines <= num_lines <= self.target_max_lines:
                best_lines = lines
                break

            if num_lines > self.target_max_lines:
                low = mid + 1
            else:
                high = mid - 1

        return best_lines

    def draw_lines(self, lines):
        """
        Draws detected lines on the resized image.

        Args:
            lines (list): A list of lines where each line is represented as (rho, theta).
        """
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(self.resized_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def process(self):
        """
        Runs the entire process of detecting and drawing lines on the image.
        """
        lines = self.adaptive_hough_threshold()
        self.draw_lines(lines)
        cv2.imshow("Lines Detected", self.resized_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()


def main():
    """
    Main function to create the detector and start the line detection process.
    """
    detector = HoughLineDetector("src/houghtransform/img/two_lines.png")
    detector.process()


if __name__ == "__main__":
    main()
