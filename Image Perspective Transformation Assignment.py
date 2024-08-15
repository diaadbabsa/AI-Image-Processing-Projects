import cv2
import numpy as np

class Points:
    def __init__(self):
        self.source_points = []
        self.destination_points = []

    def add_source_point(self, x, y):
        self.source_points.append((x, y))

    def set_default_destination_points(self, width, height):
        self.destination_points = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height)
        ]

    def get_source_points(self):
        return np.array(self.source_points, dtype=np.float32)

    def get_destination_points(self):
        return np.array(self.destination_points, dtype=np.float32)

class DocumentScanner:
    @staticmethod
    def calculate_transform_matrix(points):
        source_points = points.get_source_points()
        destination_points = points.get_destination_points()
        return cv2.findHomography(source_points, destination_points)[0]

    @staticmethod
    def transform_image(image, matrix, output_size):
        return cv2.warpPerspective(image, matrix, output_size)

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}")

    points = Points()

    points.add_source_point(483, 204) 
    points.add_source_point(853, 329)  
    points.add_source_point(890, 429) 
    points.add_source_point(486, 429)  
    height, width = 500, 500  
    points.set_default_destination_points(width, height)

    matrix = DocumentScanner.calculate_transform_matrix(points)
    transformed_image = DocumentScanner.transform_image(image, matrix, (width, height))

    cv2.imshow("Original Image", image)
    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("C:\\Users\\11\\Desktop\\trean\\t.jpg")
