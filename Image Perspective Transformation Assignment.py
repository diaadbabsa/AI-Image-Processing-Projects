import cv2
import numpy as np

# Define a class to store source and destination points
class Points:
    def __init__(self):
        # List to store source points
        self.source_points = []
        # List to store destination points
        self.destination_points = []

    def add_source_point(self, x, y):
        # Add a source point to the list
        self.source_points.append((x, y))

    def set_default_destination_points(self, width, height):
        # Set default destination points in a rectangular shape
        self.destination_points = [
            (0, 0),              # Top-left corner
            (width, 0),          # Top-right corner
            (width, height),     # Bottom-right corner
            (0, height)          # Bottom-left corner
        ]

    def get_source_points(self):
        # Convert the list of source points to a numpy array with float32 type
        return np.array(self.source_points, dtype=np.float32)

    def get_destination_points(self):
        # Convert the list of destination points to a numpy array with float32 type
        return np.array(self.destination_points, dtype=np.float32)

# Define a class for document scanning
class DocumentScanner:
    @staticmethod
    def calculate_transform_matrix(points):
        # Calculate the homography matrix using source and destination points
        source_points = points.get_source_points()
        destination_points = points.get_destination_points()
        return cv2.findHomography(source_points, destination_points)[0]

    @staticmethod
    def transform_image(image, matrix, output_size):
        # Apply the transformation to the image using the homography matrix
        return cv2.warpPerspective(image, matrix, output_size)

def main(image_path):
    # Read the image from the given path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to read the image at {image_path}")

    # Create a Points object
    points = Points()

    # Add source points (these should be manually specified according to the image)
    points.add_source_point(483, 204) 
    points.add_source_point(853, 329)  
    points.add_source_point(890, 429) 
    points.add_source_point(486, 429)  

    # Set default destination points based on the final size of the image
    height, width = 500, 500  
    points.set_default_destination_points(width, height)

    # Calculate the transformation matrix
    matrix = DocumentScanner.calculate_transform_matrix(points)
    # Apply the transformation to the image
    transformed_image = DocumentScanner.transform_image(image, matrix, (width, height))

    # Display the original image
    cv2.imshow("Original Image", image)
    # Display the transformed image
    cv2.imshow("Transformed Image", transformed_image)
    # Wait for a key press and then close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the main function if this script is executed
if __name__ == "__main__":
    main("C:\\Users\\11\\Desktop\\trean\\t.jpg")
