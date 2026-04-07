import cv2
import json

class ShapeDetector:
    """
    This system detects  geometric shapes from an image.

    Instead of using pretrained AI models, it relies on classical
    computer vision techniques. This makes the pipeline lightweight
    and interpretable.

    The goal is to convert raw visual input into structured data
    suitable for training machine learning models.
    """

    def __init__(self):
        pass

    def detect_shapes(self, image_path):
        """
        Main pipeline:

        1. Load image
        2. Convert to grayscale (simplifies processing)
        3. Reduce noise (improves detection accuracy)
        4. Detect edges
        5. Find contours (object boundaries)
        6. Approximate shape geometry
        """

        # Load image from disk
        image = cv2.imread(image_path)

        # Validate that image loaded correctly
        if image is None:
            raise ValueError("Image not found. Check file path.")

        # Convert to grayscale (removes color complexity)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to smooth noise
        # This prevents false edges during detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform edge detection
        # This highlights boundaries of objects
        edges = cv2.Canny(blurred, 50, 150)

        # Extract contours (continuous boundaries)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []

        for cnt in contours:
            # Approximate contour to a polygon
            # This reduces complexity and reveals shape structure
            approx = cv2.approxPolyDP(
                cnt, 0.04 * cv2.arcLength(cnt, True), True
            )

            # Bounding box for detected shape
            x, y, w, h = cv2.boundingRect(cnt)

            shape_type = "unknown"

            """
            Shape classification logic:

            - 3 vertices → triangle
            - 4 vertices → rectangle
            - >4 vertices → circle (approximation)

            This works because geometric shapes have predictable edge counts.
            """

            if len(approx) == 3:
                shape_type = "triangle"

            elif len(approx) == 4:
                shape_type = "rectangle"

            elif len(approx) > 4:
                shape_type = "circle"

            results.append({
                "shape": shape_type,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })

        return results


# Instantiate system
detector = ShapeDetector()

# Run detection on test image
shapes = detector.detect_shapes("test.jpg")

# Save results as structured dataset
with open("output.json", "w") as f:
    json.dump(shapes, f, indent=2)

print("Shape detection pipeline complete.")