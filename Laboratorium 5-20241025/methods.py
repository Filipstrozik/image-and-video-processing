import cv2
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from typing import Any, List


def black_pill_binarizer(image: np.ndarray, display: bool = False) -> np.ndarray:
    # Convert the image to grayscale
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image = cv2.GaussianBlur(image, (5, 5), 0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Grayscale Image")
        plt.axis("off")
        plt.show()

    # Apply thresholding
    _, image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Binarized Image")
        plt.axis("off")
        plt.show()

    # applt dilation
    kernel = np.ones((9, 9), np.uint8)

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hulls = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        convex_hulls.append(hull)
    # Fill the contours
    cv2.drawContours(image, convex_hulls, -1, (255, 255, 255), thickness=cv2.FILLED)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Filled Contours")
        plt.axis("off")
        plt.show()

    image = cv2.dilate(image, kernel, iterations=1)
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Dilated Image")
        plt.axis("off")
        plt.show()

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Binarized with edges Image")
        plt.axis("off")
        plt.show()

    # get hue from hsv image
    image_h = image_hsv[:, :, 0]

    lower_bound = 8
    upper_bound = 19

    hue_mask = cv2.inRange(image_h, lower_bound, upper_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(hue_mask, cmap="gray")
        plt.title("Hue Mask")
        plt.axis("off")
        plt.show()

    kernel = np.ones((9, 9), np.uint8)
    hue_mask = cv2.morphologyEx(hue_mask, cv2.MORPH_CLOSE, kernel)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(hue_mask, cmap="gray")
        plt.title("Hue Mask closed")
        plt.axis("off")
        plt.show()

    image = cv2.bitwise_and(image, hue_mask)

    # Apply erosion
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Eroded Image")
        plt.axis("off")
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title("Contours")
        plt.axis("off")
        plt.show()

    # convex hull
    convex_hulls = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        convex_hulls.append(hull)

    # filter convex hulls
    min_area = 200  # Minimum area threshold
    filtered_hulls = [hull for hull in convex_hulls if cv2.contourArea(hull) > min_area]
    convex_hulls = filtered_hulls

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title("Convex Hulls")
        plt.axis("off")
        plt.show()

    return convex_hulls


def yellow_pills_binarizer(image: np.ndarray, display: bool = False) -> np.ndarray:

    image_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    image_hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]

    lower_bound = 22
    upper_bound = 30

    # Create a binary mask for hue values between 25 and 29
    hue_mask = cv2.inRange(image_hue, lower_bound, upper_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(hue_mask, cmap="gray")
        plt.title("Hue Mask")
        plt.colorbar()
        plt.show()

    kernel = np.ones((5, 5), np.uint8)

    hue_mask = cv2.erode(hue_mask, kernel, iterations=2)

    hue_mask = cv2.dilate(hue_mask, kernel, iterations=1)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(hue_mask, cmap="gray")
        plt.title("Hue Mask closed")
        plt.colorbar()
        plt.show()

    circles = cv2.HoughCircles(
        hue_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=20,
        param1=50,
        param2=15,
        minRadius=8,
        maxRadius=30,
    )

    # If some circles are detected, draw them on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        output = np.zeros_like(image)
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        # Display the output image with detected circles
        if display:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title("Detected Circles using Hough Circle Transform")
            plt.axis("off")
            plt.show()
    else:
        print("No circles were detected")

    # Transform circles to contours
    contours = []
    if circles is not None:
        for x, y, r in circles:
            # Create a circular contour
            contour = np.array(
                [
                    [x + r * np.cos(theta), y + r * np.sin(theta)]
                    for theta in np.linspace(0, 2 * np.pi, 100)
                ],
                dtype=np.int32,
            )
            contours.append(contour)

    return contours


# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # blue organizer hue 90 - 110

# # Extract the hue channel
# hue_channel = image_hsv[..., 0]

# # Define the hue range for binarization
# lower_bound = 90
# upper_bound = 110
# # Create a binary mask for hue values between 90 and 110
# hue_mask = cv2.inRange(hue_channel, lower_bound, upper_bound)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(hue_mask, cmap="gray")
#     plt.title("Hue Mask")
#     plt.colorbar()
#     plt.show()

# # Find contours
# contours, _ = cv2.findContours(hue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # filter contours by area

# contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

# if display:
#     contour_image = np.zeros_like(image)
#     cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)

# # Convex hull
# convex_hulls = [cv2.convexHull(contour) for contour in contours]

# if display:
#     contour_image = np.zeros_like(image)
#     cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)

# # all pixels outside the convex hull are set to 0
# mask = np.zeros_like(image)
# cv2.drawContours(mask, convex_hulls, -1, (255, 255, 255), -1)

# image = cv2.bitwise_and(image, mask)

# # print img color model

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image[..., ::-1])
#     plt.title("Filtered Image")
#     plt.show()

# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# # Extract the 'b*' channel from the Lab image
# b_channel = image_lab[..., 2]
# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(b_channel, cmap="gray")
#     plt.title("B Channel")
#     plt.colorbar()
#     plt.show()

# # apply thresholding 182
# threshold_value = 182
# _, image_b = cv2.threshold(b_channel, threshold_value, 255, cv2.THRESH_BINARY)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image_b, cmap="gray")
#     plt.title(f"Binarized B Channel {threshold_value}")
#     plt.colorbar()
#     plt.show()
# # image_combined = image_b
# # Extract the hue channel

# hue_channel = image_hsv[..., 0]

# # Define the hue range for binarization
# lower_bound = 25
# upper_bound = 29

# # Create a binary mask for hue values between 20 and 40
# hue_mask = cv2.inRange(hue_channel, lower_bound, upper_bound)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(hue_mask, cmap="gray")
#     plt.title("Hue Mask")
#     plt.colorbar()
#     plt.show()

# # # make a bitwise NAND of otsu_thresh and edges
# image_combined = cv2.bitwise_or(image_b, hue_mask)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image_combined, cmap="gray")
#     plt.title("Combined Image")
#     plt.colorbar()
#     plt.show()

# # Apply erosion
# erosion_kernel = np.ones((7, 7), np.uint8)
# image_combined = cv2.erode(image_combined, erosion_kernel, iterations=1)
# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image_combined, cmap="gray")
#     plt.title("Eroded Image")
#     plt.colorbar()
#     plt.show()

# # Apply dilation
# dilation_kernel = np.ones((7, 7), np.uint8)
# image_combined = cv2.dilate(image_combined, dilation_kernel, iterations=1)
# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image_combined, cmap="gray")
#     plt.title("Dilated Image")
#     plt.colorbar()
#     plt.show()

# # edges

# blur = cv2.GaussianBlur(image, (5, 5), 0)
# edges = cv2.Canny(blur, 120, 190)

# # perform CLOSE operation on edges
# edges_kernel = np.ones((3, 3), np.uint8)
# edges = cv2.dilate(edges, edges_kernel, iterations=1)

# edges = cv2.bitwise_not(edges)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(edges, cmap="gray")
#     plt.title("Edges")
#     plt.colorbar()
#     plt.show()

# image = cv2.bitwise_and(image_combined, edges)

# if display:
#     plt.figure(figsize=(12, 8))
#     plt.imshow(image, cmap="gray")
#     plt.title("Final Image")
#     plt.colorbar()
#     plt.show()

# # Find contours
# contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# if display:
#     contour_image = np.zeros_like(image)
#     cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#     plt.title("Contours")
#     plt.colorbar()
#     plt.show()

# # convex hull
# convex_hulls = []
# for contour in contours:
#     hull = cv2.convexHull(contour)
#     convex_hulls.append(hull)

# # filter contours by area
# convex_hulls = [hull for hull in convex_hulls if cv2.contourArea(hull) > 300]

# if display:
#     contour_image = np.zeros_like(image)
#     cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)
#     plt.figure(figsize=(12, 8))
#     plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#     plt.title("Convex Hulls")
#     plt.colorbar()
#     plt.show()

# return convex_hulls


# BLUE PILLS
def blue_pills_binarizer(image: np.ndarray, display: bool = False) -> List[np.ndarray]:
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the S channel
    s_lower_bound = 0
    s_upper_bound = 28

    # Create a binary mask for S channel values between 0 and 30
    s_channel = image_hsv[:, :, 1]
    binary_mask_s = cv2.inRange(s_channel, s_lower_bound, s_upper_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_s, cmap="gray")
        plt.title("Binary Mask for S Channel (0 to 28)")
        plt.colorbar()
        plt.show()

    # Define the lower and upper bounds for the H channel
    h_lower_bound = 90
    h_upper_bound = 120

    # Create a binary mask for H channel values between 90 and 120
    h_channel = image_hsv[:, :, 0]
    binary_mask_h = cv2.inRange(h_channel, h_lower_bound, h_upper_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_h, cmap="gray")
        plt.title("Binary Mask for H Channel (90 to 120)")
        plt.colorbar()
        plt.show()

    # Combine the masks using bitwise AND
    combined_mask = cv2.bitwise_and(binary_mask_s, binary_mask_h)

    # make a canny edge detection
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 150)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(edges, cmap="gray")
        plt.title("Edges detected using Canny filter")
        plt.colorbar()
        plt.show()

    # Apply bitwise AND operation with the edges
    edges = cv2.bitwise_not(edges)

    combined_mask = cv2.bitwise_and(combined_mask, edges)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask for S (0 to 30) and H (90 to 120) Channels")
        plt.colorbar()
        plt.show()

    # Apply erosion to the combined mask
    kernel = np.ones((7, 7), np.uint8)
    eroded_mask = cv2.erode(combined_mask, kernel, iterations=1)

    # Find contours in the eroded mask
    contours, _ = cv2.findContours(
        eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by area
    filtered_contours = [
        contour for contour in contours if cv2.contourArea(contour) > 1000
    ]

    if display:
        contour_image = np.zeros_like(eroded_mask)
        cv2.drawContours(contour_image, filtered_contours, -1, (255, 255, 255), 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Filtered Contours")
        plt.colorbar()
        plt.show()

    # Map contours to convex hulls
    convex_hulls = [cv2.convexHull(contour) for contour in filtered_contours]

    if display:
        hull_image = np.zeros_like(eroded_mask)
        cv2.drawContours(hull_image, convex_hulls, -1, (255, 255, 255), 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(hull_image, cmap="gray")
        plt.title("Convex Hulls")
        plt.colorbar()
        plt.show()

    return convex_hulls


# BIG WHITE PILLS
def big_white_pills_binarize(image: np.ndarray, display: bool = False) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_h = image[..., 0]
    lower_hue_bound = 8
    upper_hue_bound = 18

    # Create a binary mask for hue values between 20 and 40
    binary_mask_h = cv2.inRange(image_h, lower_hue_bound, upper_hue_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_h, cmap="gray")
        plt.title("Binarized Hue Channel")
        plt.colorbar()
        plt.show()

    image_s = image[..., 1]
    lower_saturation_bound = 0
    upper_saturation_bound = 65

    # Create a binary mask for hue values between 20 and 40
    binary_mask_s = cv2.inRange(image_s, lower_saturation_bound, upper_saturation_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_s, cmap="gray")
        plt.title("Binarized Saturation Channel")
        plt.colorbar()
        plt.show()

    image_v = image[..., 2]
    lower_value_bound = 150
    upper_value_bound = 255

    # Create a binary mask for hue values between 20 and 40
    binary_mask_v = cv2.inRange(image_v, lower_value_bound, upper_value_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_v, cmap="gray")
        plt.title("Binarized Value Channel")
        plt.colorbar()
        plt.show()

    # Combine the masks using bitwise AND

    combined_mask = cv2.bitwise_and(binary_mask_h, binary_mask_v)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask h i v")
        plt.colorbar()
        plt.show()

    combined_mask = cv2.bitwise_and(combined_mask, binary_mask_s)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask h s v")
        plt.colorbar()
        plt.show()

    # Apply erosion
    erosion_kernel = np.ones((7, 7), np.uint8)
    image = cv2.erode(combined_mask, erosion_kernel, iterations=2)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Eroded Image")
        plt.colorbar()
        plt.show()

    # Apply dilation
    dilation_kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, dilation_kernel, iterations=1)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Dilated Image")
        plt.colorbar()
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Contours")
        plt.colorbar()
        plt.show()

    # Convex hull

    convex_hulls = [cv2.convexHull(contour) for contour in contours]

    # filter contours by area
    convex_hulls = [hull for hull in convex_hulls if cv2.contourArea(hull) > 2500]

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Convex Hulls")
        plt.colorbar()
        plt.show()

    return convex_hulls


# small white pills
def small_white_pills_binarize(image: np.ndarray, display: bool = False) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_h = image[..., 0]
    # Define the hue range for binarization
    lower_bound = 90
    upper_bound = 110
    # Create a binary mask for hue values between 90 and 110
    hue_mask = cv2.inRange(image_h, lower_bound, upper_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(hue_mask, cmap="gray")
        plt.title("Hue Mask")
        plt.colorbar()
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(hue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours by area

    contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)

    # Convex hull
    convex_hulls = [cv2.convexHull(contour) for contour in contours]

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)

    # all pixels outside the convex hull are set to 0
    mask = np.zeros_like(image)
    cv2.drawContours(mask, convex_hulls, -1, (255, 255, 255), -1)

    image = cv2.bitwise_and(image, mask)

    # print img color model

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.title("Filtered Image")
        plt.show()

    image_h = image[..., 0]

    lower_hue_bound = 8
    upper_hue_bound = 18

    # Create a binary mask for hue values between 20 and 40
    binary_mask_h = cv2.inRange(image_h, lower_hue_bound, upper_hue_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_h, cmap="gray")
        plt.title("Binarized Hue Channel")
        plt.colorbar()
        plt.show()

    image_s = image[..., 1]
    lower_saturation_bound = 0
    upper_saturation_bound = 65

    # Create a binary mask for hue values between 20 and 40
    binary_mask_s = cv2.inRange(image_s, lower_saturation_bound, upper_saturation_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_s, cmap="gray")
        plt.title("Binarized Saturation Channel")
        plt.colorbar()
        plt.show()

    image_v = image[..., 2]
    lower_value_bound = 150
    upper_value_bound = 255

    # Create a binary mask for hue values between 20 and 40
    binary_mask_v = cv2.inRange(image_v, lower_value_bound, upper_value_bound)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(binary_mask_v, cmap="gray")
        plt.title("Binarized Value Channel")
        plt.colorbar()
        plt.show()

    # Combine the masks using bitwise AND

    combined_mask = cv2.bitwise_and(binary_mask_h, binary_mask_v)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask h i v")
        plt.colorbar()
        plt.show()

    combined_mask = cv2.bitwise_and(combined_mask, binary_mask_s)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask h i v s")
        plt.colorbar()
        plt.show()

    # detect edges
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 150)

    # dilation
    dilation_kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, dilation_kernel, iterations=1)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(edges, cmap="gray")
        plt.title("Edges detected using Canny filter")
        plt.colorbar()
        plt.show()

    # Apply bitwise AND operation with the edges
    edges = cv2.bitwise_not(edges)

    combined_mask = cv2.bitwise_and(combined_mask, edges)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(combined_mask, cmap="gray")
        plt.title("Combined Binary Mask with Edges")
        plt.colorbar()
        plt.show()

    # Apply erosion
    erosion_kernel = np.ones((7, 7), np.uint8)
    image = cv2.erode(combined_mask, erosion_kernel, iterations=2)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Eroded Image")
        plt.colorbar()
        plt.show()

    # Apply dilation
    dilation_kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, dilation_kernel, iterations=2)

    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap="gray")
        plt.title("Dilated Image")
        plt.colorbar()
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if display:
        for contour in contours:
            print(cv2.contourArea(contour))

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Contours")
        plt.colorbar()
        plt.show()

    # Convex hull

    convex_hulls = [cv2.convexHull(contour) for contour in contours]

    # filter contours by area
    convex_hulls = [hull for hull in convex_hulls if 240 < cv2.contourArea(hull) < 1700]

    # print every area of convex hull
    if display:
        for hull in convex_hulls:
            print(cv2.contourArea(hull))

    if display:
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, convex_hulls, -1, (255, 255, 0), 3)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Convex Hulls")
        plt.colorbar()
        plt.show()

    symmetry_tolerance = 1.5
    ellipses = []
    filtered_hulls = []
    for i, contour in enumerate(convex_hulls):
        if len(contour) >= 5:  # fitEllipse requires at least 5 points
            # Extract the major and minor axes lengths
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse

            # Calculate the axis ratio
            axis_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

            if axis_ratio < symmetry_tolerance:
                ellipses.append(ellipse)
                filtered_hulls.append(contour)

    if display:
        contour_image = np.zeros_like(image)
        for ellipse in ellipses:
            cv2.ellipse(contour_image, ellipse, (255, 255, 0), 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(contour_image, cmap="gray")
        plt.title("Fitted Ellipses")
        plt.colorbar()
        plt.show()

    return filtered_hulls
