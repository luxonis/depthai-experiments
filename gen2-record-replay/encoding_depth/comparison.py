import numpy as np

# Returns a mask where both frames have pixels above some threshold
def get_union_mask(
    frame1: np.ndarray, frame2: np.ndarray, threshold: int
) -> np.ndarray:
    # First check that dimensions are the same
    assert frame1.shape == frame2.shape, "Frames must have the same dimensions"
    # Create a mask where both frames have pixels above some threshold
    mask1 = frame1 > threshold
    mask2 = frame2 > threshold
    return np.logical_and(mask1, mask2)


def mse(imageA, imageB, mask=None) -> float:
    # Compute the Mean Squared Error between two images, use the mask if provided
    if mask is None:
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    else:
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2 * mask)
    # Normalize for the number of pixels (considering the mask AND the image dimensions)
    number_of_valid_pixels_in_mask = (
        np.sum(mask) if mask is not None else imageA.shape[0] * imageA.shape[1]
    )
    colors_per_pixel = imageA.shape[2] if len(imageA.shape) > 2 else 1
    return err / (number_of_valid_pixels_in_mask * colors_per_pixel)


def psnr(imageA, imageB, max_pixel_value, mask=None) -> float:
    # Compute the PSNR
    mean_squared_err = mse(imageA, imageB, mask)
    if mean_squared_err == 0:
        return float("inf")
    return 10 * np.log10(max_pixel_value**2 / mean_squared_err)


def average_difference(imageA, imageB, mask=None) -> float:
    # Use the mask and null the difference where the depth was invalidated
    difference = np.abs(imageA.astype(np.float32) - imageB.astype(np.float32))
    if mask is not None:
        # Check that mask dimensions match image dimensions
        if mask.shape != imageA.shape:
            raise ValueError("Mask dimensions must match image dimensions")
        difference[mask == False] = 0
        average_difference = np.sum(difference) / np.sum(mask)
    else:
        average_difference = np.sum(difference) / (imageA.shape[0] * imageA.shape[1])
    return average_difference


def max_difference(imageA, imageB, mask=None) -> float:
    # Use the mask and null the difference where the depth was invalidated
    difference = np.abs(imageA.astype(np.float32) - imageB.astype(np.float32))
    if mask is not None:
        # Check that mask dimensions match image dimensions
        if mask.shape != imageA.shape:
            raise ValueError("Mask dimensions must match image dimensions")
        difference[mask == False] = 0
        max_difference = np.max(difference)
    else:
        max_difference = np.max(difference)
    return max_difference


def print_all_metrics(imageA, imageB, max_pixel_value, mask=None):
    # Compute all metrics and print them
    psnr_val = psnr(imageA, imageB, max_pixel_value, mask)
    average_difference_val = average_difference(imageA, imageB, mask)
    mse_val = mse(imageA, imageB, mask)
    max_difference_val = max_difference(imageA, imageB, mask)
    original_fillrate = np.sum(imageA > 0) / (imageA.shape[0] * imageA.shape[1])
    if mask is None:
        compressed_fillrate = np.sum(imageB > 0) / (imageB.shape[0] * imageB.shape[1])
    else:
        compressed_fillrate = np.sum(mask) / (imageA.shape[0] * imageA.shape[1])
    relative_fillrate = compressed_fillrate / original_fillrate
    matrix_json = {
        "psnr": psnr_val,
        "mse": mse_val,
        "average_difference": average_difference_val,
        "max_difference": max_difference_val,
        "original_fillrate": original_fillrate,
        "compressed_fillrate": compressed_fillrate,
        "relative_fillrate": relative_fillrate,
    }
    print(
        f"""
PSNR: {psnr_val}, MSE: {mse_val},
Average difference: {average_difference_val}, Max difference: {max_difference_val}
MSE: {mse_val}, Max difference: {max_difference_val}
Original fillrate: {original_fillrate}, Compressed fillrate: {compressed_fillrate},
Relative fillrate: {relative_fillrate}
        """
    )
    return matrix_json


if __name__ == "__main__":
    # Assuming 8-bit image, so max_pixel_value is 255
    max_pixel_value = 255

    # Dummy images for demonstration; replace these with actual numpy arrays of your images
    imageA = np.array([[255, 255, 253], [0, 0, 0], [255, 255, 255]], dtype=np.uint8)

    imageB = np.array([[255, 255, 255], [0, 0, 0], [255, 0, 255]], dtype=np.uint8)

    mask = np.array([[True, True, True], [False, False, False], [True, False, True]])

    # Compute all metrics and print them
    print_all_metrics(imageA, imageB, max_pixel_value, mask)
