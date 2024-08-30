import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from umie_datasets.config.dataset_config import MaskColor

color_palette = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Pink
        (128, 255, 0),  # Lime
        (0, 255, 128),  # Spring Green
        (128, 128, 255),# Light Blue
        (255, 128, 128),# Light Pink
        (128, 255, 128),# Light Green
        (192, 192, 192) # Silver
    ]

def visualize_segmentation_mask(image_np : np.array, mask_np: np.array, masks:  dict[str, MaskColor]) ->str:
    """
    Visualize a segmentation mask overlaid on an image.

    This function takes a grayscale image and its corresponding segmentation mask,
    along with a dictionary of mask colors, and creates a visualization where each
    segmented region is colored according to the provided color palette.

    Args:
        image_np (np.array): A 2D numpy array representing the grayscale image.
        mask_np (np.array): A 2D numpy array representing the segmentation mask.
        masks (dict[str, MaskColor]): A dictionary mapping mask names to MaskColor objects.
            Each MaskColor object should have a 'target_color' attribute.

    Returns:
        str: The file path of the saved visualization image.

    The function performs the following steps:
    1. Converts the grayscale image to RGB.
    2. Applies colors from the color palette to the segmented regions.
    3. Creates a matplotlib figure with the colored segmentation overlay.
    4. Adds a legend to identify each segmented region.
    5. Saves the figure to a temporary file and returns its path.

    Note:
        This function uses a predefined color palette and matplotlib for visualization.
        The resulting image is saved as a temporary PNG file.
    """
    import tempfile
    import os

    backtorgb = cv.cvtColor(image_np, cv.COLOR_GRAY2RGB)
    for i, (name, color) in enumerate(masks.items()):
        backtorgb[mask_np == color.target_color] = color_palette[i]

    plt.figure(figsize=(10, 8))
    plt.imshow(backtorgb)

    # Create legend patches
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=tuple(c/255 for c in color_palette[i])) for i, (name, _) in enumerate(masks.items())]

    # Add legend
    plt.legend(legend_patches, masks.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Segmentation Mask')
    plt.axis('off')
    plt.tight_layout()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        plt.savefig(tmp_file.name)
        plt.close()
        return tmp_file.name