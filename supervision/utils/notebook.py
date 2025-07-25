from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from supervision.annotators.base import ImageType
from supervision.utils.conversion import pillow_to_cv2


def plot_image(
    image: ImageType, size: tuple[int, int] = (12, 12), cmap: str | None = "gray"
) -> None:
    """
    Plots image using matplotlib.

    Args:
        image (ImageType): The frame to be displayed ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
        size (Tuple[int, int]): The size of the plot in inches.
        cmap (str): the colormap to use for single channel images.

    Examples:
        ```python
        import cv2
        import supervision as sv

        image = cv2.imread("path/to/image.jpg")

        %matplotlib inline
        sv.plot_image(image=image, size=(16, 16))
        ```
    """
    if isinstance(image, Image.Image):
        image = pillow_to_cv2(image)

    plt.figure(figsize=size)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis("off")
    plt.show()


def plot_images_grid(
    images: list[ImageType],
    grid_size: tuple[int, int],
    titles: list[str] | None = None,
    size: tuple[int, int] = (12, 12),
    cmap: str | None = "gray",
) -> None:
    """
    Plots images in a grid using matplotlib.

    Args:
       images (List[ImageType]): A list of images as ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
       grid_size (Tuple[int, int]): A tuple specifying the number
            of rows and columns for the grid.
       titles (Optional[List[str]]): A list of titles for each image.
            Defaults to None.
       size (Tuple[int, int]): A tuple specifying the width and
            height of the entire plot in inches.
       cmap (str): the colormap to use for single channel images.

    Raises:
       ValueError: If the number of images exceeds the grid size.

    Examples:
        ```python
        import cv2
        import supervision as sv
        from PIL import Image

        image1 = cv2.imread("path/to/image1.jpg")
        image2 = Image.open("path/to/image2.jpg")
        image3 = cv2.imread("path/to/image3.jpg")

        images = [image1, image2, image3]
        titles = ["Image 1", "Image 2", "Image 3"]

        %matplotlib inline
        plot_images_grid(images, grid_size=(2, 2), titles=titles, size=(16, 16))
        ```
    """
    nrows, ncols = grid_size

    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):
            images[idx] = pillow_to_cv2(img)

    if len(images) > nrows * ncols:
        raise ValueError(
            "The number of images exceeds the grid size. Please increase the grid size"
            " or reduce the number of images."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        ax.axis("off")
    plt.show()
