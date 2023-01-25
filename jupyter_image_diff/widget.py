from dataclasses import dataclass, field
from typing import Sequence

import ipywidgets as widgets
import numpy as np
from ipycanvas import Canvas


def ensure_correct_image_shape(image: np.ndarray):
    """This function ensures that the shape is always (F, H, W, C), where F is the
    number of frames, H is height, W is width, and C is the number of channels. The
    number of channels can either be 3 (RGB) or 4 (RGBA).

    Allowed input shapes are:
    - (H, W)        # Just a 2D matrix
    - (H, W, 3)     # A 2D matrix with RGB channels
    - (H, W, 4)     # A 2D matrix with RGBA channels
    - (F, H, W)     # A set of frames of 2D matrices
    - (F, H, W, 3)  # A set of frames of 2D matrices with RGB channels
    - (F, H, W, 4)  # A set of frames of 2D matrices with RGBA channels

    After ensuring correct image shape the shape will be (F, H, W, C). Note that if the
    width is 3 or 4 it may be confused with the channels dimension. If your image is
    3 or 4 pixels wide, you will have to add a channel dimension yourself.
    """
    # Check if image has channel data
    if image.shape[-1] not in (3, 4):
        # Add channel data
        image = np.stack([image, image, image], axis=-1)

    # Check if image has frame data
    if not image.ndim == 4:
        # Add frame dimension
        image = np.expand_dims(image, 0)

    assert (
        image.ndim == 4
    ), "Invalid image shape. See ensure_correct_image_shape docstring for valid image shapes."
    return image


def normalize(image: np.ndarray):
    """Normalize image to the range [0, 255]"""
    image = image - image.min()
    image = image / image.max()
    return image * 255


@dataclass
class Comparison:
    """A class to compare numpy array images in a Jupyter notebook. See example below.

    Create some random images to compare:
    >>> import numpy as np
    >>> images = [np.random.uniform(0, 255, (10, 10)) for _ in range(3)]

    Create a comparison widget and display it:
    >>> from jupyter_image_diff.widget import Comparison
    >>> comp = Comparison(images)
    >>> comp.widget  # This will display the widget if run in a notebook

    You can programmatically go the the previous or next image. If you run this in a
    different cell than the one above you will see that the widget is re-rendered:
    >>> comp.next_image()

    Update the second image:
    >>> new_image = np.random.uniform(0, 255, (10, 10))
    >>> comp.update_image(1, new_image)  # 1 is the index of the second image
    """

    images: Sequence[np.ndarray]
    state: dict = field(
        default_factory=lambda: {
            "image": 0,
            "frame": 0,
            "prev_image": 1,  # Used for diffing
            "diffing": False,
        }
    )

    canvas_width: int = 300
    canvas_height: int = 300

    def __post_init__(self):
        # Set up a hidden canvas and draw each image to be compared
        self.canvases = []
        for im in self.images:
            im = ensure_correct_image_shape(normalize(im))
            num_frames, height, width, num_channels = im.shape
            canvas = Canvas(width=width, height=height)
            canvas.put_image_data(im[self.state["frame"]])
            self.canvases.append(canvas)

        # Set up the main canvas that is displayed
        self.main_canvas = Canvas(
            width=self.canvas_width,
            height=self.canvas_height,
            layout=widgets.Layout(
                width=f"{self.canvas_width}px", height=f"{self.canvas_height}px"
            ),
        )

        # Set up canvas that will be used for diffing images
        self.diffing_canvas = Canvas()

        # Set up various UI widgets and hook up events
        def key_down_handler(key: str, shift_key: bool, ctrl_key: bool, meta_key: bool):
            if key == "ArrowRight":
                self.next_image()
            elif key == "ArrowLeft":
                self.prev_image()
            elif key == "d":
                self.toggle_diff()

        self.main_canvas.on_key_down(key_down_handler)

        self.image_label = widgets.Label()

        # UI buttons
        self.prev_button = widgets.Button(description="Previous image [←]")
        self.prev_button.on_click(lambda *_: self.prev_image())
        self.next_button = widgets.Button(description="Next image [→]")
        self.next_button.on_click(lambda *_: self.next_image())
        self.toggle_diff_button = widgets.Button(description="Toggle diff [d]")
        self.toggle_diff_button.on_click(lambda *_: self.toggle_diff())

        self.draw_canvas()

    def update_image(self, index: int, new_image: np.ndarray):
        "Set the image at index to new_image and redraw the canvas."
        new_image = ensure_correct_image_shape(normalize(new_image))
        self.images[index] = new_image
        self.canvases[index].put_image_data(new_image[self.state["frame"]])
        self.draw_canvas()

    def next_image(self):
        self.state["prev_image"] = self.state["image"]
        self.state["image"] = (self.state["image"] + 1) % len(self.canvases)
        self.draw_canvas()

    def prev_image(self):
        self.state["prev_image"] = self.state["image"]
        self.state["image"] = (self.state["image"] - 1 + len(self.canvases)) % len(
            self.canvases
        )
        self.draw_canvas()

    def toggle_diff(self):
        """Enable/disable diffing between the current image and the previous image.

        The previous immage is the image that was displayed before the current image."""
        self.state["diffing"] = not self.state["diffing"]
        self.draw_canvas()

    def draw_canvas(self):
        if self.state["diffing"]:
            im1 = ensure_correct_image_shape(
                normalize(self.images[self.state["image"]])
            )
            im2 = ensure_correct_image_shape(
                normalize(self.images[self.state["prev_image"]])
            )
            diff = (im1[self.state["frame"]] - im2[self.state["frame"]] + 255) / 2
            self.diffing_canvas.height = diff.shape[0]
            self.diffing_canvas.width = diff.shape[1]
            self.diffing_canvas.put_image_data(diff)
            copy_canvas = self.diffing_canvas
            self.image_label.value = f"Diffing images {self.state['image']+1} and {self.state['prev_image']+1}"
        else:
            copy_canvas = self.canvases[self.state["image"]]
            self.image_label.value = (
                f"Image {self.state['image']+1} of {len(self.images)}"
            )

        self.main_canvas.draw_image(
            copy_canvas, 0, 0, self.main_canvas.width, self.main_canvas.height
        )

    @property
    def widget(self):
        "Get the widget that can be rendered in a notebook."
        return widgets.VBox(
            [
                self.image_label,
                self.main_canvas,
                widgets.HBox(
                    [
                        self.prev_button,
                        self.next_button,
                    ]
                ),
                self.toggle_diff_button,
            ],
        )
