import cv2
import numpy as np
from methodtools import lru_cache

from liveprint.pose import TorsoKeypoints
from liveprint.system_setup import PRCoords


def adapt_pic(print_, image, torso: TorsoKeypoints):
    """
    This function crops and warps
    :param print_: picture to be transformed and placed over the backaground
    :param image: the background image to be augmented
    :param torso: TorsoKeypoints
    :return: transformed pic
    """
    x, y, *_ = print_.shape
    transformation_matrix = cv2.getPerspectiveTransform(
        np.float32(((0, 0), (x, 0), (0, y), (x, y))),
        np.float32(
            [
                torso.left_shoulder.coords(),
                torso.right_shoulder.coords(),
                torso.left_hip.coords(),
                torso.right_hip.coords(),
            ]
        ),
    )

    ix, iy, *_ = image.shape
    dim = (iy, ix)
    return cv2.warpPerspective(print_, transformation_matrix, dim)


def read_apng(filenames):
    return [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in filenames]


def overlay_transparent(background_img, img_to_overlay_t):
    """
    @brief      Overlays a PNG animation frame onto another image.

    @param      background_img    The background image
    @param      img_to_overlay_t  The PNG frame to be placed over the background (has alpha channel)

    @return     Background image with overlay on top
    """

    background_img = background_img.copy()
    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, mask = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    h, w, _ = overlay_color.shape

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(
        background_img, background_img, mask=cv2.bitwise_not(mask)
    )
    return cv2.add(img1_bg, overlay_color)


class ProjectableRegion:
    """
    This class is used for the extraction of the region of interest (the region
    where the image is being projected on) from webcam
    image and transforming its perspective to fit the projector dimensions.
    """
    def __init__(self, coords: PRCoords, output_width, output_height):
        self._coords = coords
        self._output_width = output_width
        self._output_height = output_height

    @lru_cache
    def _transformation_matrix(self):
        pts1 = np.float32(
            (
                self._coords.ul(),
                self._coords.ur(),
                self._coords.ll(),
                self._coords.lr(),
            )
        )
        pts2 = np.float32(
            (
                (0, 0),
                (self._output_width, 0),
                (0, self._output_height),
                (self._output_width, self._output_height),
            )
        )
        return cv2.getPerspectiveTransform(pts1, pts2)

    def of(self, webcam_img):
        """
        :param webcam_img: np.array -- image from the webcam
        :return: np.array -- extracted region of interested
        """
        return cv2.warpPerspective(
            webcam_img, self._transformation_matrix, (self._output_width, self._output_height)
        )


class Apng:
    def __init__(self, frames):
        """
        :param frames: list of the animation frames List[np.array]
        """
        self.frames = frames
        self.i = 0

    def next_frame(self):
        """
        Returns next frame of the animation.
        After the last frame is obtained, the cycle starts over.
        :return: np.array -- a frame of the animation
        """
        i = self.i
        self.i = (self.i + 1) % len(self.frames)
        return self.frames[i]


def get_apng(args):
    """
    :return: List of the animation frames
    """
    with open(args.apng) as f:
        frames = []
        for path in [l.strip() for l in f.readlines()]:
            if path:
                frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    raise FrameNotFound(f"Didn't manage to find '{path}'")
                frames.append(frame)
        return Apng(frames)


class FrameNotFound(Exception):
    pass
