

# TODO: implement tools for the automatic detection of the projectable region coordinates
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import cv2
from cv2 import aruco
from methodtools import lru_cache


class PRCoords(ABC):
    @abstractmethod
    def ul(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def ur(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def ll(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def lr(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def all(self) -> Tuple[Tuple[int, int], Tuple[int, int],
                           Tuple[int, int], Tuple[int, int]]:
        pass


class CachedPRCoords(PRCoords):
    def __init__(self, coords: PRCoords):
        self._origin = coords

    @lru_cache
    def ul(self) -> Tuple[int, int]:
        return self._origin.ul()

    def ur(self) -> Tuple[int, int]:
        return self._origin.ur()

    def ll(self) -> Tuple[int, int]:
        return self._origin.ll()

    def lr(self) -> Tuple[int, int]:
        return self._origin.lr()

    def all(self) -> Tuple[Tuple[int, int], Tuple[int, int],
                           Tuple[int, int], Tuple[int, int]]:
        return self._origin.all()


class AutoDetectedPRCoords(PRCoords):
    def __init__(self, cap, marker_image: "ArucoMarkerImage"):
        self._cap = cap
        self._marker_image = marker_image

    def _detect(self) -> "DetectedCornerMarkers":
        self._marker_image.display()
        return self._marker_image.detect(self._cap.read()[1])

    def ul(self) -> Tuple[int, int]:
        return self._detect().ul().projection_region_corner()

    def ur(self) -> Tuple[int, int]:
        return self._detect().ur().projection_region_corner()

    def ll(self) -> Tuple[int, int]:
        return self._detect().ll().projection_region_corner()

    def lr(self) -> Tuple[int, int]:
        return self._detect().lr().projection_region_corner()

    def all(self) -> Tuple[Tuple[int, int], Tuple[int, int],
                           Tuple[int, int], Tuple[int, int]]:
        markers = self._detect()
        return (
            markers.ul().projection_region_corner(),
            markers.ur().projection_region_corner(),
            markers.lr().projection_region_corner(),
            markers.ll().projection_region_corner(),
        )


class ArucoMarkerImage:
    def __init__(self, markers, proj_width, proj_height, window_name):
        self._markers = markers
        self._dimensions = [proj_height, proj_width]
        self._window_name = window_name

    def display(self):
        """
        This method prints the image returned by method .image()
        onto the window which name was specified in the 'window_name' constructor argument
        :return: None
        """
        cv2.imshow(self._window_name, self.image())

    def image(self):
        """
        This method returns an image with four or less markers in the corners of the
        white background image.

        The markers are placed in the following order:
        - upper left corner
        - upper right corner
        - lower right corner
        - lower left corner
        :return: np.array of dimensionality [proj_height, proj_width] (black & white image)
        """
        bg = 255 * np.ones(shape=self._dimensions, dtype=np.uint8)
        for i, m in enumerate(self._markers.markers()):
            if i == 0:
                x_offset = y_offset = 0
            elif i == 1:
                x_offset = self._dimensions[1] - m.shape[1]
                y_offset = 0
            elif i == 2:
                x_offset = self._dimensions[1] - m.shape[1]
                y_offset = self._dimensions[0] - m.shape[0]
            elif i == 3:
                x_offset = 0
                y_offset = self._dimensions[0] - m.shape[0]
            else:
                raise TypeError("Too many markers (Should be 4 or less)")
            bg[y_offset:y_offset + m.shape[0], x_offset:x_offset + m.shape[1]] = m
        return bg

    def detect(self, frame):
        """
        This method is used for detection of the aruco markers on the image captured
        from the web cam
        :param frame: np.array (color image)
        :return: List[np.array of shape [1, 4, 2]]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return DetectedCornerMarkers(ids, corners)


class DetectedCornerMarkers:
    def __init__(self, ids, corners):
        self._ids = ids
        self._corners = corners

    def ul(self):
        return self._get(0)

    def ur(self):
        return self._get(1)

    def lr(self):
        return self._get(2)

    def ll(self):
        return self._get(3)

    def _get(self, i):
        for marker_corners, id in zip(self._corners, self._ids):
            if id == i:
                return DetectedCornerMarker(id, marker_corners)
        raise TypeError(f"Not found marker with id {i}")


class DetectedCornerMarker:
    def __init__(self, id_, corners):
        self._id = id_
        self._corners = corners

    def projection_region_corner(self) -> Tuple[int, int]:
        """
        Returns coordinates of the corner of the projection the detected marker has been mapped to.
        For instance, if the id is 1, which means that the marker was projected on to the upper right corner of the
            output, the coordinates of the upper right corner of the detected marker are returned.
        :return:  two element list of integers
        """
        if self._id not in range(3):
            raise TypeError(f"Unknown marker id: {self._id}")
        return tuple(self._corners.squeeze()[self._id].tolist())


class ArucoMarkers:
    """
    This class is used to generate aruco markers with specified IDs
    and dimensions
    """
    def __init__(self, side: int, ids=tuple(range(4))):
        """
        :param side: the side of the marker
        :param ids: the ids to be encoded in the markers
        """
        self._ids = ids
        self._side = side

    def markers(self):
        """
        :return: a list of black & white images of markers (np.array)
        """
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        return [aruco.drawMarker(aruco_dict, i, self._side) for i in self._ids]

