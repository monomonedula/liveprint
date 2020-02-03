

# TODO: implement tools for the automatic detection of the projectable region coordinates
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import cv2
from cv2 import aruco



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


class AutoDetectedPRCoords(PRCoords):
    def __init__(self, cap, projector_dimensions):
        self._detected = False
        self._cap = cap
        self._proj_dimensions = projector_dimensions

    # TODO: cache this
    def _detect(self):
        pass
    
    def ul(self) -> Tuple[int, int]:
        if not self._detected:
            self._detect()

    def ur(self) -> Tuple[int, int]:
        if not self._detected:
            self._detect()

    def ll(self) -> Tuple[int, int]:
        if not self._detected:
            self._detect()

    def lr(self) -> Tuple[int, int]:
        if not self._detected:
            self._detect()


class ArucoMarkerImage:
    def __init__(self, markers, proj_width, proj_height, window_name):
        self._markers = markers
        self._dimensions = [proj_height, proj_width]
        self._window_name = window_name

    def display(self):
        cv2.imshow(self._window_name, self.image())

    def image(self):
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners  # shape: [1, 4, 2]


class ArucoMarkers:
    def __init__(self, side, ids=tuple(range(4))):
        self._ids = ids
        self._side = side

    def markers(self):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        return [aruco.drawMarker(aruco_dict, i, self._side) for i in self._ids]

