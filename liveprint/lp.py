from functools import reduce

import numpy as np
import cv2

from liveprint.utils import overlay_transparent, adapt_pic


class LivePrint:
    def __init__(self, projector, video_capture, cv):
        self._projector = projector
        self._cap = video_capture
        self._cv = cv

    def run(self):
        while True:
            cv2.imshow("Liveprint", self._projector.project(self._cap.read()[1]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


class Projector:
    def __init__(
        self, background, poses_factory, projectable_region_factory, animation
    ):
        self._bg = background
        self._poses_factory = poses_factory
        self._projectable_region = projectable_region_factory
        self._animation = animation

    def project(self, image):
        layers = self._bg.layers()
        roi = self._projectable_region.of(image)
        for torso in self._poses_factory.poses(roi).torso_keypoints():
            projection = adapt_pic(self._animation.next_frame(), roi, torso)
            layers.append(projection)
        return reduce(overlay_transparent, layers)


class WhiteBackground:
    def __init__(self, projector_dimensions):
        """
        :param projector_dimensions: height, width, number of channels (3)
        """
        self._proj_dims = projector_dimensions

    def layers(self):
        return [255 * np.ones(shape=self._proj_dims, dtype=np.uint8)]


class WebcamBackground:
    """
    Background used for a test run.
    Returns the projectable region from the webcam as the background.
    """
    def __init__(self, cap, projectable_region):
        """
        :param cap: opencv webcam
        :param projectable_region: ProjectableRegion instance
        """
        self._cap = cap
        self._projectabe_region = projectable_region

    def layers(self):
        return [self._projectabe_region.of(self._cap.read()[1])]
