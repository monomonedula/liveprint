import os
import pathlib

import cv2
import numpy as np

from liveprint.lp import Projector, WhiteBackground
from liveprint.utils import Apng


class FakePosesFactory:
    def poses(self, image):
        return FakePoses()


class FakePoses:
    def torso_keypoints(self, threshold=0.15):
        return iter([FakeTorsoKeypoints()])


class FakeKeypoint:
    def __init__(self, number, x, y, score):
        self.number = number
        self.x = x
        self.y = y
        self.score = score

    def threshold(self, thresh):
        return self.score >= thresh

    def coords(self):
        return int(self.x), int(self.y)


class FakeTorsoKeypoints:
    def __init__(
        self,
        left_shoulder=(740, 161,),
        right_shoulder=(875, 150,),
        left_hip=(759, 308,),
        right_hip=(862, 311,),
    ):
        self.left_shoulder = FakeKeypoint(5, *left_shoulder, 0.6)
        self.right_shoulder = FakeKeypoint(6, *right_shoulder, 0.6)
        self.left_hip = FakeKeypoint(11, *left_hip, 0.6)
        self.right_hip = FakeKeypoint(12, *right_hip, 0.6)


class FakeProjectableRegion:
    def __init__(self, output_height=768, output_width=1024):
        self._output_resolution = (output_height, output_width, 3)

    def of(self, webcam_img):
        return 255 * np.ones(shape=self._output_resolution, dtype=np.uint8)


def test_projector():
    path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_image_1.png"
    )
    projectable_region_dims = [768, 1024]
    output_image = Projector(
        WhiteBackground([*projectable_region_dims, 3]),
        FakePosesFactory(),
        FakeProjectableRegion(*projectable_region_dims),
        Apng([cv2.imread(path, cv2.IMREAD_UNCHANGED)]),
    ).project(None)
    expected_image = cv2.imread(
        os.path.join(
            pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_output_1.png"
        ),
        cv2.IMREAD_UNCHANGED
    )
    np.testing.assert_almost_equal(output_image, expected_image)


class TransparentBackground:
    def layers(self):
        return []


def test_projector_transparent_background():

    path = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_image_1.png"
    )
    projectable_region_dims = [768, 1024]
    output_image = Projector(
        TransparentBackground(),
        FakePosesFactory(),
        FakeProjectableRegion(*projectable_region_dims),
        Apng([cv2.imread(path, cv2.IMREAD_UNCHANGED)]),
    ).project(None)
    expected_image = cv2.imread(
        os.path.join(
            pathlib.Path(__file__).parent.absolute(), "..", "resources", "test_output_2.png"
        ),
        cv2.IMREAD_UNCHANGED
    )
    np.testing.assert_almost_equal(output_image, expected_image)
