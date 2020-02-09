import os
import pathlib
from unittest.mock import Mock

import cv2

from liveprint.system_setup import ArucoMarkerImage, AutoDetectedPRCoords


def test_detection():
    detected_markers = ArucoMarkerImage(Mock(), Mock(), Mock(), Mock(),).detect(
        cv2.imread(
            os.path.join(
                pathlib.Path(__file__).parent.absolute(),
                "..",
                "resources",
                "markers_test_img.jpg",
            ),
            cv2.IMREAD_COLOR,
        )
    )
    assert detected_markers.ul().projection_region_corner() == (297, 165)
    assert detected_markers.ur().projection_region_corner() == (1726, 156)
    assert detected_markers.lr().projection_region_corner() == (1696, 933)
    assert detected_markers.ll().projection_region_corner() == (255, 982)


def test_auto_detected_pr_coords():
    class FakeWebCam:
        def read(self):
            return (
                None,
                cv2.imread(
                    os.path.join(
                        pathlib.Path(__file__).parent.absolute(),
                        "..",
                        "resources",
                        "markers_test_img.jpg",
                    ),
                    cv2.IMREAD_COLOR,
                ),
            )

    class DetectOnlyMarkerImage:
        def __init__(self, origin: ArucoMarkerImage):
            self._origin = origin

        def display(self):
            pass

        def detect(self, image):
            return self._origin.detect(image)

    # noinspection PyTypeChecker
    detected_markers = AutoDetectedPRCoords(
        FakeWebCam(),
        DetectOnlyMarkerImage(ArucoMarkerImage(Mock(), Mock(), Mock(), Mock(),)),
    )
    assert detected_markers.ul() == (297, 165)
    assert detected_markers.ur() == (1726, 156)
    assert detected_markers.lr() == (1696, 933)
    assert detected_markers.ll() == (255, 982)
