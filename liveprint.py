from functools import reduce

import numpy as np
import cv2

import posenet
from lp_utils.pose import Poses
from lp_utils.utils import adapt_pic, overlay_transparent, Apng


def get_corners(args):
    """
    :return: Tuple of four int pairs denoting corners coordinates
     (Upper left, upper right, down left, down right)

    """

    def to_int_tuple(line):
        x, y = line.split()
        return int(x), int(y)

    path = args.corners_file
    with open(path) as f:
        ul, ur, dl, dr = [to_int_tuple(line) for line in f.readlines() if line.strip()]
    return ul, ur, dl, dr


def proj_dim(args):
    """
    :return: Dimensions of a projecting device
    """
    return args.proj_width, args.proj_height


class FrameNotFound(Exception):
    pass


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


class LivePrint:
    def __init__(self, projector, video_capture, cv):
        self._projector = projector
        self._cap = video_capture
        self._cv = cv

    def run(self):
        while True:
            print("frame")
            cv2.imshow("posenet", self._projector.project(self._cap.read()[1]))
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
        image = self._projectable_region.of(image)
        for torso in self._poses_factory.poses(image).torso_keypoints():
            print("torso")
            projection = adapt_pic(self._animation.next_frame(), image, torso)
            cv2.imwrite("projection_test.png", projection)
            layers.append(projection)
        return reduce(overlay_transparent, layers)


class Background:
    def __init__(self, projector_dimensions):
        """
        :param projector_dimensions: height, width, number of channels (3)
        """
        self._proj_dims = projector_dimensions

    def layers(self):
        return [255 * np.ones(shape=self._proj_dims, dtype=np.uint8)]


class WebcamBackground:
    def __init__(self, cap):
        self._cap = cap

    def layers(self):
        return [self._cap.read()[1]]


class PosesFactory:
    def __init__(self, session, output_stride, model_outputs):
        self._session = session
        self._output_stride = output_stride
        self._model_outputs = model_outputs

    def poses(self, roi) -> Poses:
        input_image, display_image, output_scale = posenet.process_input(
            roi, output_stride=self._output_stride
        )
        (
            heatmaps_result,
            offsets_result,
            displacement_fwd_result,
            displacement_bwd_result,
        ) = self._session.run(self._model_outputs, feed_dict={"image:0": input_image})
        (
            pose_scores,
            keypoint_scores,
            keypoint_coords,
        ) = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=self._output_stride,
            max_pose_detections=10,
            min_pose_score=0.15,
        )
        keypoint_coords *= output_scale
        return Poses(pose_scores, keypoint_scores, keypoint_coords)
