from functools import reduce

import tensorflow as tf
import cv2
import argparse
import numpy as np

import posenet
from lp_utils.pose import Poses
from lp_utils.utils import (
    Apng,
    ProjectableRegion,
    overlay_transparent,
    adapt_pic,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--corners_file",
    type=str,
    help="Path to a file containing"
    " integer x and y coordinates of the projectable region"
    " inside a web cam frame.\n"
    " Example is provided: corners_example.txt",
)
parser.add_argument(
    "--apng",
    type=str,
    help="Path to a file with a list" "of the apng animation frames paths.",
)
parser.add_argument("--proj_width", type=int)
parser.add_argument("--proj_height", type=int)

args = parser.parse_args()


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


def get_apng(args):
    """
    :return: List of the animation frames
    """
    path = args.apng
    with open(path) as f:
        frame_paths = [l.strip() for l in f.readlines()]
        frames = [
            cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in frame_paths if path
        ]
        return Apng(frames)


def main():
    apng = get_apng(args)
    corners = get_corners(args)
    proj_dimensions = proj_dim(args)
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg["output_stride"]
            cap = cv2.VideoCapture(0)

            cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "posenet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            LivePrint(
                Projector(
                    Background([1080, 768, 3]),
                    PosesFactory(sess, output_stride, model_outputs),
                    ProjectableRegion(*corners, *proj_dimensions),
                    apng,
                ),
                cap,
                cv2,
            ).run()
    finally:
        cv2.destroyAllWindows()


class LivePrint:
    def __init__(self, projector, video_capture, cv):
        self._projector = projector
        self._cap = video_capture
        self._cv = cv

    def run(self):
        while True:
            cv2.imshow("posenet", self._projector.project(self._cap.read()[1]).image())
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
            projection = adapt_pic(self._animation.next_frame(), image, torso)
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


if __name__ == "__main__":
    main()
