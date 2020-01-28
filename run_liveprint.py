import tensorflow as tf
import cv2
import argparse

import posenet
from liveprint import get_apng, get_corners, proj_dim, LivePrint, Projector, Background, PosesFactory, WebcamBackground
from lp_utils.utils import ProjectableRegion


def main(apng, corners, proj_dimensions):
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg["output_stride"]
            cap = cv2.VideoCapture(2)

            cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "posenet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            LivePrint(
                Projector(
                    WebcamBackground(cap),
                    # Background([1080, 1920, 3]),
                    PosesFactory(sess, output_stride, model_outputs),
                    ProjectableRegion(*corners, *proj_dimensions),
                    apng,
                ),
                cap,
                cv2,
            ).run()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
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
    apng = get_apng(args)
    corners = get_corners(args)
    proj_dimensions = proj_dim(args)
    main(apng, corners, proj_dimensions)
