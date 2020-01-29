import tensorflow as tf
import cv2
import argparse

import posenet
from liveprint import (
    get_apng,
    get_corners,
    LivePrint,
    Projector,
    Background,
    PosesFactory,
    WebcamBackground,
)
from lp_utils.utils import ProjectableRegion


def main(apng, corners, proj_dimensions, projector, cam):
    try:
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg["output_stride"]
            cap = cv2.VideoCapture(cam)

            cv2.namedWindow("posenet", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "posenet", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            LivePrint(
                Projector(
                    Background([*proj_dimensions, 3])
                    if projector
                    else WebcamBackground(
                        cap, ProjectableRegion(*corners, *proj_dimensions)
                    ),
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
        " integer x and y coordinates of the projectable region corners"
        " inside a web cam frame.\n"
        " Example is provided: corners_example.txt",
    )
    parser.add_argument(
        "--apng",
        type=str,
        help="Path to a file with a list" "of the apng animation frames paths.",
    )
    parser.add_argument(
        "--test",
        dest="projector",
        action="store_false",
        help="This option is used for the test run of the app (without a projector)."
        " The video from the web cam is used as the background when this option is selected.",
    )
    parser.set_defaults(projector=True)
    parser.add_argument(
        "--proj_width", type=int, help="The width of the projector in pixels"
    )
    parser.add_argument(
        "--proj_height", type=int, help="The height of the projector in pixels"
    )
    parser.add_argument("--webcam", type=int, help="Webcam device id (0, 1, 2 etc.)")
    parser.set_defaults(webcam=0)

    args = parser.parse_args()
    apng = get_apng(args)
    corners = get_corners(args)
    proj_dimensions = args.proj_width, args.proj_height
    main(apng, corners, proj_dimensions, args.projector, args.webcam)
