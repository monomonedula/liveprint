from functools import reduce

import cv2
import numpy as np
import tensorflow as tf
import posenet

from posenet.utils import get_adjacent_keypoints
from lp_utils.pose import Pose, Keypoints, TorsoKeypoints, Keypoint, Poses


def draw_picture(
        img, instance_scores, keypoint_scores, keypoint_coords,
        picture,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def adapt_pic(pic, image, torso: TorsoKeypoints):
    """
    :param pic: picture to be transformed
    :param image: image to be augmented with pic
    :param torso: tuple(lsh, rsh, lh, rh)
        lsh: left shoulder coordinates
        rsh: right shoulder coordinates
        lh: left hip coordinates
        rh: right hip coordinates
    :return: transformed pic
    """
    torso = [getattr(torso, part).coords(True) for part in ('right_shoulder',
                                                            'left_shoulder',
                                                            'right_hip',
                                                            'left_hip')]

    x, y, *_ = pic.shape
    pts1 = np.float32(((0, 0), (y, 0), (0, x), (y, x)))

    pts2 = np.float32(torso)
    transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    ix, iy, *_ = image.shape
    dim = (iy, ix)
    return cv2.warpPerspective(pic, transformation_matrix, dim)


# def get_torso_keypoints(keypoints):
#     kp = keypoints[RIGHT_SHOULDER], keypoints[LEFT_SHOULDER], \
#          keypoints[RIGHT_HIP], keypoints[LEFT_HIP]
#     return tuple(
#         (int(x), int(y))
#         for y, x in kp
#     )


def convert_gif_to_frames(gif):
    # Initialize the frame number and create empty frame list
    frame_num = 0
    frame_list = []

    # Loop until there are frames left
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not
            okay, frame = gif.read()
            # Append to empty frame list
            if not okay:
                break
            frame_list.append(frame)
            # Break if there are no other frames to read
            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

    return frame_list


def read_apng(filenames):
    return [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in filenames]


def get_kpts(impath):
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']
        input_image, draw_image, output_scale = posenet.read_imgfile(impath,
                                                                     scale_factor=1.0,
                                                                     output_stride=output_stride)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)
        keypoint_coords *= output_scale
        return pose_scores, keypoint_scores, keypoint_coords, draw_image


def overlay_transparent(background_img, img_to_overlay_t):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    background_img = background_img.copy()

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, mask = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    # mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    # roi = background_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(background_img, background_img, mask=cv2.bitwise_not(mask))

    # cv2.namedWindow('callibration', cv2.WND_PROP_FULLSCREEN)

    # Mask out the logo from the logo image.
    # img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    return cv2.add(img1_bg, overlay_color)


class Stage1Transormator:
    def __init__(self, ul, ur, dl, dr, output_width, output_height):
        self._coords = (ul, ur, dl, dr)
        self._output_resolution = (output_width, output_height)
        pts1 = np.float32(self._coords)
        pts2 = np.float32(((0, 0),
                          (output_width, 0),
                          (0, output_height),
                          (output_width, output_height)))
        self._transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def __call__(self, webcam_img):
        return cv2.warpPerspective(webcam_img,
                                   self._transformation_matrix,
                                   self._output_resolution)


class Apng:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def next_frame(self):
        i = self.i
        self.i = (self.i + 1) % len(self.frames)
        return self.frames[i]


def process_frame(webcam_frame, apng_frame, transform1, projection,
                  sess, output_stride, model_outputs):
    projected_region = transform1(webcam_frame)
    input_image, display_image, output_scale = posenet.process_input(projected_region,
                                                                     output_stride=output_stride)
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image}
    )
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=0.15)

    keypoint_coords *= output_scale

    poses = Poses(pose_scores, keypoint_scores, keypoint_coords)

    bg = 255 * np.ones(shape=[768, 1024, 3], dtype=np.uint8)
    torso_projections = projection(projected_region, apng_frame, poses, bg)
    return torso_projections


def do_projection(image, animation_frame, poses: Poses, overlay_bg=None):
    layers = []
    if overlay_bg is not None:
        layers.append(overlay_bg)

    for t in map(TorsoKeypoints.from_pose, poses.threshold(0.15)):
        projection = adapt_pic(animation_frame, image, t)
        layers.append(projection)

    return reduce(lambda p1, p2: overlay_transparent(p1, p2),
                  layers)
