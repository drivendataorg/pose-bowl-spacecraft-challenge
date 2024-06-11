import cv2
import numpy as np
import os
import random

from scipy.spatial.transform import Rotation as R


class Localizator():
    def __init__(self, data_path, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        self._data_path = data_path
        self._K = np.array([
            [5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
            [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
        ])
        self._sift = cv2.SIFT_create( #5000)
            nfeatures=5000,
            # contrastThreshold=0.04,
            # edgeThreshold=10,
            sigma=1.4
        )
        self._matcher = cv2.BFMatcher()

    def _get_image_path(self, chain_id, image_id):
        return os.path.join(os.path.join(
            self._data_path, "images", chain_id, f"{image_id:03}.png"))

    def _get_kps(self, img):
        return self._sift.detectAndCompute(img, None)

    def _get_matches(self, descs1, descs2):
        SNN_threshold = 0.8
        matches = self._matcher.knnMatch(descs1, descs2, k=2)

        # Apply ratio test
        snn_ratios = []
        tentatives = []
        for m, n in matches:
            if m.distance < SNN_threshold * n.distance:
                tentatives.append(m)
                snn_ratios.append(m.distance / n.distance)

        sorted_indices = np.argsort(snn_ratios)
        tentatives = list(np.array(tentatives)[sorted_indices])
        return tentatives

    @staticmethod
    def _n(q):
        if q[3] < 0:
            q = -q
        return q / np.linalg.norm(q)

    def predict_chain(self, chain_id, ranges):
        if not np.isnan(ranges[0]):
            starting_dist = ranges[0]
            dist_set = True
        else:
            starting_dist = 100.
            dist_set = False
        spacecraft_coord = np.array([starting_dist, 0., 0.])

        reference_image_path = self._get_image_path(chain_id, 0)
        img_ref = cv2.cvtColor(
            cv2.imread(reference_image_path), cv2.COLOR_BGR2RGB)
        kps_ref, descs_ref = self._get_kps(img_ref)

        trajectory = [np.array([0., 0., 0., 1., 0., 0., 0.])]
        for image_id in range(1, len(ranges)):
            if not dist_set and not np.isnan(ranges[image_id]):
                starting_dist = ranges[image_id]
                dist_set = True

            current_image_path = self._get_image_path(chain_id, image_id)
            img = cv2.cvtColor(
                cv2.imread(current_image_path), cv2.COLOR_BGR2RGB)

            try:
                kps, descs = self._get_kps(img)
                tentatives = self._get_matches(descs, descs_ref)

                matches_A = np.float32(
                    [kps_ref[m.trainIdx].pt for m in tentatives])
                matches_B = np.float32(
                    [kps[m.queryIdx].pt for m in tentatives])
                E, E_mask = cv2.findEssentialMat(
                    matches_A, matches_B,
                    cameraMatrix=self._K, method=cv2.USAC_ACCURATE,
                    prob=0.9999, threshold=0.5)

                np_points_ref = np.array([
                    kps_ref[match.trainIdx].pt
                    for match, inlier in zip(tentatives, E_mask)
                    if inlier])
                np_points = np.array([
                    kps[match.queryIdx].pt
                    for match, inlier in zip(tentatives, E_mask)
                    if inlier])

                _, rotation, _, _ = cv2.recoverPose(
                    E, np_points_ref, np_points, self._K)

                pred_quat = self._n(
                    R.from_matrix(rotation).as_quat())[np.array([3, 2, 0, 1])]
                pred_coord = np.array([0., 0., 0.])
                
                if abs(pred_quat[0]) < 0.90:
                    raise ValueError
            except:
                pred_quat = np.array([1., 0., 0., 0.])
                if image_id >= 10:
                    pred_coord = 0.25 * spacecraft_coord
                else:
                    pred_coord = np.array([0., 0., 0.])

            trajectory.append(np.concatenate([pred_coord, pred_quat]))

        return np.array(trajectory)
