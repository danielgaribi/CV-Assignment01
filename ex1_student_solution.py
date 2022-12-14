"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        # return homography
        A = []
        for src, dst in zip(match_p_src.T, match_p_dst.T):
            A.append([-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[0], src[1] * dst[0], dst[0]])
            A.append([0, 0, 0, -src[0], -src[1], -1, src[0] * dst[1], src[1] * dst[1], dst[1]])

        A = np.asarray(A)
        _, _, vh = np.linalg.svd(A, full_matrices=True)
        H = (vh.T.conj()[:, 8]).reshape(3, 3)

        return H

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        h1 = homography[0, :]
        h2 = homography[1, :]
        h3 = homography[2, :]
        forward_map = np.zeros((dst_image_shape[0], dst_image_shape[1], 3), dtype=int)

        for y_src in range(src_image.shape[0]):
            for x_src in range(src_image.shape[1]):
                vec = [x_src, y_src, 1]
                x_target = int(np.dot(h1, vec) / np.dot(h3, vec))
                y_target = int(np.dot(h2, vec) / np.dot(h3, vec))
                if 0 <= y_target < dst_image_shape[0] and 0 <= x_target < dst_image_shape[1]:
                    forward_map[y_target, x_target, :] = src_image[y_src, x_src, :]

        return forward_map

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        Y, X = np.mgrid[:src_image.shape[0], :src_image.shape[1]]

        forward_map = np.zeros((3, src_image.shape[1], src_image.shape[0]), dtype=int)
        forward_map[0, :, :] = X.T
        forward_map[1, :, :] = Y.T
        forward_map[2, :, :] = np.ones((src_image.shape[1], src_image.shape[0]))

        temp = np.tensordot(homography, forward_map, axes=(1, 0))
        dst_img_coordinates = np.round(np.divide(temp[0:2, :, :], temp[2, :, :]))

        INVALID_COORDINATES_VALUE = -1
        dst_img_coordinates[0, :, :][dst_img_coordinates[0, :, :] >= dst_image_shape[1]] = INVALID_COORDINATES_VALUE
        dst_img_coordinates[0, :, :][dst_img_coordinates[0, :, :] < 0] = INVALID_COORDINATES_VALUE
        dst_img_coordinates[1, :, :][dst_img_coordinates[1, :, :] >= dst_image_shape[0]] = INVALID_COORDINATES_VALUE
        dst_img_coordinates[1, :, :][dst_img_coordinates[1, :, :] < 0] = INVALID_COORDINATES_VALUE
        src_valid_idx = np.where(dst_img_coordinates != INVALID_COORDINATES_VALUE)
        dst_idx = dst_img_coordinates[:, src_valid_idx[1], src_valid_idx[2]]

        forward_map = np.zeros(dst_image_shape, dtype=int)
        forward_map[dst_idx[1].astype(int), dst_idx[0].astype(int)] = src_image[src_valid_idx[2], src_valid_idx[1]]

        return forward_map

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """

        num_points = len(match_p_src[0])

        homogen_src_points = np.asarray([match_p_src[0], match_p_src[1], [1] * num_points])  # add third coordinate
        trans_src_p = homography @ homogen_src_points                                       # transform using homography
        trans_src_p = np.divide(trans_src_p[0:2, :], trans_src_p[2])                        # normalize by 3rd coord.
        distances = np.linalg.norm(match_p_dst - trans_src_p, axis=0)                       # calculate distances
        good_dist = distances[distances < max_err]
        num_good_points = len(good_dist)
        fit_percent, dist_mse = num_good_points / num_points, np.mean(good_dist) if num_good_points else 10 ** 9
        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model

        num_points = len(match_p_src[0])

        homogen_src_points = np.asarray([match_p_src[0], match_p_src[1], [1] * num_points])  # add third coordinate
        trans_src_p = homography @ homogen_src_points                                       # transform using homography
        trans_src_p = np.divide(trans_src_p[0:2, :], trans_src_p[2])                        # normalize by 3rd coord.
        distances = np.linalg.norm(match_p_dst - trans_src_p, axis=0)                       # calculate distances
        good_points = distances < max_err
        return match_p_src[:, good_points], match_p_dst[:, good_points]

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        w = inliers_percent
        t = max_err
        p = 0.99  # parameter determining the probability of the algorithm to succeed
        d = 0.5  # the minimal probability of points which meets with the model
        n = 4  # number of points sufficient to compute the model
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        num_points = len(match_p_dst[0])
        homography = None
        cur_err = 10 ** 9
        for i in range(k):
            rand_p_indx = np.random.randint(0, num_points, size=4)
            temp_homography = self.compute_homography_naive(match_p_src[:, rand_p_indx], match_p_dst[:, rand_p_indx])
            src_meet_points, dst_meet_points = self.meet_the_model_points(temp_homography, match_p_src, match_p_dst, t)
            num_meet_points = src_meet_points.shape[1]
            if num_meet_points / num_points > d:
                new_homography = self.compute_homography_naive(src_meet_points, dst_meet_points)
                _, dist_mse = self.test_homography(new_homography, match_p_src, match_p_dst, t)
                if dist_mse < cur_err:
                    homography = new_homography

        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # (1) Create a mesh-grid of columns and rows of the destination image.
        Y, X = np.mgrid[:dst_image_shape[0], :dst_image_shape[1]]

        # (2) Create a set of homogenous coordinates for the destination image using the mesh-grid from (1).
        dst_coord_matrix = np.stack([X.T, Y.T, np.ones_like(X.T)])  # trans. because image coord.are opposite

        # (3) Compute the corresponding coordinates in the source image using the backward projective homography.
        matching_source_points = np.tensordot(backward_projective_homography, dst_coord_matrix, axes=(1, 0))
        matching_source_points = np.divide(matching_source_points[0:2, :, :], matching_source_points[2, :, :]).T

        # (4) Create the mesh-grid of source image coordinates.
        Y, X = np.mgrid[:src_image.shape[0], :src_image.shape[1]]

        # (5) For each color channel (RGB): Use scipy's interpolation.griddata with an appropriate configuration to compute the bi-cubic interpolation of the projected coordinates.
        new_dst_img = griddata((Y.flatten(), X.flatten()), src_image[Y.flatten(), X.flatten()],
                         (matching_source_points[:, :, 1], matching_source_points[:, :, 0]), method='linear')

        new_dst_img = np.nan_to_num(new_dst_img)

        return new_dst_img


    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        # (1) Build the translation matrix from the pads.
        T = np.identity(3)
        T[0,2] = -pad_left
        T[1,2] = -pad_up

        # (2) Compose the backward homography and the translation matrix together.
        H = backward_homography @ T

        # (3) Scale the homography as learnt in class.
        H = H / np.linalg.norm(H)

        return H

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""

        # (1) Compute the forward homography and the panorama shape.
        forward_homography = self.compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)

        # (2) Compute the backward homography.
        backward_homography = self.compute_homography(match_p_dst, match_p_src, inliers_percent, max_err)

        # (3) Add the appropriate translation to the homography so that the source image will plant in place.
        panorama_rows_num, panorama_cols_num, padStruct = self.find_panorama_shape(src_image, dst_image, forward_homography)
        panorama_shape = (panorama_rows_num, panorama_cols_num, 3)
        translated_backward_homography =  self.add_translation_to_backward_homography(backward_homography, padStruct.pad_left, padStruct.pad_up)

        # (4) Compute the backward warping with the appropriate translation.
        backward_warp = self.compute_backward_mapping(translated_backward_homography, src_image, panorama_shape)

        # (5) Create the an empty panorama image and plant there the destination image.
        panorama_image = np.zeros(panorama_shape)
        panorama_image[padStruct.pad_up:padStruct.pad_up+dst_image.shape[0], padStruct.pad_left:padStruct.pad_left+dst_image.shape[1], :] = dst_image[:, :, :]

        # (6) place the backward warped image in the indices where the panorama image is zero.
        none_dst_image_pixels = np.ones_like(panorama_image, dtype=bool)
        none_dst_image_pixels[padStruct.pad_up:padStruct.pad_up+dst_image.shape[0], padStruct.pad_left:padStruct.pad_left+dst_image.shape[1], :] = False
        panorama_image[none_dst_image_pixels] = backward_warp[none_dst_image_pixels]
        """
        It took me some time to understand this, so I writing this down for latter use (Now Iam 99.9% sure):
        
        1. You create a forward homography.
        2. Using it you use the given function to calculate the panorama size, as well as "the position of the
            destination image in the panorama". This positioning is described in the pad structure (returned by the
            same function)
        3. Now we calculate the backward homography and add to it the appropriate translation.
        4. We create a black image of the size of the panorama.
        5. We use the compute_backward_panorama function with the source_img, the destination image which will be
           the (currently black) panorama image and the backward homography with translation. This will plant the
           source image into the panorama (to its correct location).
        6. Now we just put the destination image into the panorama (its position is calculated in section 2), but
           only in the places where the panorama is still black!! 
        """
        return panorama_image / 255