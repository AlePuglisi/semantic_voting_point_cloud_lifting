"""
Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use 
under the terms of the LICENSE.md file.

For inquiries contact  george.drettakis@inria.fr
-----

@file colmap_loader.py
@brief Utils to load COLMAP project files

Original Owner: Ruijie Zhu
Author: Alessandro Puglisi 
License: MIT
"""

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(m.model_id, m) for m in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(m.model_name, m) for m in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file (corrupted or wrong format?)")
    return struct.unpack(endian_character + format_char_sequence, data)


# --------------------------------------------------------------------------- #
# points3D
# NOTE: both readers now return (ids, xyzs, rgbs, errors); ids are required to
# match COLMAP's Image.point3D_ids to rows of the arrays.
# --------------------------------------------------------------------------- #
def read_points3D_text(path):
    """points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[] (track has NO length column)."""
    ids, xyzs, rgbs, errors = [], [], [], []
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            ids.append(int(elems[0]))
            xyzs.append([float(x) for x in elems[1:4]])
            rgbs.append([int(x) for x in elems[4:7]])
            errors.append(float(elems[7]))
    return (np.array(ids, dtype=np.int64),
            np.array(xyzs, dtype=np.float64).reshape(-1, 3),
            np.array(rgbs, dtype=np.uint8).reshape(-1, 3),
            np.array(errors, dtype=np.float64).reshape(-1, 1))


def read_points3D_binary(path_to_model_file):
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        ids = np.empty(num_points, dtype=np.int64)
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3), dtype=np.uint8)
        errors = np.empty((num_points, 1))
        for p in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            ids[p] = props[0]
            xyzs[p] = props[1:4]
            rgbs[p] = props[4:7]
            errors[p] = props[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(8 * track_length, 1)  # skip track (image_id, point2D_idx) pairs
    return ids, xyzs, rgbs, errors


# --------------------------------------------------------------------------- #
# cameras
# --------------------------------------------------------------------------- #
def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            if model not in CAMERA_MODEL_NAMES:
                raise ValueError(f"Unknown COLMAP camera model: {model}")
            cameras[camera_id] = Camera(id=camera_id, model=model,
                                        width=int(elems[2]), height=int(elems[3]),
                                        params=np.array(tuple(map(float, elems[4:]))))
    return cameras


def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id = props[0], props[1]
            model = CAMERA_MODEL_IDS[model_id]
            params = read_next_bytes(fid, 8 * model.num_params, "d" * model.num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model.model_name,
                                        width=props[2], height=props[3],
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name_bytes = b""
            c = read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name_bytes += c
                c = read_next_bytes(fid, 1, "c")[0]
            image_name = name_bytes.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([np.array(x_y_id_s[0::3], dtype=np.float64),
                                   np.array(x_y_id_s[1::3], dtype=np.float64)])
            point3D_ids = np.array(x_y_id_s[2::3], dtype=np.int64)
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                     camera_id=camera_id, name=image_name,
                                     xys=xys.reshape(-1, 2), point3D_ids=point3D_ids)
    return images


def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = " ".join(elems[9:])  # names may contain spaces
                elems = fid.readline().split()    # may be empty if no 2D points
                xys = np.column_stack([np.array(elems[0::3], dtype=np.float64),
                                       np.array(elems[1::3], dtype=np.float64)])
                point3D_ids = np.array(elems[2::3], dtype=np.int64)
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec,
                                         camera_id=camera_id, name=image_name,
                                         xys=xys.reshape(-1, 2), point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()