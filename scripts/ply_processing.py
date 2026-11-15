
"""
@file ply_processing.py
@brief Point Cloud 2D-to-3D Semantic Lifting from a COLMAP project and single channel semantic masks
 
@details
Projects/associates COLMAP sparse points with 2D semantic masks and assigns a
label (and color) to every 3D point through a voting strategy.
 
Pipeline:
  1. Locate and load the COLMAP model (binary or text) -> find_colmap_model(), load_colmap_model()
  2. For every registered image, find and read its semantic mask -> ImageIndex, load_label_image()
  3. Collect one (point, label) observation per point visible in the view:
       - 'majority' / 'prob': project every 3D point into the view -> project_points()
       - 'corr': use the 2D keypoints COLMAP matched to 3D points (tracks)
  4. Vote over all the observations of each point -> vote()
  5. Write the labelled point cloud as a PLY file -> storePly()
 
Original Author: Ruijie Zhu
Author: Alessandro Puglisi 
License: MIT
"""

import argparse
import os
import sys
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from colmap_loader import (
    read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary,
    read_intrinsics_text, read_extrinsics_text, read_points3D_text,
    qvec2rotmat,
)


## Supported image/mask extensions, in priority order (lossless formats first).
IMG_EXTS = ('.png', '.tif', '.tiff', '.bmp', '.webp', '.jpg', '.jpeg')  # priority order (lossless first)



# --- COLMAP model discovery / loading ---

def find_colmap_model(project_dir):
    """
    @brief Locate a COLMAP model (cameras/images/points3D) inside a project folder.

    @details Searches, in this order: `sparse/0`, `sparse`, `colmap` and the project
             root. In each candidate the binary files (`.bin`) are preferred over the
             text files (`.txt`).

    @param project_dir Path to the project folder.
    @return Tuple `(model_dir, fmt)` where `fmt` is `'bin'` or `'txt'`,
            or `None` if no complete model was found.
    """
    candidates = [os.path.join(project_dir, p) for p in ('sparse/0', 'sparse', 'colmap')] + [project_dir]
    for d in candidates:
        if all(os.path.isfile(os.path.join(d, f)) for f in ('cameras.bin', 'images.bin', 'points3D.bin')):
            return d, 'bin'
        if all(os.path.isfile(os.path.join(d, f)) for f in ('cameras.txt', 'images.txt', 'points3D.txt')):
            return d, 'txt'
    return None


def load_colmap_model(model_dir, fmt):
    """
    @brief Load cameras, images and 3D points of a COLMAP model.

    @param model_dir Folder containing the COLMAP files (see find_colmap_model()).
    @param fmt       `'bin'` for binary files, anything else for text files.
    @return Tuple `(cameras, images, ids, xyz, rgb)`:
            - cameras: dict camera_id -> Camera
            - images:  dict image_id -> Image (pose, name, 2D keypoints, point3D_ids)
            - ids:     (N,) int64 array of 3D point ids
            - xyz:     (N,3) float array of 3D coordinates
            - rgb:     (N,3) uint8 array of the original COLMAP colors
    """
    if fmt == 'bin':
        cameras = read_intrinsics_binary(os.path.join(model_dir, 'cameras.bin'))
        images = read_extrinsics_binary(os.path.join(model_dir, 'images.bin'))
        ids, xyz, rgb, _ = read_points3D_binary(os.path.join(model_dir, 'points3D.bin'))
    else:
        cameras = read_intrinsics_text(os.path.join(model_dir, 'cameras.txt'))
        images = read_extrinsics_text(os.path.join(model_dir, 'images.txt'))
        ids, xyz, rgb, _ = read_points3D_text(os.path.join(model_dir, 'points3D.txt'))
    return cameras, images, ids, xyz, rgb



# --- Image lookup (extension/case independent, masks and images may differ) ---

class ImageIndex:
    """
    @brief Index of the image files of a folder, used to match COLMAP image names.

    @details Finds the file corresponding to a COLMAP image name regardless of its
             extension or letter case, so images (.jpg) and masks (.png) may use
             different formats. Lookup order: exact relative path, same path without
             extension (case-insensitive), same base name without folders.
             If several extensions match, lossless formats win (see IMG_EXTS).
    """

    def __init__(self, directory):
        """
        @brief Scan a folder recursively and build the lookup tables.

        @param directory Folder containing the images/masks.
        @throws FileNotFoundError If the folder contains no supported image file.
        """
        self.directory = directory
        self._exact, self._stem, self._base = {}, {}, {}
        for root, _, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, directory).replace(os.sep, '/')
                self._exact[rel] = full
                self._stem.setdefault(os.path.splitext(rel)[0].lower(), []).append(full)
                self._base.setdefault(os.path.splitext(f)[0].lower(), []).append(full)
        if not self._exact:
            raise FileNotFoundError(f"No image files found in {directory}")

    @staticmethod
    def _best(paths):
        """
        @brief Pick the preferred file among candidates with different extensions.

        @param paths List of candidate file paths.
        @return The path whose extension comes first in IMG_EXTS.
        """
        return sorted(paths, key=lambda p: IMG_EXTS.index(os.path.splitext(p)[1].lower()))[0]

    def find(self, image_name):
        """
        @brief Find the file matching a COLMAP image name.

        @param image_name Image name as stored in COLMAP (may contain subfolders).
        @return Full path of the matching file, or `None` if there is none.
        """
        name = image_name.replace('\\', '/')
        if name in self._exact:
            return self._exact[name]
        stem = os.path.splitext(name)[0].lower()
        if stem in self._stem:
            return self._best(self._stem[stem])
        base = os.path.splitext(os.path.basename(name))[0].lower()
        if base in self._base:  # e.g. COLMAP name has a subfolder but masks are flat
            return self._best(self._base[base])
        return None


_warned = set()
def _warn_once(msg):
    """
    @brief Print a warning only the first time a given message is emitted.

    @param msg Warning text.
    """
    if msg not in _warned:
        _warned.add(msg)
        print(f"[warning] {msg}")


def load_label_image(path):
    """
    @brief Read a semantic mask as a single-channel label image.

    @details The file is read unchanged (keeps 8/16-bit values). Multi-channel masks
             are reduced to their first channel. A warning is issued for JPEG masks,
             since lossy compression corrupts label values near edges.

    @param path Path of the mask file.
    @return (H,W) array of labels.
    @throws FileNotFoundError If OpenCV cannot read the file.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if img.ndim == 3:
        _warn_once("Mask has several channels; using the first one as label.")
        img = img[:, :, 0]
    if path.lower().endswith(('.jpg', '.jpeg')):
        _warn_once("JPEG masks are lossy: label values can be corrupted near edges. Prefer PNG masks.")
    return img


def load_color_image(path, size_wh):
    """
    @brief Read a color image as RGB, resized to a target size.

    @param path    Path of the color image.
    @param size_wh Target size `(width, height)`; nearest-neighbour resize is used
                   if the image has a different size (keeps colors unmixed).
    @return (H,W,3) uint8 array in RGB order.
    @throws FileNotFoundError If OpenCV cannot read the file.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read color image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, PLY wants RGB
    if (img.shape[1], img.shape[0]) != tuple(size_wh):
        img = cv2.resize(img, tuple(size_wh), interpolation=cv2.INTER_NEAREST)
    return img



# --- Label -> color ---

class ID2RGBConverter:
    """
    @brief Maps object/label ids to unique, reproducible RGB colors.

    @details Each new label receives a random 24-bit id (never reused) that is split
             into its R, G, B bytes. The generator is seeded, so runs are repeatable.
    """

    def __init__(self, seed=0):
        """
        @brief Create the converter.

        @param seed Seed of the random generator.
        """
        self.rng = np.random.default_rng(seed)
        self.all_id = set()
        self.obj_to_id = {}

    @staticmethod
    def _id_to_rgb(id_):
        """
        @brief Split a 24-bit integer id into its three color bytes.

        @param id_ Integer id.
        @return (3,) uint8 array `[byte0, byte1, byte2]`.
        """
        return np.array([(id_ >> (8 * i)) & 255 for i in range(3)], dtype=np.uint8)

    def convert(self, obj):
        """
        @brief Get the color of a label, creating it the first time it is seen.

        @param obj Label value.
        @return Tuple `(id, rgb)`: the 24-bit id and its (3,) uint8 RGB color.
        """
        obj = int(obj)
        if obj not in self.obj_to_id:
            while True:
                id_ = int(self.rng.integers(255, 256 ** 3))
                if id_ not in self.all_id:
                    break
            self.obj_to_id[obj] = id_
            self.all_id.add(id_)
        id_ = self.obj_to_id[obj]
        return id_, self._id_to_rgb(id_)



# --- Projection ---

def project_points(xyz, R, t, camera):
    """
    @brief Project 3D world points into an image (vectorized).

    @details World -> camera transform `X_cam = R * X + t`, perspective division,
             lens distortion (SIMPLE_RADIAL, RADIAL, OPENCV; other models are
             treated as pinhole with a warning) and intrinsics. Pixel coordinates
             follow the COLMAP convention (center of the top-left pixel is (0.5, 0.5)),
             hence use `floor` to get a pixel index. No occlusion test is performed.

    @param xyz    (N,3) world coordinates.
    @param R      (3,3) world-to-camera rotation matrix.
    @param t      (3,) world-to-camera translation.
    @param camera COLMAP Camera (model name and parameters).
    @return Tuple `(u, v, valid)`: pixel coordinates (N,) and a boolean mask that is
            False for points behind the camera or with non-finite projections.
    """
    cam = xyz @ R.T + t.reshape(1, 3)
    z = cam[:, 2]
    valid = z > 1e-6
    zs = np.where(valid, z, 1.0)
    x, y = cam[:, 0] / zs, cam[:, 1] / zs
    m, p = camera.model, camera.params

    if m in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:  # PINHOLE, OPENCV, FULL_OPENCV, FOV, OPENCV_FISHEYE, THIN_PRISM_FISHEYE
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]

    r2 = x * x + y * y
    if m == 'SIMPLE_RADIAL':
        f = 1 + p[3] * r2
        x, y = x * f, y * f
    elif m == 'RADIAL':
        f = 1 + p[3] * r2 + p[4] * r2 * r2
        x, y = x * f, y * f
    elif m == 'OPENCV':
        k1, k2, p1, p2 = p[4:8]
        rad = 1 + k1 * r2 + k2 * r2 * r2
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = x * rad + dx, y * rad + dy
    elif m not in ('PINHOLE', 'SIMPLE_PINHOLE'):
        _warn_once(f"Camera model {m}: lens distortion is ignored (projection may be slightly off).")

    u, v = fx * x + cx, fy * y + cy
    valid &= np.isfinite(u) & np.isfinite(v)
    return u, v, valid



# --- Voting ---

def vote(n_points, rows, labels, strategy, ignore_background, rng):
    """
    @brief Assign one label per 3D point from all its per-view observations.

    @details Observations are counted per `(point, label)` pair, then:
             - `'majority'` / `'corr'`: the most frequent label wins (ties -> smallest label).
             - `'prob'`: a label is sampled with probability proportional to its count.
             Points without any (remaining) observation keep `has_label = False`.

    @param n_points          Total number of 3D points.
    @param rows              (M,) point row index of each observation.
    @param labels            (M,) label of each observation.
    @param strategy          `'majority'`, `'corr'` or `'prob'`.
    @param ignore_background If True, observations with label 0 are discarded before voting.
    @param rng               numpy Generator, used by the `'prob'` strategy.
    @return Tuple `(final_label, has_label)`: (N,) int64 labels (0 if unlabelled)
            and an (N,) boolean mask of the points that received a label.
    """
    final = np.zeros(n_points, dtype=np.int64)
    has = np.zeros(n_points, dtype=bool)
    if ignore_background:
        k = labels != 0
        rows, labels = rows[k], labels[k]
    if rows.size == 0:
        return final, has

    L = int(labels.max()) + 1
    uk, cnt = np.unique(rows.astype(np.int64) * L + labels, return_counts=True)  # sorted by (row, label)
    r, lab = uk // L, uk % L

    if strategy == 'prob':
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        sizes = np.diff(np.r_[starts, len(r)])
        totals = np.add.reduceat(cnt, starts)
        csum = np.cumsum(cnt)
        before = np.r_[0, csum[:-1]][starts]
        cum = csum - np.repeat(before, sizes)
        thr = np.repeat(rng.random(len(starts)) * totals, sizes)
        hit = np.flatnonzero(cum > thr)
        hr = r[hit]
        sel = hit[np.r_[True, hr[1:] != hr[:-1]]]
    else:  # majority / corr: most frequent label (ties -> smallest label)
        order = np.lexsort((-cnt, r))
        rs = r[order]
        sel = order[np.r_[True, rs[1:] != rs[:-1]]]

    final[r[sel]] = lab[sel]
    has[r[sel]] = True
    return final, has



# --- PLY output ---

def storePly(path, xyz, rgb, label):
    """
    @brief Save a labelled point cloud as a PLY file.

    @details Vertex properties: x, y, z, nx, ny, nz (normals set to 0), red, green,
             blue and `label`. The label type is the smallest of uint8/uint16/uint32
             able to hold the largest label.

    @param path  Output file path.
    @param xyz   (N,3) point coordinates.
    @param rgb   (N,3) uint8 colors.
    @param label (N,) or (N,1) point labels.
    """
    label = np.asarray(label).reshape(-1)
    mx = int(label.max()) if label.size else 0
    ldt = 'u1' if mx < 256 else ('u2' if mx < 65536 else 'u4')
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('label', ldt)]
    el = np.zeros(len(xyz), dtype=dtype)
    el['x'], el['y'], el['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el['red'], el['green'], el['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    el['label'] = label
    PlyData([PlyElement.describe(el, 'vertex')]).write(path)



# --- Main lifting routine ---

def lift_project(project_dir, algorithm, output_ply_name, mask_dir_name, ignore_background, seed):
    """
    @brief Run the whole semantic lifting for one COLMAP project.

    @details Loads the model and the masks, gathers the per-view observations
             (projection for `majority`/`prob`, keypoint tracks for `corr`; mask
             coordinates are rescaled if the mask resolution differs from the camera's),
             votes, colors the points by label (from `color_mask/` if present, otherwise
             with ID2RGBConverter) and writes the PLY. Unlabelled points keep their
             original COLMAP color and label 0. Images without a matching mask are skipped.

    @param project_dir       Project folder (contains the masks and the COLMAP model).
    @param algorithm         `'majority'`, `'prob'` or `'corr'`.
    @param output_ply_name   Bare file name (saved next to the COLMAP model) or a path.
    @param mask_dir_name     Name of the semantic mask folder inside the project.
    @param ignore_background Ignore label 0 while voting.
    @param seed              Seed for random colors and probabilistic voting.
    @throws FileNotFoundError If the COLMAP model or the mask folder is missing.
    @throws RuntimeError      If no mask matches any COLMAP image.
    """
    found = find_colmap_model(project_dir)
    if found is None:
        raise FileNotFoundError(
            f"No COLMAP model (cameras/images/points3D .bin or .txt) found in {project_dir} "
            f"(searched sparse/0, sparse, colmap and the project root)")
    model_dir, fmt = found
    print(f"  COLMAP model: {model_dir} ({fmt})")

    label_dir = os.path.join(project_dir, mask_dir_name)
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Mask folder not found: {label_dir}")
    color_dir = os.path.join(project_dir, 'color_mask')
    color_index = ImageIndex(color_dir) if os.path.isdir(color_dir) else None
    label_index = ImageIndex(label_dir)

    cameras, images, ids, xyz, rgb = load_colmap_model(model_dir, fmt)
    print(f"  {len(images)} images, {len(cameras)} cameras, {len(ids)} points")

    order = np.argsort(ids)
    sorted_ids = ids[order]
    rng = np.random.default_rng(seed)
    converter = ID2RGBConverter(seed)
    label_colors = {}
    all_rows, all_labels, missing = [], [], []

    for k, image in enumerate(images.values()):
        cam = cameras[image.camera_id]
        mask_path = label_index.find(image.name)
        if mask_path is None:
            missing.append(image.name)
            continue
        label_img = load_label_image(mask_path)
        h, w = label_img.shape
        sx, sy = w / cam.width, h / cam.height  # masks may have another resolution than COLMAP's

        if algorithm == 'corr':
            pid = image.point3D_ids
            keep = pid >= 0
            pid = pid[keep]
            u, v = image.xys[keep, 0], image.xys[keep, 1]
            pos = np.clip(np.searchsorted(sorted_ids, pid), 0, len(sorted_ids) - 1)
            ok = sorted_ids[pos] == pid
            rows, u, v = order[pos[ok]], u[ok], v[ok]
        else:
            R = qvec2rotmat(image.qvec)
            u, v, valid = project_points(xyz, R, image.tvec, cam)
            rows, u, v = np.flatnonzero(valid), u[valid], v[valid]

        px = np.floor(u * sx).astype(np.int64)
        py = np.floor(v * sy).astype(np.int64)
        inb = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        rows, px, py = rows[inb], px[inb], py[inb]
        labels = label_img[py, px].astype(np.int64)

        # one representative color per label
        uniq, first = np.unique(labels, return_index=True)
        new = [(l, i) for l, i in zip(uniq.tolist(), first) if l not in label_colors]
        if new:
            color_img = None
            if color_index is not None:
                cp = color_index.find(image.name)
                if cp is not None:
                    color_img = load_color_image(cp, (w, h))
            for l, i in new:
                label_colors[l] = (color_img[py[i], px[i]] if color_img is not None
                                   else converter.convert(l)[1])

        all_rows.append(rows)
        all_labels.append(labels)
        if (k + 1) % 50 == 0 or k + 1 == len(images):
            print(f"  processed {k + 1}/{len(images)} views")

    if missing:
        print(f"[warning] no mask found for {len(missing)}/{len(images)} images (e.g. {missing[:3]}); skipped")
    if not all_rows:
        raise RuntimeError(f"No mask matched any COLMAP image. Check names in {label_dir} vs COLMAP image names.")

    final, has = vote(len(ids), np.concatenate(all_rows), np.concatenate(all_labels),
                      algorithm, ignore_background, rng)

    out_rgb = rgb.astype(np.uint8).copy()  # unlabeled points keep their COLMAP color
    for l in np.unique(final[has]).tolist():
        out_rgb[has & (final == l)] = label_colors[l]
    print(f"  {has.sum()}/{len(ids)} points received a label; labels: {np.unique(final[has]).tolist()}")

    if os.path.isabs(output_ply_name) or os.path.dirname(output_ply_name):
        out_path = output_ply_name
    else:
        out_path = os.path.join(model_dir, output_ply_name)
    storePly(out_path, xyz, out_rgb, final)
    print(f"  Point cloud saved to {out_path}")


def main(args):
    """
    @brief Process one project, or every project found in a folder.

    @details If `args.project_path` is itself a COLMAP project it is processed alone;
             otherwise each of its sub-folders containing a COLMAP model is processed
             (one PLY per project, saved in its model folder). When `ignore_background`
             is not given, it defaults to on for `corr` and off for the other strategies.

    @param args Parsed command line arguments (see build_parser()).
    @throws FileNotFoundError If no COLMAP project is found.
    """
    root = args.project_path
    if find_colmap_model(root) is not None:
        projects = [root]
    else:  # folder containing several projects
        projects = [os.path.join(root, d) for d in sorted(os.listdir(root))
                    if os.path.isdir(os.path.join(root, d))
                    and find_colmap_model(os.path.join(root, d)) is not None]
    if not projects:
        raise FileNotFoundError(f"No COLMAP project found in {root}")

    ignore_bg = args.ignore_background
    if ignore_bg is None:
        ignore_bg = (args.algorithm == 'corr')  # original behaviour

    for p in projects:
        print(f"Processing {p} with '{args.algorithm}' voting...")
        out = args.output_ply_name
        if len(projects) > 1:
            out = os.path.basename(out)  # one file per project, inside its model folder
        lift_project(p, args.algorithm, out, args.mask_dir_name, ignore_bg, args.seed)

HELP_TEXT = """
How to run:
  python ply_processing.py --project_path <path/to/project> \\
                              --algorithm <majority|prob|corr> \\
                              --output_ply_name <points3D_corr.ply | path/to/output.ply>

Examples:
  python ply_processing.py --project_path datasets/my_scene
  python ply_processing.py --project_path datasets/my_scene --algorithm majority --output_ply_name points3D_majority.ply
  python ply_processing.py --project_path datasets/my_scene --algorithm corr --no-ignore_background

Expected project structure (COLMAP .bin or .txt):
  <project_path>/
      object_mask/        single-channel semantic masks (PNG recommended), named like the COLMAP images
      color_mask/         (optional) colored masks
      sparse/0/           cameras.bin, images.bin, points3D.bin   (or the .txt equivalents,
                          also accepted directly in sparse/ or colmap/)

Notes:
  --project_path may also be a folder containing several such projects.
  Image and mask extensions may differ (e.g. images .jpg, masks .png).
  --output_ply_name: a bare file name is saved next to the COLMAP model; a path is used as given.
"""

class FriendlyArgumentParser(argparse.ArgumentParser):
    """
    @brief ArgumentParser that explains how to run the script when the arguments are wrong.
    """

    def error(self, message):
        """
        @brief Report an argument error, print the usage and the run instructions, then exit.

        @param message Description of the error given by argparse.
        """
        sys.stderr.write(f"\n[error] Invalid arguments: {message}\n\n")
        self.print_usage(sys.stderr)
        sys.stderr.write(HELP_TEXT)
        sys.exit(2)


def build_parser():
    """
    @brief Build the command line parser.

    @return A FriendlyArgumentParser with the options `--project_path`, `--algorithm`,
            `--output_ply_name`, `--mask_dir_name`, `--ignore_background` and `--seed`.
    @throws AttributeError On Python < 3.9 (`argparse.BooleanOptionalAction` is missing).
    """
    parser = FriendlyArgumentParser(
        description="Preprocess 3D point clouds with semantic labels using various voting strategies.",
        epilog=HELP_TEXT, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_path', type=str, default='datasets/colmap_project',
                        help='A COLMAP project, or a folder containing several projects')
    parser.add_argument('--algorithm', type=str, default='corr', choices=['majority', 'prob', 'corr'],
                        help='majority / prob: project all points in every view; corr: use COLMAP keypoint tracks')
    parser.add_argument('--output_ply_name', type=str, default='points3D_corr.ply',
                        help='File name (saved next to the COLMAP model) or a full output path')
    parser.add_argument('--mask_dir_name', type=str, default='object_mask',
                        help='Name of the semantic mask folder inside the project')
    parser.add_argument('--ignore_background', action=argparse.BooleanOptionalAction, default=None,
                        help='Ignore label 0 when voting (default: on for corr, off otherwise)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random colors / prob sampling')
    return parser


def parse_args():
    """
    @brief Parse and validate the command line arguments.

    @details Besides argparse's own checks, verifies that `--project_path` is an existing
             folder, that `--output_ply_name` ends with `.ply` and that its folder (if any)
             exists. Any problem prints the error and the run instructions and exits with code 2.

    @return The parsed `argparse.Namespace`.
    """
    try:
        parser = build_parser()
    except AttributeError:  # argparse.BooleanOptionalAction needs Python >= 3.9
        sys.stderr.write("\n[error] This script requires Python 3.9 or newer.\n")
        sys.exit(2)

    args = parser.parse_args()  # unknown/missing/invalid values end in parser.error()

    if not os.path.isdir(args.project_path):
        parser.error(f"--project_path '{args.project_path}' is not an existing directory")
    if not args.output_ply_name.lower().endswith('.ply'):
        parser.error(f"--output_ply_name '{args.output_ply_name}' must end with '.ply'")
    out_dir = os.path.dirname(args.output_ply_name)
    if out_dir and not os.path.isdir(out_dir):
        parser.error(f"output directory '{out_dir}' does not exist")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.stderr.write(f"\n[error] {e}\n")
        sys.stderr.write(HELP_TEXT)
        sys.exit(1)