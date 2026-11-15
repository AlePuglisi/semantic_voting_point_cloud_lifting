# semantic_voting_point_cloud_lifting
preprocessing 3D point clouds with semantic labels by projecting them onto 2D images and assigning colors/labels through various voting strategies. <br/>
The methodology is based on the pre-processing scripts from [ObjectGS](https://github.com/RuijieZhu94/ObjectGS). 

It implements Multiple Voting Algorithms: 
- majority voting
- probability-based voting
- correlation-based voting
## Requirements

- OpenCV (`cv2`)
- NumPy
- PLY file handling (`plyfile`)
- COLMAP loader utilities ([`colmap_loader.py`](./scripts/colmap_loader.py))

## Expected Dataset Structure

### Option 1 : COLMAP bin project
If working with COLMAP gui, this format is exported at the end of a reconstruction as: <br/>
`File > Export all models` <br/>
And selecting the `sparse/` workspace to export it.

```text
project/
    ├── images/                # RGB Images (raw or color enhanced)
    |   ├── frame001.jpg
    |   ├── ...
    |   └── frame00N.jpg
    ├── object_mask/           # Single Channel Semantic Masks (e.g. from SAM3 with instances grouped in classes)
    |   ├── frame001.jpg
    |   ├── ...
    |   └── frame00N.jpg
    └── sparse/                # Sparse SfM Reconstruction (COLMAP)
         └── 0/
             ├── cameras.bin
             ├── frames.bin
             ├── images.bin
             └── points3D.bin
```

### Option 2 : COLMAP txt project 

If working with COLMAP gui, this format is exported at the end of a reconstruction as: <br/>
`File > Export model as text` <br/>
And selecting the `sparse/` workspace to export it.

```text
project/
    ├── images/                # RGB Images (raw or color enhanced)
    |   ├── frame001.jpg
    |   ├── ...
    |   └── frame00N.jpg
    ├── object_mask/           # Single Channel Semantic Masks (e.g. from SAM3 with instances grouped in classes)
    |   ├── frame001.jpg
    |   ├── ...
    |   └── frame00N.jpg
    └── sparse/                # Sparse SfM Reconstruction (COLMAP)
        ├── cameras.txt
        ├── frames.txt
        ├── images.txt
        ├── points3D.txt
        ├── project.ini
        └── rigs.txt
```

## Quick Usage 

```bash
python ply_processing.py --project_path /<path/to/project> --algorithm <voting_strategy> --output_ply_name <path/to/output.ply>
```

### Command Line Arguments

- `--project_path`: Path to the dataset directory (default: `datasets/lerf_mask`)
- `--algorithm`: Voting algorithm to use. Options: `majority`, `prob`, `corr` (default: `corr`)
- `--output_ply_name`: Output PLY file name (default: `points3D_corr.ply`)

## How it Works 

## Acknowledgements
The code for semantic lifting used here is based on ObjectGS pre-processing module
```bibtex
@inproceedings{zhu2025objectgs,
  title={ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting},
  author={Zhu, Ruijie and Yu, Mulin and Xu, Linning and Jiang, Lihan and Li, Yixuan and Zhang, Tianzhu and Pang, Jiangmiao and Dai, Bo},
  booktitle={International Conference on Computer Vision (ICCV), 2025},
  year={2025}
}
```