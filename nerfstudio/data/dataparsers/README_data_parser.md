# Lightning-NeRF ICRA 2024

:page_facing_up: Lightning NeRF: Efficient Hybrid Scene Representation for Autonomous Driving

:boy: Junyi Cao, Zhichao Li, Naiyan Wang, Chao Ma

**Please consider citing our paper if you find it interesting or helpful to your research.**
```
@article{cao2024lightning,
  title={{Lightning NeRF}: Efficient Hybrid Scene Representation for Autonomous Driving},
  author={Cao, Junyi and Li, Zhichao and Wang, Naiyan and Ma, Chao},
  journal={arXiv preprint arXiv:2403.05907},
  year={2024}
}
```

---

### Custom Dataparser
[NeRFStudio](https://github.com/nerfstudio-project/nerfstudio/) uses the class `DataParser` to load data for model training and evaluation, as shown below.
![img](https://docs.nerf.studio/_images/pipeline_parser-light.png)

Since the current project has yet to provide corresponding `DataParser` implementations for [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) and [Argoverse2 (Sensor Dataset)](https://argoverse.github.io/user-guide/datasets/sensor.html), one needs to implement the `DataParser` on his own.

For implementation, you can refer to the official guidance [here](https://docs.nerf.studio/developer_guides/pipelines/dataparsers.html).  Alternatively, you can use our custom dataparsers included in this folder.

To use our custom dataparsers:
1. Copy `kitti_dataparser.py` and `argo_dataparser.py` to `nerfstudio/data/dataparsers/` directory.
1. Import the dataparsers in `nerfstudio/data/datamanagers/base_datamanager.py`:
    ```python
    # Omit many lines above ...
    from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
    from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
    from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
    # NOTE HERE: Import dataparser for Argoverse 2
    from nerfstudio.data.dataparsers.argo_dataparser import ArgoDataParserConfig
    # NOTE HERE: Import dataparser for KITTI-360
    from nerfstudio.data.dataparsers.kitti_dataparser import KittiDataParserConfig
    from nerfstudio.data.datasets.base_dataset import InputDataset
    # Omit many lines below ...
    ```
1. Define the name of the dataparser in `nerfstudio/data/datamanagers/base_datamanager.py`:
    ```python
    AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
        tyro.extras.subcommand_type_from_defaults(
            {
                "nerfstudio-data": NerfstudioDataParserConfig(),
                "minimal-parser": MinimalDataParserConfig(),
                "arkit-data": ARKitScenesDataParserConfig(),
                "blender-data": BlenderDataParserConfig(),
                "instant-ngp-data": InstantNGPDataParserConfig(),
                "nuscenes-data": NuScenesDataParserConfig(),
                "dnerf-data": DNeRFDataParserConfig(),
                "phototourism-data": PhototourismDataParserConfig(),
                "dycheck-data": DycheckDataParserConfig(),
                "scannet-data": ScanNetDataParserConfig(),
                "sdfstudio-data": SDFStudioDataParserConfig(),
                "nerfosr-data": NeRFOSRDataParserConfig(),
                "sitcoms3d-data": Sitcoms3DDataParserConfig(),
                "argo-data": ArgoDataParserConfig(),    # NOTE HERE #
                "kitti-data": KittiDataParserConfig(),  # NOTE HERE #
            },
            prefix_names=False,  # Omit prefixes in subcommands themselves.
        )
    ]
    ```
1. Replace `${dataparser_name}` with the corresponding name (e.g., `argo-data` or `kitti-data`) in [our provided training script](https://github.com/VISION-SJTU/Lightning-NeRF?tab=readme-ov-file#training).