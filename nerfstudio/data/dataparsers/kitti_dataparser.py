"""Data parser for KITTI-360 dataset"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
from typing_extensions import Literal

import torch
from rich.console import Console

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)


@dataclass
class KittiDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: Kitti)
    """target class to instantiate"""
    data: Path = Path("data/Kitti/sample_001")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """
    Sets the bounding cube to have edge length of this size.
    The longest dimension of the Friends axis-aligned bbox will be scaled to this value.
    """
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_depth: bool = False
    """Whether to load depth map for depth supervision"""
    depth_unit_scale_factor: float = 1.0
    """Scales the depth values to meters. Default value is 1 as the meter is used by this dataset."""


@dataclass
class Kitti(DataParser):
    """Kitti Dataset"""

    config: KittiDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # correct data path
        data_root = self.config.data / split
        
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []

        for i, meta_file in enumerate((data_root / "metas").iterdir()):
            meta = load_from_json(meta_file)
            
            image_filename = data_root / "images" / f"{meta['frame_id']}_{meta['camera']}.png"
            mask_filename = data_root / "masks" / f"{meta['frame_id']}_{meta['camera']}.png"
            depth_filename = data_root / "depths" / f"{meta['frame_id']}_{meta['camera']}.npy"

            intrinsics = torch.tensor(meta["intrinsic"], dtype=torch.float32)
            camtoworld = torch.tensor(meta["camtoworld"], dtype=torch.float32)

            # append data
            image_filenames.append(image_filename)
            if mask_filename.exists():
                mask_filenames.append(mask_filename)
            if self.config.load_depth and depth_filename.exists():
                depth_filenames.append(depth_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)
        
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth map filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)
        
        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1
        
        # global meta file
        global_meta_file = self.config.data / "meta_data.json"
        global_meta = load_from_json(global_meta_file)
        
        if "orientation_override" in global_meta:
            orientation_method = global_meta["orientation_override"]
            CONSOLE.log(f"[yellow]Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method
        
        # camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
        #     camera_to_worlds,
        #     method=orientation_method,
        #     center_method=self.config.center_method,
        # )
        transform = torch.tensor(global_meta["transform"], dtype=torch.float32)
        camera_to_worlds = transform @ camera_to_worlds
        
        # Scale poses
        # scale_factor = 1.0
        # if self.config.auto_scale_poses:
        #     scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))
        scale_factor = global_meta["scale"]
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        # scene box from meta data
        meta_scene_box = global_meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(aabb=aabb)
        
        if "camera_model" in global_meta:
            camera_type = CAMERA_MODEL_TO_TYPE[global_meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        height, width = global_meta["height"], global_meta["width"]
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=camera_type,
        )
        assert cameras.distortion_params is None, "Would like to disable distortion, but failed"
            
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform,
            metadata={
                'scale': scale_factor, 'trans': transform,
                'depth_filenames': depth_filenames if len(depth_filenames) > 0 else None,
                'depth_unit_scale_factor': self.config.depth_unit_scale_factor,
            }
        )
        
        CONSOLE.log(f"KITTI-360 dataset loaded with "
                    f"#images {len(image_filenames)}, "
                    f"#masks {len(mask_filenames)}, "
                    f"#depth {len(depth_filenames)}.")
        return dataparser_outputs
