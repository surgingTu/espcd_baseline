import argparse
import logging
from pathlib import Path

import torch

from sharp.utils import camera as camera_utils
from sharp.utils import gsplat as gsplat_utils
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import SceneMetaData, save_ply, load_ply
from sharp.cli.render import render_gaussians

LOGGER = logging.getLogger("predict_free")
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input-path", type=str, required=True, help="Path to an image or a folder containing images.")
    p.add_argument("-o", "--output-path", type=str, required=True, help="Folder to save .ply (and optional .mp4).")
    p.add_argument("-c", "--checkpoint-path", type=str, default="", help="Path to the .pt checkpoint. If empty, download default.")
    p.add_argument("--render", action="store_true", help="Whether to render trajectory for checkpoint (CUDA only).")
    p.add_argument("--device", type=str, default="default", help="Device to run on: default/cpu/mps/cuda")
    p.add_argument("--f-px", type=float, default=240.0, help="Focal length in pixels (assume principal point at image center).")
    p.add_argument("-v", "--verbose", action="store_true", help="Activate debug logs.")
    p.add_argument("--eye", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    p.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 1.0], metavar=("X", "Y", "Z"))
    return p.parse_args()


def pick_device(device: str) -> str:
    if device != "default":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def render_one_image(gaussians, f_px: float, width: int, height: int, out_dir, stem: str, device: torch.device, eye: torch.Tensor, target: torch.Tensor):
    """
    渲染单帧，并把 color/depth/alpha 存成 png（不写 mp4）。
    world/camera 坐标约定沿用项目默认：OpenCV (x右/y下/z前)，extrinsics 为 w2c。
    """
    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2.0, 0],
            [0, f_px, (height - 1) / 2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )

    cam_model = camera_utils.create_camera_model(gaussians, intrinsics, resolution_px=(width, height))

    eye_position = torch.tensor(eye, device=torch.device("cpu"))
    look_at_position = torch.tensor(target, device=torch.device("cpu"))
    print("eye_position", eye_position)
    camera_info = cam_model.compute(eye_position, look_at_position)
    print("camera_info.extrinsics", camera_info.extrinsics)

    renderer = gsplat_utils.GSplatRenderer(color_space="linearRGB")
    rendering = renderer(
        gaussians.to(device),
        extrinsics=camera_info.extrinsics[None].to(device),
        intrinsics=camera_info.intrinsics[None].to(device),
        image_width=camera_info.width,
        image_height=camera_info.height,
    )

    # 直接落盘：stem.color.png / stem.depth.png / stem.alpha.png
    gsplat_utils.write_renderings(rendering, out_dir, stem)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    device_str = pick_device(args.device)
    LOGGER.info("Using device %s", device_str)

    height, width = 320, 480

    f_px = float(args.f_px)

    ply_path = input_path / "encoder_input.ply"
    gaussians, _ = load_ply(ply_path)

    render_one_image(
        gaussians=gaussians,
        f_px=f_px,
        width=width,
        height=height,
        out_dir=output_path,
        stem="encoder_input",
        device=torch.device(device_str),
        eye=args.eye,
        target=args.target,
    )


if __name__ == "__main__":
    main()