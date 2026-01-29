from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import sys
sys.path.append("/home/surgingtu/libs/sharp")

from sharp.models import PredictorParams, create_predictor
from sharp.utils import camera as camera_utils
from sharp.utils import gsplat as gsplat_utils
from sharp.utils import io as io_utils
from sharp.cli.predict import predict_image  # 这是纯函数，不依赖 click


DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

def _pick_device(device: str) -> str:
    if device != "default":
        return device
    # 对 SHARP+gsplat：建议默认直接用 cuda（没有就报错，而不是回退）
    return "cuda"


@dataclass
class SharpNovelViewSynthesizer:
    device: str = "default"
    checkpoint_path: Optional[str] = None
    model_url: str = DEFAULT_MODEL_URL
    color_space: str = "linearRGB"   # 跟你脚本一致

    _predictor: Any = None

    def __post_init__(self):
        self.device = _pick_device(self.device)
        self._device = torch.device(self.device)

    def load(self):
        self.device = _pick_device(self.device)
        self._device = torch.device(self.device)

        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "SHARP(gsplat) 渲染后端要求 CUDA，但当前 worker 环境没有可用 CUDA。"
                "请确认：1) worker 使用的是 CUDA 版 PyTorch；2) 没有丢失 CUDA 环境变量；3) 能在该 python 中 torch.cuda.is_available()==True。"
            )

        if self.checkpoint_path:
            state_dict = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.model_url, progress=True, map_location="cpu")

        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(self.device)

        self._predictor = predictor

    @torch.no_grad()
    def synthesize(
        self,
        image_rgb: np.ndarray,
        f_px: float,
        eye_pos: Sequence[float] = (0.0, 0.0, 0.0),
        look_at: Optional[Sequence[float]] = (0.0, 0.0, 1.0),
        out_size: Optional[Sequence[int]] = None,  # (width, height)
    ) -> np.ndarray:
        """
        输入：单张RGB图 + 相机参数
        输出：dict(color/depth/alpha) 的 numpy，不落盘
        """
        self.load()

        print("self._device:", self._device)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be HxWx3")

        h, w = image_rgb.shape[:2]
        width, height = (w, h) if out_size is None else (int(out_size[0]), int(out_size[1]))

        # 1) predict gaussians（在 SHARP 的 world 坐标系里）
        gaussians = predict_image(self._predictor, image_rgb, float(f_px), self._device)

        # 2) build intrinsics（OpenCV pinhole）
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self._device,
            dtype=torch.float32,
        )



        cam_model = camera_utils.create_camera_model(
            gaussians, intrinsics, resolution_px=(width, height)
        )

        eye_t = torch.as_tensor(eye_pos, device=self._device, dtype=torch.float32)
        look_t = None if look_at is None else torch.as_tensor(look_at, device=self._device, dtype=torch.float32)

        cam_info = cam_model.compute(eye_t, look_t)

        # 强制保证给 gsplat 的矩阵在 CUDA + float32 + contiguous
        extr = cam_info.extrinsics.to(device=self._device, dtype=torch.float32).contiguous()
        intr = cam_info.intrinsics.to(device=self._device, dtype=torch.float32).contiguous()

        print("self._device:", self._device)
        print("extr device/dtype/contig:", extr.device, extr.dtype, extr.is_contiguous())
        print("intr device/dtype/contig:", intr.device, intr.dtype, intr.is_contiguous())

        i = 0
        assert i == 1, "i is not 1"

        # gaussians 可能是自定义结构，内部 tensor 未必真的搬到了 cuda
        # 如果 gaussians 有属性（例如 means/colors/opacities 等），也打印一下：
        for name in ["means", "scales", "quats", "opacities", "colors"]:
            if hasattr(gaussians, name):
                t = getattr(gaussians, name)
                if isinstance(t, torch.Tensor):
                    print(name, t.device, t.dtype, t.shape)

        assert extr.is_cuda, "extrinsics is not CUDA"
        assert intr.is_cuda, "intrinsics is not CUDA"

        # 4) render single frame
        renderer = gsplat_utils.GSplatRenderer(color_space="linearRGB")
        rendering = renderer(
            gaussians.to(self._device),
            extrinsics=extr[None],          # [1,4,4]
            intrinsics=intr[None],          # [1,4,4] 或 [1,3,3]/[1,4,4] 取决于实现
            image_width=int(cam_info.width),
            image_height=int(cam_info.height),
        )

        # 5) to numpy
        color = (rendering.color[0].permute(1, 2, 0).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        # depth = rendering.depth[0, 0].cpu().numpy()
        # alpha = rendering.alpha[0, 0].cpu().numpy()

        return color