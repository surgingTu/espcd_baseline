# sharp_worker_server.py
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from typing import Optional

import numpy as np
from PIL import Image

# 注意：这里可以 import sharp，因为这个脚本会由 sharp 环境的 python 启动
from sharp.api.novel_view import SharpNovelViewSynthesizer


def read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model_url", default=None)
    args = ap.parse_args()
    print("args", args, file=sys.stderr, flush=True)

    synth = SharpNovelViewSynthesizer(device=args.device, checkpoint_path=args.checkpoint)
    if args.model_url is not None:
        synth.model_url = args.model_url
    synth.load()

    while True:
        try:
            header_len = struct.unpack("<I", read_exact(4))[0]
        except EOFError:
            break

        header = json.loads(read_exact(header_len).decode("utf-8"))
        img_len = int(header["img_len"])
        f_px = float(header["f_px"])
        eye = header.get("eye", [0.0, 0.0, 0.0])
        lookat = header.get("lookat", [0.0, 0.0, 1.0])

        out_w = header.get("out_w", None)
        out_h = header.get("out_h", None)
        out_size = None
        if out_w is not None and out_h is not None:
            out_size = (int(out_w), int(out_h))

        png_bytes = read_exact(img_len)
        img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"), dtype=np.uint8)

        # synthesize -> np.uint8[H,W,3]
        color = synth.synthesize(img, f_px=f_px, eye_pos=eye, look_at=lookat, out_size=out_size)

        out_buf = io.BytesIO()
        Image.fromarray(color).save(out_buf, format="PNG")
        out_bytes = out_buf.getvalue()

        sys.stdout.buffer.write(struct.pack("<I", len(out_bytes)))
        sys.stdout.buffer.write(out_bytes)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
