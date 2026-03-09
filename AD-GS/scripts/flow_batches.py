import json
import os
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import imageio
import glob
import tqdm

from colmap_utils import read_cameras_binary
from models import get_model
from utils import project_points, get_intrins, get_proj_mat, base_conv

from utils.side_by_side_html import visualize_flow

np.set_printoptions(precision=4, suppress=True)

import torch.nn as nn
import torch.nn.functional as F
from models.core.utils import grid_sample
from models.core.utils import frame_utils
import libs.waymo_utils as waymo_utils
from models.utils import forward_interpolation

import torch.nn.functional as nnf


def generate_image_row(image, caption, width):
    return f"""
    <tr>
        <td><img src='{image}' alt='{caption}' width='{width}'/></td>
        <td>{caption}</td>
    </tr>
    """


def generate_side_by_side_html(image_path, semantic_mask_path, flow_fwd_path, flow_bwd_path, pointcloud_path, output_path):
    # ... unchanged body ...
    try:
        image, semantic_mask, flow_fwd, flow_bwd, pointcloud = None, None, None, None, None
        if image is not None:
            image = imageio.imread(image_path).repeat(3, axis=-1) / 255.0
        if semantic_mask is not None:
            semantic_mask = imageio.imread(semantic_mask_path).repeat(3, axis=-1) / 255.0
        if flow_fwd is not None:
            flow_fwd = np.load(flow_fwd_path)['flo_fwd']
        if flow_bwd is not None:
            flow_bwd = np.load(flow_bwd_path)['flo_bwd']
        if pointcloud is not None:
            pointcloud = np.load(pointcloud_path)['pointcloud']
    except:
        pass
    return


def warp(image, flow, mode='bilinear', padding_mode='zeros'):
    if image.ndim == 3:
        image = image[None]
    b, c, h, w = image.shape
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing='ij'), dim=-1).to(flow.device)

    flow_norm = torch.zeros_like(flow)
    flow_norm[..., 0] = flow[..., 0] / (w - 1) * 2
    flow_norm[..., 1] = flow[..., 1] / (h - 1) * 2

    grid = grid + flow_norm
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    warped = F.grid_sample(image, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return warped


def warp_im_back(im, flo):
    B, _, H, W = im.shape
    flo = flo.permute(0, 2, 3, 1)
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    grid = torch.stack((xx, yy), 2).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    if im.is_cuda:
        grid = grid.cuda()

    vgrid = grid - flo
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    output = nnf.grid_sample(im, vgrid, align_corners=True, padding_mode='zeros')
    mask = torch.ones(im.size()).cuda()
    mask = nnf.grid_sample(mask, vgrid, align_corners=True, padding_mode='zeros')
    mask = (mask > 0.9999).float()
    return output * mask


def generate_waymo_flow(
        path, 
        downsample,
        slide_window,
        num_cams=1,
        start_idx=0,          # ✅ Added
        end_idx=None):        # ✅ Added

    """
    Main flow generation function — now supports slicing the dataset
    using start_idx and end_idx so you can run in batches.
    """

    model_path = 'libs/pretrained/raft-sintel.pth'
    model, model_args = get_model("toy", model_path)
    # model.cuda()
    model.cpu()
    model.backbone_type = torch.nn.DataParallel(model.backbone_type)
    model.eval()

    image_folder      = os.path.join(path, "image")
    colmap_cameras    = read_cameras_binary(str(os.path.join(path, "sparse", "cameras.bin")))
    image_files       = sorted(os.listdir(image_folder))
    indices           = [int(i.split('.')[0]) for i in image_files]

    images = []
    for file in image_files:
        img = imageio.imread(os.path.join(image_folder, file))
        h, w, _ = img.shape

        ds = downsample
        dh = int(h / ds)
        dw = int(w / ds)
        img_resized = torch.tensor(img).float()\
            .permute(2, 0, 1).unsqueeze(0) / 255.0
        img_resized = nnf.interpolate(img_resized, (dh, dw), mode='nearest')
        images.append(img_resized)

    images = torch.cat(images, dim=0)

    os.makedirs(os.path.join(path, f"flows_{downsample}"), exist_ok=True)
    flow_folder = os.path.join(path, f"flows_{downsample}")

    total = images.shape[0]

    # ✅ Batch slicing
    if end_idx is None or end_idx > total:
        end_idx = total

    iter_count = 0
    for idx in tqdm.tqdm(range(start_idx, end_idx), desc="Processing"):

        now    = indices[idx]
        now_id = idx

        ####################################
        # Flow Forward
        ####################################
        ola_list_fwd = [i for i in range(now - slide_window, now) if i in indices]
        ola_indices_fwd = [indices.index(i) for i in ola_list_fwd]

        if len(ola_indices_fwd) > 0:
            flows_forward = []
            image_0 = images[now_id]
            for ref_idx in ola_indices_fwd:
                image_1 = images[ref_idx]

                with torch.no_grad():
                    # flo_fwd, flo_bwd = model(image_0.cuda(), image_1.cuda(), iters=20, test_mode=True)
                    flo_fwd, flo_bwd = model(image_0, image_1, iters=20, test_mode=True)

                flows_forward.append(flo_fwd[0].cpu().permute(1, 2, 0).numpy())

            flow_name = os.path.join(flow_folder, f"{now:06d}.npz")
            np.savez(flow_name, flo_fwd=np.stack(flows_forward))
            iter_count += 1

        ####################################
        # Flow Backward
        ####################################
        ola_list_bwd = [i for i in range(now + 1, now + slide_window + 1) if i in indices]
        ola_indices_bwd = [indices.index(i) for i in ola_list_bwd]

        if len(ola_indices_bwd) > 0:
            flows_backward = []
            image_1 = images[now_id]
            for ref_idx in ola_indices_bwd:
                image_0 = images[ref_idx]

                with torch.no_grad():
                    flo_fwd, flo_bwd = model(image_0, image_1, iters=20, test_mode=True)

                flows_backward.append(flo_fwd[0].cpu().permute(1, 2, 0).numpy())

            flow_name = os.path.join(flow_folder, f"{now:06d}.npz")
            existing = dict(np.load(flow_name)) if os.path.exists(flow_name) else {}
            existing["flo_bwd"] = np.stack(flows_backward)
            np.savez(flow_name, **existing)
            iter_count += 1

        ####################################
        # Optional visualization (unchanged)
        ####################################

        viz_id = visit_counter.get(now, 0)
        visit_counter[now] = viz_id + 1
        fwd_v = (viz_id - slide_window) if viz_id >= slide_window else None
        bwd_v = viz_id if viz_id < slide_window else None

        if fwd_v is not None:
            viz_name = os.path.join(flow_folder, f"{now:06d}_fwd_{fwd_v}.png")
            visualize_flow(
                image_0.permute(1, 2, 0).numpy(),
                flows_forward[fwd_v],
                viz_name,
                "forward flow",
            )

        if bwd_v is not None:
            viz_name = os.path.join(flow_folder, f"{now:06d}_bwd_{bwd_v}.png")
            visualize_flow(
                image_0.permute(1, 2, 0).numpy(),
                flows_backward[bwd_v],
                viz_name,
                "backward flow",
            )


# ================================================================
# ✅ NEW: Automatic batching wrapper
# ================================================================
def generate_waymo_flow_in_batches(
        path,
        downsample,
        slide_window,
        num_cams=1,
        batch_size=200):

    image_folder = os.path.join(path, "image")
    image_files  = sorted(os.listdir(image_folder))
    total        = len(image_files)

    for start in range(0, total, batch_size):
        end = start + batch_size
        print(f"\n🔥 Processing batch {start}–{end}\n")

        generate_waymo_flow(
            path,
            downsample,
            slide_window,
            num_cams=num_cams,
            start_idx=start,
            end_idx=end
        )


