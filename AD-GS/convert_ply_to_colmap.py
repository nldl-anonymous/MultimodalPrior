from plyfile import PlyData, PlyElement
import numpy as np

SH_C0 = 0.28209479177387814

def convert_and_rescale(ad_ply, colmap_ref_ply, out_ply):
    ad = PlyData.read(ad_ply)
    ref = PlyData.read(colmap_ref_ply)

    v_ad = ad['vertex'].data
    v_ref = ref['vertex'].data

    # Compute scene scale
    ad_xyz = np.vstack([v_ad['x'], v_ad['y'], v_ad['z']]).T
    ref_xyz = np.vstack([v_ref['x'], v_ref['y'], v_ref['z']]).T

    ad_extent = np.linalg.norm(ad_xyz.max(0) - ad_xyz.min(0))
    ref_extent = np.linalg.norm(ref_xyz.max(0) - ref_xyz.min(0))

    scale_factor = ref_extent / ad_extent
    print("Scale factor:", scale_factor)

    # Apply scale
    x = v_ad['x'] * scale_factor
    y = v_ad['y'] * scale_factor
    z = v_ad['z'] * scale_factor

    # normals
    nx = v_ad['nx']
    ny = v_ad['ny']
    nz = v_ad['nz']

    # SH color
    sh = np.stack([
        v_ad['f_dc_0'],
        v_ad['f_dc_1'],
        v_ad['f_dc_2']
    ], axis=1)

    rgb = sh * SH_C0 + 0.5
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    n = len(v_ad)

    vertex = np.empty(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('t', 'f4'), ('obj', 'i4')
    ])

    vertex['x'] = x
    vertex['y'] = y
    vertex['z'] = z
    vertex['nx'] = nx
    vertex['ny'] = ny
    vertex['nz'] = nz
    vertex['red'] = rgb[:,0]
    vertex['green'] = rgb[:,1]
    vertex['blue'] = rgb[:,2]
    vertex['t'] = 0
    vertex['obj'] = 0

    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(out_ply)
    print("Saved:", out_ply)


convert_and_rescale(
    "points3d_ADGaussian.ply",
    "points3d_colmap.ply",
    "points3d_ADGaussian_to_colmap.ply"
)
