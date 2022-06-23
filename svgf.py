from utils import *
import logging
import numpy as np
import torch
from torch import linalg


def backproject(curr, prev_camera_info):
    """
    compute the reprojected pixel coordinate for each pixel, regardless the consistency
    :return: [H, W, (x,y,z,w)] (z and w will not be used)
    """

    view_prev = prev_camera_info['view']
    proj = prev_camera_info['proj']
    vert_res, hori_res, _ = curr['color'].shape

    # (x, y, z) -> (x, y, z, w=1.0)
    homogeneous_w = torch.ones(curr['world_pos'].shape[0:2]).unsqueeze(dim=2)
    world_p = torch.cat((curr['world_pos'], homogeneous_w), dim=2)

    # world space -> view space -> clip space
    reproj_view_p = torch.matmul(world_p, view_prev)
    reproj_p = torch.matmul(reproj_view_p, proj)
    homogeneous_w = reproj_p[:,:,3].unsqueeze(dim=2)
    reproj_p = reproj_p / homogeneous_w

    reproj_coord = torch.add(reproj_p, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    reproj_coord = torch.mul(reproj_coord, torch.tensor([1.0/2.0 * hori_res, 1.0/2.0 * vert_res, 1.0, 1.0]))

    return reproj_coord

def test_consistency(curr, prev, reproj_coord):
    """
    compare the two samples' depths and normals to determine if they are consistent
    
    :param curr:
    :param prev:
    :param reproj_coord: [H, W, (x,y,z,w)] represents where each pixel was located in the previous frame
    
    :return: [H, W] consistency mask
    """

    # for discarding backround-intersected pixels
    ray_scene_intersected = curr['world_pos'][:, :, 2] <= 1.0

    reproj_x = reproj_coord[:, :, 0]
    reproj_y = reproj_coord[:, :, 1]

    # check out of border
    inside = ray_scene_intersected \
             & (reproj_x >= 0) \
             & (reproj_x < hori_res) \
             & (reproj_y >= 0) \
             & (reproj_y < vert_res)

    # 리프로젝션된 좌표가 프레임 경계를 넘지 않는지 테스트한다.
    # 이미지 경계를 벗어나지 않는 valid한 리프로젝션 좌표는 그대로 사용하고, 
    # 그렇지 않은 경우 좌표 (0, 0, 0, 0)을 임시로 채워넣는다.
    # (0, 0, 0, 0)을 넣어도 문제가 되지 않는다: 어짜피 마지막에 mask에 inside를 곱하기 때문에 나머지는 다 False로 처리됨
    reproj_coord = torch.where(inside.unsqueeze(-1), reproj_coord, torch.zeros(4)).long()
    reproj_x = reproj_coord[:, :, 0]
    reproj_y = reproj_coord[:, :, 1]
    
    # compare curr depth and reprojected depth
    similar_depth = torch.abs(curr['world_pos'][:, :, 2] - prev['world_pos'][reproj_y, reproj_x, 2]) < 5.0

    # compare curr normal and reprojected normal 
    similar_normal = torch.linalg.vector_norm(curr['normal'][:, :] - prev['normal'][reproj_y, reproj_x], dim=2) < 5.0

    mask = torch.logical_and(inside, similar_depth)
    mask = torch.logical_and(mask, similar_normal)
    
    return mask

def lerp_masked(curr, history, alpha, consistent):
    """
    :param consistent: [H, W] boolean mask
    :return: temporally alpha-blended color buffer
    """

    consistent = consistent.unsqueeze(-1)
    
    # reset the invalid color history
    history = torch.where(consistent, history, curr)

    return lerp(curr, history, alpha)

def lerp(curr, history, alpha):
    return alpha * curr + (1.0 - alpha) * history

def temporal_accumulate(curr, prev, reproj_coord):
    """
	p: reproj_coord (i.e., the (0, 0) point of the interpolated pixel)
    A, B, C, D: 2x2 neighborhood pixels
	+-----+-----+
	| A   |    B|
    |   p-+---+ |
	+---|-+---|-+
    | C | |   |D|
	|   +-+---+ |
	+-----+-----+
	"""
    vert_res, hori_res = reproj_coord.shape[0:2]

    x = reproj_coord[:, :, 0].unsqueeze(dim=2)
    y = reproj_coord[:, :, 1].unsqueeze(dim=2)
    offsets = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]) # z and w offsets are not used

    # use the fractional portion of the coordinate
    x = torch.frac(x)
    y = torch.frac(y)

    # bilinear interpolation weight for each tap
    bilinear_w = torch.concat([(1 - y) * (1 - x), y * (1 - x), (1 - y) * x, y * x], dim=2)

    any_valid = torch.zeros_like(prev['history_len'], dtype=torch.bool)
    valid = [None] * 4

    for tap in range(0, 4):
        valid[tap] = test_consistency(curr, prev, reproj_coord + offsets[tap])
        any_valid = torch.logical_or(any_valid, valid[tap])
        bilinear_w[:, :, tap] = bilinear_w[:, :, tap] * valid[tap] # now invalid taps have 0 weight
    
    prev['history_len'][any_valid] += 1
    prev['history_len'][~any_valid] = 0
    
    sum_w = torch.sum(bilinear_w, dim=2) # sum of valid tap weights for each pixel
    
    # expand for RGBA color channels
    sum_w = sum_w.unsqueeze(dim=2).expand(*sum_w.shape[0:2], 4) 
 
    sum_contrib = torch.zeros_like(sum_w)
        
    for tap in range(0, 4):
        coord = reproj_coord[valid[tap]] + offsets[tap]
        coord = coord.long()
        coord_x = coord[:, 0]
        coord_y = coord[:, 1]
        sum_contrib[valid[tap]] += prev['color'][coord_y, coord_x] * bilinear_w[valid[tap]][:, tap].unsqueeze(-1)

    color = sum_contrib / sum_w
    color = color.nan_to_num()

    # TODO: additional 3x3 filtering for inconsistent pixels

    """
    TODO: temporal variance estimation
    - estimate per-pixel luminance variance using mu1, mu2 (the first and second raw moments)
    - temporally accumulate these moments, reusing the geometric tests
    - estimate temporal variance from integrated moments using the simple formular var = mu2 - mu1^2
    - < 4 frames after disocclusion, instead estimate the variance spatially, using a 7x7 bilater filter

    history length를 보존해야 한다. (++history_length)
    per-pixel luminance moments를 매 프레임 누적해나간다. (moment integration)
    만약 history length가 4 이상이면 integrated moments를 이용해서 배리언스를 계산한다.
    history length가 4 미만이면 대신 spatial variance를 계산한 후 bilateral 필터링한다.

    bilateral filter에서 퍼픽셀 웨이트 계산하기
    float computeWeight(
        float depthCenter, float depthP, float phiDepth,
        float3 normalCenter, float3 normalP, float phiNormal,
        float luminanceIllumCenter, float luminanceIllumP, float phiIllum)
    {
        const float weightNormal  = pow(saturate(dot(normalCenter, normalP)), phiNormal);
        const float weightZ       = (phiDepth == 0) ? 0.0f : abs(depthCenter - depthP) / phiDepth;
        const float weightLillum  = abs(luminanceIllumCenter - luminanceIllumP) / phiIllum;

        const float weightIllum   = exp(0.0 - max(weightLillum, 0.0) - max(weightZ, 0.0)) * weightNormal;

        return weightIllum;
    }

    
    """

    # per-pixel moment calculation and temporal integration
    moment1 = color
    moment2 = moment1 * moment1
    moment1 = lerp(moment1, prev['moment1'], 0.2)
    moment2 = lerp(moment2, prev['moment2'], 0.2)

    # temporal variance
    # variance = moment2 - moment1 * moment1

    accum = {}
    accum['color'] = color
    accum['moment1'] = moment1
    accum['moment2'] = moment2
    accum['history_len'] = prev['history_len']
    accum['world_pos'] = prev['world_pos']
    accum['normal'] = prev['normal']

    return accum, any_valid

    """
    # [H, W, Surrounding corner, Channel] = [1000, 900, 4, 4]
    
    # valid한 좌표만을 인덱싱할 수 있을까?

    #valid = torch.ones(valid.shape, dtype=torch.bool)

    tap_contrib = prev[corner_y * valid, corner_x * valid] * bilinear_w
    sum_tap = torch.sum(tap_contrib, dim=2)
    sum_w = torch.where(sum_w >= 0.01, sum_w, torch.ones(4))
    sum_tap = sum_tap / sum_w

    return sum_tap
    
    sum_tap = torch.sum(tap_contrib, dim=2)

    sum_w = torch.where(sum_w >= 0.01, sum_w, torch.ones(4))
    sum_tap = sum_tap / sum_w
    
    sum_tap[(sum_w >= 0.01)] = sum_tap[(sum_w >= 0.01)] / sum_w[(sum_w >= 0.01)]
    
    return sum_tap
    """

def estimate_variance(curr, accum):
    return NotImplemented
    if accum['history_len'] < 4:
        # spatial variance
        pass
    else:
        # temporal variance
        pass

def masked_indices(mask):
    """
    :return: the indicies tensor of True elements for given mask tensor
             its tensor shape is same as the mask 
    """
    # make the indices tensor (each element represents its coordinate)
    vert_res, hori_res = mask.shape[0:2]
    y = torch.linspace(0, vert_res - 1, steps=vert_res, dtype=torch.long)
    x = torch.linspace(0, hori_res - 1, steps=hori_res, dtype=torch.long)
    y, x = torch.meshgrid(y, x)
    grid = torch.stack((x, y), dim=2)

    return grid[mask]
    
def atrous_filter(accum, step):
    """
    baseline implementation: [Dammertz2010] styile a-trous filtering (w/o variance esti.)
    - 5-level wavelet transform
    - atrous_filter 함수는 1개의 레벨만 처리함 (step > 0)
    - 5x5 filtering
    """

    vert_res, hori_res = accum['color'].shape[0:2]
    sigma_depth = 4
    sigma_normal = 32
    sigma_luminance = 8

    # allow for smaller illumination variations to be smoothed
    for i in range(0, step):
        sigma_luminance *= 2.0**(-i)
    
    kernel = torch.tensor([[1/16, 1/16, 1/16, 1/16, 1/16],
                           [1/16, 1/4,  1/4,  1/4,  1/16],
                           [1/16, 1/4,  3/8,  1/4,  1/16],
                           [1/16, 1/4,  1/4,  1/4,  1/16],
                           [1/16, 1/16, 1/16, 1/16, 1/16]], dtype=torch.float)
                           
    filtered = torch.zeros_like(accum['color'])
    sum_w = torch.zeros_like(accum['color'])
    y = torch.linspace(0, vert_res - 1, steps=vert_res, dtype=torch.long)
    x = torch.linspace(0, hori_res - 1, steps=hori_res, dtype=torch.long)
    y, x = torch.meshgrid(y, x)
    y = y.unsqueeze(-1)
    x = x.unsqueeze(-1)

    for yy in range(-2, 3):
        for xx in range(-2, 3):
            
            kernel_w = kernel[yy+2, xx+2]
            # not center = (yy != 0) | (xx != 0)
            v = y + yy * step
            u = x + xx * step
            inside = (u >= 0) & (u < hori_res) & (v >= 0) & (v < vert_res)
            inside = inside.squeeze(-1)

            masked_idx = masked_indices(inside)
            center_x = masked_idx[:, 0]
            center_y = masked_idx[:, 1]
            p_x = masked_idx[:, 0] + xx * step
            p_y = masked_idx[:, 1] + yy * step

            depth_diff = (accum['world_pos'][center_y, center_x][:, 2] - accum['world_pos'][p_y, p_x][:, 2]).unsqueeze(-1)
            w_depth = torch.exp(-linalg.norm(depth_diff, dim=-1) / sigma_depth**2)
            normal_diff = (accum['normal'][center_y, center_x] - accum['normal'][p_y, p_x])
            w_normal = torch.exp(-linalg.norm(normal_diff, dim=-1) / sigma_normal**2)
            luminance_diff = (accum['color'][center_y, center_x] - accum['color'][p_y, p_x])
            w_luminance = torch.exp(-linalg.norm(luminance_diff, dim=-1) / sigma_luminance**2)

            weight = w_depth * w_normal * w_luminance
            weight = weight.unsqueeze(-1)
            kernel_w = kernel_w.expand(weight.shape)
            w = weight * kernel_w

            sum_w[center_y, center_x] += w
            filtered[center_y, center_x] += accum['color'][p_y, p_x] * w
    
    # normalize by weights
    accum['color'] = filtered / sum_w

    return accum


frames = []
camera_infos = []
frame_step = 1
for i in range(0, 151, frame_step):
    frame, camera_info = load_frame(i)
    frames.append(frame)
    camera_infos.append(camera_info)

vert_res, hori_res = frames[0]['color'].shape[0:2]

accum = {}

for i in range(1, len(frames)):
    curr = frames[i]
    prev = frames[i - 1]
    curr_camera_info = camera_infos[i]
    prev_camera_info = camera_infos[i - 1]

    reproj_coord = backproject(curr, prev_camera_info)

    accum['normal'] = prev['normal']
    accum['world_pos'] = prev['world_pos']
    if i == 1:
        accum['color'] = prev['color']
        accum['moment1'] = prev['color']
        accum['moment2'] = prev['color']
        accum['history_len'] = torch.zeros((vert_res, hori_res), dtype=torch.int)
    
    accum, consistency = temporal_accumulate(curr, accum, reproj_coord)
    # variance = estimate_variance(curr, accum)

    accum['color'] = lerp_masked(curr['color'], accum['color'], 0.2, consistency)
    # the filtered color from the "first" wavelet itration as our color history
    # used to temporally integrate with future frames
    accum = atrous_filter(accum, 1)
    # thus we need to modify this part, as the paper's explanation
    accum = atrous_filter(accum, 2)
    accum = atrous_filter(accum, 4)
    
curr['color'] = torch.permute(curr['color'], (2, 0, 1)).cpu().numpy()
accum['color'] = torch.permute(accum['color'], (2, 0, 1)).cpu().numpy()
print_srgb_comparison([curr['color'], accum['color']],['curr', 'accum'])
