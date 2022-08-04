import logging
import OpenEXR
import numpy as np
import torch
import Imath
import matplotlib.pyplot as plt

device = 0
torch.set_default_tensor_type(torch.cuda.FloatTensor)
logging.basicConfig(
    format='%(asctime)s:%(levelname)s: %(message)s',
    level=logging.INFO
)

def txt_matricies_to_numpy(txt_path):
    lines = []
    with open(txt_path) as f:
        for line in f.readlines():
            lines.append(line.split(' ')[:-1])

    view = np.array(lines[4:8], dtype=float)
    proj = np.array(lines[8:12], dtype=float)

    return view, proj

def exr_to_numpy(exr_path, channels):
    img = OpenEXR.InputFile(str(exr_path))
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_str = img.channels(channels, float_type)

    out_channels = []
    for channel_str in channels_str:
        out_channels.append(np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1))

    return np.stack(out_channels)

def print_srgb_comparison(data_list, title_list):
    fig = plt.figure()
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    for i, data in enumerate(data_list):
        data = data[0:3]
        data = data.transpose((1, 2, 0))
        data = correct_gamma(data)

        ax = fig.add_subplot(1, len(data_list), i+1)
        ax.imshow(data)
        ax.axis('off')
        ax.set_title('{}'.format(title_list[i]))
    plt.show()

def correct_gamma(data):
    return np.power(data, (1.0/2.2))

def load_frame(frame_i):
    frame = {}

    frame['color'] = exr_to_numpy('dataset/RunningEstimateXyza_1spp_R_{}.exr'.format(frame_i), 'RGBA')
    frame['world_pos'] = exr_to_numpy('dataset/Feature_WorldPosition_1spp_R_{}.exr'.format(frame_i), 'RGB')
    frame['normal'] = exr_to_numpy('dataset/Feature_NormalOrientation_1spp_R_{}.exr'.format(frame_i), 'RGB')

    frame['color'] = frame['color'].transpose((1, 2, 0))
    frame['world_pos'] = frame['world_pos'].transpose((1, 2, 0))
    frame['normal'] = frame['normal'].transpose((1, 2, 0))

    camera_info = {}
    camera_info['view'], camera_info['proj'] = txt_matricies_to_numpy('dataset/matrix_R_{}.txt'.format(frame_i))

    frame['color'] = torch.from_numpy(frame['color']).float().to(device)
    frame['world_pos'] = torch.from_numpy(frame['world_pos']).float().to(device)
    frame['normal'] = torch.from_numpy(frame['normal']).float().to(device)
    camera_info['view'] = torch.from_numpy(camera_info['view']).float().to(device)
    camera_info['proj'] = torch.from_numpy(camera_info['proj']).float().to(device)

    # logging.info('Frame #{} loaded'.format(frame_i))

    return frame, camera_info