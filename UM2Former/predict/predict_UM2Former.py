import torch
import os
import numpy as np
from utils.dataset import WHU_OHS_Dataset
from Nets.UM2former import UM2Former, UM2Former_OP8
from torchsummary import summary
from osgeo import gdal
from tqdm import tqdm
from matplotlib import pyplot as plt
from einops import rearrange
from PIL import Image


class_name_dict = ['0Paddy field', '1Dry farm', '2Woodland', '3Shrubbery', '4Sparse woodland',
                   '5Other forest land', '6High-covered grassland', '7Medium-covered grassland',
                   '8Low-covered grassland', '9River/canal', '10Lake', '11Reservoir/pond', '12Beach land',
                   '13Shoal', '14Urban built-up', '15Rural settlement', '16Other construction land',
                   '17Sand', '18Gobi', '19Saline-alkali soil', '20Marshland', '21Bare land', '22Bare rock', '23Ocean']

color_map_dict = np.array([
    [255, 215, 0],  # 金黄色
    [0, 255, 255],  # 青色
    [34, 139, 34],  # 森林绿
    [127, 255, 0],  # 黄绿
    [0, 255, 0],  # 绿
    [0, 201, 87],  # 翠绿
    [115, 74, 18],  # 标土棕
    [160, 82, 45],  # 赫色
    [255, 255, 0],  # 黄色
    [135, 206, 235],  # 天蓝
    [65, 105, 225],  # 品蓝
    [8, 46, 84],  # 靛青
    [160, 102, 211],  # jasoa
    [61, 145, 64],  # 钴绿色
    [128, 42, 42],  # 棕色
    [255, 99, 71],  # 番茄红
    [250, 128, 114],  # 橙红色
    [255, 192, 203],  # 粉红色
    [255, 0, 255],  # 深红色
    [208, 32, 144],  # violet red
    [199, 21, 133],  # medium violet red
    [176, 48, 96],  # maroon
    [255, 20, 147],  # deep pink
    [139, 137, 137]])  # snow4


def draw(image: np.ndarray, label: np.ndarray):
    h, w = image.shape
    image_flatten = image.flatten()
    label_flatten = label.flatten()
    image_draw = np.zeros((h*w, 3))
    label_draw = np.zeros((h*w, 3))
    palette = color_map_dict
    palette = palette * 1.0 / 255
    cls_list = np.unique(label)
    cls_list = cls_list[1:]

    inds = np.where(label_flatten == -1)[0]

    for cls in cls_list:
        inds1 = np.where(label_flatten == cls)[0]
        label_draw[inds1, 0] = palette[cls, 0]
        label_draw[inds1, 1] = palette[cls, 1]
        label_draw[inds1, 2] = palette[cls, 2]
        inds2 = np.where(image_flatten == cls)[0]
        image_draw[inds2, 0] = palette[cls, 0]
        image_draw[inds2, 1] = palette[cls, 1]
        image_draw[inds2, 2] = palette[cls, 2]

    for i in range(0, 3):
        label_draw[inds, i] = 1
        image_draw[inds, i] = 1

    image_draw = np.reshape(image_draw, (h, w, 3))
    label_draw = np.reshape(label_draw, (h, w, 3))

    return image_draw, label_draw


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def main():
    print('Build model ...')
    model_name = 'UM2Former'
    model = UM2Former(num_classes=24).to(device)

    summary(model, (32, 512, 512))

    # Load model (model of final epoch or best model evaluated on the validation set)
    model_path = 'D:/PyCharm/UM2Former/train/modal_param/S3_UM2Former/UM2Former_final.pth'
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Load data ...')
    data_root = 'G:/WHU_OHS/'
    image_prefix = ['S3']

    data_path_test_image = os.path.join(data_root, 'val', 'image')

    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_test_image)):
        for fname in fnames:
            if is_image_file(fname):
                for i in image_prefix:
                    if i + '_' in fname:
                        image_path = os.path.join(data_path_test_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        test_image_list.append(image_path)
                        test_label_list.append(label_path)

    assert len(test_image_list) == len(test_label_list)

    test_dataset = WHU_OHS_Dataset(image_file_list=test_image_list, label_file_list=test_label_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print('Predicting.')

    with torch.no_grad():
        model.eval()
        for data, label, name in tqdm(test_loader):
            data = data.to(device)
            pred = model(data)
            output = pred[0, :, :, :].argmax(axis=0)
            output = output.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            output = output.astype(np.uint8)
            if name == ['S3_0124.tif']:
                label109 = rearrange(label, 'c h w -> (c h) w')
                output109 = output

        output109, label109 = draw(output109, label109)

        plt.subplot(121)
        plt.imshow(np.array(label109))
        plt.subplot(122)
        plt.imshow(output109)
        plt.show()

        output_image = (output109 * 255).astype(np.uint8)  # 确保数据范围为 [0, 255]
        output_image = Image.fromarray(output_image)

        output_image.save('S3_0124.png', 'PNG', quality=95)


if __name__ == '__main__':
    main()