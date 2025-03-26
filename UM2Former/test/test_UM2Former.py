import torch
import os
import numpy as np
from utils.dataset import WHU_OHS_Dataset
from Nets.UM2former import UM2Former, UM2Former_OP8
from tqdm import tqdm

class_num = 24

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMG_EXTENSION = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSION)


def genConfusionMatrix(numClass, imgPredict, imgLabel):
    mask = (imgLabel != -1)
    label = numClass * imgLabel[mask] + imgPredict[mask]
    count = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def main():
    print('Build model ...')
    model_name = 'UM2Former'
    model = UM2Former(num_classes=class_num).to(device)

    # Load final epoch or best model evaluated on the validation set
    model_path = 'D:/PyCharm/UM2Former/train/modal_param/O21_UM2Former/UM2Former_final.pth'  # final 235 final 240 195
    model.load_state_dict(torch.load(model_path), strict=False)
    print("Loaded trained model.")

    print('Load data ...')
    data_root = 'G:/WHU_OHS/'
    image_prefix = ['O21']

    data_path_test_image = os.path.join(data_root, 'ts', 'image')

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

    print('Testing.')

    with torch.no_grad():
        model.eval()
        confusionmat = torch.zeros([class_num, class_num])
        confusionmat = confusionmat.to(device)

        for data, label, _ in tqdm(test_loader):
            data = data.to(device)
            label = label.to(device)

            pred = model(data)

            output = pred[0, :, :, :].argmax(axis=0)
            label = label[0, :, :]

            confusionmat_tmp = genConfusionMatrix(class_num, output, label)
            confusionmat = confusionmat + confusionmat_tmp

    confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 1e-7

    pa = a / (c + eps)
    ua = a / (b + eps)
    print('PA:', pa)
    print('UA:', ua)

    f1 = 2 * pa * ua / (pa + ua + eps)
    print('F1:', f1)

    mean_f1 = np.nanmean(f1)
    print('mean F1:', mean_f1)

    oa = np.sum(a) / np.sum(confusionmat)
    print('OA:', oa)

    pe = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (oa - pe) / (1 - pe)
    print('Kappa:', Kappa)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    print('Iou:', IoU)
    print('mIoU:', mIoU)


if __name__ == '__main__':
    main()