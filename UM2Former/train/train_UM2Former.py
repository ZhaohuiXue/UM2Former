import torch
import time
import datetime
import os
from utils.dataset import WHU_OHS_Dataset
from Nets.UM2former import UM2Former, UM2Former_OP8
from utils.EvaluateHelper import evaluate
import torch.optim as optim
from tqdm import tqdm
from ptflops import get_model_complexity_info

num_classes = 24
conf_mat = True

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

    # ################################################### build model ###################################################
    print('Build model ...')
    model_name = 'UM2Former'
    model = UM2Former(num_classes=num_classes).to(device)

    FLOPs, params = get_model_complexity_info(model, (32, 512, 512),
                                             as_strings=True, print_per_layer_stat=True)
    print(f"Total computational complexity: {FLOPs}")
    print(f"Total number of parameters: {params}")

    epoch_num = 250
    batch_size = 4

    # ############################################# dataset & dataloader ################################################
    print('Load data ...')

    data_root = 'G:/WHU_OHS/'

    # Choose which image to use for training
    # Remenber use list!!!!!!!!!!!!!!!!!!!!!
    image_prefix = ['T1']  # T1 S3 S8 T5 O21

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    data_path_train_image = os.path.join(data_root, 'tr', 'image')
    data_path_val_image = os.path.join(data_root, 'val', 'image')

    train_image_list = []
    train_label_list = []
    val_image_list = []
    val_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_train_image)):
        for fname in fnames:
            if is_image_file(fname):
                for i in image_prefix:
                    if i + '_' in fname:
                        image_path = os.path.join(data_path_train_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        train_image_list.append(image_path)
                        train_label_list.append(label_path)

    for root, paths, fnames in sorted(os.walk(data_path_val_image)):
        for fname in fnames:
            if is_image_file(fname):
                for j in image_prefix:
                    if j + '_' in fname:
                        image_path = os.path.join(data_path_val_image, fname)
                        label_path = image_path.replace('image', 'label')
                        assert os.path.exists(label_path)
                        assert os.path.exists(image_path)
                        val_image_list.append(image_path)
                        val_label_list.append(label_path)

    assert len(train_image_list) == len(train_label_list)
    assert len(val_image_list) == len(val_label_list)

    train_dataset = WHU_OHS_Dataset(image_file_list=train_image_list, label_file_list=train_label_list)
    val_dataset = WHU_OHS_Dataset(image_file_list=val_image_list, label_file_list=val_label_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=2, pin_memory=True)

    #################################################### optimizer ######################################################
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    # T1:lr=0.0005, weight decay=0 / O1-T8:lr=0.0001, weightdecay=0.0001
    optimizer = optim.Adam(params_to_optimize, lr=0.0005, betas=(0.9, 0.999), weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    model_path = './model/' + image_prefix[0] + '_' + model_name + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    print('Start training.')
    min_val_loss = 999

    start_time = time.time()
    times = torch.zeros(epoch_num)
    for epoch in range(epoch_num):
        print('Epoch: %d/%d' % (epoch + 1, epoch_num))
        print('Current learning rate: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']))

        model.train()
        batch_index = 0
        loss_sum = 0

        for data, label, _ in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            starter.record()
            pred = model(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # calculate inference time
            times[epoch] = curr_time

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            lr_schduler.step()

            loss_sum = loss_sum + loss.item()
            batch_index = batch_index + 1
            average_loss_cur = loss_sum / batch_index
            if batch_index % 10 == 0:
                print('training loss %.6f' % average_loss_cur)

        average_loss = loss_sum / batch_index
        print('Epoch [%d/%d] training loss %.6f' % (epoch + 1, epoch_num, average_loss))

        with torch.no_grad():
            model.eval()
            # T1:conf_mat=TRUE / to save time O1-T8:conf_mat=FALSE
            if conf_mat:
                conf, val_index, val_loss_sum = evaluate(model, val_loader, device=device,
                                                             num_classes=num_classes, criterion=criterion,
                                                             val_index=0, val_loss_sum=0, conf_mat=conf_mat)
                average_val_loss = val_loss_sum / val_index
                print('Epoch [%d/%d] validation loss %.6f\n' % (epoch + 1, epoch_num, average_val_loss))
                print(conf)
            else:
                val_index, val_loss_sum = evaluate(model, val_loader, device=device,
                                                   num_classes=num_classes, criterion=criterion,
                                                   val_index=0, val_loss_sum=0, conf_mat=conf_mat)
                average_val_loss = val_loss_sum / val_index
                print('Epoch [%d/%d] validation loss %.6f\n' % (epoch + 1, epoch_num, average_val_loss))

            if average_val_loss < min_val_loss:
                min_val_loss = average_val_loss
                # Update the best model evaluated on the validation set
                torch.save(model.state_dict(), model_path + model_name + '_update_' + str(epoch) + '.pth')

        # Save model regularly
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_path + model_name + '_' + str(epoch) + '.pth')

    # Save model for the final epoch
    torch.save(model.state_dict(), model_path + model_name + '_final.pth')
    # record the running time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))


if __name__ == '__main__':
    main()