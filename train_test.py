from data_loader import load_train_datasets
import torch
from torch import nn, optim
from datetime import datetime
import copy
import logging
from my_model import ResNet152_net2_Multi_16_2

# from net import ResNet152WithGram
# 二分类训练代码
from net2 import resnet152

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Using GPU ", device)


def calc_epoch_time(start_time, end_time):
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    data = end - start  # 天数小时分钟秒
    days = (end - start).days  # 天数
    hours = str(data).split(':')[0].replace(' ', '')  # 小时
    minutes = str(data).split(':')[1]  # 分钟
    seconds = str(data).split(':')[2]  # 秒
    total_time = str(days * 24 * 60 + 60 * int(hours) + int(minutes)) + 'm:' + str(seconds) + 's'  # ×分钟x秒
    return total_time


def optimizer(emo_params):
    print('loading adam optimizer...')

    emo_optimizer = torch.optim.SGD(emo_params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=emo_optimizer, step_size=10, gamma=0.1)

    return emo_optimizer, scheduler


# 损失函数的设计
def loss_func():
    print('loading entropy loss...')
    # class_weights = torch.tensor(
    #     [2181 / 2277, 2181 / 1267, 2181 / 783, 2181 / 953, 2181 / 3776, 2181 / 2111, 2181 / 2197, 2181 / 4084])
    # class_weights = class_weights.to(device)
    class_weights = torch.tensor(
        [1, 10])
    emo_loss_func = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return emo_loss_func


def translabel(label8, map_8to2):
    label2 = torch.zeros((label8.shape), dtype=label8.dtype)
    for i in range(label2.shape[0]):
        label2[i] = map_8to2[label8[i].item()]
    return label2


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train_model(isr_model, data_name, dataloaders, isr_optimizer, scheduler, isr_loss, class_name):
    epochs = 50
    logger = get_logger('/root/lxy/lxy/lxy/ML-ISR/log_new/ResNet152_net2_Multi_162_2.log')
    isr_loss.weight = torch.tensor([25, 1], dtype=torch.float32).to(device)
    logger.info('start training!')

    classess = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
    map_8to2 = {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0}

    logger.info("training is starting...")
    best_model_weights = copy.deepcopy(isr_model.state_dict())
    best_acc = 0.0
    # epoch_acc = -1
    for epoch in range(epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        # logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logger.info('-' * 10)
        for phase in ['train', 'test']:
            start_time = datetime.now()  # 获取当前系统时间
            if phase == 'train':
                isr_model.train()
            elif phase == 'test':
                isr_model.eval()
                # continue

            running_loss_1 = 0.0
            running_loss_2 = 0.0
            running_loss_3 = 0.0
            # running_corrects_1 = 0
            # running_corrects_2 = 0
            # running_corrects = 0
            running_corrects_1_2label = 0
            running_corrects_2_2label = 0
            running_corrects_2label = 0
            cont_LoT = 0
            cont_PoT = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels8 = translabel(labels, map_8to2)
                labels8 = labels8.to(device)
                isr_optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    emo_predicts_high_2label, emo_predicts_low_2label = isr_model(inputs)

                    loss_1_2label = isr_loss(emo_predicts_high_2label, labels8)
                    loss_2_2label = isr_loss(emo_predicts_low_2label, labels8)
                    loss_3_2label = isr_loss(emo_predicts_high_2label + emo_predicts_low_2label, labels8)

                    # loss_1 = isr_loss(emo_predicts_1, labels)
                    # loss_2 = isr_loss(emo_predicts_2, labels)
                    # loss_3 = isr_loss(emo_predicts_1+emo_predicts_2, labels)
                    # final_loss = loss_1+loss_2+loss_3+loss_1_2label+loss_2_2label+loss_3_2label
                    final_loss = loss_1_2label + loss_2_2label + loss_3_2label

                    # _, emo_predicts = torch.max(torch.softmax(emo_predicts_1,dim=1)+torch.softmax(emo_predicts_2,dim=1), 1)
                    #
                    # _, emo_predicts_1 = torch.max(emo_predicts_1, 1)
                    # _, emo_predicts_2 = torch.max(emo_predicts_2, 1)

                    _, emo_predicts_2label = torch.max(
                        torch.softmax(emo_predicts_high_2label, dim=1) + torch.softmax(emo_predicts_low_2label, dim=1),
                        1)

                    _, emo_predicts_1_2label = torch.max(emo_predicts_high_2label, 1)
                    _, emo_predicts_2_2label = torch.max(emo_predicts_low_2label, 1)
                    for i in range(labels8.shape[0]):
                        if labels8[i].item() == 0:
                            cont_LoT += 1
                            if emo_predicts_2label[i].item() == 0: cont_PoT += 1
                    if phase == 'train':
                        final_loss.backward()
                        isr_optimizer.step()
                running_loss_1 += loss_1_2label.item() * inputs.size(0)
                running_loss_2 += loss_2_2label.item() * inputs.size(0)
                running_loss_3 += final_loss.item() * inputs.size(0)

                # running_corrects_1 += torch.sum(emo_predicts_1 == labels.data)
                # running_corrects_2 += torch.sum(emo_predicts_2 == labels.data)

                running_corrects_1_2label += torch.sum(emo_predicts_1_2label == labels8.data)
                running_corrects_2_2label += torch.sum(emo_predicts_2_2label == labels8.data)

                # running_corrects += torch.sum(emo_predicts == labels.data)
                running_corrects_2label += torch.sum(emo_predicts_2label == labels8.data)

            if phase == 'train':
                scheduler.step()
            emo_lr = isr_optimizer.state_dict()['param_groups'][0]['lr']

            epoch_loss_1 = running_loss_1 / len(dataloaders[phase].dataset)
            epoch_loss_2 = running_loss_2 / len(dataloaders[phase].dataset)
            epoch_loss = running_loss_3 / len(dataloaders[phase].dataset)

            # epoch_acc_1 = running_corrects_1.double() / len(dataloaders[phase].dataset)
            # epoch_acc_2 = running_corrects_2.double() / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_acc_1_2label = running_corrects_1_2label.double() / len(dataloaders[phase].dataset)
            epoch_acc_2_2label = running_corrects_2_2label.double() / len(dataloaders[phase].dataset)
            epoch_acc_2label = running_corrects_2label.double() / len(dataloaders[phase].dataset)

            if phase == 'test' and epoch_acc_2label > best_acc:
                best_model_weights = copy.deepcopy(isr_model.state_dict())
                best_acc = epoch_acc_2label
                # 保存模型

                model_save_path = "/root/lxy/lxy/lxy/ML-ISR/model_state_dict/" + 'FI_new' + '/'
                torch.save({
                    'model_state_dict': isr_model.state_dict(),
                }, model_save_path + 'FI' + '_train_best_final_ResNet152_net2_Multi_162_2.pth')

            end_time = datetime.now()  # 获取当前系统时间
            epoch_time = calc_epoch_time(start_time, end_time)
            logger.info(
                '%5s_loss:  %.5f   %.5f  %.5f  lr: %.12f  accuracy:  %.3f%%   %.3f%%   %.3f%%  %5s_time: %s 阳性率%.3f' % (
                phase, epoch_loss_1, epoch_loss_2, epoch_loss, emo_lr, epoch_acc_1_2label * 100,
                epoch_acc_2_2label * 100, epoch_acc_2label * 100, phase, epoch_time, cont_PoT / cont_LoT))
            # print('accuracy2:  %.3f%%   %.3f%%   %.3f%%' % (epoch_acc_1_2label* 100,epoch_acc_2_2label* 100, epoch_acc_2label * 100))

            # logger.info('%5s_loss: %.5f  lr: %.12f  accuracy: %.3f%%  %5s_time: %s' % (
            # phase, epoch_loss, emo_lr, epoch_acc * 100, phase, epoch_time))

    # logger.info('finish training!')
    logger.info("finishing training!")
    logger.info("best_test_acc: %s", best_acc)
    if 'FI' in data_name:
        isr_model.load_state_dict(best_model_weights)
        # print("best_test_acc:", best_acc)

        # logger.info("best_val_acc:", best_acc)
    return isr_model, isr_optimizer


def main():
    data_name, dataloaders, class_name = load_train_datasets(
        data_dir="/root/lxy/lxy/lxy/FI_test")
    model_save_path = "/root/lxy/lxy/lxy/ML-ISR/model_state_dict/" + 'FI_new' + '/'
    # isr_model = resnet152()
    isr_model = ResNet152_net2_Multi_16_2()
    # torch.save(isr_model.state_dict(),"resnet152.pth")
    # isr_model.fc = torch.nn.Linear(2048,8)
    # isr_model.load_state_dict(torch.load("/data/lxj/MldrNet/resnet152-394f9c45.pth"))
    # isr_model.fc=torch.nn.Linear(in_features=2048,out_features=8)
    isr_model = isr_model.to(device=device)
    isr_loss = loss_func()
    isr_optimizer, scheduler = optimizer(isr_model.parameters())
    for params_name, param in isr_model.named_parameters():
        if param.requires_grad == True:
            print("  ", params_name)

    isr_model, isr_optimizer = train_model(isr_model=isr_model, data_name=data_name, dataloaders=dataloaders,
                                           isr_optimizer=isr_optimizer, scheduler=scheduler, isr_loss=isr_loss,
                                           class_name=class_name)


if __name__ == '__main__':
    main()

