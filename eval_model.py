from data_loader import load_test_datasets
import torch
from datetime import datetime
import os
import logging
from my_model import  ResNet152_net2_Multi_16
from torch.utils.tensorboard import SummaryWriter

"""
   预测模型
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


# 日志文件
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


def eval_model(isr_model, data_name, dataloaders, class_name, weights):
    cls_num = len(class_name)
    writer = SummaryWriter(log_dir='visual_results')
    target_folder = 'misclassified_images'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    misclassified_images = []
    correct_labels = []
    predicted_labels = []
    results = []
    if 'FI' in data_name:
        datas = [[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 ]
    elif 'emotion_roi' in data_name:
        datas = [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]
    else:
        datas = [[0, 0],
                 [0, 0]]

    class_correct, class_total = [0] * cls_num, [0] * cls_num

    # logger = get_logger('/data/kangbo/pycharmWorkspace/ML-ISR/log/test/FI_best_resnet152_test_gram_aff_fi_abstract.log')  # 日志文件路径

    # logger.info('start testing!')

    print("testing is starting...")
    isr_model.eval()
    with torch.no_grad():
        # 在测试阶段进行循环遍历
        for phase in ['test']:
            running_corrects = 0  # 初始化正确分类计数
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 将图像数据移动到 GPU 或 CPU 上
                labels = labels.to(device)  # 将标签移动到 GPU 或 CPU 上
                emo_predicts_1, emo_predicts_2 = isr_model(inputs)  # 使用模型进行预测
                emo_predicts_1 = torch.softmax(emo_predicts_1, dim=1)
                emo_predicts_2 = torch.softmax(emo_predicts_2, dim=1)
                # print(emo_predicts_1)
                # print(emo_predicts_2)
                _, emo_predicts = torch.max(torch.softmax(emo_predicts_1, dim=1) + torch.softmax(emo_predicts_2, dim=1), 1)

                # _, emo_predicts = torch.max(emo_predicts, 1)   # 获取预测的标签
                # print("emo_predicts=", emo_predicts)
                running_corrects += torch.sum(emo_predicts == labels.data)  # 计算正确分 类的数量
                # print("running_corrects=", running_corrects)
                is_correct = (emo_predicts == labels.data)  # 判断每个样本是否被正确分类
                # print("is_correct=", is_correct)
                data_2 = emo_predicts.cpu().numpy()  # 将预测标签移动到 CPU 并转换为 NumPy 数组
                data_1 = labels.data.cpu().numpy()  # 将真实标签移动到 CPU 并转换为 NumPy 数组

                # 更新混淆矩阵（datas 变量）以记录分类错误的情况
                for i in range(len(emo_predicts)):
                    datas[data_1[i]][data_2[i]] += 1
                # 计算每个类别的正确分类和总样本数
                for i in range(len(emo_predicts)):
                    emotion_label = emo_predicts[i]
                    class_total[labels.data[i]] += 1
                    correct_count = is_correct[i].item()
                    class_correct[emotion_label] += correct_count
        # 计算测试精度
        epoch_acc = running_corrects.double() / len(dataloaders['test'].dataset)
        print('accuracy: %.3f%%' % (epoch_acc * 100))
        # logger.info('accuracy: %.3f%%' % (epoch_acc * 100))
        for data in datas:
            print(data)
        #   计算每个类别的精度并记录
        for i in range(cls_num):
            if class_total[i] == 0: class_total[i] = 1  # 避免除以零错误
            print('Accuracy of %11s: %.3f%%' % (class_name[i], 100.0 * class_correct[i] / class_total[i]))
            print(class_correct[i])
            print(class_total[i])
            # logger.info('Accuracy of %11s: %.3f%%' % (class_name[i], 100.0 * class_correct[i] / class_total[i]))
            # logging.info('Class Correct: %d' % class_correct[i])
            # logging.info('Class Total: %d' % class_total[i])

    return epoch_acc


def eval_rationality():
    data_name, dataloaders, class_name = load_test_datasets(data_dir="/root/lxy/lxy/lxy/FI_test")
    model_save_path = "/root/lxy/lxy/lxy/ML-ISR/model_state_dict/" + 'FI_new/' + 'FI_train_best_final_ResNet152_net2_Multi_16_4.pth'
    weight_file = torch.load(model_save_path)

    isr_model = ResNet152_net2_Multi_16()


    isr_model = isr_model.to(device=device)
    print('loading ' + model_save_path + ' model_state_dict...')
    isr_model.load_state_dict(weight_file['model_state_dict'])
    for params_name, param in isr_model.named_parameters():
        if param.requires_grad == True:
            print("  ", params_name)
    # print(list(isr_model.parameters()))
    eval_model(isr_model=isr_model, data_name=data_name, dataloaders=dataloaders, class_name=class_name,
               weights=weight_file)


def main():
    eval_rationality()


if __name__ == '__main__':
    main()
