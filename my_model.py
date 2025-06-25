
import torch
import torch.nn as nn
import numpy as np
from net2 import resnet152
import cv2
from triplet_attention import TripletAttention
# 模型代码
def gram_matrix(input):
    # 获取批处理大小、深度、高度和宽度
    b, c, h, w = input.size()
    # 将特征映射调整为二维矩阵形式
    features = input.view(b, c, h * w)
    # 计算特征与其转置的乘积来获得Gram矩阵
    # 使用bmm(batch matrix-matrix product)来处理批次
    G = torch.bmm(features, features.transpose(1, 2))
    # 标准化Gram矩阵的值
    return G.div(c * h * w)

# 高低，添加颜色信息，并添加se，triple_attention，整体模型
class ResNet152_net2_Multi_16(nn.Module):
    def __init__(self, img_dim=2048, fine_tune=False, num_class=8):
        super().__init__()
        print("loading model...")
        self.resnet = resnet152(pretrained=True)

        self.style1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 64, out_features=64)
        )
        self.style2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*256, out_features=256)
        )

         # 决定是否冻结预训练模型的一部分或全部层。如果目标任务的数据集很小，通常建议冻结预训练模型的卷积层，只训练新添加的层。这有助于防止在小数据集上发生过拟合。
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.load_state_dict(torch.load("/data/kangbo/pycharmWorkspace/ML-ISR/resnet152.pth"))
        # self.mobilenetv2 = models.mobilenet_v2(pretrained=True)

        self.in_dim = self.resnet.fc.in_features
        self.img_dim = img_dim
        self.num_class = num_class
        self.fla = nn.Flatten()
        self.resnet.fc = nn.Linear(self.img_dim, self.num_class)
        self.fc = nn.Linear(512,8)

        self.fc_gram = nn.Linear(64+256+256,512)
        self.fc_color = nn.Linear(18,64)
        self.fc_color_2 = nn.Linear(64,256)
        # self.fc2_low=nn.Linear(512,2)
        # self.fc2_high=nn.Linear(2048,2)
        self.te1 = TripletAttention(64)
        self.te2 = TripletAttention(256)
        self.te3 = TripletAttention(2048)
        self.se = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.fc_all = nn.Linear(2048+512,2048)
        # self.feature_fusion = FeatureFusion(512, 2048, 1024)


    def forward(self, image):

        # 提取 RGB 颜色信息
        rgb_mean = torch.mean(image, dim=(2, 3))
        rgb_std = torch.std(image, dim=(2, 3))
        rgb_skewness = torch.mean(
            ((image - rgb_mean.unsqueeze(-1).unsqueeze(-1)) / rgb_std.unsqueeze(-1).unsqueeze(-1)) ** 3, dim=(2, 3))

        # 将图像转换为 numpy 数组
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        # image_np = image_np.to("cuda")

        # 转换颜色空间为 HSV
        hsv_image = np.zeros_like(image_np)
        for i in range(len(image_np)):
            hsv_image[i] = cv2.cvtColor(image_np[i], cv2.COLOR_RGB2HSV)

        # 计算 HSV 颜色信息的一阶、二阶和三阶矩
        hsv_mean = np.mean(hsv_image, axis=(1, 2))
        hsv_std = np.std(hsv_image, axis=(1, 2))
        hsv_skewness = np.mean(((hsv_image - hsv_mean.reshape(-1, 1, 1, 3)) / hsv_std.reshape(-1, 1, 1, 3)) ** 3,
                               axis=(1, 2))
        hsv_mean = torch.tensor(hsv_mean).to(image.device)
        hsv_std = torch.tensor(hsv_std).to(image.device)
        hsv_skewness = torch.tensor(hsv_skewness).to(image.device)


        color_info = torch.cat((rgb_mean, rgb_std, rgb_skewness, hsv_mean, hsv_std, hsv_skewness), dim=1)

        color_info = self.fc_color(color_info)
        color_info = self.fc_color_2(color_info)
        color_info = self.se(color_info)

        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)  #64*112*112
        # x0_0 = self.te1(x0)
        x = self.resnet.maxpool(x0)  # 第一阶段进行普通卷积 变成原来1/4

        # 其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。

        x1 = self.resnet.layer1(x)  #256*56*56
        # x1_1 = self.te2(x1)
        x2 = self.resnet.layer2(x1)  #512*28*28
        x3 = self.resnet.layer3(x2)  #1024*14*14
        x4 = self.resnet.layer4(x3)  #2048*7*7

        x0 = self.te1(x0)
        x1 = self.te2(x1)
        x11 = gram_matrix(x0)
        x22 = gram_matrix(x1)
        x11 = self.style1(x11)
        x22 = self.style2(x22)
        x12 = torch.cat((color_info,x11,x22),dim=1)
        # x12 = torch.cat((x11,x22),dim=1)
        x12 = self.fc_gram(x12)

        x4 = self.te3(x4)
        x = self.resnet.avgpool(x4)
        x = torch.flatten(x, 1)

        # x = self.feature_fusion(x12,x)


        # x = self.resnet.fc(x)
        # x = torch.concat((x,x12),dim=1)
        # x = self.fc_all(x)
        x = self.resnet.fc(x)
        # if isinstance(self.resnet.fc,nn.Linear):
        #     print(self.resnet.fc)

        return x,self.fc(x12)


# 高低，添加颜色信息，并添加se，triple_attention,二分类
class ResNet152_net2_Multi_16_2(nn.Module):
    def __init__(self, img_dim=2048, fine_tune=False, num_class=2):
        super().__init__()
        print("loading model...")
        self.resnet = resnet152(pretrained=True)

        self.style1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 64, out_features=64)
        )
        self.style2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*256, out_features=256)
        )

         # 决定是否冻结预训练模型的一部分或全部层。如果目标任务的数据集很小，通常建议冻结预训练模型的卷积层，只训练新添加的层。这有助于防止在小数据集上发生过拟合。
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.load_state_dict(torch.load("/data/kangbo/pycharmWorkspace/ML-ISR/resnet152.pth"))
        # self.mobilenetv2 = models.mobilenet_v2(pretrained=True)

        self.in_dim = self.resnet.fc.in_features
        self.img_dim = img_dim
        self.num_class = num_class
        self.fla = nn.Flatten()
        self.resnet.fc = nn.Linear(self.img_dim, self.num_class)
        self.fc = nn.Linear(512,2)

        self.fc_gram = nn.Linear(64+256+256,512)
        self.fc_color = nn.Linear(18,64)
        self.fc_color_2 = nn.Linear(64,256)
        # self.fc2_low=nn.Linear(512,2)
        # self.fc2_high=nn.Linear(2048,2)
        self.te1 = TripletAttention(64)
        self.te2 = TripletAttention(256)
        self.te3 = TripletAttention(2048)
        self.se = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.fc_all = nn.Linear(2048+512,2048)
        # self.feature_fusion = FeatureFusion(512, 2048, 1024)
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self, image):

        # 提取 RGB 颜色信息
        rgb_mean = torch.mean(image, dim=(2, 3))
        rgb_std = torch.std(image, dim=(2, 3))
        rgb_skewness = torch.mean(
            ((image - rgb_mean.unsqueeze(-1).unsqueeze(-1)) / rgb_std.unsqueeze(-1).unsqueeze(-1)) ** 3, dim=(2, 3))

        # 将图像转换为 numpy 数组
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        # image_np = image_np.to("cuda")

        # 转换颜色空间为 HSV
        hsv_image = np.zeros_like(image_np)
        for i in range(len(image_np)):
            hsv_image[i] = cv2.cvtColor(image_np[i], cv2.COLOR_RGB2HSV)

        # 计算 HSV 颜色信息的一阶、二阶和三阶矩
        hsv_mean = np.mean(hsv_image, axis=(1, 2))
        hsv_std = np.std(hsv_image, axis=(1, 2))
        hsv_skewness = np.mean(((hsv_image - hsv_mean.reshape(-1, 1, 1, 3)) / hsv_std.reshape(-1, 1, 1, 3)) ** 3,
                               axis=(1, 2))
        hsv_mean = torch.tensor(hsv_mean).to(image.device)
        hsv_std = torch.tensor(hsv_std).to(image.device)
        hsv_skewness = torch.tensor(hsv_skewness).to(image.device)


        color_info = torch.cat((rgb_mean, rgb_std, rgb_skewness, hsv_mean, hsv_std, hsv_skewness), dim=1)

        color_info = self.fc_color(color_info)
        color_info = self.fc_color_2(color_info)
        color_info = self.se(color_info)

        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)  #64*112*112
        # x0_0 = self.te1(x0)
        x = self.resnet.maxpool(x0)  # 第一阶段进行普通卷积 变成原来1/4

        # 其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。

        x1 = self.resnet.layer1(x)  #256*56*56
        # x1_1 = self.te2(x1)
        x2 = self.resnet.layer2(x1)  #512*28*28
        x3 = self.resnet.layer3(x2)  #1024*14*14
        x4 = self.resnet.layer4(x3)  #2048*7*7

        x0 = self.te1(x0)
        x1 = self.te2(x1)
        x11 = gram_matrix(x0)
        x22 = gram_matrix(x1)
        x11 = self.style1(x11)
        x22 = self.style2(x22)
        x12 = torch.cat((color_info,x11,x22),dim=1)
        # x12 = torch.cat((x11,x22),dim=1)
        x12 = self.fc_gram(x12)

        x4 = self.te3(x4)
        x = self.resnet.avgpool(x4)
        x = torch.flatten(x, 1)

        # x = self.feature_fusion(x12,x)


        # x = self.resnet.fc(x)
        # x = torch.concat((x,x12),dim=1)
        # x = self.fc_all(x)
        x = self.resnet.fc(x)
        # if isinstance(self.resnet.fc,nn.Linear):
        #     print(self.resnet.fc)

        return x,self.fc(x12)


# 无颜色
class ResNet152_net2_Multi_20(nn.Module):
    def __init__(self, img_dim=2048, fine_tune=False, num_class=8):
        super().__init__()
        print("loading model...")
        self.resnet = resnet152(pretrained=True)

        self.style1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 64, out_features=64)
        )
        self.style2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*256, out_features=256)
        )

         # 决定是否冻结预训练模型的一部分或全部层。如果目标任务的数据集很小，通常建议冻结预训练模型的卷积层，只训练新添加的层。这有助于防止在小数据集上发生过拟合。
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.load_state_dict(torch.load("/data/kangbo/pycharmWorkspace/ML-ISR/resnet152.pth"))
        # self.mobilenetv2 = models.mobilenet_v2(pretrained=True)

        self.in_dim = self.resnet.fc.in_features
        self.img_dim = img_dim
        self.num_class = num_class
        self.fla = nn.Flatten()
        self.resnet.fc = nn.Linear(self.img_dim, self.num_class)
        self.fc = nn.Linear(512,8)

        self.fc_gram = nn.Linear(64+256,512)
        self.fc_color = nn.Linear(18,64)
        self.fc_color_2 = nn.Linear(64,256)
        # self.fc2_low=nn.Linear(512,2)
        # self.fc2_high=nn.Linear(2048,2)
        self.te1 = TripletAttention(64)
        self.te2 = TripletAttention(256)
        self.te3 = TripletAttention(2048)
        self.se = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.fc_all = nn.Linear(2048+512,2048)
        # self.feature_fusion = FeatureFusion(512, 2048, 1024)


    def forward(self, image):

        # 提取 RGB 颜色信息
        rgb_mean = torch.mean(image, dim=(2, 3))
        rgb_std = torch.std(image, dim=(2, 3))
        rgb_skewness = torch.mean(
            ((image - rgb_mean.unsqueeze(-1).unsqueeze(-1)) / rgb_std.unsqueeze(-1).unsqueeze(-1)) ** 3, dim=(2, 3))

        # 将图像转换为 numpy 数组
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        # image_np = image_np.to("cuda")

        # 转换颜色空间为 HSV
        hsv_image = np.zeros_like(image_np)
        for i in range(len(image_np)):
            hsv_image[i] = cv2.cvtColor(image_np[i], cv2.COLOR_RGB2HSV)

        # 计算 HSV 颜色信息的一阶、二阶和三阶矩
        hsv_mean = np.mean(hsv_image, axis=(1, 2))
        hsv_std = np.std(hsv_image, axis=(1, 2))
        hsv_skewness = np.mean(((hsv_image - hsv_mean.reshape(-1, 1, 1, 3)) / hsv_std.reshape(-1, 1, 1, 3)) ** 3,
                               axis=(1, 2))
        hsv_mean = torch.tensor(hsv_mean).to(image.device)
        hsv_std = torch.tensor(hsv_std).to(image.device)
        hsv_skewness = torch.tensor(hsv_skewness).to(image.device)


        color_info = torch.cat((rgb_mean, rgb_std, rgb_skewness, hsv_mean, hsv_std, hsv_skewness), dim=1)

        color_info = self.fc_color(color_info)
        color_info = self.fc_color_2(color_info)
        color_info = self.se(color_info)

        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)  #64*112*112
        # x0_0 = self.te1(x0)
        x = self.resnet.maxpool(x0)  # 第一阶段进行普通卷积 变成原来1/4

        # 其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。

        x1 = self.resnet.layer1(x)  #256*56*56
        # x1_1 = self.te2(x1)
        x2 = self.resnet.layer2(x1)  #512*28*28
        x3 = self.resnet.layer3(x2)  #1024*14*14
        x4 = self.resnet.layer4(x3)  #2048*7*7

        x0 = self.te1(x0)
        x1 = self.te2(x1)
        x11 = gram_matrix(x0)
        x22 = gram_matrix(x1)
        x11 = self.style1(x11)
        x22 = self.style2(x22)
        # x12 = torch.cat((color_info,x11,x22),dim=1)
        x12 = torch.cat((x11,x22),dim=1)
        x12 = self.fc_gram(x12)

        x4 = self.te3(x4)
        x = self.resnet.avgpool(x4)
        x = torch.flatten(x, 1)

        # x = self.feature_fusion(x12,x)


        # x = self.resnet.fc(x)
        # x = torch.concat((x,x12),dim=1)
        # x = self.fc_all(x)
        x = self.resnet.fc(x)
        # if isinstance(self.resnet.fc,nn.Linear):
        #     print(self.resnet.fc)

        return x,self.fc(x12)

# 无多层特征提取模块
class ResNet152_Gram_color(nn.Module):
    def __init__(self, img_dim=4096, fine_tune=False, num_class=8):
        super().__init__()
        print("loading model...")
        self.resnet = resnet152(pretrained=True)
        # 决定是否冻结预训练模型的一部分或全部层。如果目标任务的数据集很小，通常建议冻结预训练模型的卷积层，只训练新添加的层。这有助于防止在小数据集上发生过拟合。
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # self.mobilenetv2 = models.mobilenet_v2(pretrained=True)

        self.style1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64 * 64, out_features=64)
        )
        self.style2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 256, out_features=256)
        )
        # self.style3 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(in_features=512 * 512, out_features=512)
        # )
        self.droupout = nn.Dropout(p=0.5)
        self.style_fc = nn.Linear(in_features= 64+256, out_features=2048)
        self.se = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
        self.fc_color = nn.Linear(18, 64)
        self.fc_color_2 = nn.Linear(64, 256)

       # self.classification = nn.Linear(in_features=2 * 2048, out_features=8)

        self.in_dim = self.resnet.fc.in_features
        self.img_dim = img_dim
        self.num_class = num_class
        self.fla = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.resnet.fc = nn.Linear(self.img_dim, self.num_class)

    def forward(self, image):
        # 提取 RGB 颜色信息
        rgb_mean = torch.mean(image, dim=(2, 3))
        rgb_std = torch.std(image, dim=(2, 3))
        rgb_skewness = torch.mean(
            ((image - rgb_mean.unsqueeze(-1).unsqueeze(-1)) / rgb_std.unsqueeze(-1).unsqueeze(-1)) ** 3, dim=(2, 3))

        # 将图像转换为 numpy 数组
        image_np = image.permute(0, 2, 3, 1).cpu().numpy()
        # image_np = image_np.to("cuda")

        # 转换颜色空间为 HSV
        hsv_image = np.zeros_like(image_np)
        for i in range(len(image_np)):
            hsv_image[i] = cv2.cvtColor(image_np[i], cv2.COLOR_RGB2HSV)

        # 计算 HSV 颜色信息的一阶、二阶和三阶矩
        hsv_mean = np.mean(hsv_image, axis=(1, 2))
        hsv_std = np.std(hsv_image, axis=(1, 2))
        hsv_skewness = np.mean(((hsv_image - hsv_mean.reshape(-1, 1, 1, 3)) / hsv_std.reshape(-1, 1, 1, 3)) ** 3,
                               axis=(1, 2))
        hsv_mean = torch.tensor(hsv_mean).to(image.device)
        hsv_std = torch.tensor(hsv_std).to(image.device)
        hsv_skewness = torch.tensor(hsv_skewness).to(image.device)

        color_info = torch.cat((rgb_mean, rgb_std, rgb_skewness, hsv_mean, hsv_std, hsv_skewness), dim=1)

        color_info = self.fc_color(color_info)
        color_info = self.fc_color_2(color_info)
        color_info = self.se(color_info)

        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x)  #64*112*112
        # gram_1 =
        x = self.resnet.maxpool(x0)  # 第一阶段进行普通卷积 变成原来1/4

        # 其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。

        x1 = self.resnet.layer1(x)  #256*56*56
        x2 = self.resnet.layer2(x1)  #512*28*28
        x3 = self.resnet.layer3(x2)  #1024*14*14
        x4 = self.resnet.layer4(x3)  #2048*7*7

        # x = self.resnet.avgpool(x4)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)
        # x = self.resnet.fc(x)
        # if isinstance(self.resnet.fc,nn.Linear):
        #     print(self.resnet.fc)

        x11 = gram_matrix(x0)
        x22 = gram_matrix(x1)
        # x33 = gram_matrix(x2)

        x1 = self.style1(x11)
        x2 = self.style2(x22)
        # x3 = self.style3(x33)

        # x12 = torch.concat([x1, x2,color_info], dim=1)
        x12 = torch.concat([x1, x2], dim=1)
        # x123 = torch.concat([x1, x2, x3], dim=1)
        # x123=self.droupout(x123)
        x12 = self.style_fc(x12)

        x = self.resnet.avgpool(x4)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        x124 = torch.concat([x, x12], dim=1)
        # x1234 = self.fc1(x1234)
        x124=self.droupout(x124)
        x124 = self.resnet.fc(x124)

        return x124

