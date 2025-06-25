import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as dataloader
import os

# data_dir = "./datasets/twitter_1_5"
input_size = 224
batch_size = 32

def load_train_datasets_val(data_dir):
    data_transforms = {
        'train':transforms.Compose([
            # transforms.Resize((input_size, input_size)),
            # # transforms.RandomResizedCrop(size=input_size, scale=(0.2, 0.8), ratio=(1.0, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # resnet_gram数据处理方式
            transforms.RandomResizedCrop(224, scale=(256, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # transforms.Resize(256),
            # transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        ),
        'val':transforms.Compose([
            # 在验证集中进行数据预处理
            # transforms.Resize((input_size, input_size)),
            # # transforms.CenterCrop([input_size, input_size]),# 从中心裁剪指定大小的区域
            # # transforms.RandomHorizontalFlip(),# 随机水平翻转
            # transforms.ToTensor(), # 将图像转换为 PyTorch 的 Tensor 格式
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# 标准化图像

            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


            # # resnet_gram的测试集处理方式,（验证集本身的处理）
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]



        ),
        # 'test':transforms.Compose([
        #     transforms.Resize((input_size, input_size)),
        #     # transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        # ),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    print(len(image_datasets["train"]))
    # dataloader_dict = {x: dataloader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataloader_dict = {'train': 'train', 'val': 'val'}
    dataloader_dict['train'] = dataloader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_dict['val'] = dataloader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)

    class_name = image_datasets['train'].classes
    data_name = data_dir.split('/')[-1]
    count_data = len(image_datasets['train'])
    print(data_name, " train_datasets", "count: ", count_data)
    print(class_name)
    return data_name, dataloader_dict, class_name


def load_train_datasets(data_dir):
    data_transforms = {
        'train':transforms.Compose([
            # transforms.Resize((input_size, input_size)),
            # # transforms.RandomResizedCrop(size=input_size, scale=(0.2, 0.8), ratio=(1.0, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # resnet_gram数据处理方式
            transforms.RandomResizedCrop(224, scale=(256, 480)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            # transforms.Resize(256),
            # transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        ),
        'test':transforms.Compose([
            # 在验证集中进行数据预处理
            # transforms.Resize((input_size, input_size)),
            # # transforms.CenterCrop([input_size, input_size]),# 从中心裁剪指定大小的区域
            # # transforms.RandomHorizontalFlip(),# 随机水平翻转
            # transforms.ToTensor(), # 将图像转换为 PyTorch 的 Tensor 格式
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# 标准化图像

            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


            # # resnet_gram的测试集处理方式,（验证集本身的处理）
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]



        ),
        # 'test':transforms.Compose([
        #     transforms.Resize((input_size, input_size)),
        #     # transforms.CenterCrop(input_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        # ),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    print(len(image_datasets["train"]))
    # dataloader_dict = {x: dataloader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataloader_dict = {'train': 'train', 'test': 'test'}
    dataloader_dict['train'] = dataloader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader_dict['test'] = dataloader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)

    class_name = image_datasets['train'].classes
    data_name = data_dir.split('/')[-1]
    count_data = len(image_datasets['train'])
    print(data_name, " train_datasets", "count: ", count_data)
    print(class_name)
    return data_name, dataloader_dict, class_name

def load_test_datasets(data_dir):

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        ),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloader_dict = {x: dataloader(image_datasets[x], batch_size=1, shuffle=True, num_workers=2) for x in ['test']}

    class_name = image_datasets['test'].classes
    count_data = len(image_datasets['test'])
    data_name = data_dir.split('/')[-1]
    print(data_name, " test_datasets", "count: ", count_data)
    print(class_name)

    # for inputs, labels in dataloader_dict['test']:
    #     aa = inputs
    #     bb = labels
    #     a=1

    return  data_name, dataloader_dict, class_name

def main():
    load_train_datasets()
    load_test_datasets()

if __name__=='__main__':
    main()

