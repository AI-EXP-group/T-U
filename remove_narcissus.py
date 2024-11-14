import torch.optim as optim
from util import *
from models import RES

class MyCIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, label = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def remove_backdoor(lab=None,category=None,lamda_1=None,alpha_2=None):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_wipe_out = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    transform_after = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),  # cifar10
    ])
    # Load model weights
    path_model = 'poisoned_model_weight_path'
    para = torch.load(path_model)
    model = RES.ResNet18()
    # Because the size of the CIFAR10 dataset is small, modify the convolutional kernel size
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(para)
    # copy_model is used when calculating KD losses
    copy_model = RES.ResNet18()
    copy_model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    copy_model.fc = torch.nn.Linear(512, 10)
    copy_model.load_state_dict(para)
    model = model.to(device)
    copy_model=copy_model.to(device)
    # Load the data
    ori_wipe_out = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,transform=transform_wipe_out)
    ori_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    # Required when testing ASR
    attack_trigger = torch.load('target class trigger path')
    clean_data = []  # Store clean pictures
    patch_data = []  # Stores the data of the pasted trigger
    test_data = []  # Store the data for testing
    for z in range(10):
        per_data = [[image, label] for image, label in ori_test if label == z]
        wipe_out_data = [[image, label] for image, label in ori_wipe_out if label == z]
        num = int(len(per_data) * 0.5)
        clean_data.extend(wipe_out_data[:num])
        test_data.extend((per_data[num:]))
    for i in range(10):
        defense_trigger = torch.load('defense trigger path')
        patch_trigger = [[torch.clamp(image + defense_trigger[0],0,1), label] for image, label in clean_data]
        patch_data.extend(patch_trigger)
    Loss1_data = MyCIFAR10Dataset(patch_data, transform_after)
    patch_data_dataloader = DataLoader(Loss1_data, batch_size=200, shuffle=True)
    Loss2_data = MyCIFAR10Dataset(clean_data, transform_after)
    clean_data_dataloader = DataLoader(Loss2_data, batch_size=20, shuffle=True)
    # Prepare the test ACC data
    acc_data = MyCIFAR10Dataset(test_data, transform_after)
    acc_data_dataloader = DataLoader(acc_data, batch_size=256, shuffle=True)
    # Prepare the ASR data for testing
    asr_data = [[image + attack_trigger[0] * 2, label] for image, label in test_data if label != lab]
    asr_data = MyCIFAR10Dataset(asr_data, transform_after)
    asr_data_dataloader = DataLoader(asr_data, batch_size=128, shuffle=True)
    mse = nn.MSELoss()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sche = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    model.eval()
    total = 0
    correct = 0
    for image, label in acc_data_dataloader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model.forward(image, reture_feature=False)
        # 得到预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 更新计数器
        total += label.size(0)
        correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print("The original ACC was：" + str(accuracy))

    total = 0
    correct = 0
    for image, label in asr_data_dataloader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model.forward(image, reture_feature=False)
        # 得到预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 更新计数器
        total += label.size(0)
        correct += (predicted == lab).sum().item()
    asr = 100 * correct / total
    print("The original ASR was：" + str(asr))

    for epoch in range(30):
        loss_list = []
        model.train()
        dataloader1_iter = iter(patch_data_dataloader)
        dataloader2_iter = iter(clean_data_dataloader)
        num_batches = min(len(dataloader1_iter), len(dataloader2_iter))
        for batch_idx in range(num_batches):
            inputs1, targets1 = next(dataloader1_iter)
            inputs1, targets1 = inputs1.to(device), targets1.to(device)
            inputs2, targets2 = next(dataloader2_iter)
            inputs2, targets2 = inputs2.to(device), targets2.to(device)
            outputs1 = model.forward(inputs1, reture_feature=False)
            outputs3=model.forward(inputs2, reture_feature=True)
            outputs4=copy_model.forward(inputs2, reture_feature=True)
            loss1 = criterion(outputs1, targets1)
            loss3=mse(outputs3,outputs4)
            loss =lamda_1*loss1+alpha_2*loss3
            optimizer.zero_grad()
            loss.backward()
            loss_list.append(float(loss.data))
            optimizer.step()
        sche.step()
        ave_loss = np.average(np.array(loss_list))
        print('Epoch:%d, Loss: %.06f' % (epoch, ave_loss))
