# 核心组件
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 数据变换
transform = transforms.Compose([
    transforms.ToTensor(),          # 转为张量
    transforms.Normalize((0.5,), (0.5,)),  # 标准化
])

# 数据加载器
dataset = CustomDataset(X, Y, transform)
dataloader = DataLoader(dataset, 
                       batch_size=32,
                       shuffle=True,
                       num_workers=4)  # 多进程加载