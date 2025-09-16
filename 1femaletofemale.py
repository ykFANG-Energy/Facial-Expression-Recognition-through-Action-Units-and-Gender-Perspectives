import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

# 定义数据集类
class FaceDataset(Dataset):
    def __init__(self, images, vectors, transform=None):
        self.images = images.astype(np.float32)  # 确保图像为float32类型
        self.vectors = vectors.astype(np.float32)  # 确保向量为float32类型
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        vector = self.vectors[idx]
        if self.transform:
            image = self.transform(image)
        return image, vector

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    device_index = torch.cuda.current_device()
    print("PyTorch is using GPU")
except RuntimeError:
    device_index = -1
    print("PyTorch is using CPU")


# 路径设置
images_path = "/home/fangyuankun/cnn/stimuli_ecnu_AU/all_face"
details_path = "/home/fangyuankun/cnn/stimuli_ecnu_AU/vector_files/face_to_face/to_f2f.csv"

# 加载CSV数据
details = pd.read_csv(details_path)

# 定义加载和预处理图像的函数
def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (150, 200))  # 调整图像大小以适应模型输入
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # 归一化
    return img

# 加载图像数据
images = []
vectors = []
for _, row in details.iterrows():
    img_path = os.path.join(images_path, row['face_file'].split('/')[-1])
    print(f"Checking image path: {img_path}")
    if os.path.exists(img_path):
        img = load_image(img_path)
        images.append(img)
        emotion_vector = np.fromstring(row['AUamp'][1:-1], sep=' ')
        vectors.append(emotion_vector)
    else:
        print(f"File not found: {img_path}")

# 将列表转换为numpy数组
X = np.array(images).astype(np.float32)  # 确保图像数据为float32类型
y = np.array(vectors).astype(np.float32)  # 确保向量数据为float32类型

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('已分割数据集')

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建数据集和数据加载器
train_dataset = FaceDataset(X_train, y_train, transform=transform)
test_dataset = FaceDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 定义模型
class FaceModel(nn.Module):
    def __init__(self, output_size):
        super(FaceModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 9 * 12, 512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(nn.ReLU()(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.drop1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

# 自定义余弦相似度损失函数
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true / y_true.norm(dim=1, keepdim=True)
        y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
        return 1 - (y_true * y_pred).sum(dim=1).mean()

# # 自定义余弦相似度评估指标
# def cosine_similarity(y_true, y_pred):
#     y_true = y_true / y_true.norm(dim=1, keepdim=True)
#     y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
#     return (y_true * y_pred).sum(dim=1).mean()

# 创建模型
output_size = y_train.shape[1]
model = FaceModel(output_size).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CosineSimilarityLoss()

print('下一步是训练模型')

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 评估模型
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    test_loss = running_loss / len(test_loader.dataset)
    return test_loss

# 训练和评估模型
train_model(model, train_loader, criterion, optimizer, num_epochs=100)
print('训练中')
test_loss = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')
print(f"Test Cosine Similarity: {1 - test_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), '/home/fangyuankun/cnn/stimuli_ecnu_AU/model/female_to_female_statedict.pth')

# 如果需要，将模型移回 CPU
model = model.to("cpu")

# 结束
print("模型已保存，训练完成。")

