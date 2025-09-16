import torch.nn as nn
import numpy as np
#from torch.testing._internal.common_utils import args
from torch.utils.data import Dataset, DataLoader
class FaceModel(nn.Module):
    def __init__(self, output_size=42):
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
# %%
# Pipeline #1 : Generating AUs from images
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import trange

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.ToTensor()
])

path = '/home/fangyuankun/cnn/stimuli_ecnu_AU/all_to_all3.pth'
final_model = FaceModel()
final_model.to(device)
final_model.load_state_dict(torch.load(path))
final_model.eval()
face_files = '/home/fangyuankun/cnn/stimuli_ecnu_AU/test_images'
AUs = None
for i in trange(face_files.shape[0]):
        path = face_files[i]
        # Input (Picture) Preprocessing
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0 
        img = preprocess(img).to(torch.float32)
        img = img.unsqueeze(0)
        with torch.no_grad(): 
                output = final_model(img)
        predicted_vector = output.numpy()   
        if AUs is None:
                AUs = predicted_vector
        else:
                AUs = np.concatenate((AUs, predicted_vector), axis=0)
print('Generated AUs: ', AUs.shape)