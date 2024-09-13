
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
from unidecode import unidecode
import torch
import torch.nn as nn
device = "mps" if torch.backends.mps.is_available(
) else "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
labels = ['tức giận', 'ghê tởm', 'sợ hãi', 'vui vẻ',
          'buồn', 'bất ngờ', 'tự nhiên', 'khinh miệt']

# labels = ['tức giận','ghê tởm','đáng sợ','vui vẻ','tự nhiên','buồn','ngạc nhiên']
class VGG4(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG4, self).__init__()

        # Các lớp Convolution và Pooling
        self.features = nn.Sequential(
            # Block 1
            # Đầu vào 3 kênh (RGB), đầu ra 64 kênh
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # MaxPooling

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Các lớp Fully Connected
        self.classifier = nn.Sequential(
            nn.Linear(18432, 2048),  # 128 * 56 * 56: kích thước sau MaxPooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten tensor để đưa vào Fully Connected layers
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_vgg():
    model = VGG4(num_classes=7)
    model.to(device)
    model.load_state_dict(torch.load('models/vgg.pth', map_location=device))
    model.eval()
    return model
def build_mbf_den_trang():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=7, bias=True),
            nn.Softmax(-1)
        )
    model.load_state_dict(torch.load(
            'models/mbf_dentrang.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

class EmotionRecognition:
    def __init__(self):
        # self.model = build_mbf_den_trang()
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=8, bias=True),
            nn.Softmax(-1)
        )
        self.model.load_state_dict(torch.load(
            'models/best_mobilenetv2.pth', map_location=device))
        self.model.to(device)
        self.model.eval()
        self.model = build_vgg()
        # self.model = build_mbf_den_trang()


    def predict(self, image):
        image = Image.fromarray(image).convert('L').convert('RGB')
        image_trans = transform(image)
        image_trans = image_trans.unsqueeze(0).to(device)
        predictions = self.model(image_trans)
        predicted_label = torch.argmax(predictions, dim=1).item()

        return unidecode(labels[predicted_label]), predictions[0][predicted_label]
