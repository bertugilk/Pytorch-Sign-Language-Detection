import torch
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import cv2
import os

from Model.network import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0),"\n")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_data = pd.read_csv(r"C:\Users\bertu\PycharmProjects\Sign Language Recognition\Data\sign_mnist_train.csv")
test_data = pd.read_csv(r"C:\Users\bertu\PycharmProjects\Sign Language Recognition\Data\sign_mnist_test.csv")

model=Net()
model.to(device)

if torch.cuda.is_available():
    checkpoint = r"C:\Users\bertu\PycharmProjects\Sign Language Recognition\weights\model.pt"
    checkpoints=torch.load(checkpoint)
    try:
        checkpoints.eval()
    except AttributeError as error:
        print(error)

    model.load_state_dict(checkpoints)
    model.eval()

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

#--------------------------------------------------------------------------------------------------------------------------

class SignsLanguageDataset(Dataset):

    def __init__(self, train=True, transform=None):

        self.train = train
        self.transform = transform

        if self.train == True:
            self.signs_lang_dataset = train_data
        else:
            self.signs_lang_dataset = test_data

        self.X_set = self.signs_lang_dataset.iloc[:, 1:].values
        self.y_set = self.signs_lang_dataset.iloc[:, 0].values

        self.X_set = np.reshape(self.X_set, (self.X_set.shape[0], 1, 28, 28)) / 255
        self.y_set = np.array(self.y_set)

    def __getitem__(self, index):

        image = self.X_set[index, :, :]

        label = self.y_set[index]

        sample = {'image_sign': image, 'label': label}

        return sample

    def __len__(self):
        return self.X_set.__len__()

test_dl = DataLoader(SignsLanguageDataset(train = False),shuffle=True)

data = test_data.iloc[:,1:].values.reshape(test_data.shape[0],1,28,28)
data = torch.Tensor(data).to(device)
y_true = test_data.iloc[:,0].values.reshape(test_data.shape[0],1).squeeze()
y_pred_tensor = model(data)
y_pred = y_pred_tensor.cpu().detach().numpy()
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_true,y_pred))

#--------------------------------------------------------------------------------------------------------------------------

def preProccess(img):
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res2 = torch.from_numpy(res1)
    res3 = res2.type(torch.FloatTensor).cuda()
    return res3

with torch.no_grad():
    image = cv2.imread(r"..\test_images\a.jpg")
    new_image=preProccess(image)
    out = model(new_image)
    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)
    pred = out.max(1, keepdim=True)[1]

    if float(probs[0, 0]) < 0.4:
        text = 'Sign not detected'
    else:
        text = signs[str(int(pred))]
        accuracy = '{:.2f}'.format(float(probs[0, 0])) + '%'

    print("Prediction = ",text)
    print("Accuracy = ",accuracy)

    plt.subplot(3,3,1)
    plt.title("Pred: "+text)
    plt.axis('off')
    plt.imshow(image)
    plt.show()
