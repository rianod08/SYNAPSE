import torch
from torch.nn import (ReLU, MaxPool1d, Conv1d, BatchNorm1d, Linear, Dropout, LeakyReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, LogSoftmax, Flatten)
from torch.optim import SGD
import pandas as pd
import csv
from sklearn.model_selection import KFold
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_eeg_dataset(npz_file):
    """
    Loads the EEG dataset from a .npz file.

    Args:
        npz_file (str): Path to the .npz file.

    Returns:
        tuple: A tuple containing samples (numpy array) and labels (numpy array).
    """
    data = np.load(npz_file)
    samples = data['samples']
    labels = data['labels']
    zero = []
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []
    for i in labels:
      if i == 0:
        zero.append(i)
      elif i == 1:
        one.append(i)
      elif i == 2:
        two.append(i)
      elif i == 3:
        three.append(i)
      elif i == 4:
        four.append(i)
      elif i == 5:
        five.append(i)
      elif i == 6:
        six.append(i)
      elif i == 7:
        seven.append(i)
      elif i == 8:
        eight.append(i)
      elif i == 9:
        nine.append(i)
    print(len(zero), len(one), len(two), len(three), len(four), len(five), len(six), len(seven), len(eight), len(nine))


    return samples, labels
samples, labels = load_eeg_dataset("/content/drive/MyDrive/eeg_dataset.npz")
print(samples.shape)



def calculate_accuracy(loader, model, criterion):
    model.eval()  
    correct = 0
    total = 0
    total_loss = 0  

    with torch.no_grad(): 
        for inputs, labels in loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

           
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) 

           
            loss = criterion(outputs, labels)
            total_loss += loss.item() 

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)  
    return accuracy, avg_loss

def normalize_eeg(data):
  
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array.")
    if data.ndim != 3:
        raise ValueError("Input data must have dimensions (trials, channels, time).")

  
    channel_means = np.mean(data, axis=(0, 2), keepdims=True)  
    channel_stds = np.std(data, axis=(0, 2), keepdims=True)   


    channel_stds = np.where(channel_stds == 0, 1e-10, channel_stds)

   
    normalized_data = (data - channel_means) / channel_stds

    return normalized_data


import numpy as np

def augment_and_expand_eeg_dataset_with_labels(eeg_data, labels, num_samples, noise_std=0.01, time_shift_max=100, crop_length=None, pad_value=0):

    augmented_samples = []
    augmented_labels = []

    for sample, label in zip(eeg_data, labels):

        for _ in range(num_samples):
            augmented_sample = augment_eeg_data(
                sample,
                noise_std=noise_std,
                time_shift_max=time_shift_max,
                crop_length=crop_length,
                pad_value=pad_value,
            )
            augmented_samples.append(augmented_sample)
            augmented_labels.append(label) 

 
    augmented_samples = np.array(augmented_samples)
    augmented_labels = np.array(augmented_labels)

    expanded_data = np.concatenate([eeg_data, augmented_samples], axis=0)
    expanded_labels = np.concatenate([labels, augmented_labels], axis=0)

    return expanded_data, expanded_labels



def augment_eeg_data(eeg_data, noise_std=0.01, time_shift_max=100, crop_length=None, pad_value=0):

    augmented_data = eeg_data.copy()

    noise = np.random.normal(0, noise_std, augmented_data.shape)
    augmented_data += noise

    time_shift = np.random.randint(-time_shift_max, time_shift_max + 1)
    if time_shift > 0:
        augmented_data = np.pad(augmented_data, ((0, 0), (time_shift, 0)), mode='constant')[:, :eeg_data.shape[1]]
    elif time_shift < 0:
        augmented_data = np.pad(augmented_data, ((0, 0), (0, -time_shift)), mode='constant')[:, -time_shift:]

    if crop_length is not None:
        if crop_length < augmented_data.shape[1]:
            # Crop to desired length
            start_idx = np.random.randint(0, augmented_data.shape[1] - crop_length + 1)
            augmented_data = augmented_data[:, start_idx:start_idx + crop_length]
        elif crop_length > augmented_data.shape[1]:
            # Pad to desired length
            pad_width = crop_length - augmented_data.shape[1]
            augmented_data = np.pad(augmented_data, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value)

    return augmented_data
samples = np.array(samples, dtype=np.float32)

samples1 = samples
print(samples1.shape)

samples = samples1.shape[0]    
channels = 144   
time_points = 1024  
original_dataset = samples1  
original_labels = labels  


num_augmented_samples_per_original = 10
expanded_dataset, expanded_labels = augment_and_expand_eeg_dataset_with_labels(
    original_dataset,
    original_labels,
    num_samples=num_augmented_samples_per_original,
    noise_std=0.008,
    time_shift_max=10,
    crop_length=1024,
    pad_value=0
)

print(expanded_labels[:10])
print(labels[:10])

x_train, x_test, y_train, y_test = train_test_split(
            expanded_dataset, expanded_labels, test_size=0.3, random_state=40
)


x_train = normalize_eeg(x_train)
x_test = normalize_eeg(x_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print(y_train)
print(expanded_labels[10:])
print(labels[10:])
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            Conv1d(144,144, kernel_size=5,padding=2),
            BatchNorm1d(144),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=2, stride=2)




        )
        self.cnn_layer2 = Sequential(
            Conv1d(144, 100,kernel_size=5, padding=2),
            BatchNorm1d(100),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=2, stride=2),
            Dropout(0.2)




        )
        self.cnn_layer3 = Sequential(
            Conv1d(100, 50, kernel_size=5,padding=2),
            BatchNorm1d(50),
            LeakyReLU(0.1),
            MaxPool1d(kernel_size=2, stride=2),
            Dropout(0.2)




        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 144, 1024)  # Dummy input with batch size 1
            x = self.cnn_layers(dummy_input)
            x = self.cnn_layer2(x)
            x = self.cnn_layer3(x)
            self.in_features = x.numel()


        self.linear_layer1 = Sequential(
            Linear(in_features=self.in_features,out_features=1024),
            LeakyReLU(0.1),
            Dropout(0.2)


        )
        self.linear_layer2 = Sequential(
            Linear(in_features=1024,out_features=512),
            LeakyReLU(0.1),
            Dropout(0.2)

        )
        self.linear_layer3 = Sequential(
            Linear(in_features=512,out_features=10)
        )
        self.flatten = Sequential(
            Flatten() # probably has to be changed
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = self.cnn_layer2(x)

        x = self.cnn_layer3(x)

        x = self.flatten(x)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        return x


model = Net()
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_file, label,  transform=None):
        self.df = csv_file
        self.transform = transform
        self.label = label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        scan = (self.df[index])
        label = self.label[index]
        if self.transform:
            scan = self.transform(scan)
        scan = scan.astype(np.float32)
        return scan, label

#peak was with 0.005 learning rate and 5e-5 decay think i

test_dataset = CustomDataSet(csv_file=x_test, label=(y_test))
train_dataset = CustomDataSet(csv_file=x_train, label=(y_train))



criterion = CrossEntropyLoss()
num_epochs = 250
train_loss_list = []


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to compute and display the confusion matrix
def plot_confusion_matrix(loader, model):
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed during testing
        for inputs, labels in loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the class with highest score

            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


import torch
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_epochs =50
model = Net()
model = model.to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5) # 0.005 as well
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
accuracylist = []
test_accuracylist = []
losslist = []
test_losslist = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    y_true = []
    y_pred = []

    total_correct = 0
    total_samples = 0
    model.train()
    for i, (scan, labels) in enumerate(train_loader):

        scan = scan.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(scan)


        loss = criterion(outputs, labels)

 

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        

    accuracy = 100 * total_correct / total_samples
    print("Accuracy: ", accuracy)
    accuracylist.append(accuracy)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("F1 Score:", f1)  
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")
    test_accuracy, loss = calculate_accuracy(test_loader, model, criterion)
    print(f"Test Accuracy: {test_accuracy}%")
    print(f"Test Loss: {loss}%")
    losslist.append(train_loss_list[-1])
    test_losslist.append(loss)
    test_accuracylist.append(test_accuracy)


import matplotlib.pyplot as plt


plot_confusion_matrix(test_loader, model)
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


smoothed_accuracy = smooth_curve(accuracylist)
smoothed_test_accuracy = smooth_curve(test_accuracylist)
smoothed_loss = smooth_curve(losslist)
smoothed_test_loss = smooth_curve(test_losslist)


epochs_smooth = list(range(len(smoothed_accuracy)))


plt.figure(figsize=(8, 6))
plt.plot(epochs_smooth, smoothed_accuracy, label='Test Accuracy', color='blue')
plt.plot(epochs_smooth, smoothed_test_accuracy, label='Train Accuracy', color='red')


plt.title('Train and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epochs_smooth, smoothed_loss, label='Test Loss', color='blue')
plt.plot(epochs_smooth, smoothed_test_loss, label='Train Loss', color='red')


plt.title('Smoothed Train and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.grid(True)
plt.show()
