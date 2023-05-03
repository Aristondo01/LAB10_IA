from ImageReader import ImageReader
from RN import RN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

Lector = ImageReader()

Lector.read_images(50)

train, test = Lector.get_train_and_test()

train_loader = DataLoader(train)
test_loader = DataLoader(test)

model = RN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epocas_dict = {}
num_epochs = 10
for epoch in range(num_epochs):
    perdida = 0
    for i, (images, labels) in enumerate(train_loader):
        # Clean gradient before new batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Call backward propagation
        loss.backward()
        optimizer.step()
        if (i+1) % 2500 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        perdida += loss.item()
    epocas_dict[epoch] = perdida/len(train_loader)

print('Loss per epoch:')
for key in epocas_dict:
    print('Epoch {}: {}'.format(key+1, epocas_dict[key]))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))