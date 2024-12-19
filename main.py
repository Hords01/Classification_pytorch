import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

#setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using dev: {device}")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")


# transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#path to data, and setting the loaders
egitim_seti_yolu = 'data/Pediatric_Chest_X_ray_Pneumonia/train/'
egitim_seti = datasets.ImageFolder(root=egitim_seti_yolu,transform=transform)
test_seti_yolu = 'data/Pediatric_Chest_X_ray_Pneumonia/test/'
test_seti = datasets.ImageFolder(root=test_seti_yolu, transform=transform)

egitim_seti_boyutu = int(0.8 * len(egitim_seti))
val_boyutu = len(egitim_seti) - egitim_seti_boyutu
egitim_seti, val_seti = random_split(egitim_seti, [egitim_seti_boyutu, val_boyutu])


egitim_loader = DataLoader(egitim_seti, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_seti, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_seti, batch_size=32, shuffle=False)


print(f"etiketler: {test_seti.class_to_idx}")


print(f"egitim seti uzunlugu: {len(egitim_seti)}")
print(f"val seti uzunlugu: {len(val_seti)}")
print(f"test seti uzunlugu: {len(test_seti)}")

print(f"egitim batch sayısı: {len(egitim_loader)}")
print(f"test batch sayısı: {len(test_loader)}")
print(f"val batch sayısı: {len(val_loader)}")

for images, labels in egitim_loader:
    print(images.shape)
    break

#Model
model = models.resnet50(pretrained=True)
sinif_adlari = ['Normal', 'Pneumonia']
model.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)

for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(sinif_adlari))
model = model.to(device)

#model optimizer and criterion
learning_rate = 0.001

sinif_sayisi_sayaci = Counter(label for _, label in egitim_loader.dataset)
sinif_sayisi = [sinif_sayisi_sayaci[i] for i in range(len(sinif_adlari))]
sinif_agirliklari = 1.0 / torch.tensor(sinif_sayisi, dtype=torch.float)
sinif_agirliklari = sinif_agirliklari / sinif_agirliklari.sum()

criterion = nn.CrossEntropyLoss(weight=sinif_agirliklari).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

#model training
epochs = 5
egitim_kaybi = []
val_kaybi = []
val_acc_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in egitim_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    egitim_loss = running_loss / len(egitim_loader)
    egitim_kaybi.append(egitim_loss)

    model.eval()
    val_kayb = 0.0
    dogru = 0
    toplam = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_kayb += loss.item()
            _, tahmin = torch.max(outputs, 1)
            toplam += labels.size(0)
            dogru += (tahmin == labels).sum().item()

    val_kaybi.append(val_kayb / len(val_loader))
    val_acc = dogru / toplam
    val_acc_list.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{epochs}, Egitim Kaybı(Train loss): {egitim_loss:.5f}, Doğruluk Kesinliği(Val Acc): {val_acc:.5f}")

print("finito")

# test accuracy
test_acc = 0.0
butun_tahminler = []
butun_etiketler = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, tahmin = torch.max(outputs, 1)

        test_acc += (tahmin == labels).sum().item()
        butun_tahminler.extend(tahmin.cpu().numpy())
        butun_etiketler.extend(labels.cpu().numpy())

test_acc /= len(test_seti)
print(f"Test Dogrulugu: {test_acc:.5f}")

#confusion matrix
conf_matrix = confusion_matrix(butun_etiketler, butun_tahminler, labels=list(range(len(sinif_adlari))))
goruntule = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=sinif_adlari)

plt.figure(figsize=(8,8))
goruntule.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

#training loss vs val loss
plt.figure(figsize=(10,5))
plt.plot(egitim_kaybi, label='Egitim Kaybı')
plt.plot(val_kaybi, label='Dogrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Egitim vs Dogrulama Kaybı')
plt.legend()
plt.show()

#val acc for each epoch
plt.figure(figsize=(10, 5))
plt.plot(val_acc_list, label='Dogrulama Kesinligi')
plt.xlabel('Epoch')
plt.ylabel('Dogruluk')
plt.title('Epoch Başına Dogrulama Kesinligi')
plt.legend()
plt.show()

