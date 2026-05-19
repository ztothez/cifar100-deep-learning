# CIFAR-100 CNN Experiment: Local VS Code / Terminal Results Walkthrough

### Deep Learning Course Assignment
This section documents a local terminal execution of the CIFAR-100 CNN experiment. The model was run as a standalone Python script from VS Code / terminal instead of inside a notebook. The purpose was to confirm that the same image classification workflow could run locally, save result plots as image files, and print training progress directly in the terminal.

**Dataset:** CIFAR-100  
**Framework:** PyTorch  
**Execution:** Local Python virtual environment through VS Code / terminal  
**Device:** PyTorch CUDA GPU execution shown in terminal output
## 1. Purpose of this run
This run tested the CIFAR-100 CNN workflow as a standalone Python script. The script loaded the CIFAR-100 dataset, selected a smaller 10-class animal subset, normalized the images, converted the data into PyTorch tensors, trained a convolutional neural network, saved result plots, and printed the final test accuracy in the terminal.

This report is separate from the Google Colab notebook and the AMD Cloud / ROCm walkthrough. It focuses only on the local terminal runs and the results visible in the VS Code / terminal screenshots.
## 2. Execution environment
The script was executed from a Python virtual environment in the terminal. The terminal output repeatedly showed:
```text
Training on: cuda
```

This confirms that the PyTorch script used GPU acceleration for these local runs. A warning from `torchvision.datasets.cifar.py` appeared several times:
```text
VisibleDeprecationWarning: dtype(): align should be passed as Python or NumPy boolean
```

This warning did not stop the experiment. The CIFAR-100 files were already downloaded and verified, and the training process continued normally.
## 3. Full script used for the terminal run
The full script was run as `main.py`. It loads CIFAR-100, filters the dataset to animal classes, creates PyTorch tensors and DataLoaders, trains a CNN using LeakyReLU and Adam, saves plots, and prints the final test accuracy.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True) 

X_train = np.array(trainset.data)
y_train = np.array(trainset.targets)
X_test = np.array(testset.data)
y_test = np.array(testset.targets)

# Filter to 10 classes
selected_classes = [3, 21, 27, 31, 38, 42, 43, 88, 72, 97] # animal classes
class_names = ['bear', 'chimpanzee', 'crocodile', 'elephant', 'kangaroo', 'leopard', 'lion', 'tiger', 'wolf', 'seal']  

train_mask = np.isin(y_train, selected_classes)
X_train, y_train = X_train[train_mask], y_train[train_mask]

test_mask = np.isin(y_test, selected_classes)
X_test, y_test = X_test[test_mask], y_test[test_mask]

label_map = {orig: new for new, orig in enumerate(selected_classes)}
y_train = np.array([label_map[l] for l in y_train])
y_test = np.array([label_map[l] for l in y_test])

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train).permute(0, 3, 1, 2)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test).permute(0, 3, 1, 2)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=16, shuffle=False)

# Visualize some training images
plt.figure(figsize=(10, 10))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(X_train[i])
	plt.xlabel(class_names[y_train[i]])
plt.savefig('sample_images.png')

# Build the CNN model
class CNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 32, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(32, 32, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d((2, 2)),
			nn.Dropout2d(0.25),
			nn.Conv2d(32, 64, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(64, 64, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d((2, 2)),
			nn.Dropout2d(0.25),
			nn.Conv2d(64, 128, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.Conv2d(128, 128, (3, 3), padding=1),
			nn.LeakyReLU(0.1),
			nn.MaxPool2d((2, 2)),
			nn.Dropout2d(0.25),
		)
		
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(2048, 512),
			nn.LeakyReLU(0.1),
			nn.Dropout(0.5),
			nn.Linear(512, 10),
		)
		def forward(self, x):
			x = self.features(x)
			x = self.classifier(x)
			return x

# Setup device, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on: {device}")

# Train the model
EPOCHS = 30
train_accs, val_accs, train_losses, val_losses = [], [], [], []

for epoch in range(EPOCHS):
	model.train()
	correct, total, running_loss = 0, 0, 0.0
	
	for X_batch, y_batch in train_loader:
		X_batch, y_batch = X_batch.to(device), y_batch.to(device)
		
		optimizer.zero_grad()
		outputs = model(X_batch)
		loss = criterion(outputs, y_batch)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
		correct += (outputs.argmax(1) == y_batch).sum().item()
		total += y_batch.size(0)
	
	train_accs.append(correct / total)
	train_losses.append(running_loss / len(train_loader))
	
	model.eval()
	correct, total, running_loss = 0, 0, 0.0
	
	with torch.no_grad():
		for X_batch, y_batch in test_loader:
			X_batch, y_batch = X_batch.to(device), y_batch.to(device)
			outputs = model(X_batch)
			loss = criterion(outputs, y_batch)
			running_loss += loss.item()
			correct += (outputs.argmax(1) == y_batch).sum().item()
			total += y_batch.size(0)
			
	val_accs.append(correct / total)
	val_losses.append(running_loss / len(test_loader))
	print(f"Epoch {epoch+1}/{EPOCHS} | Train acc: {train_accs[-1]:.4f} | Val acc: {val_accs[-1]:.4f}")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('results.png')
print("Plot saved to results.png")

# Predict on test images
def plot_predictions(index):
	model.eval()
	img_tensor = X_test_t[index].unsqueeze(0).to(device)
	with torch.no_grad():
		pred_probs = model(img_tensor)
	
	pred_label = class_names[pred_probs.argmax(1).item()]
	true_label = class_names[y_test[index]]
	
	plt.imshow(X_test[index])
	plt.title(f"True: {true_label} | Pred: {pred_label}")
	plt.axis('off')
	plt.savefig(f'prediction_{index}.png')

for i in range(5):
	plot_predictions(i)

print(f"\nFinal test accuracy: {val_accs[-1]*100:.2f}%")
```
## 4. Dataset setup
The script used CIFAR-100 and selected a 10-class animal subset: `bear`, `chimpanzee`, `crocodile`, `elephant`, `kangaroo`, `leopard`, `lion`, `tiger`, `wolf`, and `seal`.

The selected CIFAR-100 class indices were:
```text
[3, 21, 27, 31, 38, 42, 43, 88, 72, 97]
```

The original CIFAR-100 labels were remapped into labels from 0 to 9 so the CNN could classify the selected subset correctly. The image pixel values were normalized to the range 0–1 before training.

The script also saved a sample image grid as:
```text
sample_images.png
```
## 5. Model and training setup
The CNN used three convolutional blocks. Each block used convolutional layers, LeakyReLU activation, max pooling, and dropout. The classifier flattened the final feature maps and passed them through a dense layer before producing predictions for the 10 selected classes.

The main model configuration was:

|Component|Setting|
|---|---|
|Model type|Convolutional Neural Network|
|Activation|LeakyReLU|
|Optimizer|Adam|
|Learning rate|0.001|
|Loss function|CrossEntropyLoss|
|Batch size|16 in the full terminal script|
|Output classes|10|
|Saved plots|`sample_images.png`, `results.png`, prediction images|
The script printed training and validation accuracy for each epoch in the terminal. It also saved output images to disk, including the sample image grid, accuracy/loss plots, and prediction examples.
## 6. Saved output images from the script
This section contains the image files generated by the Python script and pulled from the environment. These are not terminal screenshots. They are saved plot/image outputs created by the code.
### 6.1 Sample images
The script saved the selected animal-class sample grid as `sample_images.png`.

![[sample_images 1 1.png]]
**Figure 1.** Sample images from the selected CIFAR-100 animal-class subset used in the local terminal run.
### 6.2 Training curves
The script saved training and validation accuracy/loss curves to image files such as `results.png`, `baseline_results.png`, and other run-specific result plots.
![[results.png]]
**Figure 2.** Saved training and validation accuracy/loss curves from the local terminal run, showing training progress and signs of overfitting.

![[baseline_results.png]]
**Figure 3.** Saved baseline training and validation accuracy/loss curves from the local terminal run.

![[run2_best_config.png]]
**Figure 4.** Saved best-configuration training curves using LeakyReLU, Adam, and batch size 32.

![[run3_sgd.png]]
**Figure 5.** Saved SGD optimizer training curves using LeakyReLU, SGD learning rate 0.01, and batch size 64.
### 6.3 Prediction examples
The script saved prediction examples as separate image files. These show both correct and incorrect predictions on visually similar animal classes.

![[prediction_0.png]]
**Figure 6a.** Prediction example where the true class was wolf and the model predicted leopard.

![[prediction_1.png]]
**Figure 6b.** Prediction example where the true class was lion and the model predicted leopard.

![[prediction_2.png]]
**Figure 6c.** Prediction example where the true class was seal and the model correctly predicted seal.

![[prediction_3.png]]
**Figure 6d.** Prediction example where the true class was chimpanzee and the model predicted bear.

![[prediction_4.png]]
**Figure 6e.** Prediction example where the true class was leopard and the model correctly predicted leopard.
## 7. Terminal screenshot evidence

This section contains terminal screenshots. These screenshots are used as evidence that the script ran in the local terminal, used CUDA, printed epoch-by-epoch training progress, and produced the final test accuracy values.

![[Pasted image 20260517143354.png|204]]
**Figure 7.** Terminal output confirming that the local PyTorch script used CUDA GPU acceleration.

![[Screenshot from 2026-05-17 00-24-38.png]]
**Figure 8.** Baseline-style terminal run reaching **61.50%** final test accuracy.

![[Screenshot from 2026-05-17 00-25-10.png]]
**Figure 9.** Additional baseline-style terminal run reaching **60.50%** final test accuracy.

![[Screenshot from 2026-05-17 00-25-23.png]]
![[Screenshot from 2026-05-17 00-25-56 1.png]]
**Figure 10.** Deeper 50-epoch terminal run reaching **54.90%** final test accuracy.

![[Screenshot from 2026-05-17 00-26-08 1.png]]
**Figure 11 LeakyReLU + Adam terminal run reaching **67.30%** final test accuracy.

![[Screenshot from 2026-05-17 00-30-35.png]]
**Figure 12.** SGD optimizer terminal run reaching **55.30%** final test accuracy after 30 epochs.

![[Screenshot from 2026-05-17 00-32-51.png]]
![[Screenshot from 2026-05-17 00-33-02.png]]
![[Screenshot from 2026-05-17 00-33-06.png]]
**Figure 13.** Extended SGD optimizer terminal run reaching **59.20%** final test accuracy after 80 epochs.

![[Screenshot from 2026-05-17 00-38-22.png]]
**Figure 14.** Tuned terminal run reaching **68.00%** final test accuracy.

![[Screenshot from 2026-05-17 00-42-35.png]]
![[Screenshot from 2026-05-17 00-42-39.png]]
**Figure 15.** Best visible local terminal result, reaching **69.90%** final test accuracy.

![[Screenshot from 2026-05-17 00-44-33.png]]
**Figure 16.** Shorter 15-epoch terminal run reaching **65.80%** final test accuracy.

![[Screenshot from 2026-05-17 01-01-24.png]]
**Figure 17.** Final animal-class terminal run reaching **59.10%** final test accuracy.

Optional extra evidence:
![[Screenshot from 2026-05-17 01-02-36.png]]
**Optional Figure.** Poor local terminal run reaching **15.50%**, suggesting an unsuccessful configuration or training issue.

![[Screenshot from 2026-05-17 01-05-57.png]]
**Optional Figure.** Additional local terminal run reaching **57.20%** final test accuracy.

![[Screenshot from 2026-05-17 01-13-25.png]]
**Optional Figure.** Additional local terminal run reaching **60.70%** final test accuracy.

![[Screenshot from 2026-05-17 01-15-14.png]]
**Optional Figure.** Additional local terminal run reaching **59.30%** final test accuracy.
## 8. Terminal run results
Several local terminal runs were captured in the screenshots. The visible final test accuracies are summarized below.

|Run shown in terminal screenshots|Visible configuration / context|Final test accuracy|
|---|---|--:|
|Baseline-style run|30 epochs, CUDA|**61.50%**|
|Additional baseline-style run|30 epochs, CUDA|**60.50%**|
|Deeper / 50-epoch run|50 epochs, CUDA|**54.90%**|
|LeakyReLU + Adam run|30 epochs, CUDA|**67.30%**|
|SGD run|30 epochs, CUDA|**55.30%**|
|Extended SGD run|80 epochs, CUDA|**59.20%**|
|Additional animal-class run|30 epochs, CUDA|**59.70%**|
|Tuned run|30 epochs, CUDA|**68.00%**|
|Extended tuned run|50 epochs, CUDA|**69.90%**|
|Shorter tuned run|15 epochs, CUDA|**65.80%**|
|Final animal-class run|30 epochs, CUDA|**59.10%**|
|Failed / poor run|30 epochs, CUDA|**15.50%**|
|Additional run|30 epochs, CUDA|**57.20%**|
|Additional run|30 epochs, CUDA|**60.70%**|
|Additional run|30 epochs, CUDA|**59.30%**|
The strongest visible local terminal result was **69.90% final test accuracy**, achieved by an extended tuned run. The next strongest visible result was **68.00%**, followed by **67.30%** from the LeakyReLU + Adam run.

The lowest visible result was **15.50%**, which suggests that one run failed to train properly or used a poor configuration. Since most other runs reached around 55–70%, this result should be treated as an unsuccessful run rather than representative model performance.
## 9. Result interpretation
The local terminal runs confirm that the CIFAR-100 CNN workflow could be executed outside a notebook environment. The repeated `Training on: cuda` output shows that PyTorch used GPU acceleration for the terminal runs.

The saved output images show what the script produced during execution. The sample image grid confirms that the animal-class subset was loaded correctly. The training curve plots show that the model learned from the training data, but also that validation accuracy often plateaued while training accuracy continued to rise. This indicates overfitting, especially in the animal-class runs.

The results varied noticeably between runs. Final test accuracy ranged from **15.50%** to **69.90%**, although the **15.50%** run appears to be an unsuccessful or poorly configured run. Most successful runs were in the range of approximately **55% to 70%**.

The baseline-style runs reached **61.50%** and **60.50%**. The deeper 50-epoch run reached only **54.90%**, showing that increasing depth or training longer did not automatically improve performance. The LeakyReLU + Adam run improved to **67.30%**, which supports the earlier observation that LeakyReLU helped the model train better.

The SGD optimizer runs reached **55.30%** after 30 epochs and **59.20%** after 80 epochs. This shows that SGD improved with longer training but still did not outperform the stronger Adam-based runs.

The best visible local terminal result was **69.90%**, achieved by an extended tuned run. This was close to the strongest notebook results and shows that the local terminal setup was capable of producing competitive results when the configuration was suitable.
## 10. Conclusion
The local VS Code / terminal experiment successfully ran the CIFAR-100 CNN workflow as a standalone PyTorch script. The terminal output confirmed GPU acceleration through `Training on: cuda`, and the script saved result plots and prediction images to files.

Across the visible terminal runs, the best final test accuracy was **69.90%**. Other strong results included **68.00%** and **67.30%**, while weaker runs such as **54.90%**, **55.30%**, and **15.50%** showed that not every configuration trained successfully.

Overall, the terminal results support the same conclusion as the notebook experiments: CNN performance on CIFAR-100 depends strongly on model configuration, activation function, optimizer, batch size, training duration, and selected classes. The best local terminal result came from a tuned run, while deeper models or SGD-based configurations did not automatically improve performance.