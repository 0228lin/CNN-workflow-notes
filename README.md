# CNN-workflow-notes
*A beginner-friendly guide to understanding Convolutional Neural Networks (CNNs) — with intuitive examples, annotated code, and visual diagrams.*

---

## 🖼️ CNN Workflow Overview

A classic CNN architecture typically includes:

- **Input Layer**
- **Convolutional Layer**
- **Activation Function (ReLU)**
- **Pooling Layer**
- **Flattening**
- **Fully Connected Layer**
- **Softmax Output Layer**

### 📊 CNN Structure Diagram

![CNN Workflow](https://pica.zhimg.com/v2-df0296c00a39a5089b24c632e0db4aa8_1440w.jpg)

---
## 🔍 Step 1: Input Layer

**Function**: Accepts raw image data (e.g., grayscale or RGB).  
**Format**: Typically 2D or 3D tensors — `[Height x Width x Channels]`.

Input is loaded via `DataLoader` or NumPy arrays, normalized for consistency.

---

## 🧩 Step 2: Convolution Layer (`w`, `b`)

**Principle**: Convolution captures local dependencies by looking at neighborhoods in the image, reducing dimensionality while preserving structure. It’s like scanning for important textures!  
**Purpose**: A mathematical operation where small matrices called filters or kernels slide over the image to detect local patterns (e.g., edges, corners).
- `Weights (w)`: Weight matrix (filter)  
- `Bias (b)`: Bias term to adjust output. --- A scalar added after the convolution to shift the output and help fit the data better.
- `Feature Map`: The resulting map after applying the kernel and bias, showing where features (like vertical edges) were found.

### 🧠 Key Concepts:
- **Local connectivity**: Each neuron sees only a small part of the input.
- **Weight sharing**: Same filter used across the entire image.
- **Output**: Feature maps that highlight patterns.

### 📦 Example:

```python
import numpy as np

# 3x3 grayscale image
image = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

# 2x2 filter (weights)
filter = np.array([[1, 0],
                   [0, -1]])

bias = 1

def simple_conv(img, filt, b):
    window = img[:2,:2]                  # Extract patch
    weighted = window * filt             # Element-wise multiply
    summed = np.sum(weighted)            # Sum values
    result = summed + b                  # Add bias
    return result

print(simple_conv(image, filter, bias))  # Output: -3
```

---

## 💡 Step 3: Activation Layer (ReLU (Rectified Linear Unit))

`ReLU` : A function defined as $$ f(x) = \max(0, x) $$ --- turns all negative values to zero and keeps positive values unchanged.  

### 📦 ReLU Example:

```python
import numpy as np
x = np.array([-2, 0, 3])
relu = lambda x: np.maximum(0, x)
print(relu(x))  # Output: [0 0 3]
```

### 🧠 Why ReLU?
- Filters out negative noise.
- It introduces non-linearity, enabling the network to learn complex, non-linear mappings between inputs and outputs.  
📌 Alternatives: Sigmoid and Tanh functions are also used, but ReLU tends to perform better in deep networks.
---

## 🏊 Step 4: Pooling Layer

**Purpose**: Downsample feature maps to reduce computation and retain dominant information. 

### Types:
- **Max Pooling**: Selects the maximum value from a patch of the feature map. 
- **Average Pooling**: Averages the values in the patch.
![**Pooling**](https://pic2.zhimg.com/v2-1519745e58e78e28cfd0bf3aee6e2cb7_1440w.jpg)

`Kernel Size`: Defines the dimensions of the patch being pooled.
`Stride`: Controls how far the window moves between pooling operations.

### 📦 Max Pooling Example:

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1.0,2.0],
                  [3.0,4.0]]).view(1,1,2,2)  # [Batch, Channel, H, W]

pooled = F.max_pool2d(x, kernel_size=2)
print(pooled)  # Output: tensor([[[[4.0]]]])
```

### 🧠 Why Pooling?
- Maintains spatial hierarchy.
- Reduces parameter count and overfitting.
- Pooling reduces spatial dimensions, creating a more compact representation. It adds translational invariance — meaning the model still works even if the object shifts slightly.

---

## 📏 Step 5: Flattening

**Purpose**: Convert 2D feature maps into 1D vector for fully connected layers.

### 📦 Example:

```python
import torch
x = torch.tensor([[1, 2],
                  [3, 4]])

flat = x.view(-1)  # Reshape to 1D
print(flat)  # Output: tensor([1, 2, 3, 4])
```

---

## 🧠 Step 6: Fully Connected Layer

**Purpose**: Map extracted features to class scores.

### 📦 Example:

```python
import torch.nn as nn
import torch

fc = nn.Linear(4, 3)  # 4 features ➜ 3 classes
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(fc(x))  # Raw logits (unnormalized scores)
```

---

## 🔢 Step 7: One-Hot Encoding

**Purpose**: Represent categorical labels in binary format.

### 📦 Example:

```python
from sklearn.preprocessing import OneHotEncoder

labels = [['cat'], ['dog'], ['bird']]
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(labels)
print(encoded)
```

**Output:**
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

---

## 📊 Step 8: Softmax Function

**Purpose**: Convert logits to probabilities for classification.

### 📦 Example:

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(probs)  # tensor([0.659, 0.242, 0.099])
```

### 🧠 Why Softmax?
- Normalizes predictions.
- Ensures total probability sums to 1.

---

## 📉 Step 9: Loss Function

**Purpose**: Measure how far predictions deviate from actual labels.

### Cross Entropy Loss (for classification):

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

outputs = torch.tensor([[1.0, 2.0, 0.1]])  # logits
labels = torch.tensor([1])  # correct class index

loss = criterion(outputs, labels)
print(loss.item())  # Example: ~0.74
```

---

## 🔁 Step 10: Training Cycle

```text
1. Forward Pass ➜ Compute output
2. Loss Calculation ➜ Measure errors
3. Backward Pass ➜ Update weights (w, b)
4. Repeat for many epochs
```

---

## ⚙️ Hyperparameter Settings

- **Kernel Init**: Gaussian or uniform initialization
- **Padding**: Preserves edge features
- **Stride**: Controls filter step size (affects resolution)



---

## 💡 2. Activation Function (ReLU)

Adds non-linearity so CNNs can learn complex patterns.

```python
import numpy as np
x = np.array([-3, 0, 2])
relu = lambda x: np.maximum(0, x)
print(relu(x))  # Output: [0, 0, 2]
```

---

## 🏊 3. Pooling Layer

Reduces spatial dimensions while keeping important features.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1.0,2.0],[3.0,4.0]]).view(1,1,2,2)
out = F.max_pool2d(x, kernel_size=2)
print(out)  # tensor([[[[4.0]]]])
```

---

## 📏 4. Flatten

Prepares data for fully connected layers.

```python
import torch
x = torch.tensor([[1,2],[3,4]])
flat = x.view(-1)
print(flat)  # tensor([1, 2, 3, 4])
```

---

## 🧠 5. Fully Connected Layer

Acts like a traditional neural network, mapping features to class scores.

```python
import torch.nn as nn
fc = nn.Linear(4, 3)  # 4 inputs, 3 output classes
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(fc(x))
```

---

## 🔢 6. One-Hot Encoding

Converts class labels into binary format for classification.

```python
from sklearn.preprocessing import OneHotEncoder
labels = [['cat'], ['dog'], ['bird']]
encoder = OneHotEncoder(sparse=False)
print(encoder.fit_transform(labels))
```

---

## 📊 7. Softmax Function

Turns raw scores into probability distributions.

```python
import torch.nn.functional as F
import torch
logits = torch.tensor([2.0, 1.0, 0.1])
print(F.softmax(logits, dim=0))  # tensor([0.659, 0.242, 0.099])
```

---

## 📉 8. Loss Function

Measures prediction error. Cross Entropy is commonly used.

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
outputs = torch.tensor([[1.0, 2.0, 0.1]])
labels = torch.tensor([1])  # true label is class index 1

loss = criterion(outputs, labels)
print(loss.item())  # Lower is better
```

---

## ⚙️ 9. Training Process

CNN training involves:

- Forward pass through layers
- Calculating loss
- Backpropagation to update `w` and `b`
- Repeating over multiple epochs

---

## 🛠️ 10. Hyperparameters

- **Kernel Initialization**: Often Gaussian-distributed
- **Padding**: Preserves edge information
- **Stride**: Controls filter movement; larger stride = coarser features

---

## 📚 References

This guide is inspired by structured explanations from Chinese deep learning tutorials and annotated diagrams. For a deeper dive, refer to the full CNN architecture breakdown in [this Zhihu image](https://pica.zhimg.com/v2-df0296c00a39a5089b24c632e0db4aa8_1440w.jpg).



---

## ✅ Summary

```text
Image ➜ Convolution ➜ ReLU ➜ Pooling ➜ Flatten ➜ FC Layer ➜ Softmax ➜ CrossEntropy Loss ➜ Update (w, b) ➜ Repeat
```

---

## 📚 References

- Diagram credit: [Zhihu CNN Schematic](https://pica.zhimg.com/v2-df0296c00a39a5089b24c632e0db4aa8_1440w.jpg)
- Concepts adapted from: “CNN卷积神经网络原理” tutorial series



