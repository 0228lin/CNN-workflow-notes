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

`ReLU` : A function defined as 

$$
f(x) = \max(0, x)
$$

--- turns all negative values to zero and keeps positive values unchanged.  

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
![Stride](https://pic4.zhimg.com/v2-f72b4e145e7e99f3d3a01813ee24c767_1440w.jpg)

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

**Purpose**: Convert 2D feature maps into 1D vector for fully connected layers because `Fully connected layers` expect vectors, not matrices

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

*This layer acts like the "brain" that makes final decisions based on all the features extracted by earlier layers.*

**Concept**: Neurons in this layer are connected to every element in the previous layer.  

**Function**: Learns global patterns by integrating all features.  

`Weights & Biases`: Every connection has a weight (how strong it is) and a bias (adjusts output).  


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

**Purpose**: Represent categorical labels in binary format.(Only one "1", rest all "0")

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
*Softmax ensures each output is interpretable as a probability and lets the model "choose" the most likely class.*
**Purpose**: Convert logits to probabilities for classification.(把只有计算机能看懂的数字变成人类能看懂的概率 `normalization`，概率最大的就是那个分类，用于classification，所有概率加和为1)
**Formula**: 

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

![Softmax](https://pic1.zhimg.com/v2-a511bf904037c009998665bdc78e2f4a_1440w.jpg)  
**Output**: A probability distribution across all possible classes (e.g., [0.1, 0.7, 0.2] for 3 classes).

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

**Purpose**: Measure how far predictions deviate from actual labels.  The loss function quantifies how wrong the model is — and drives learning by guiding gradient descent to reduce error.

### Cross Entropy Loss (for classification):  Measures the difference between predicted probabilities and actual labels.  
**Function**: 

$$ \text{Loss} = - \sum y \log(p) $$  

- `y` : the true label (often one-hot encoded)
- `p` : the predicted probability

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

`Epoch`: One full pass through the training dataset.
`Batch Size`: Number of samples processed before the model updates.
`Learning Rate`: Controls how big a step the optimizer takes.
`Gradient Descent`: Algorithm used to update weights and minimize loss.

```text
1. Forward Pass ➜ Compute output
2. Loss Calculation ➜ Measure errors
3. Backward Pass ➜ Update weights (w, b)
4. Repeat for many epochs
```
---

## 🛠️ 10. Hyperparameters

- **Kernel Initialization**: Often Gaussian-distributed
- **Padding**: Preserves edge information --- 用像素填充输入图像的边界，这样边界信息被卷积核扫描到的次数不会远低于中间信息的扫描次数，从而保持边界信息价值，也可以使所有的输入图像尺寸一致  
![Padding](https://pic1.zhimg.com/v2-608f1fe51c1e6d202e539fe362d9f9ba_1440w.jpg)  
- **Stride**: Controls filter movement; larger stride = coarser features --- 卷积核工作的时候，每次滑动的格子数，默认是1，但是也可以自行设置，步幅越大，扫描次数越少，得到的特征也就越“粗糙”

---

---

## ✅ Summary

```text
Image ➜ Convolution ➜ ReLU ➜ Pooling ➜ Flatten ➜ FC Layer ➜ Softmax ➜ CrossEntropy Loss ➜ Update (w, b) ➜ Repeat
```

---

## 📚 References
- Concepts & Diagrams adapted from: [一文掌握CNN卷积神经网络](https://zhuanlan.zhihu.com/p/104776627)



