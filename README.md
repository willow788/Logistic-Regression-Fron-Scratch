<div align="center">

# 🎯 Logistic Regression From Scratch

### *Building Binary Classification from the Ground Up*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/willow788/Logistic-Regression-Fron-Scratch?style=social)](https://github.com/willow788/Logistic-Regression-Fron-Scratch/stargazers)

---

</div>

## 📚 Overview

This repository contains a **complete implementation of Logistic Regression** built entirely from scratch using Python and NumPy. No scikit-learn shortcuts here!  🚫

Perfect for those who want to understand the mathematical foundations and inner workings of one of machine learning's most fundamental algorithms.

---

## ✨ Features

- 🔢 **Pure NumPy Implementation** - No high-level ML libraries
- 📊 **Gradient Descent Optimization** - Step-by-step parameter updates
- 📈 **Performance Metrics** - Accuracy, precision, recall, and F1-score
- 🎨 **Visualization Tools** - Beautiful confusion matrices and ROC curves
- 🧪 **Model Improvements** - Advanced techniques and optimizations
- 📓 **Jupyter Notebooks** - Interactive and well-documented code

---

## 🗂️ Repository Structure

```
Logistic-Regression-Fron-Scratch/
│
├── Main jupyter notebook/    # Core implementation notebooks
├── Improved Model/           # Enhanced versions with optimizations
├── . gitignore               # Git ignore file
└── README.md                # You are here!  📍
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
NumPy
Pandas
Matplotlib
Seaborn
Jupyter Notebook
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/willow788/Logistic-Regression-Fron-Scratch.git
cd Logistic-Regression-Fron-Scratch
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn jupyter
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

4. **Open and run the notebooks! ** 🎉

---

## 🧮 Mathematical Foundation

Logistic Regression uses the **sigmoid function** to map predictions to probabilities: 

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- **z = w₁x₁ + w₂x₂ + ...  + wₙxₙ + b** (linear combination)
- **Cost Function**: Binary Cross-Entropy Loss
- **Optimization**: Gradient Descent

---

## 📊 What You'll Learn

<table>
<tr>
<td width="50%">

### 🎓 Core Concepts
- Sigmoid activation function
- Cost function derivation
- Gradient computation
- Parameter optimization

</td>
<td width="50%">

### 🛠️ Implementation Skills
- Vectorized operations
- Training loop design
- Model evaluation
- Hyperparameter tuning

</td>
</tr>
</table>

---

## 📈 Key Components

### 1️⃣ **Initialization**
Set up weights and bias terms

### 2️⃣ **Forward Propagation**
Compute predictions using sigmoid function

### 3️⃣ **Cost Calculation**
Measure model error with binary cross-entropy

### 4️⃣ **Backward Propagation**
Calculate gradients for parameter updates

### 5️⃣ **Parameter Update**
Adjust weights using gradient descent

### 6️⃣ **Prediction**
Make classifications on new data

---

## 🎯 Use Cases

- 📧 **Spam Detection** - Email classification
- 🏥 **Medical Diagnosis** - Disease prediction
- 💳 **Fraud Detection** - Transaction classification
- 🎬 **Sentiment Analysis** - Review categorization

---

## 🔍 Model Improvements

The `Improved Model` directory includes:

- ⚡ **Regularization** (L1/L2) to prevent overfitting
- 🎲 **Feature scaling** for faster convergence
- 🔄 **Mini-batch gradient descent** for efficiency
- 📉 **Learning rate scheduling** for better optimization

---

## 🤝 Contributing

Contributions are welcome! Feel free to: 

- 🐛 Report bugs
- 💡 Suggest new features
- 🔧 Submit pull requests
- ⭐ Star the repository

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**willow788**

[![GitHub](https://img.shields.io/badge/GitHub-willow788-181717?style=flat&logo=github)](https://github.com/willow788)

---

## 🙏 Acknowledgments

### 📚 Learning Resources
- **[GeeksforGeeks (GFG)](https://www.geeksforgeeks.org/)** - Excellent article and tutorial on implementing Logistic Regression from scratch

### 📊 Dataset
- **[Kaggle](https://www.kaggle.com/)** - For providing the dataset used in this project

### 🎨 Visualization Enhancement
- **[GitHub Copilot](https://github.com/features/copilot)** - For enhancing the confusion matrix and ROC curve visualizations, making them more informative and visually appealing

### 💡 Special Thanks
- Inspired by Andrew Ng's Machine Learning course
- Built with passion for understanding ML fundamentals
- Community feedback and contributions

---

<div align="center">

### 💫 If you found this helpful, please give it a ⭐! 

**Happy Learning! 🚀**

</div>
