# 🧱 Concrete Compressive Strength Prediction

## 📋 Project Overview
This project predicts the **compressive strength of concrete** using both **machine learning** and **deep learning** techniques.  
Concrete compressive strength is a key factor in civil engineering and depends on the mix design — cement, slag, fly ash, water, superplasticizer, aggregates, and curing age.

---

## 📊 Dataset Information
- **Dataset Name:** Concrete Compressive Strength  
- **Instances:** 1030  
- **Attributes:** 8 input variables + 1 output variable  
- **Type:** Multivariate regression  
- **Source:** Prof. I-Cheng Yeh, Chung-Hua University, Taiwan  
- **Reference Paper:** *Modeling of Strength of High-Performance Concrete Using Artificial Neural Networks (1998)*  

### Variables
| Feature | Type | Unit | Description |
|----------|------|------|-------------|
| Cement | Quantitative | kg/m³ | Component 1 |
| Blast Furnace Slag | Quantitative | kg/m³ | Component 2 |
| Fly Ash | Quantitative | kg/m³ | Component 3 |
| Water | Quantitative | kg/m³ | Component 4 |
| Superplasticizer | Quantitative | kg/m³ | Component 5 |
| Coarse Aggregate | Quantitative | kg/m³ | Component 6 |
| Fine Aggregate | Quantitative | kg/m³ | Component 7 |
| Age | Quantitative | Days (1–365) | Curing time |
| **Compressive Strength** | Quantitative | MPa | Target Variable |

---

## 🧠 Models Implemented

### Regression Models
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting  
- XGBoost  
- Support Vector Regressor (SVR)  
- Artificial Neural Network (ANN)

### Deep Learning Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(120, activation='relu'),
    Dense(80, activation='relu'),
    Dense(64, activation='relu'),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])
```
Epochs: 15
Batch Size: 15
Optimizer: Adam
Loss Function: Mean Squared Error

📈 Evaluation
Train R²: 0.61
Test R²: 0.63
Metrics: MSE, RMSE, R² Score
Visualizations
Residual plots to examine prediction accuracy.
Strength distribution histograms for test data.

🔄 Classification Extension

To expand the regression problem into classification:
ConcreteClass: Categorized strength levels (e.g., Low, Medium, High).
Green: Indicates eco-friendly mixtures.
Plasticizer: Converted numeric column to categorical (Yes/No).
This enabled the application of classification algorithms like Logistic Regression, Decision Trees, and Neural Networks.

🧪 Results Summary
Model	Task	Metric	Result
Linear Regression	Regression	R²	0.61
ANN	Regression	R²	0.63
Decision Tree	Classification	Accuracy	0.82
Random Forest	Classification	Accuracy	0.88
⚙️ Technologies Used

Python 3.10+

Libraries: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

IDE: Visual Studio Code, Jupyter Notebook

📂 Project Structure
📦 Concrete-Compressive-Strength
 ┣ 📜 data.csv
 ┣ 📜 regression_model.ipynb
 ┣ 📜 classification_model.ipynb
 ┣ 📜 deep_learning_model.py
 ┣ 📜 plots/
 ┣ 📜 README.md
 ┗ 📜 requirements.txt

🧾 Citation

If you use this dataset, please cite:

Yeh, I-Cheng. Modeling of Strength of High-Performance Concrete Using Artificial Neural Networks.
Cement and Concrete Research, Vol. 28, No. 12, pp. 1797–1808 (1998)

🪪 License

This project is licensed under the MIT License – free to use, modify, and distribute.

model.compile(loss='mean_squared_error', optimizer='adam')
