<H3>Name: CHIDROOP M J</H3>
<H3>Register no. : 212225240029</H3>
<H3>Date: 31/07/2026</H3>
<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupyter Notebook

# RELATED THEORETICAL CONCEPT:

A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        ```v(i) = w(i)*x(i)
           W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)
        ```
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
### MOUNT GOOGLE DRIVE:
```py
from google.colab import drive
drive.mount('/content/drive')
```

### IMPORT LIBRARIES:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SklearnPerceptron
```

### LOAD DATASET:
```py
file_path = "/content/drive/My Drive/neural_networks./train.csv"
df = pd.read_csv(file_path)
print(df.head())
print(df.info())
print(df.isnull().sum())
```

### Feature Engineering and Preprocessing:
```py
def preprocess_titanic(df, is_train=True):
    df_copy = df.copy()
    
    df_copy['Age'] = df_copy['Age'].fillna(df_copy['Age'].median())
    df_copy['Embarked'] = df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0])
    df_copy['Fare'] = df_copy['Fare'].fillna(df_copy['Fare'].median())
    
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
    
    df_copy['Title'] = df_copy['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Lady': 'Rare',
        'Countess': 'Rare', 'Capt': 'Rare', 'Sir': 'Rare', 'Don': 'Rare',
        'Jonkheer': 'Rare'
    }
    df_copy['Title'] = df_copy['Title'].map(title_mapping)
    
    df_copy['Sex'] = df_copy['Sex'].map({'male': 0, 'female': 1})
    df_copy['Embarked'] = df_copy['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df_copy = pd.get_dummies(df_copy, columns=['Title'], prefix='Title')
    
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    if is_train:
        columns_to_drop.append('Survived')
    df_copy = df_copy.drop(columns=[col for col in columns_to_drop if col in df_copy.columns])
    
    return df_copy

df_processed = preprocess_titanic(df, is_train=True)
print(df_processed.head())
```

### Feature Selection and Scaling:
```py
X = df_processed.select_dtypes(include=[np.number]).values
y = df['Survived'].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
X_scaled = (X - x_mean) / x_std

print(f"Mean after scaling: {X_scaled.mean(axis=0)[:5]}")
print(f"Std after scaling: {X_scaled.std(axis=0)[:5]}")
```

### Train-Test Split:
```py
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Training survival rate: {y_train.mean()*100:.1f}%")
print(f"Test survival rate: {y_test.mean()*100:.1f}%")
```
### Perceptron Class Implementation:
```py
class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self.misclassified_samples = []
        
    def fit(self, X, y, n_iter=100):
        self._b = 0.0
        self._w = np.zeros(X.shape[1])
        self.misclassified_samples = []
        
        for epoch in range(n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch+1}")
                break
    
    def f(self, X):
        return np.dot(X, self._w) + self._b
    
    def predict(self, X):
        return np.where(self.f(X) >= 0, 1, -1)
```

### Train the Perceptron:
```py
y_train_binary = np.where(y_train == 0, -1, 1)

perceptron = Perceptron(learning_rate=0.1)
perceptron.fit(X_train, y_train_binary, n_iter=100)

print(f"Final weights: {perceptron._w}")
print(f"Final bias: {perceptron._b}")
print(f"Final training errors: {perceptron.misclassified_samples[-1]}")
```

### Model Evaluation:
```py
y_pred_binary = perceptron.predict(X_test)
y_pred = np.where(y_pred_binary == -1, 0, 1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
```

### Error Analysis:
```py
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(perceptron.misclassified_samples) + 1), 
         perceptron.misclassified_samples, 
         color='blue', marker='o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Number of Misclassifications')
plt.title('Training Errors Over Iterations')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Initial errors: {perceptron.misclassified_samples[0]}")
print(f"Final errors: {perceptron.misclassified_samples[-1]}")
```

### Test Set Predictions:
```py
test_file_path = "/content/drive/My Drive/neural_networks./test.csv"
df_test = pd.read_csv(test_file_path)

passenger_ids = df_test['PassengerId']
df_test_processed = preprocess_titanic(df_test, is_train=False)

X_test_final = df_test_processed.select_dtypes(include=[np.number]).values
X_test_final_scaled = (X_test_final - x_mean) / x_std

predictions_binary = perceptron.predict(X_test_final_scaled)
predictions = np.where(predictions_binary == -1, 0, 1)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
print(f"\nPrediction distribution:\n{submission['Survived'].value_counts()}")
```

### Compare with Sklearn Perceptron:
```py
X_2d_train = X_train[:, :2]
y_train_2d = np.where(y_train == 0, -1, 1)

model_2d = Perceptron(learning_rate=0.1)
model_2d.fit(X_2d_train, y_train_2d, n_iter=100)

x_min, x_max = X_2d_train[:, 0].min() - 0.5, X_2d_train[:, 0].max() + 0.5
y_min, y_max = X_2d_train[:, 1].min() - 0.5, X_2d_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d_train[y_train==1, 0], X_2d_train[y_train==1, 1], 
            color='red', marker='o', label='Survived')
plt.scatter(X_2d_train[y_train==0, 0], X_2d_train[y_train==0, 1], 
            color='blue', marker='x', label='Not Survived')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.title('Perceptron Decision Boundary')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()
```

### Final Results Summary:
```py
print("="*60)
print("PERCEPTRON IMPLEMENTATION RESULTS")
print("="*60)
print(f"Dataset: Titanic - Machine Learning from Disaster")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Learning rate: {perceptron.learning_rate}")
print(f"Total epochs: {len(perceptron.misclassified_samples)}")
print(f"Initial training errors: {perceptron.misclassified_samples[0]}")
print(f"Final training errors: {perceptron.misclassified_samples[-1]}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print("="*60)
```




# OUTPUT:
### Titanic Dataset - 3D visualization:
<img width="648" height="658" alt="image" src="https://github.com/user-attachments/assets/d89f06ed-1119-4268-9e9d-130a8ae82477" />

### Training Dataset - Binary Classification:
<img width="841" height="701" alt="image" src="https://github.com/user-attachments/assets/bb22c67d-e820-41b8-ba5f-2a858a26e060" />

### Perceptron Decision Boundary:
<img width="857" height="701" alt="image" src="https://github.com/user-attachments/assets/2b1f051a-81cf-47a4-bbde-b40636d2a4d0" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Titanic - Disaster Dataset.

 
