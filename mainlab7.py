import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Exercitiul 1
diabetes = load_diabetes()

# Exercitiul 2
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(df.head())

# Exercitiul 3
print("\nCaracteristici disponibile:")
print(diabetes.feature_names)

# Exercitiul 4
print("\nInformatii statistice:")
print(df.describe())

# Exercitiul 5
plt.figure()
plt.hist(df['bmi'], bins=20, edgecolor='black')
plt.title('Histograma BMI')
plt.xlabel('BMI')
plt.ylabel('Frecventa')
plt.show()

# Exercitiul 6
plt.figure()
plt.scatter(df['bmi'], diabetes.target, label='BMI', alpha=0.5)
plt.scatter(df['age'], diabetes.target, label='Varsta', alpha=0.5)
plt.title('BMI si Varsta vs Tinta')
plt.xlabel('Caracteristica')
plt.ylabel('Progresia bolii')
plt.legend()
plt.show()

# Exercitiul 7
X_bmi = df[['bmi']]
y = diabetes.target

X_train_bmi, X_test_bmi, y_train, y_test = train_test_split(X_bmi, y, test_size=0.2, random_state=42)

model_bmi = LinearRegression()
model_bmi.fit(X_train_bmi, y_train)

y_pred_bmi = model_bmi.predict(X_test_bmi)

plt.figure()
plt.scatter(X_test_bmi, y_test, color='black', label='Date de testare')
plt.plot(X_test_bmi, y_pred_bmi, color='blue', linewidth=3, label='Linia de regresie')
plt.title('Regresie liniara simpla (BMI)')
plt.xlabel('BMI')
plt.ylabel('Progresia bolii')
plt.legend()
plt.show()

mse_bmi = mean_squared_error(y_test, y_pred_bmi)
print(f"\nMSE pentru regresia simpla (BMI): {mse_bmi}")

# Exercitiul 8
X_multi = df[['bmi', 'bp']]

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

print(f"\nCoeficienti model regresie multipla:")
print(f"Coeficient BMI: {model_multi.coef_[0]}")
print(f"Coeficient BP: {model_multi.coef_[1]}")

y_pred_multi = model_multi.predict(X_test_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)
print(f"Scorul R² pentru regresia multipla: {r2_multi}")