import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. EXPLORAREA SETULUI DE DATE
# ==========================================
iris = load_iris()
X, y = iris.data, iris.target

print("--- EXERCIȚIUL 1 ---")
print(f"Dimensiuni date: {X.shape[0]} exemple cu {X.shape[1]} atribute.")
print(f"Atribute: {iris.feature_names}")
print(f"Clase: {iris.target_names}\n")


# ==========================================
# 2. ÎMPĂRȚIREA DATELOR (80% Train, 20% Test)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("--- EXERCIȚIUL 2 ---")
print(f"Set Antrenare: {X_train.shape}")
print(f"Set Testare:   {X_test.shape}\n")


# ==========================================
# 3. PREPROCESAREA DATELOR (SCALARE)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("--- EXERCIȚIUL 3 ---")
print("Primele 3 exemple ÎNAINTE de scalare:\n", X_train[:3])
print("\nPrimele 3 exemple DUPĂ scalare:\n", X_train_scaled[:3], "\n")


# ==========================================
# 4. CONSTRUIREA MODELULUI KNN (k=3)
# ==========================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print("--- EXERCIȚIUL 4 ---")
print(f"Acuratețe model (k=3): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(f"Acuratețe model (k=3): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")


# ==========================================
# 5. IMPACTUL VALORII K (ANALIZĂ ȘI GRAFIC)
# ==========================================
k_range = range(1, 16)
scoruri = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    scoruri.append(model.score(X_test_scaled, y_test))

# Generare Grafic
plt.figure(figsize=(8, 4))
plt.plot(k_range, scoruri, marker='o', color='green')
plt.title('Acuratețea în funcție de k')
plt.xlabel('Valoarea lui k')
plt.ylabel('Scor Acuratețe')
plt.grid(True)
plt.show()

print("--- EXERCIȚIUL 5 ---")
print("Valoarea optimă pare a fi k=3 sau k=5. Un k prea mare (ex. 15) ar putea începe să ignore nuanțele locale ale datelor.\n")


# ==========================================
# 6. EVALUAREA DETALIATĂ
# ==========================================
print("--- EXERCIȚIUL 6 ---")
print("Matricea de Confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de Clasificare:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ==========================================
# 7. VIZUALIZARE ȘI PREDICȚIE MANUALĂ
# ==========================================
# Scatter plot simplu
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=iris.target_names[y], palette='bright')
plt.title('Distribuția speciilor (Petal Length vs Width)')
plt.show()

print("--- EXERCIȚIUL 7 ---")
print("Introduceți date pentru o floare nouă:")
try:
    val_utilizator = [float(input(f"{nume}: ")) for nume in iris.feature_names]
    val_scaled = scaler.transform([val_utilizator])
    rezultat = knn.predict(val_scaled)
    print(f"\nRezultat: Floarea aparține speciei **{iris.target_names[rezultat[0]]}**")
except:
    print("Eroare la introducerea datelor.")