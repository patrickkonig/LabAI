import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# #EX1, #EX2, #EX3: Încărcare, Afișare DataFrame și Feature Names
wine = load_wine(as_frame=True)
df = wine.frame

print("--- EX 2: Primele 5 rânduri ---")
print(df.head())

print("\n--- EX 3: Caracteristici disponibile ---")
print(wine.feature_names)


# #EX4: Antrenarea unui arbore de decizie (specific)
# a. Caracteristici: alcohol și flavanoids
X_4 = df[['alcohol', 'flavanoids']]
y = wine.target

# b. Limitați adâncimea la 2
clf_4 = DecisionTreeClassifier(max_depth=2, random_state=42)
clf_4.fit(X_4, y)

# c. Afișați structura arborelui
print("\n--- EX 4c: Generare grafic arbore... ---")
plt.figure(figsize=(12, 8))
plot_tree(clf_4, feature_names=['alcohol', 'flavanoids'], class_names=wine.target_names, filled=True)
plt.show()

# d. Interpretare (print în consolă pentru claritate)
print("\n--- EX 4d: Interpretare noduri ---")
print("Nodul 0 (Rădăcină): Verifică de obicei 'flavanoids'. Dacă sunt <= 0.84, clasifică direct ca clasa 2 (Vin tip 3).")
print("Nodul 1: Dacă flavanoids > 0.84, verifică 'alcohol' pentru a separa clasa 0 de clasa 1.")


# #EX5: Arbore complet (max_depth=None) și acuratețe
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

clf_5 = DecisionTreeClassifier(max_depth=None, random_state=42)
clf_5.fit(X_train, y_train)

y_pred = clf_5.predict(X_test)
print(f"\n--- EX 5: Acuratețe pe setul de testare: {accuracy_score(y_test, y_pred) * 100:.2f}% ---")


# #EX6: Importanța caracteristicilor (toate cele 13)
importances = clf_5.feature_importances_
importance_df = pd.DataFrame({'Caracteristică': wine.feature_names, 'Importanță': importances})
importance_df = importance_df.sort_values(by='Importanță', ascending=False)

print("\n--- EX 6: Importanța caracteristicilor ---")
print(importance_df)
print(f"\nCea mai influentă caracteristică este: {importance_df.iloc[0]['Caracteristică']}")


# #EX7: Bonus - Mini-arbore pe hârtie (simulat)
# a. Subset 6 exemple
subset = df[['alcohol', 'flavanoids', 'target']].sample(6, random_state=42)
print("\n--- EX 7a: Subset 6 exemple ---")
print(subset)

# b. Calcul Gini Impurity (Rădăcină)
def gini(labels):
    counts = labels.value_counts()
    return 1 - sum((c/len(labels))**2 for c in counts)

print(f"\n--- EX 7b: Gini Rădăcină: {gini(subset['target']):.4f} ---")

# c. Propunere split (ex: alcohol > 13.0)
split_val = 13.0
stanga = subset[subset['alcohol'] <= split_val]
dreapta = subset[subset['alcohol'] > split_val]

print(f"\n--- EX 7c: Split propus (alcohol > {split_val}) ---")
print(f"Exemple stânga: {len(stanga)}, Exemple dreapta: {len(dreapta)}")

# d. Decizie split
gini_stanga = gini(stanga['target'])
gini_dreapta = gini(dreapta['target'])
gini_total = (len(stanga)/6 * gini_stanga) + (len(dreapta)/6 * gini_dreapta)

print(f"Gini ponderat după split: {gini_total:.4f}")
print("Decizie: Dacă Gini ponderat < Gini rădăcină, split-ul este BUN.")