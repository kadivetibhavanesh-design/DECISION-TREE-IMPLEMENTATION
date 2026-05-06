from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of Decision Tree Model:", accuracy)

plt.figure(figsize=(12,8))

plot_tree(
    model,
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names
)

plt.title("Decision Tree Visualization")
plt.show()

print("\nConclusion:")
print("The Decision Tree model was successfully implemented using the Iris dataset.")
print("The model achieved good accuracy and classified the flowers effectively.")
