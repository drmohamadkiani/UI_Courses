from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

model=MLPClassifier(hidden_layer_sizes=(1000,100),verbose=True,activation='logistic')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))