from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.regularizers import l2

import constants

def classifier(epochs=200, batch_size=200):
  return KerasClassifier(build_fn=simple_nn, epochs=epochs, batch_size=batch_size)

def simple_nn(alpha=0.0001, num_hidden_neurons=100):
  model = Sequential()
  model.add(Dense(num_hidden_neurons, activation='relu', input_dim=len(constants.FEATURES), kernel_regularizer=l2(alpha)))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  return model

def main():
  import data
  X, y = data.get_features_and_labels(modified=False, max_events=100)
  model = classifier()
  model.fit(X.values, y.values)
  X_test, y_test = data.get_features_and_labels(modified=True, max_events=100)
  score = model.score(X_test.values, y_test.values)
  print('\nscore = {}'.format(score))

if __name__ == '__main__':
  main()
