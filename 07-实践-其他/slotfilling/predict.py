from keras.models import Sequential

model = Sequential()

model.load_weights('best_model_weights.h5', by_name=False)