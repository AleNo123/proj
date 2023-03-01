import pickle
import pylsl
import time
from pylsl import StreamInlet
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Conv1D

stream_info = pylsl.resolve_stream()
print(stream_info)
probes_duration=750

inlet = StreamInlet(stream_info[0], max_chunklen=probes_duration)
# ---------------------------------
# chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
# print(chunk)
# print(len(chunk))
# ---------------------------------
chunk_arr= []
chunk_res = []
chunk_r = [0]
with open('data.bin', mode='rb') as file:
    chunk_arr = pickle.load(file)
with open('data2.bin',mode='rb') as f:
    chunk_res = pickle.load(f)
# ---------------------------------
# for x in range(10):
#     chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
#     chunk = chunk[:675]
#     chunk_arr.append(chunk)
#     chunk_res.append(chunk_r)
# with open('data.bin', mode='wb') as file:
#     pickle.dump(chunk_arr, file)
# with open('data2.bin', mode='wb') as file:
#     pickle.dump(chunk_res,file)
# # ---------------------------------
chunk_res = keras.utils.to_categorical(chunk_res,3)
chunk_arr = np.expand_dims(chunk_arr, axis=3)
# ---------------------------------
# chunk_res = np.expand_dims(chunk_res, axis=4)
# ---------------------------------
model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu',input_shape=(675,30,1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(3, activation='softmax'),
])
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(chunk_arr, chunk_res, batch_size=32, epochs=15, validation_split=0.2)
chunk, t_ = inlet.pull_chunk(timeout=3)  # Chunk [750x31(?)]
chunk = chunk[:675]
x = np.expand_dims(chunk, axis=0)
res = model.predict(x)
print(res)
print(max(res[0]))
