import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # nur cpu nutzung (extra version für cpu)

import tensorflow as tf # bibiothek neurale netzwerke
import numpy as np # bib für zahlenberechnung

km_input = float(input("Bitte geben Sie einen Kilometerwert ein: "))  # Abfrage gleich zu Beginn

km = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
miles = np.array([0.621371, 3.106855, 6.21371, 12.42742, 31.06855, 62.1371, 124.2742, 310.6855, 621.371], dtype=float)


model = tf.keras.Sequential([  #erstellung des Modelles, Sequential = schichten werden nacheinandder bearbeitet
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1)
])


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(km, miles, epochs=3000, verbose=0)  


km_test = np.array([km_input])  
prediction = model.predict(km_test)
print(f"{km_test[0]} Kilometer sind ungefähr {prediction[0][0]:.5f} Meilen")
