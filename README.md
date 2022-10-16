# cats_and_dogs

Rapport modification des paramètres - teta learning rate - regularisation lambda - optimisation Adam ou RMS prop - activation fonction

Meilleur résultat (94% accuracy) avec ces paramètres :

```
param5 = {
'loss' : 'binary_crossentropy',
'optimizer' : 'adam',
'lr' : 0.001,
'activation' : 'relu',
'ratioTTV' : (0.70,0.15,0.10), # Ratio train,test,validation
'epoch' : 50,
'batch_size': 15
}

```

Le modèle :

```

model5 = Sequential() #Meilleur résultat avec ce modèle

model5.add(Conv2D(32, (3, 3), activation=param5['activation'], input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model5.add(BatchNormalization())
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(64, (3, 3), activation=param5['activation']))
model5.add(BatchNormalization())
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(128, (3, 3), activation=param5['activation']))
model5.add(BatchNormalization())
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Conv2D(256, (3, 3), activation=param5['activation']))
model5.add(BatchNormalization())
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.25))

model5.add(Flatten())
model5.add(Dense(512, activation=param5['activation']))
model5.add(BatchNormalization())
model5.add(Dropout(0.5))
model5.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model5.compile(loss=param5['loss'], optimizer=tf.keras.optimizers.Adam(
    learning_rate=param5['lr']
)
, metrics=['accuracy'])

```
