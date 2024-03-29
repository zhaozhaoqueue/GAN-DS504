def gan_model():
	# Generator
	g = Sequential()
	g.add(Dense(256,input_dim = z_dim))
	g.add(Activation("relu"))
	g.add(Dense(784, activation='sigmoid')) 
	g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	# Discrinimator
	d = Sequential()
	d.add(Dense(256))
	d.add(Activation("relu"))
	d.add(Dense(1, activation='sigmoid'))
	d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	# GAN
	d.trainable = False
	inputs = Input(shape=(z_dim, ))
	hidden = g(inputs)
	output = d(hidden)
	gan = Model(inputs, output)
	gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	## test git

	return g

def save_model():
	# serialize model to JSON
	model_json = g.to_json()
	with open("generator.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	g.save_weights("generator.h5")

