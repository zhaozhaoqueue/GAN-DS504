# Training
def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
    for _ in range(batchCount):  
        # Create a batch by drawing random index numbers from the training set
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
        # Create noise vectors for the generator
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
        
        # Generate the images from the noise
        generated_images = g.predict(noise)
        X = np.concatenate((image_batch, generated_images))
        # Create labels
        y = np.zeros(2*BATCH_SIZE)
        y[:BATCH_SIZE] = 1

        # Train discriminator on generated images
        d.trainable = True
        d_loss = d.train_on_batch(X, y)

        # Train generator
        noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
        y2 = np.ones(BATCH_SIZE)
        d.trainable = False
        g_loss = gan.train_on_batch(noise, y2)