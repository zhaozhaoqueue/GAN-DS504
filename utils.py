def basic_plot():
	# Generate images
	np.random.seed(504)
	h = w = 28
	num_gen = 25

	z = np.random.normal(size=[num_gen, z_dim])
	generated_images = g.predict(z)

	# plot of generation
	n = np.sqrt(num_gen).astype(np.int32)
	I_generated = np.empty((h*n, w*n))
	for i in range(n):
	    for j in range(n):
	        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

	plt.figure(figsize=(4, 4))
	plt.axis("off")
	plt.imshow(I_generated, cmap='gray')
	plt.show()