import numpy as np

def ReLU(z):
	return z.clip(0,)

def d_relu(z):
	return z>0

def softmax(z):
	ez = np.exp(z)
	# print("ez sum", z, ez.sum())
	return ez / ez.sum(axis=0)

def one_hot(y):
	enc_y = np.zeros((y.size, y.max()+1))
	enc_y[np.arange(y.size), y] = 1
	return enc_y

def decode_label(enc_y):
	return enc_y.argmax(axis=1)


class zukNet:
	def __init__(self, X_train, y_train, hidden_dim=16) -> None:
		# Number of samples: m, Number of features: n = 28*28 = 784
		self.m, self.n = X_train.shape

		self.X = X_train           # (m, n)
		self.label = y_train       # (m, 1)
		self.y = one_hot(y_train)  # (m, 10)

		# Parameters
		self.w1 = np.random.rand(hidden_dim, self.n) - 0.5
		self.b1 = np.random.rand(hidden_dim, 1) - 0.5

		self.w2 = np.random.rand(self.y.shape[1], hidden_dim) - 0.5
		self.b2 = np.random.rand(self.y.shape[1], 1) - 0.5

	def forward_prop(self, X):
		self.z1 = self.w1 @ X.T + self.b1       # (10, m)
		self.A1 = ReLU(self.z1)                 # (10, m)
		self.z2 = self.w2 @ self.A1 + self.b2   # (10, m)
		# print(self.z2)

		# final output
		return softmax(self.z2)              # (10, m)

	def backward_prop(self):
		self.dz2 = self.A2 - self.y.T                   # (10, m)
		# print(f"dz2 sum: {(self.dz2**2).sum():.2f}")

		self.dw2 = (self.dz2 @ self.A1.T) / self.m      # (10, 10)
		# print(f"A1 sum: {(self.z1).sum()}")
		self.db2 = self.dz2.sum(axis=1) / self.m

		self.dz1 = (self.w2.T @ self.dz2) * d_relu(self.z1)  # (10, m)
		self.dw1 = self.dz1 @ self.X / self.m                # (10, n)
		self.db1 = self.dz1.sum(axis=1) / self.m

	def update_pars(self, lr):
		self.w1 -= lr * self.dw1
		self.b1 -= lr * self.db1.reshape(self.b1.shape)
		self.w2 -= lr * self.dw2
		self.b2 -= lr * self.db2.reshape(self.b2.shape)

	def get_accuracy(self, prediction, label):
		return (label == prediction).sum() / len(prediction)

	def train(self, lr=0.1, max_iters=200):
		for i in range(max_iters):
			self.A2 = self.forward_prop(self.X)
			self.backward_prop()
			self.update_pars(lr)
			if i % 10 == 0:
				pred = decode_label(self.A2.T)
				print(f"Iteration {i}: accuracy={self.get_accuracy(pred, self.label):.3f}")

	def predict_single(self, vec):
		"""
		vec : (n, )
		"""
		z1 = self.w1 @ vec + self.b1.flatten()
		z2 = self.w2 @ ReLU(z1) + self.b2.flatten()
		print(z1.shape, z2.shape)
		return z2.argmax()
	
	def predict(self, x_test):
		z1 = self.w1 @ x_test.T + self.b1  # (10, m)
		A1 = ReLU(z1)                 # (10, m)
		z2 = self.w2 @ A1 + self.b2   # (10, m)

		return z2.argmax(axis=0)

	def analyze_test(self, x_test, y_test):
		prediction = self.predict(x_test)
		accuracy = self.get_accuracy(prediction, y_test)
		return accuracy

