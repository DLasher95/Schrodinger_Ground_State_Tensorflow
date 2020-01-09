import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import progressbar
from schrodinger_solver import solver
from grid_creator import grid
from tri_matrix import matrix
from tise_solver import tise
from sho_equation import sho_eq

L = 64 # We will use a 64 x 64 grid to train fairly fast, but still keeping a lot of accuracy.

# Build Deep NN - Use a series of repeating "modules" comprised of reducing and
# nonreducing layers. 2 functions will construct these layers:
def reducing(_in):
    """
       A reducing convolutional layer has 64 filters of size 3x3.
       We use stride 2 to half the data size.
       We use ReLU activation
    """
    return tf.contrib.layers.conv2d(_in, 64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)

def nonreducing(_in):
    """
       A nonreducing convolutional layer has 16 filters of size 4x4.
       We use stride 1 to preserve the data size.
       We use ReLU activation.
    """
    return tf.contrib.layers.conv2d(_in, 16, kernel_size=4, stride=1, activation_fn=tf.nn.relu)

# Convolutional neural network is then comprised of repeating blocks of reducing-nonreducing-nonreducing.
# These layers feed into two fully-connected layers which reduce to a single, continuous solution.
def CNN(_in):
    net = tf.reshape(_in, (-1, L, L, 1))
    #If you're using 256x256 potentials, you'll want 6 modules.
    #We'll use 4 since we're using 64x64 potentials
    #  e.g. for 256x256 use   for moduleID in range(6):
    for moduleID in range(4):
        net = nonreducing(nonreducing(reducing(net)))
    net = tf.reshape(net, (-1, 4*4*16))
    net = tf.contrib.layers.fully_connected(net, 1024, activation_fn=tf.nn.relu)
    net = tf.contrib.layers.fully_connected(net, 1, activation_fn=None)
    return net

# Set up the feed and optimization ops
#data comes in a [ batch * L * L * 1 ] tensor, and labels a [ batch * 1] tensor
x = tf.placeholder(tf.float32, (None, L, L, 1), name='input_image')
y = tf.placeholder(tf.float32, (None, 1))

predicted = CNN(x)
#define the loss function
loss = tf.reduce_mean(tf.square(y-predicted))
#create an optimizer, a training op, and an init op
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# Initialize the variables
sess = tf.InteractiveSession()
sess.run(init)

# Get training and testing data from solver.
# (This is 40k example, 10K reserved for testing)
S = solver(limit=20, L=L, number=50000) #Use OUR import
data, labels = S.generate_file()
train_data = data[:40000]
test_data = data[40000:]
train_labels = labels[:40000]
test_labels = labels[40000:]

# Train NN for 100 epochs.
BATCH_SIZE = 1000
EPOCHS = 100

with progressbar.ProgressBar(max_value=EPOCHS,
                             widgets=[progressbar.Percentage(),
                                      progressbar.Bar(),
                                      progressbar.DynamicMessage('loss'), '|',
                                      progressbar.ETA()],
                             redirect_stdout=False).start() as bar:
    for epoch in range(EPOCHS):
        for batch in range(train_data.shape[0] / BATCH_SIZE): # range WAS xrange, py 2
            _, loss_val = sess.run([train_step, loss],
                                   feed_dict={
                                       x: train_data[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE],
                                       y: train_labels[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                                   }
                                   )

            bar.update(epoch, loss=float(loss_val))

saver = tf.train.Saver()
saver.save(sess, save_path='./chkpts/')

# Use testing set to make predictions. Not all predictions will fit in memory at once,
# so we will iterate through in batches.
BATCH_SIZE = 1000 #Didn't we already do this above?
bar = progressbar.ProgressBar()
prediction = []
for batch in range(test_data.shape[0] / BATCH_SIZE): # range WAS xrange, py 2
    batch_predictions = sess.run(predicted,
                                 feed_dict={
                                     x: test_data[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE],
                                 }
                                 )
    prediction.append(batch_predictions.flatten())

prediction = np.array(prediction).flatten()

# Plot results on histogram
fig, ax = plt.subplots(1,1, figsize=(8,8))
counts, xedges, yedges = np.histogram2d(test_labels.flatten(), prediction.flatten(), bins=100)

ax.pcolormesh(xedges, yedges, counts.T, cmap='Reds')

ax.set_xlabel("True energy")
ax.set_ylabel("Predicted energy")
ax.grid(alpha=0.3)
ax.set_title("Median absolute error: {0} mHa".format(np.median(np.abs(prediction.flatten() - test_labels.flatten()))*1000.))
fig.show()

"""
Purpose: Use Deep Learning to solve for the ground-state of simple 2D quantum Schrodinger problems.

Author: Dylan Lasher
"""
"""
# Create a finite grid with L x L grid points extending from -20 to +20 bohr:
my_grid = grid(limit = 20, L = 256) #TODO: Why not make these variables and use here?

# Construct tri-diagonal matrix with fringes. Break down sparse matrix:
my_matrix = matrix(my_grid.L, my_grid.dx, my_grid.dy)

# Test using Simple Harmonic Oscillator (SHO)
# Manually generate three potentials
pot_0 = sho_eq.V_SHO(my_grid.mesh, 1, 1, 0, 0)
pot_1 = sho_eq.V_SHO(my_grid.mesh, 2, 1, 0, 0)
pot_2 = sho_eq.V_SHO(my_grid.mesh, 1, 5, 2, 5)

# plot the potentials
# View with either PyCharm Professional in Scientific mode or Spyder
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].pcolormesh(my_grid.x, my_grid.y, pot_0, cmap='plasma')
axs[1].pcolormesh(my_grid.x, my_grid.y, pot_1, cmap='plasma')
axs[2].pcolormesh(my_grid.x, my_grid.y, pot_2, cmap='plasma')
axs[1].set_yticks([])
axs[2].set_yticks([])
_ = [axs[i].set_xlabel(r"$x$, [bohr]") for i in range(3)]
axs[0].set_ylabel(r"$y$, [bohr]")
fig.show()

# Generate several potentials randomly and compare finite-difference method
# to the analytical solution
np.random.seed(123)
number = 10
kx = np.random.rand(number) * 0.16
ky = np.random.rand(number) * 0.16
cx = (np.random.rand(number) - 0.5) * 16
cy = (np.random.rand(number) - 0.5) * 16

for i in range(number):
    E, psi = tise().solve(sho_eq.V_SHO(my_grid.mesh, kx[i], ky[i], cx[i], cy[i]), my_grid.L, my_matrix.T)
    numerical = np.real(E[0])
    analytical = 0.5 * (np.sqrt(kx[i]) + np.sqrt(ky[i]))  # Analytical energy for SHO
    print("Numerical: {0:8.5f}\tAnalytical: {1:8.5f}\tError: {2:8.5f}% ".format(numerical, analytical, 100. * np.abs(
        numerical - analytical) / analytical))
"""