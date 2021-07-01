import numpy as np
import scipy.optimize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')

def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)

data = np.array([[-186.89663369, -676.73636943, 1699.79144579],
                 [-184.59955245, -538.30600097, 1721.77542783],
                 [-185.62922672, -403.61001797, 1776.75451066],
                 [-182.69561286, -254.41219693, 1795.73799446],
                 [ -66.62311878, -679.38497313, 1696.98992162],
                 [ -63.40015054, -546.61363209, 1735.9762627 ],
                 [ -60.50409394, -410.84554471, 1790.96165794],
                 [ -56.77905528, -263.22856598, 1828.94669781],
                 [  57.38448341, -707.45417166, 1757.19827594],
                 [  61.55889308, -551.56528085, 1739.18279415],
                 [  68.26620642, -414.74070542, 1790.17431001],
                 [  72.96378945, -261.00914688, 1785.16001681],
                 [ 190.06170273, -565.79561825, 1771.3965315 ],
                 [ 197.00272448, -418.86526063, 1790.38694068],
                 [ 209.12362644, -275.37060143, 1854.38643544]])
c, normal = fitPlaneLTSQ(data)

# plot fitted plane
maxx = np.max(data[:,0])
maxy = np.max(data[:,1])
minx = np.min(data[:,0])
miny = np.min(data[:,1])
fig.show()

point = np.array([0.0, 0.0, c])
d = -point.dot(normal)

# plot original points
ax.scatter(data[:, 0], data[:, 1], data[:, 2])

# compute needed points for plane plotting
xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]
print(xx)
print(xx)
print(yy)
print('z')
print(z)
# plot plane
surface = ax.plot_surface(xx, yy, z, alpha=0.2)
print(surface._vec)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(10,2000)
plt.draw()

print("Round 2")

#------------------------------------------------------
data = np.array([[-186.89663369, -676.73636943, 1699.79144579],
                 [-184.59955245, -538.30600097, 1721.77542783],
                 [-185.62922672, -403.61001797, 1776.75451066],
                 [-182.69561286, -254.41219693, 1795.73799446],
                 [ -66.62311878, -679.38497313, 1696.98992162],
                 [ -63.40015054, -546.61363209, 1735.9762627 ],
                 [ -60.50409394, -410.84554471, 1790.96165794],
                 [ -56.77905528, -263.22856598, 1828.94669781],
                 [  57.38448341, -707.45417166, 1757.19827594],
                 [  61.55889308, -551.56528085, 1739.18279415],
                 [  68.26620642, -414.74070542, 1790.17431001],                 
                 [  68.26620642, -414.74070542, 1790.17431001],
                 [            0,             0,             0],
                 [  72.96378945, -261.00914688, 1785.16001681],
                 [ 190.06170273, -565.79561825, 1771.3965315 ],
                 [ 197.00272448, -418.86526063, 1790.38694068],
                 [ 209.12362644, -275.37060143, 1854.38643544]])
c, normal = fitPlaneLTSQ(data)

# plot fitted plane
maxx = np.max(data[:,0])
maxy = np.max(data[:,1])
minx = np.min(data[:,0])
miny = np.min(data[:,1])

point = np.array([0.0, 0.0, c])
d = -point.dot(normal)

# plot original points
# ax.scatter(data[:, 0], data[:, 1], data[:, 2])

plt.pause(1)

# compute needed points for plane plotting
xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]

sur = np.vstack((xx.flatten(),yy.flatten(), z.flatten()))
sur = np.vstack((sur, np.ones((1,4))))
# plot plane
print("Vector before : \n{}".format(surface._vec))
surface._vec = sur
print("Vector After : \n{}".format(surface._vec))

# surface.draw()
# = ax.plot_surface(xx, yy, z, alpha=0.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(10,2000)
plt.draw()
plt.pause(20)




