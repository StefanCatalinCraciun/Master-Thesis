#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:30:24 2023

@author: stefan
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import struct

def load_wavefield_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the wavefield
        metadata = struct.unpack('iiiii', f.read(5 * 4))
        print("Metadata: ", metadata)
        nx, ny, nt, dx, Nb = map(int, metadata)
        
        # Read the wavefield data
        data = np.fromfile(f, dtype=np.float64)
        
    # Reshape the data to the original shape
    wavefield = data.reshape((nx, ny, nt))
    
    return wavefield, nt, dx, Nb

def load_seismogram_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the seismogram
        metadata = np.fromfile(f, dtype=float, count=5)
        print(metadata)
        nr, nt, dt, dx, Nb = metadata
        nr = int(nr)
        nt = int(nt)
        dx = int(dx)
        Nb = int(Nb)
        
        # Read the wavefield data
        data = np.fromfile(f, dtype=float)
        
    # Reshape the data to the original shape
    seismogram = data.reshape(nr, nt)
    return seismogram, nt, dx, dt, Nb

def init():
    im.set_data(data[0,:,:])
    return (im,)

# animation function. This is called sequentially
def animate(i):
    data_slice = data[i,:,:]
    im.set_data(data_slice)
    return (im,)

'''
### Seismogram ###
seismogram, nt, dx, dt, Nb = load_seismogram_from_bin("Visualization/seismogram.bin")
print(seismogram.shape)
#print(nt,dx,dt,Nb)
#plt.imshow(seismogram.T)

#data = seismogram
time = np.linspace(0 * dt, nt * dt, nt)

# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(seismogram, 1)
vmax = np.percentile(seismogram, 98.5)

plt.figure(facecolor='white', figsize=(8, 6))
labelsize = 16
plt.title('Seismogram', fontsize=labelsize)
plt.xlabel("Distance (m)", fontsize=labelsize)
plt.ylabel("Time (ms)", fontsize=labelsize)
extent = [0, (seismogram.shape[0]) * dx, nt * dt * 1000, 0]
plt.imshow(seismogram.T, cmap='bwr', vmin=-vmax, vmax=vmax, extent=extent, aspect = 'auto', interpolation='spline16')
plt.xlim(0, (seismogram.shape[0]) * dx)
plt.ylim(nt * dt * 1000, 0)

# Set the extent values: left, right, bottom, and top
#, extent=[0,500,300,0]

# Create arrays of tick positions and labels
#num_ticks = 6
#y_ticks = np.arange(0, nt, step=200)
#y_tick_labels = [f"{t * 1000:.1f}" for t in time[y_ticks]]

# Set the tick positions and labels for the y axis
#plt.yticks(y_ticks, y_tick_labels)
plt.savefig("Visualization/Seismogram.svg", format="svg", bbox_inches = 'tight')
plt.show()

### Cut Seismogram ####
Cut  = seismogram[Nb:-Nb, :]

print(Cut.shape)
print("Shape of Rec.p ", seismogram.shape)

### Seismogram Cut ###
# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
vmax = np.percentile(seismogram, 98.5)

plt.figure(facecolor='white', figsize=(8, 6))
labelsize = 16
plt.title('Seismogram', fontsize=labelsize)
plt.xlabel("Distance (m)", fontsize=labelsize)
plt.ylabel("Time (ms)", fontsize=labelsize)
extent = [0, Cut.shape[0] * dx, nt * dt * 1000, 0]
plt.imshow(Cut.T, cmap='bwr', vmin=-vmax, vmax=vmax, extent=extent, aspect = 'auto', interpolation='spline16')
plt.xlim(0, (Cut.shape[0]) * dx)
plt.ylim(nt * dt * 1000, 0)

# Set the extent values: left, right, bottom, and top
#, extent=[0,500,300,0]

# Create arrays of tick positions and labels
#num_ticks = 6
#y_ticks = np.arange(0, nt, step=200)
#y_tick_labels = [f"{t * 1000:.1f}" for t in time[y_ticks]]

# Set the tick positions and labels for the y axis
#plt.yticks(y_ticks, y_tick_labels)
plt.savefig("Visualization/Seismogram_Cut.svg", format="svg", bbox_inches = 'tight')
plt.show()


'''


##### Wavefield ###
#######################################
wavefield, nt, dx, Nb = load_wavefield_from_bin("Visualization/wavefield.bin")
print(wavefield.shape)
data = wavefield.T

vmax=5e-11
plt.imshow(data[100,:,:], cmap = 'bwr', vmin=-vmax, vmax=vmax, interpolation='spline16')
plt.show()

fig, ax = plt.subplots()
ax.labelsize = 5
# Create arrays of tick positions and labels
num_ticks = 6
x_ticks = np.linspace(0, wavefield.shape[0]-1, num_ticks, dtype=int)
y_ticks = np.linspace(0, wavefield.shape[1]-1, num_ticks)
x_tick_labels = [f"{x * dx:.1f}" for x in x_ticks]
y_tick_labels = [f"{y * dx:.1f}" for y in y_ticks]

# Set the tick positions and labels for the x and y axes
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, fontsize=8)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels, fontsize=8)

# Set axis labels
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")

ax.set_xlim(0, (wavefield.shape[0]-1))
ax.set_ylim(wavefield.shape[1]-1, 0)

# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
#vmax = np.percentile(wavefield, 98.5)
#vmin=-1e-10
vmax=5e-11

im = ax.imshow(data[0,:,:], cmap = 'bwr', vmin=-vmax, vmax=vmax, interpolation='spline16')
#im2 = ax.imshow((vp.T)/1000, cmap=plt.cm.gray, interpolation='nearest', alpha=0.4)

#spline16
#bwr
#coolwarm_r


# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nt, interval=1, blit=True)


writervideo = animation.FFMpegWriter(fps=60, bitrate=5000)
anim.save('Visualization/wavefield.mp4', writer=writervideo, dpi=300)
plt.close()


print('OK')


'''
########################33#
copy = wavefield[Nb:-Nb, Nb:-Nb, :]
print(copy.shape)
data = copy.T
print(data.shape)

fig, ax = plt.subplots()
# Create arrays of tick positions and labels
num_ticks = 6
x_ticks = np.linspace(0, copy.shape[0]-1, num_ticks)
y_ticks = np.linspace(0, copy.shape[1]-1, num_ticks)
x_tick_labels = [f"{x * dx:.1f}" for x in x_ticks]
y_tick_labels = [f"{y * dx:.1f}" for y in y_ticks]

# Set the tick positions and labels for the x and y axes
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, fontsize=8)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels, fontsize=8)

# Set axis labels
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")

ax.set_xlim(0, (copy.shape[0]-1))
ax.set_ylim(copy.shape[1]-1, 0)

# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
vmax = np.percentile(wavefield, 98.5)


#, vmin=-1e-10, vmax=1e-10

im = ax.imshow(data[0,:,:], cmap = 'bwr', vmin=-vmax, vmax=vmax, interpolation='spline16')
#im2 = ax.imshow((vp.T)/1000, cmap=plt.cm.gray, interpolation='nearest', alpha=0.4)

#spline16
#bwr
#coolwarm_r


# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nt, interval=1, blit=True)


writervideo = animation.FFMpegWriter(fps=60, bitrate=5000)
anim.save('Visualization/wavefield_cut.mp4', writer=writervideo, dpi=300)
plt.close()
'''