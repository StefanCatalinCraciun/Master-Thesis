"""
Visualization libary

This script contains functions for visualizing the wavefield, seismogram and traces. 
It loads the data from the binary files and plots them.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Internal Functions 
# ----------------
def load_wavefield_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the wavefield
        metadata = np.fromfile(f, dtype=float, count=5)
        #print(metadata)
        nx, ny, nt, dx, Nb = metadata
        
        # Read the wavefield data
        data = np.fromfile(f, dtype=np.float64)
        print(data.shape)
        nx = int(nx)
        ny = int(ny)
        nt = int(nt)
        dx = int(dx)
        Nb = int(Nb)
        print(nx, ny, nt, dx, Nb)
    # Reshape the data to the original shape
    wavefield = data.reshape((nt, ny, nx))
    return wavefield, nt, dx, Nb

def load_seismogram_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the seismogram
        metadata = np.fromfile(f, dtype=float, count=5)
        #print(metadata)
        nr, nt, dt, dx, Nb = metadata
        nr = int(nr)
        nt = int(nt)
        dx = int(dx)
        Nb = int(Nb)
        
        # Read the seismogram data
        data = np.fromfile(f, dtype=np.float64)
        
    # Reshape the data to the original shape
    seismogram = data.reshape(nt, nr)
    return seismogram, nt, dx, dt, Nb


def load_trace_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the seismogram
        metadata = np.fromfile(f, dtype=float, count=2)
        #print(metadata)
        nt, dt = metadata
        nt = int(nt)
        dt = float(dt)
        
        # Read the trace data
        data = np.fromfile(f, dtype=float)
        
    # Reshape the data to the original shape
    trace = data.reshape(nt)
    return trace, nt, dt

def load_trace_analytical_from_bin(file_path):
    with open(file_path, "rb") as f:
        # Read the metadata (shape) of the seismogram
        metadata = np.fromfile(f, dtype=float, count=2)
        #print(metadata)
        nt, dt = metadata
        nt = int(nt)
        dt = float(dt)
        
        # Read the trace data
        data = np.fromfile(f, dtype=float)
        
    # Reshape the data to the original shape
    trace_analytical = data.reshape(nt)
    return trace_analytical, nt, dt

def init():
    im.set_data(data[0,:,:])
    return (im,)

# Animation function. This is called sequentially
def animate(i):
    data_slice = data[i,:,:]
    im.set_data(data_slice)
    return (im,)

"""
1 Trace Plot 

"""
trace, nt, dt = load_trace_from_bin("Visualization/trace.bin")
print(trace.shape)
time = np.linspace(0 * dt, nt * dt, nt)

plt.figure(facecolor='white', figsize=(8, 6))
labelsize = 16
plt.title('Trace', fontsize=labelsize)
plt.xlabel("Time (s)", fontsize=labelsize)
plt.ylabel("Amplitude", fontsize=labelsize)
plt.plot(time, trace, 'b-',lw=3,label="FD solution")
plt.xlim(time[0], time[-1])
plt.legend()
plt.grid()
plt.savefig("Visualization/Trace.svg", format="svg", bbox_inches = 'tight')
plt.show()

"""
Analytical Trace Plot 

"""
trace_analytical, nt, dt = load_trace_from_bin("Visualization/trace_analytical.bin")
print(trace_analytical.shape)
plt.figure(facecolor='white', figsize=(8, 6))
labelsize = 16
plt.title('Analytical Solution vs FD Solution', fontsize=labelsize)
plt.xlabel("Time (s)", fontsize=labelsize)
plt.ylabel("Amplitude", fontsize=labelsize)
plt.plot(time, trace, 'b-',lw=3,label="FD solution")
plt.plot(time, trace_analytical, 'r--',lw=3,label="Analytical solution")
plt.xlim(time[0], time[-1])
plt.legend()
plt.grid()
plt.savefig("Visualization/Trace_Comparison.svg", format="svg", bbox_inches = 'tight')
plt.show()


"""
Seismogram 

"""
seismogram, nt, dx, dt, Nb = load_seismogram_from_bin("Visualization/seismogram.bin")
#print(seismogram.shape)
#print(nt,dx,dt,Nb)
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
extent = [0, (seismogram.shape[1]) * dx, nt * dt * 1000, 0]
plt.imshow(seismogram, cmap='bwr', vmin=-vmax, vmax=vmax, extent=extent, aspect = 'auto', interpolation='spline16')
plt.xlim(0, (seismogram.shape[1]) * dx)
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
Cut  = seismogram[:, Nb:-Nb]

#print(Cut.shape)
#print("Shape of Rec.p ", seismogram.shape)

"""
Seismogram Cut

"""
# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
vmax = np.percentile(seismogram, 98.5)

plt.figure(facecolor='white', figsize=(8, 6))
labelsize = 16
plt.title('Seismogram', fontsize=labelsize)
plt.xlabel("Distance (m)", fontsize=labelsize)
plt.ylabel("Time (ms)", fontsize=labelsize)
extent = [0, Cut.shape[1] * dx, nt * dt * 1000, 0]
plt.imshow(Cut, cmap='bwr', vmin=-vmax, vmax=vmax, extent=extent, aspect = 'auto', interpolation='spline16')
plt.xlim(0, (Cut.shape[1]) * dx)
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


"""
Wavefield

"""
#######################################
wavefield, nt, dx, Nb = load_wavefield_from_bin("Visualization/wavefield.bin")
print(wavefield.shape)

data = wavefield

fig, ax = plt.subplots()
ax.labelsize = 5
# Create arrays of tick positions and labels
num_ticks = 6
x_ticks = np.linspace(0, wavefield.shape[2]-1, num_ticks, dtype=int)
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

ax.set_xlim(0, (wavefield.shape[2]-1))
ax.set_ylim(wavefield.shape[1]-1, 0)

# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
#vmax = np.percentile(wavefield, 99.5)
#, vmin=-1e-10, 
vmax=5e-10

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


"""
Wavefield Cut

"""
#######################################
copy = wavefield[:, Nb:-Nb, Nb:-Nb]
#print(copy.shape)
data = copy
print(data.shape)

fig, ax = plt.subplots()
# Create arrays of tick positions and labels
num_ticks = 6
x_ticks = np.linspace(0, copy.shape[2]-1, num_ticks)
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

ax.set_xlim(0, (copy.shape[2]-1))
ax.set_ylim(copy.shape[1]-1, 0)

# Calculate vmin and vmax based on percentiles
#vmin = np.percentile(wavefield, 1)
vmax = np.percentile(data, 98.5)


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
print('Done')
