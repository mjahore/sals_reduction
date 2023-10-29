#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Some parameters we will need later. These are instrument
# specific.
bin_size = 1        # size of bin (in pixels) for averaging
px_to_mm = 0.0048   # 1 pixel = 0.0048 mm
n_ref    = 1.3317   # Refractive index of solvent
l_wavel  = 625.00   # Laser wavelength in nm
q2       = 102.00   # Distance from Lens 2 (L2) to detector (D) (units: mm)
l1       = 15.00    # Distance from Sample (S) to Lens 1 (L1) (units: mm))
l2       = 22.50    # Distance from Beamstop (BS) to L2 (units: mm)

# Ensure the code is called correctly.
if (len(sys.argv) != 5):
	print("\n\tUsage: sals_reduction.py [bs_x_pos] [bs_y_pos] [sample_filename.tiff] [background_filename.tiff]\n")
	quit()
else:
	x_cen = int(sys.argv[1]) # Beamstop center (X) on image (units: pixel)
	y_cen = int(sys.argv[2]) # Beamstop center (Y) on image (units: pixel)
	filen = sys.argv[3]      # Sample filename
	bkg   = sys.argv[4]      # Background/empty cell filename
	base_fn, ext_fn = filen.split(".")

# Read the TIFF stack.
im = io.imread(filen)

# Get the number of frames.
tiff_frames = im.shape[0]

# Assuming image is square. tiff_dim is pixels in each dimensions.
tiff_dim    = im.shape[1]

# Go through each image and integrate the intensity I(x,y,t) -> I(x,y)
I_xy  = np.zeros(im[1].shape, dtype=float, order='C')
dI_xy = np.zeros(im[1].shape, dtype=float, order='C') 
for image in im:
	dI_xy += image
	I_xy += image/tiff_frames

# Uncertainty for random process = sqrt(counts)
dI_xy = np.sqrt(dI_xy)

# The intensity scale is in arbitrary units, so no need to multiply by dt or whatever
# since that is, well, also arbitrary.
fig, ax = plt.subplots()
im = ax.imshow(I_xy)
fig.colorbar(im, ax=ax, label='Total Scattering Intensity (arb. units)')

# This is for labeling our axis values:
plt.xlabel(r"$x\; (\mathrm{px})$", fontsize=16)
plt.ylabel(r"$y\; (\mathrm{px})$", fontsize=16)
plt.plot(int(x_cen), int(y_cen), marker="x", markersize=16, markeredgecolor="red")
plt.show()
fig.savefig(base_fn + "_sals.png")

# Now let's azimuthally average I_xy.
min_idx = 50  # Offset from beam stop center.
max_idx = 512 # Maximum distance to travel from beam stop center.
I_avg   = np.zeros(int(tiff_dim/bin_size), dtype=float, order='C')
I_cnt   = np.zeros(int(tiff_dim/bin_size), dtype=int, order='C')
for i in range(x_cen-max_idx, x_cen+max_idx):
	for j in range(y_cen-max_idx,y_cen+max_idx):
		dx = i - x_cen
		dy = j - y_cen
		dr = np.sqrt(dx*dx + dy*dy) # Distance from beam stop center, in pixels.
		bin_id = int(dr/bin_size)
		I_avg[bin_id] += I_xy[i,j]	
		I_cnt[bin_id] += 1

q_vals = np.zeros(int(tiff_dim/bin_size), dtype=float, order='C') 
for i in range(int(tiff_dim/bin_size)):
	if (I_cnt[i] == 0):
		I_cnt[i] = 1
	I_avg[i] /= I_cnt[i]
	# Calculate q for each of the bins. Units: Angstrom^-1.
	q_vals[i] = 4.00 * 3.14159 * n_ref / (l_wavel * 10.0) * np.sin(0.5 * (i*bin_size)*px_to_mm / (q2*l1/l2))

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
plt.plot(q_vals, I_avg, marker="o")
plt.ylabel(r"Scattering Intensity, I(q) (arb. units)", fontsize=14)
plt.xlabel(r"Scattering Vector, q ($\AA^{-1}$)", fontsize=14)
plt.show()
fig.savefig(base_fn + "_1dsals.png")

# Save reduced data
np.savetxt(base_fn + "_reduced.dat",  np.c_[q_vals, I_avg], newline="\n", delimiter=' ')

exit
