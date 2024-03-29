#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Some parameters we will need later. These are instrument
# specific.
use_min_max = 0     # restrict q region in reduced data?
skip_bkg = 0        # Skip background subtraction? (why???)
bin_size = 1        # size of bin (in pixels) for averaging
px_to_mm = 0.0048   # 1 pixel = 0.0048 mm
n_ref    = 1.3317   # Refractive index of solvent
l_wavel  = 625.00   # Laser wavelength in nm
l1       = 15.00    # Distance from Sample (S) to Lens 1 (L1) (units: mm))
l2       = 70.00    # Distance from Beamstop (BS) to L2 (units: mm)
EFL2     = 25.43    # Effective focal length of L2 (units: mm)

# For calculating q (units: inverse Angstrom)
q2      = (l2*EFL2)/(l2 - EFL2)
q_coeff = 4.00*3.14158*n_ref/(10.0*l_wavel)
 
# Ensure the code is called correctly.
if (len(sys.argv) != 7):
	print("\n\tUsage: sals_reduction.py [bs_x_pos] [bs_y_pos] [sample_filename.tiff] [bkg_bs_x] [bkg_bs_y] [background_filename.tiff]\n")
	quit()
else:
	x_cen  = int(sys.argv[1]) # Beamstop center (X) on image (units: pixel)
	y_cen  = int(sys.argv[2]) # Beamstop center (Y) on image (units: pixel)
	filen  = sys.argv[3]      # Sample filename
	bx_cen = int(sys.argv[4]) # Beamstop center (X) for background (units: pixel)
	by_cen = int(sys.argv[5]) # Beamstop center (Y) for background (units: pixel)
	bkg_fn = sys.argv[6]      # Background/empty cell filename
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

# We are expecting a file background.tif to be in this directory that
# contains any scattering from cuvette/solvent.
bkg        = io.imread(bkg_fn)
bkg_frames = im.shape[0]
B_xy       = np.zeros(im[1].shape, dtype=float, order='C')
for image in bkg:
	B_xy += image/bkg_frames

# Now let's azimuthally average I_xy.
if (use_min_max == 0):
	min_q = 0 # Units: inverse angstrom
	max_q = 4e-4 #  ""      ""      ""
	min_idx = 0
	max_idx = 569
else:
	min_idx = 175  # Offset from beam stop center.
	max_idx = 600 # Maximum disaance to travel from beam stop center.
	min_q = q_coeff * np.sin(0.5 * int(min_idx/bin_size)*px_to_mm * l2/(q2*l1))
	max_q = q_coeff * np.sin(0.5 * int(max_idx/bin_size)*px_to_mm * l2/(q2*l1))

I_avg   = np.zeros(int(tiff_dim/bin_size), dtype=float, order='C')
I_cnt   = np.zeros(int(tiff_dim/bin_size), dtype=int, order='C')
bkg_avg   = np.zeros(int(tiff_dim/bin_size), dtype=float, order='C')
bkg_cnt   = np.zeros(int(tiff_dim/bin_size), dtype=int, order='C')
for i in range(x_cen-max_idx, x_cen+max_idx):
	for j in range(y_cen-max_idx,y_cen+max_idx):
		# Azimuthally average intensity:
		dx = i - x_cen
		dy = j - y_cen
		dr = np.sqrt(dx*dx + dy*dy) # Distance from beam stop center, in pixels.
		bin_id = int(dr/bin_size)
		if (bin_id > int(min_idx/bin_size) and bin_id < int(max_idx/bin_size)):
			I_avg[bin_id] += I_xy[i,j]	
			I_cnt[bin_id] += 1

		# Azimuthally average background intensity:
		dx = i - bx_cen
		dy = j - by_cen
		dr = np.sqrt(dx*dx + dy*dy) # Distance from beam stop center, in pixels.
		bin_id = int(dr/bin_size)
		if (bin_id > int(min_idx/bin_size) and bin_id < int(max_idx/bin_size)):
			bkg_avg[bin_id] += B_xy[i,j]	
			bkg_cnt[bin_id] += 1
	
q_vals = np.zeros(int(tiff_dim/bin_size), dtype=float, order='C') 
for i in range(int(tiff_dim/bin_size)):
	if (I_cnt[i] == 0):
		I_cnt[i] = 1

	if (bkg_cnt[i] ==0):
		bkg_cnt[i] = 1

	I_avg[i] /= I_cnt[i]
	bkg_avg[i] /= bkg_cnt[i]

	# Calculate q for each of the bins. Units: Angstrom^-1.
	q_vals[i] = q_coeff * np.sin(0.5 * (i*bin_size)*px_to_mm * l2/(q2*l1))

# Subtract off the background:
I_avg -= bkg_avg

# Uncertainty for random process = sqrt(counts)
dI_xy = np.sqrt(dI_xy)

# Correct for background scattering.
I_xy -= B_xy

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

quit(0)
