import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import configparser
import h5py
from lookup import lookup as ISMCLookup

# formatting for matplot
plt.rcParams.update({'font.size':14})

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("configfile", help="Path to a suitable configuration .ini file")
parser.add_argument("lookupfile", help="Path to an ISMC lookup table file.")
args = parser.parse_args()

# check input file
parser = configparser.ConfigParser()
parser.read(args.configfile)

#====Graph Bounds====#
R_d_max = float(parser.get('lookup','R_d_max'))
R_d_min = float(parser.get('lookup','R_d_min'))
T_d_max = float(parser.get('lookup','T_d_max'))
T_d_min = float(parser.get('lookup','T_d_min'))

perc_unc_min = float(parser.get('lookup','perc_unc_min'))
perc_unc_max = float(parser.get('lookup','perc_unc_max'))
#========#

# read lookup tables
lookup_tables = h5py.File(args.lookupfile)
muaL = np.array(lookup_tables["mua_lookup"]).T
musL = np.array(lookup_tables["mus_lookup"]).T
muaL_unc = np.array(lookup_tables["mua_lookup_uncertainty"]).T
musL_unc = np.array(lookup_tables["mus_lookup_uncertainty"]).T
mua_points = np.array(lookup_tables["mua_points"]).T
mus_points = np.array(lookup_tables["mus_points"]).T

muas = np.array(lookup_tables["muas"])
muss = np.array(lookup_tables["muss"])

Rvals = np.array(lookup_tables["Rvals"])
Tvals = np.array(lookup_tables["Tvals"])
n = float(np.array(lookup_tables["n"]))
n_unc = float(np.array((lookup_tables["n_unc"])))
g = float(np.array((lookup_tables["g"])))
t = float(np.array((lookup_tables["t"])))
R = bool(np.array(lookup_tables["R"]))
R_pw = float(np.array(lookup_tables["R_pw"]))
R_pw_unc = float(np.array(lookup_tables["R_pw_unc"]))
R_fs = float(np.array(lookup_tables["R_fs"]))
R_fs_unc = float(np.array(lookup_tables["R_fs_unc"]))
R_fp = float(np.array(lookup_tables["R_fp"]))
R_fp_unc = float(np.array(lookup_tables["R_fp_unc"]))
R_fd = float(np.array(lookup_tables["R_fd"]))
R_fd_unc = float(np.array(lookup_tables["R_fd_unc"]))
R_angle = float(np.array(lookup_tables["R_angle"]))
R_angle_unc = float(np.array(lookup_tables["R_angle_unc"]))
T = bool(np.array(lookup_tables["T"]))
T_pw = float(np.array(lookup_tables["T_pw"]))
T_pw_unc = float(np.array(lookup_tables["T_pw_unc"]))
T_fs = float(np.array(lookup_tables["T_fs"]))
T_fs_unc = float(np.array(lookup_tables["T_fs_unc"]))
T_fp = float(np.array(lookup_tables["T_fp"]))
T_fp_unc = float(np.array(lookup_tables["T_fp_unc"]))
T_fd = float(np.array(lookup_tables["T_fd"]))
T_fd_unc = float(np.array(lookup_tables["T_fd_unc"]))
T_angle = float(np.array(lookup_tables["T_angle"]))
T_angle_unc = float(np.array(lookup_tables["T_angle_unc"]))

separate = bool(np.array(lookup_tables["separate"]))
specular_included = bool(np.array(lookup_tables["specular_included"]))

print("==== Simulation Configuration Used to Generate the Lookup Tables ====\n")
print("n: %s +/- %s\ng: %s\nt: %s (cm)\nR: %s\nR_pw: %s +/- %s\nR_fs: %s +/- %s\nR_fp: %s +/- %s\nR_fd: %s +/- %s\nR_angle: %s +/- %s\nT: %s\nT_pw: %s +/- %s\nT_fs: %s +/- %s\nT_fp: %s +/- %s\nT_fd: %s +/- %s\nT_angle: %s +/- %s\nSeparate: %s\nSpecular Included: %s" %(n, n_unc, g, t, R, R_pw, R_pw_unc, R_fs, R_fs_unc, R_fp, R_fp_unc, R_fd, R_fd_unc, R_angle, R_angle_unc, T, T_pw, T_pw_unc, T_fs, T_fs_unc, T_fp, T_fp_unc, T_fd, T_fd_unc, T_angle, T_angle_unc, separate, specular_included))
print("\n========\n")


grid_x, grid_y = np.mgrid[0:100:complex(muaL.shape[0]), 0:100:complex(muaL.shape[0])]

prefix = parser.get('lookup','filename_prefix')
mua_filename= prefix+"_mua_lookup.png"
mus_filename=prefix+"_mus_lookup.png"
mua_uncertainty_filename=prefix+"_mua_lookup_uncertainty.png"
mus_uncertainty_filename=prefix+"_mus_lookup_uncertainty.png"



# make figures
fig, ax = plt.subplots()

#im = ax.imshow(np.flip(mua_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=plt.get_cmap("viridis"), vmin=0)
im = ax.pcolor(grid_x, grid_y, muaL_unc.T, vmin=perc_unc_min, vmax=perc_unc_max, cmap=plt.get_cmap("viridis"))
ax.set_title("Uncertainty From Simulations\nin $\mu_a$ Lookup Table",y=1.05)
#ax.plot(mua_points[:,0], mua_points[:,1], 'k.', alpha=0.5, ms=1)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Percent Uncertainty in $\mu_a$ (%)")
ax.set_xlabel("Diffuse Reflectance (%)")
ax.set_xlim([R_d_min,R_d_max])
ax.set_ylabel("Diffuse Transmittance (%)")
ax.set_ylim([T_d_min,T_d_max])
plt.tight_layout()
plt.savefig(mua_uncertainty_filename,dpi=500)


fig, ax = plt.subplots()

#im = ax.imshow(np.flip(mus_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=plt.get_cmap("viridis"), vmin=0)
im = ax.pcolor(grid_x, grid_y, musL_unc.T, vmin=perc_unc_min, vmax=perc_unc_max, cmap=plt.get_cmap("viridis"))
ax.set_title("Uncertainty From Simulations\nin $\mu_s'$ Lookup Table",y=1.05)
#ax.plot(mus_points[:,0], mus_points[:,1], 'k.', alpha=0.5, ms=1)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Percent Uncertainty in $\mu_s'$ (%)")
ax.set_xlabel("Diffuse Reflectance (%)")
ax.set_xlim([R_d_min,R_d_max])
ax.set_ylabel("Diffuse Transmittance (%)")
ax.set_ylim([T_d_min,T_d_max])
plt.tight_layout()
plt.savefig(mus_uncertainty_filename,dpi=500)


fig, ax = plt.subplots()

#im = ax.imshow(np.flip(mua_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=plt.get_cmap("viridis"), vmin=0)
im = ax.pcolor(grid_x, grid_y, muaL.T, norm=colors.LogNorm(vmin=muas.min(),vmax=muas.max()), cmap=plt.get_cmap("viridis"))
ax.set_title("Absorption Coefficient Lookup Table",y=1.05)
ax.plot(mua_points[:,0], mua_points[:,1], 'k.', alpha=0.5, ms=1)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Absorption Coefficient, $\mu_a$ (1/cm)")
ax.set_xlabel("Diffuse Reflectance (%)")
ax.set_xlim([R_d_min,R_d_max])
ax.set_ylabel("Diffuse Transmittance (%)")
ax.set_ylim([T_d_min,T_d_max])
plt.tight_layout()
plt.savefig(mua_filename,dpi=500)


fig, ax = plt.subplots()

#im = ax.imshow(np.flip(mus_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=plt.get_cmap("viridis"), vmin=0)
im = ax.pcolor(grid_x, grid_y, musL.T, norm=colors.LogNorm(vmin=muss.min(),vmax=muss.max()), cmap=plt.get_cmap("viridis"))
ax.set_title("Reduced Scattering Coefficient Lookup Table",y=1.05)
ax.plot(mus_points[:,0], mus_points[:,1], 'k.', alpha=0.5, ms=1)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Reduced Scattering Coefficient, $\mu_s'$ (1/cm)")
ax.set_xlabel("Diffuse Reflectance (%)")
ax.set_xlim([R_d_min,R_d_max])
ax.set_ylabel("Diffuse Transmittance (%)")
ax.set_ylim([T_d_min,T_d_max])
plt.tight_layout()
plt.savefig(mus_filename,dpi=500)

# perform the lookup
U = int(parser.get('lookup','measurement_distributions_samples')) # number of times to sample measurement uncertainty distributions
R_avg_array, R_unc_array = np.array(parser.get('experiment', 'M_R').split("\n"),dtype=float)*100, np.array(parser.get('experiment','M_R_unc').split("\n"),dtype=float)*100
T_avg_array, T_unc_array = np.array(parser.get('experiment','M_T').split("\n"),dtype=float)*100, np.array(parser.get('experiment','M_T_unc').split("\n"),dtype=float)*100

# perform several lookups
for l in range(np.size(R_avg_array)):
    R_avg = R_avg_array[l]
    T_avg = T_avg_array[l]
    R_unc = R_unc_array[l]
    T_unc = T_unc_array[l]

    mua_array = np.zeros(U)
    mua_perc_unc_array = np.zeros(U)
    mus_array = np.zeros(U)
    mus_perc_unc_array = np.zeros(U)

    for i in range(U):

        R,T = np.random.normal(R_avg,R_unc), np.random.normal(T_avg,T_unc)

        mua_array[i], R_exe, T_exe = ISMCLookup(muaL,Rvals,Tvals,R,T,verbose=False)
        #mua_perc_unc_array[i], R_exe, T_exe = lookup(muaL_unc,Rvals,Tvals,R,T,verbose=False)

        mus_array[i], R_exe, T_exe = ISMCLookup(musL,Rvals,Tvals,R,T,verbose=False)
        #mus_perc_unc_array[i], R_exe, T_exe = lookup(musL_unc,Rvals,Tvals,R,T,verbose=False)

    #mua_unc_array = mua_array*mua_perc_unc_array/100
    #mus_unc_array = mus_array*mus_perc_unc_array/100

    # to find the uncertainty in mua and mus, combine the two uncertainties: the uncertainty of the table
    # and the uncertainty caused by the measurement uncertainties

    # uncertainty from table
    mua_perc_unc_1, R_exe, T_exe = ISMCLookup(muaL_unc,Rvals,Tvals,R,T,verbose=False)
    mus_perc_unc_1, R_exe, T_exe = ISMCLookup(musL_unc,Rvals,Tvals,R,T,verbose=False)

    mua=np.mean(mua_array)
    mus=np.mean(mus_array)

    # uncertainty from measurement
    mua_perc_unc_2 = np.std(mua_array)*100/mua
    mus_perc_unc_2 = np.std(mus_array)*100/mus

    # combined uncertainty
    mua_perc_unc = np.sqrt(mua_perc_unc_1**2 + mua_perc_unc_2**2)
    mus_perc_unc = np.sqrt(mus_perc_unc_1**2 + mus_perc_unc_2**2)

    print("mua = %f +/- %f for R_d = %f +/- %f and T_d = %f +/- %f" %(mua, mua*mua_perc_unc/100, R_avg, R_unc, T_avg, T_unc))
    print("mus' = %f +/- %f for R_d = %f and +/- %f T_d = %f +/- %f" %(mus, mus*mus_perc_unc/100, R_avg, R_unc, T_avg, T_unc))
