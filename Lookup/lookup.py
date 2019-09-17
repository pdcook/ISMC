import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import interpolate
from math import isnan
import argparse
import configparser
import operator
import os
import sys
import h5py
import hdf5storage
sys.path.append("../")
from ISMC import ISMC as pythonISMC

"""
This script generates the lookup tables for mua and mus', as well as the uncertainty in those
tables propagated forward through the uncertainty in the sphere properties as well as the
refractive index.

The scrip can also perform a lookup given a measurement of R_d and T_d with uncertainty
to back-propagate the uncertainty in the measurements to give a final mua, mus' with total
uncertainty
"""

if __name__ == '__main__':
    # formatting for matplot
    plt.rcParams.update({'font.size':14})

    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--Python', dest='python', help="Run the simuations with Python.", action='store_true')
    parser.add_argument('--CUDA', dest='cuda', help="Run the simuations with CUDA. Must also provide a CUDA source and GPU IDs with --CUDAsource and --GPU respectively.", action='store_true')
    parser.add_argument('--CUDAsource', dest="source", help="Path to a suitable compiled CUDA source to run the simulation from if CUDA is the desired environment.")
    parser.add_argument('--GPU', dest="GPU", help="Device number(s) of the GPU(s) to run the simulation on. Must be specified if --CUDA is used. Use 'nvidia-smi' to list available devices. Example: --GPU 0 2", nargs='*')
    parser.add_argument("configfile", help="Path to a suitable configuration .ini file")
    args = parser.parse_args()

    if not (args.python or args.cuda):
        raise RuntimeError("Please specify a simulation environment with --CUDA or --Python")

    # check input file
    parser = configparser.SafeConfigParser()
    parser.read(args.configfile)

    #====Sphere Parameters====#
    R       = int(parser.get('spheres','R_sphere'))          # boolean to determine if the reflection sphere is present or not
    R_pw    = float(parser.get('spheres','R_pw'))              # reflectance of reflection sphere
    R_pw_unc= float(parser.get('spheres','R_pw_unc'))          # uncertainty in reflectance of reflection sphere
    R_fs    = float(parser.get('spheres','R_fs'))              # sample port fraction of reflection sphere
    R_fs_unc= float(parser.get('spheres','R_fs_unc'))          # 5% uncertainty in sample port fraction of reflection sphere
    R_fp    = float(parser.get('spheres','R_fp'))              # source port fraction of reflection sphere
    R_fp_unc= float(parser.get('spheres','R_fp_unc'))          # 5% uncertainty in source port fraction of reflection sphere
    R_fd    = float(parser.get('spheres','R_fd'))              # detector fraction of reflection sphere
    R_fd_unc= float(parser.get('spheres','R_fd_unc'))          # 5% uncertainty in detector fraction of reflection sphere
    R_f     = R_fs + R_fp + R_fd                        # total port fraction of reflection sphere
    R_f_unc = np.sqrt(R_fs**2 + R_fp**2 + R_fd**2)      # uncertainty in total port fraction of reflection sphere
    R_angle = float(parser.get('spheres','R_angle'))  #degrees # angle threshold for specular reflection if no Rsphere is present
    R_angle_unc = float(parser.get('spheres','R_angle_unc'))   # uncertainty in angle threshold for specular reflection
    specular_included =  int(parser.get('spheres','specular_included')) # boolean to determine if specular reflection is included or not

    T       = int(parser.get('spheres','T_sphere'))          # boolean to determine if the transmission sphere is present or not
    T_pw    = float(parser.get('spheres','T_pw'))              # reflectance of transmission sphere
    T_pw_unc= float(parser.get('spheres','T_pw_unc'))          # uncertainty in reflectance of transmission sphere
    T_fs    = float(parser.get('spheres','T_fs'))              # sample port fraction of transmission sphere
    T_fs_unc= float(parser.get('spheres','T_fs_unc'))          # 5% uncertainty in sample port fraction of transmission sphere
    T_fp    = float(parser.get('spheres','T_fp'))              # optional port fraction of transmission sphere
    T_fp_unc= float(parser.get('spheres','T_fp_unc'))          # 5% uncertainty in optional port fraction of transmission sphere
    T_fd    = float(parser.get('spheres','T_fd'))              # detector fraction of transmission sphere
    T_fd_unc= float(parser.get('spheres','T_fd_unc'))          # 5% uncertainty in detector fraction of transmission sphere
    T_f     = T_fs + T_fp + T_fd                        # total port fraction of transmission sphere
    T_f_unc = np.sqrt(T_fs**2 + T_fp**2 + T_fd**2)      # uncertainty in total port fraction of transmission sphere
    T_angle = float(parser.get('spheres','T_angle'))  #degrees # angle threshold for unscattered transmission if no Tsphere is present
    T_angle_unc = float(parser.get('spheres','T_angle_unc'))   # uncertainty in angle threshold for unscattered transmittance
    #========#

    #====Sample Configuration====#

    # can specify number of photons and number of runs
    #  or uncertainty threshold
    # specifying uncertainty threshold will yield
    #  the same or better results MUCH faster

    if (parser.has_option('lookup','N') and parser.has_option('lookup','runs')):
        N = int(parser.get('lookup','N'))
        runs = int(parser.get('lookup','runs'))
    else:
        uncertainty = float(parser.get('lookup','uncertainty'))
        estimate = float(parser.get('lookup','estimate'))

    separate = int(parser.get('spheres','separate'))

    uncertainty_runs = int(parser.get('lookup','input_distributions_samples')) # number of times to sample distributions for refractive index and
                            # sphere parameters
    n = float(parser.get('sample','n'))
    n_unc = float(parser.get('sample','n_unc'))
    g = float(parser.get('sample','g'))
    t = float(parser.get('sample','t'))

    N_mu_a = int(parser.get('lookup','N_mua'))
    mu_a_start = float(parser.get('lookup','mua_start'))
    mu_a_base = float(parser.get('lookup','mua_base'))

    N_mu_s = int(parser.get('lookup','N_mus'))
    mu_s__start = float(parser.get('lookup','mus_start'))
    mu_s__base = float(parser.get('lookup','mus_base'))
    #========#

    #====Graph Bounds====#
    R_d_max = float(parser.get('lookup','R_d_max'))
    R_d_min = float(parser.get('lookup','R_d_min'))
    T_d_max = float(parser.get('lookup','T_d_max'))
    T_d_min = float(parser.get('lookup','T_d_min'))

    perc_unc_min = float(parser.get('lookup','perc_unc_min'))
    perc_unc_max = float(parser.get('lookup','perc_unc_max'))
    #========#

    #=============================================================================================#
    #=============================================================================================#
    #=============================================================================================#

    mu_a = np.array([mu_a_start*mu_a_base**i for i in range(N_mu_a)])    # range of mua
    mu_s_ = np.array([mu_s__start*mu_a_base**i for i in range(N_mu_s)])   # range of mus'

    rows = len(mu_a)
    cols = len(mu_s_)

    # distributions for n and sphere properties

    n_dist = np.random.normal(n,n_unc,uncertainty_runs)
    R_pw_dist = np.random.normal(R_pw,R_pw_unc,uncertainty_runs)
    R_fs_dist = np.random.normal(R_fs,R_fs_unc,uncertainty_runs)
    R_fp_dist = np.random.normal(R_fp,R_fp_unc,uncertainty_runs)
    R_fd_dist = np.random.normal(R_fd,R_fd_unc,uncertainty_runs)
    R_angle_dist = np.random.normal(R_angle,R_angle_unc,uncertainty_runs)
    T_pw_dist = np.random.normal(T_pw,T_pw_unc,uncertainty_runs)
    T_fs_dist = np.random.normal(T_fs,T_fs_unc,uncertainty_runs)
    T_fp_dist = np.random.normal(T_fp,T_fp_unc,uncertainty_runs)
    T_fd_dist = np.random.normal(T_fd,T_fd_unc,uncertainty_runs)
    T_angle_dist = np.random.normal(T_angle,T_angle_unc,uncertainty_runs)

# ==== ==== ==== CUDA IMPLEMENTATION ==== ==== ==== #
# parse the args and formulate the command that will run the simulations
def runCUDA(uncertainty_runs):

    R_d = np.empty((uncertainty_runs, rows, cols))
    T_d = np.empty((uncertainty_runs, rows, cols))

    for r in range(uncertainty_runs):

        print("SAMPLE %d/%d OF DISTRIBUTIONS FOR n AND SPHERE PROPERTIES" %(r+1,uncertainty_runs))
        print("n: %f g: %f t:%f R_pw: %f R_fs: %f R_fp: %f R_fd: %f T_pw: %f T_fs: %f T_fp: %f T_fd: %f" %(n_dist[r], g, t, R_pw_dist[r], R_fs_dist[r], R_fp_dist[r], R_fd_dist[r], T_pw_dist[r], T_fs_dist[r], T_fp_dist[r], T_fd_dist[r]))

        if not args.GPU:
            raise RuntimeError("Must supply GPU IDs if --CUDA is specified. Use -h or --help for usage.")
        if not args.source:
            raise RuntimeError("Must supply compiled CUDA source if --CUDA is specified. Use -h or --help for usage.")
        cmd = "%s --GPU \"" %args.source
        for dev in args.GPU:
            cmd = cmd + str(dev) + " "
        try:
            cmd = cmd + "\" -N %d --runs %d " %(N, runs)
        except:
            cmd = cmd + "\" --unc %f " %(uncertainty)
        try:
            cmd = cmd + "--est %f " %(estimate)
        except:
            pass
        cmd = cmd + "-n %f -g %f -t %f --Nmua %d --muaS %f --muaB %f --Nmus %d --musS %f --musB %f --Rangle %f --Tangle %f " %(n_dist[r], g, t, N_mu_a, mu_a_start, mu_a_base, N_mu_s, mu_s__start, mu_s__base, R_angle_dist[r], T_angle_dist[r])
        if R:
            cmd = cmd + "--Rsphere \"%f %f %f %f\" " %(R_pw_dist[r], R_fs_dist[r], R_fp_dist[r], R_fd_dist[r])
        if T:
            cmd = cmd + "--Tsphere \"%f %f %f %f\" " %(T_pw_dist[r], T_fs_dist[r], T_fp_dist[r], T_fd_dist[r])
        if separate:
            cmd = cmd + "--separate "
        if specular_included:
            cmd = cmd + "--specularIncluded "

        cmd = cmd + "--go "

        os.system(cmd)

        R_d_filename   = "Diffuse_Reflectance.csv" 
        T_d_filename  = "Diffuse_Transmittance.csv"

        R_d_data    = np.genfromtxt(R_d_filename,dtype=None,encoding=None,delimiter=",", invalid_raise=False)[2:,2:]
        T_d_data   = np.genfromtxt(T_d_filename,dtype=None,encoding=None,delimiter=",", invalid_raise=False)[2:,2:]

        x,y = np.shape(R_d_data)

        for i in range(x):
            for j in range(y):
                try:
                    R_d_data[i][j] = np.float(R_d_data[i][j].replace('"',''))
                except:
                    pass
        for i in range(x):
            for j in range(y):
                try:
                    T_d_data[i][j] = np.float(T_d_data[i][j].replace('"',''))
                except:
                    pass

        R_d_data    = np.delete(R_d_data,rows,0).astype(float)
        T_d_data    = np.delete(T_d_data,rows,0).astype(float)

        R_d[r]   = R_d_data[0:rows,:]
        #R_d_std  = R_d_data[rows+1:,:]

        T_d[r]   = T_d_data[0:rows,:]
        #T_d_std  = T_d_data[rows+1:,:]

    R_d_std = np.std(R_d, axis = 0)
    T_d_std = np.std(T_d, axis = 0)

    R_d = np.mean(R_d, axis = 0)
    T_d = np.mean(T_d, axis = 0)

    return R_d, R_d_std, T_d, T_d_std

# ==== ==== ==== ==== ==== ==== ==== #

# ==== ==== ==== PYTHON IMPLEMENTATION ==== ==== ==== #

def sample_space(n, R_pw, R_fs, R_fp, R_fd, R_angle, T_pw, T_fs, T_fp, T_fd, T_angle):

    R_f = R_fs+R_fd+R_fp
    T_f = T_fs+T_fd+T_fp

    R_d = np.zeros((rows,cols))
    R_d_std = np.zeros((rows,cols))
    T_d = np.zeros((rows,cols))
    T_d_std = np.zeros((rows,cols))

    mu_s = mu_s_/(1-g) if g!=1 else mu_s_

    for mua_i in range(len(mu_a)):
        for mus_i in range(len(mu_s)):
            mua = mu_a[mua_i]
            mus = mu_s[mus_i]
            R_d_array = np.zeros(runs)
            T_d_array = np.zeros(runs)
            for run in range(runs):
                print("RUN: %d | mu_a: %f mu_s': %f" %(run+1, mua, mu_s_[mus_i]))
                mut=mua+mus
                if separate:
                    _, R_d_array[run], _, _, _= pythonISMC(N,n,g,t,mua,mus,mut, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, False, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)
                    _, _, _, T_d_array[run], _= pythonISMC(N,n,g,t,mua,mus,mut, False, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)
                else:
                    _, R_d_array[run], _, T_d_array[run], _ = pythonISMC(N,n,g,t,mua,mus,mut,R,R_pw,R_fs,R_fp,R_fd,R_f,R_angle,specular_included,T,T_pw,T_fs,T_fp,T_fd,T_f,T_angle) 
            R_d[mua_i][mus_i] = np.mean(R_d_array)
            R_d_std[mua_i][mus_i] = np.std(R_d_array)
            T_d[mua_i][mus_i] = np.mean(T_d_array)
            T_d_std[mua_i][mus_i] = np.std(T_d_array)

    return R_d, R_d_std, T_d, T_d_std

def runPYTHON(O):

    R_d = np.zeros((O,rows,cols))
    if runs>1:
        R_d_std = np.zeros((rows,cols))
    T_d = np.zeros((O,rows,cols))
    if runs>1:
        T_d_std = np.zeros((rows,cols))

    for r in range(O):
        print("SAMPLE %d/%d OF DISTRIBUTIONS FOR n AND SPHERE PROPERTIES" %(r+1,uncertainty_runs))

        R_d_, R_d_std_, T_d_, T_d_std_ = sample_space(n_dist[r], R_pw_dist[r], R_fs_dist[r], R_fp_dist[r], R_fd_dist[r], R_angle_dist[r], T_pw_dist[r], T_fs_dist[r], T_fp_dist[r], T_fd_dist[r], T_angle_dist[r])

        R_d[r] = R_d_
        if runs>1:
            R_d_std += (R_d_std_)**2
        T_d[r] = T_d_
        if runs>1:
            T_d_std += (T_d_std_)**2

    if runs>1:
        R_d_std = np.sqrt(R_d_std)
        T_d_std = np.sqrt(T_d_std)
    else:
        R_d_std = np.std(R_d,axis=0)
        T_d_std = np.std(T_d,axis=0)

    return np.mean(R_d,axis=0), R_d_std, np.mean(T_d,axis=0), T_d_std

# ==== ==== ==== ==== ==== ==== ==== #

# generate the lookup tables #

# blatantly copied directly from stack overflow
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def makelookup(refl, trans, refl_std, trans_std, N, uncertainty = False, M = 100, method="cubic", graph=False, show=False, mua_filename=None, mus_filename=None, mua_uncertainty_filename=None, mus_uncertainty_filename=None):
    mua_lookup = []
    mus_lookup = []
    Rstd_lookup = []
    Tstd_lookup = []
    for row in range(len(refl)):
        if row == 0:
            continue
        for col in range(len(refl[row])):
            if col == 0:
                continue
            # contruct arrays in the form of: (R, T, val)
            mua_lookup.append([float(refl[row][col])*100, float(trans[row][col])*100, float(mu_a[row])])                # R, T, mua
            mus_lookup.append([float(refl[row][col])*100, float(trans[row][col])*100, float(mu_s_[col])])               # R, T, mus'
            Rstd_lookup.append([float(refl[row][col])*100, float(trans[row][col])*100, float(refl_std[row][col])*100])  # R, T, Rstdev
            Tstd_lookup.append([float(refl[row][col])*100, float(trans[row][col])*100, float(trans_std[row][col])*100]) # R, T, Tstdev

    # sort arrays by R first then T
    mua_lookup_sorted = np.array(sorted(mua_lookup, key=operator.itemgetter(0, 1)))
    mus_lookup_sorted = np.array(sorted(mus_lookup, key=operator.itemgetter(0, 1)))
    Rstd_lookup_sorted = np.array(sorted(Rstd_lookup, key=operator.itemgetter(0, 1)))
    Tstd_lookup_sorted = np.array(sorted(Tstd_lookup, key=operator.itemgetter(0, 1)))

    # grid goes from 0 to 100 on both x and y axis with N points
    grid_x, grid_y = np.mgrid[0:100:complex(N), 0:100:complex(N)]

    # contruct arrays for the points in the form of (R, T)
    mua_points = np.array([[mua_lookup_sorted[i,0], mua_lookup_sorted[i,1]] for i in range(len(mua_lookup_sorted[:,0]))])
    mus_points = np.array([[mus_lookup_sorted[i,0], mus_lookup_sorted[i,1]] for i in range(len(mus_lookup_sorted[:,0]))])
    Rstd_points = np.array([[Rstd_lookup_sorted[i,0], Rstd_lookup_sorted[i,1]] for i in range(len(Rstd_lookup_sorted[:,0]))])
    Tstd_points = np.array([[Tstd_lookup_sorted[i,0], Tstd_lookup_sorted[i,1]] for i in range(len(Tstd_lookup_sorted[:,0]))])

    # contruct arrays of just the values
    mua_values = mua_lookup_sorted[:,2]
    mus_values = mus_lookup_sorted[:,2]
    Rstd_values = Rstd_lookup_sorted[:,2]
    Tstd_values = Tstd_lookup_sorted[:,2]

    # interpolate the data
    mua_lookup_sorted_interp = interpolate.griddata(mua_points, mua_values, (grid_x, grid_y), method=method)
    mus_lookup_sorted_interp = interpolate.griddata(mus_points, mus_values, (grid_x, grid_y), method=method)
    Rstd_lookup_sorted_interp = interpolate.griddata(Rstd_points, Rstd_values, (grid_x, grid_y), method=method)
    Tstd_lookup_sorted_interp = interpolate.griddata(Tstd_points, Tstd_values, (grid_x, grid_y), method=method)

    Rvals = np.linspace(0,100,N)
    Tvals = np.linspace(0,100,N)

    # find the uncertainty in the interpolation via sampling gaussian distributions in a monte carlo fashion
    if uncertainty:

        mua_uncertainty = np.empty((N,N))
        mua_uncertainty[:] = np.nan
        mus_uncertainty = np.empty((N,N))
        mus_uncertainty[:] = np.nan

        for i in range(N):
            for j in range(N):
                R = Rvals[i]
                T = Tvals[j]
                R_std = Rstd_lookup_sorted_interp[i][j]
                T_std = Tstd_lookup_sorted_interp[i][j]
                if not (isnan(R_std) or isnan(T_std) or R_std<0 or T_std<0):
                    muas=[]
                    muss=[]
                    #print("R: %4.4f,    Rstd: %4.4f,    T: %4.4f,    Tstd: %4.4f" %(R,R_std,T,T_std))
                    R_dist = np.random.normal(R,R_std,M)
                    T_dist = np.random.normal(T,T_std,M)
                    for k in range(M):
                        R_idx = find_nearest(Rvals,R_dist[k])
                        T_idx = find_nearest(Tvals,T_dist[k])
                        muas.append(mua_lookup_sorted_interp[R_idx][T_idx])
                        muss.append(mus_lookup_sorted_interp[R_idx][T_idx])
                    R_idx = find_nearest(Rvals,R)
                    T_idx = find_nearest(Tvals,T)
                    mua_uncertainty[i,j] = 100*np.std(np.array(muas))/mua_lookup_sorted_interp[R_idx][T_idx]
                    mus_uncertainty[i,j] = 100*np.std(np.array(muss))/mus_lookup_sorted_interp[R_idx][T_idx]

        # graph uncertainties
        if graph:

            cmap = plt.get_cmap("viridis")
            cmap.set_bad('white',1.)

            fig, ax = plt.subplots()

            #im = ax.imshow(np.flip(mua_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=cmap vmin=0)
            im = ax.pcolor(grid_x, grid_y, mua_uncertainty, vmin=perc_unc_min, vmax=perc_unc_max, cmap=cmap)
            ax.set_title("Uncertainty From Simulations\nin $\mu_a$ Lookup Table",y=1.05)
            #ax.plot(mua_points[:,0], mua_points[:,1], 'k.', alpha=0.5, ms=1)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Percent Uncertainty in $\mu_a$ (%)")
            ax.set_xlabel("Diffuse Reflectance (%)")
            ax.set_xlim([R_d_min,R_d_max])
            ax.set_ylabel("Diffuse Transmittance (%)")
            ax.set_ylim([T_d_min,T_d_max])
            plt.tight_layout()
            if show:
                plt.show()
            if mua_uncertainty_filename and not show:
                plt.savefig(mua_uncertainty_filename,dpi=500)


            fig, ax = plt.subplots()

            #im = ax.imshow(np.flip(mus_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=cmap vmin=0)
            im = ax.pcolor(grid_x, grid_y, mus_uncertainty, vmin=perc_unc_min, vmax=perc_unc_max, cmap=cmap)
            ax.set_title("Uncertainty From Simulations\nin $\mu_s'$ Lookup Table",y=1.05)
            #ax.plot(mus_points[:,0], mus_points[:,1], 'k.', alpha=0.5, ms=1)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Percent Uncertainty in $\mu_s'$ (%)")
            ax.set_xlabel("Diffuse Reflectance (%)")
            ax.set_xlim([R_d_min,R_d_max])
            ax.set_ylabel("Diffuse Transmittance (%)")
            ax.set_ylim([T_d_min,T_d_max])
            plt.tight_layout()
            if show:
                plt.show()
            if mus_uncertainty_filename and not show:
                plt.savefig(mus_uncertainty_filename,dpi=500)


    # graph lookup tables
    if graph:

        cmap = plt.get_cmap("viridis")
        cmap.set_bad('white',1.)

        fig, ax = plt.subplots()

        #im = ax.imshow(np.flip(mua_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=cmap vmin=0)
        im = ax.pcolor(grid_x, grid_y, mua_lookup_sorted_interp, norm=colors.LogNorm(vmin=mu_a.min(),vmax=mu_a.max()), cmap=cmap)
        ax.set_title("Absorption Coefficient Lookup Table",y=1.05)
        ax.plot(mua_points[:,0], mua_points[:,1], 'k.', alpha=0.5, ms=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Absorption Coefficient, $\mu_a$ (1/cm)")
        ax.set_xlabel("Diffuse Reflectance (%)")
        ax.set_xlim([R_d_min,R_d_max])
        ax.set_ylabel("Diffuse Transmittance (%)")
        ax.set_ylim([T_d_min,T_d_max])
        plt.tight_layout()
        if show:
            plt.show()
        if mua_filename and not show:
            plt.savefig(mua_filename,dpi=500)


        fig, ax = plt.subplots()

        #im = ax.imshow(np.flip(mus_lookup_sorted_interp.T,0), extent=(0,100,0,100),cmap=cmap vmin=0)
        im = ax.pcolor(grid_x, grid_y, mus_lookup_sorted_interp, norm=colors.LogNorm(vmin=mu_s_.min(),vmax=mu_s_.max()), cmap=cmap)
        ax.set_title("Reduced Scattering Coefficient Lookup Table",y=1.05)
        ax.plot(mus_points[:,0], mus_points[:,1], 'k.', alpha=0.5, ms=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Reduced Scattering Coefficient, $\mu_s'$ (1/cm)")
        ax.set_xlabel("Diffuse Reflectance (%)")
        ax.set_xlim([R_d_min,R_d_max])
        ax.set_ylabel("Diffuse Transmittance (%)")
        ax.set_ylim([T_d_min,T_d_max])
        plt.tight_layout()
        if show:
            plt.show()
        if mus_filename and not show:
            plt.savefig(mus_filename,dpi=500)

    return mua_lookup_sorted_interp.T, mua_uncertainty.T, mus_lookup_sorted_interp.T, mus_uncertainty.T, Rvals, Tvals, mua_points, mus_points

def lookup(table, Rvals, Tvals, R, T, verbose=False):
    diff = np.Inf
    for R_i in range(len(Rvals)):
        if np.abs(Rvals[R_i]-R) < diff:
            diff = np.abs(Rvals[R_i]-R)
            R_exe = Rvals[R_i]

    diff = np.Inf
    for T_i in range(len(Tvals)):
        if np.abs(Tvals[T_i]-T) < diff:
            diff = np.abs(Tvals[T_i]-T)
            T_exe = Tvals[T_i]

    if verbose:
        print("R: %f --> %f" %(R,R_exe))
        print("T: %f --> %f" %(T,T_exe))

    R_i = Rvals.tolist().index(R_exe)
    T_i = Tvals.tolist().index(T_exe)

    val = table[T_i][R_i]

    return val, R_exe, T_exe



#========================================================#

if __name__=='__main__':
    # run the simulations with the specified language
    if args.python:
        R_d, R_d_std, T_d, T_d_std = runPYTHON(uncertainty_runs)
    if args.cuda:
        R_d, R_d_std, T_d, T_d_std = runCUDA(uncertainty_runs)

    P=int(parser.get('lookup','dimensions'))   # number of points to be interpolated to. the final graph will be NxN
    M=int(parser.get('lookup','simulation_distributions_samples'))    # number of times to sample the uncertainty distributions to get the uncertainty in mua and mus' 

    # the makelookup function takes the data for R and T, the standard deviations in those data sets, N, a bool to determine whether or not it should calculate the uncertainties, M, the interpolation method, a bool to tell it if it should graph the table or just generate it, and filenames for all of the graphs
    # it outputs the lookup tables for mua and mus', and the R and T values associated with that lookup table
    prefix = parser.get('lookup','filename_prefix')

    base_muaL, base_mua_uncL, base_musL, base_mus_uncL, base_Rvals, base_Tvals, mua_points, mus_points = makelookup(R_d, T_d, R_d_std, T_d_std, P, uncertainty=True, M=M, method='cubic', graph=True, mua_filename= prefix+"_mua_lookup.png", mus_filename=prefix+"_mus_lookup.png", mua_uncertainty_filename=prefix+"_mua_lookup_uncertainty.png", mus_uncertainty_filename=prefix+"_mus_lookup_uncertainty.png")

    hdf5storage.write({'mua_lookup':base_muaL, 'mus_lookup':base_musL, "mua_lookup_uncertainty":base_mua_uncL, "mus_lookup_uncertainty":base_mus_uncL, "Rvals":base_Rvals, "Tvals":base_Tvals, "n":n, "n_unc":n_unc, "g":g, "t":t, "R":R, "R_pw":R_pw, "R_pw_unc":R_pw_unc, "R_fs":R_fs, "R_fs_unc":R_fs_unc, "R_fp":R_fp, "R_fp_unc":R_fp_unc, "R_fd":R_fd, "R_fd_unc":R_fd_unc, "R_angle":R_angle, "R_angle_unc":R_angle_unc, "separate":separate, "specular_included":specular_included, "T":T, "T_pw":T_pw, "T_pw_unc":T_pw_unc, "T_fs":T_fs, "T_fs_unc":T_fs_unc, "T_fp":T_fp, "T_fp_unc":T_fp_unc, "T_fd":T_fd, "T_fd_unc":T_fd_unc, "T_angle":T_angle, "T_angle_unc":T_angle_unc, "mua_points":mua_points, "mus_points":mus_points, "muas":mu_a, "muss":mu_s_}, '.', '%s_lookup_tables.h5'%prefix ,matlab_compatible=True)

    #==== An example of a lookup for a measured R and T ====#

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

            mua_array[i], R_exe, T_exe = lookup(base_muaL,base_Rvals,base_Tvals,R,T,verbose=False)
            #mua_perc_unc_array[i], R_exe, T_exe = lookup(base_mua_uncL,base_Rvals,base_Tvals,R,T,verbose=False)

            mus_array[i], R_exe, T_exe = lookup(base_musL,base_Rvals,base_Tvals,R,T,verbose=False)
            #mus_perc_unc_array[i], R_exe, T_exe = lookup(base_mus_uncL,base_Rvals,base_Tvals,R,T,verbose=False)

        #mua_unc_array = mua_array*mua_perc_unc_array/100
        #mus_unc_array = mus_array*mus_perc_unc_array/100

        # to find the uncertainty in mua and mus, combine the two uncertainties: the uncertainty of the table
        # and the uncertainty caused by the measurement uncertainties

        # uncertainty from table
        mua_perc_unc_1, R_exe, T_exe = lookup(base_mua_uncL,base_Rvals,base_Tvals,R,T,verbose=False)
        mus_perc_unc_1, R_exe, T_exe = lookup(base_mus_uncL,base_Rvals,base_Tvals,R,T,verbose=False)

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
