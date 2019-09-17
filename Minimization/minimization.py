# minimization.py
#====FOREWORD====#
"""
This script serves as an example of how to work backwards from experimental results for the
total transmittance and total reflectance of a sample with a known refractive index, anisotropy,
and thickness to find the sample's absorption coefficient and reduced scattering coefficient.

This program was written for Python 3.6.3.

This software is being provided "as is", without any express or implied warranty. In
particular, the authors do not make any representation or warranty of any kind concerning the
merchantability of this software or its fitness for any particular purpose.
"""
#========#

from scipy.optimize import minimize
import numpy as np
import argparse
import os
import configparser
import sys
sys.path.append("../")
from ISMC import ISMC as pythonISMC

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('--Python', dest='python', help="Run the simuations with Python.", action='store_true')
parser.add_argument('--CUDA', dest='cuda', help="Run the simuations with CUDA. Must also provide a CUDA source and GPU IDs with --CUDAsource and --GPU respectively.", action='store_true')
parser.add_argument('--CUDAsource', dest="source", help="Path to a suitable compiled CUDA source to run the simulation from if CUDA is the desired environment.")
parser.add_argument('--GPU', dest="GPU", help="Device number of the GPU to run the simulation on. Must be specified if --CUDA is used. Use 'nvidia-smi' to list available devices. Example: --GPU 0")
parser.add_argument("configfile", help="Path to a suitable configuration .ini file")
args = parser.parse_args()

if not (args.python or args.cuda):
    raise RuntimeError("Please specify a simulation environment with --CUDA or --Python")

# parse input file
parser = configparser.SafeConfigParser()
parser.read(args.configfile)

N           = int(parser.get('minimization','N'))     # number of photons to use in each simulation
uncertainty_runs = int(parser.get('minimization','input_distributions_samples'))  # number of times to sample distributions for refractive index, sphere parameters, and experimental measurements
#====Measured Values====#
if (parser.has_option('experiment','M_R')):
	R_d_exact = np.array(parser.get('experiment', 'M_R').split("\n"),dtype=float)[0]  # experimental value for diffuse reflectance - 'None' if not measured
else:
	R_d_exact = None

if (parser.has_option('experiment','M_R_unc')):
	R_d_exact_unc  = np.array(parser.get('experiment', 'M_R_unc').split("\n"),dtype=float)[0] # uncertainty in experimental value for diffuse reflectance - must be specified if R_d_exact is
else:
	R_d_exact_unc = None

if (parser.has_option('experiment','M_S')):
	R_s_exact = np.array(parser.get('experiment', 'M_S').split("\n"),dtype=float)[0] # experimental value for specular reflectance - 'None' if not measured
else:
	R_s_exact = None

if (parser.has_option('experiment','M_S_unc')):
	R_s_exact_unc = np.array(parser.get('experiment', 'M_S_unc').split("\n"),dtype=float)[0] # uncertainty in experimental value for specular reflectance - must be specified if R_s_exact is
else:
	R_s_exact_unc = None

if (parser.has_option('experiment','M_T')):
	T_d_exact = np.array(parser.get('experiment', 'M_T').split("\n"),dtype=float)[0] # experimental value for diffuse transmittance - 'None' if not measured
else:
	T_d_exact = None

if (parser.has_option('experiment','M_T_unc')):
	T_d_exact_unc = np.array(parser.get('experiment', 'M_T_unc').split("\n"),dtype=float)[0] # uncertainty in experimental value for diffuse transmittance - must be specified if T_d_exact is
else:
	T_d_exact_unc = None

if (parser.has_option('experiment','M_U')):
	T_u_exact = np.array(parser.get('experiment', 'M_U').split("\n"),dtype=float)[0] # experimental value for direct transmittance - 'None' if not measured
else:
	T_u_exact = None

if (parser.has_option('experiment','M_U_unc')):
	T_u_exact_unc = np.array(parser.get('experiment', 'M_U_unc').split("\n"),dtype=float)[0] # uncertainty in experimental value for diffuse reflectance - must be specified if T_u_exact is
else:
	T_u_exact = None

#====Sample Properties====#
n 			= float(parser.get('sample','n'))     # refractive index of the sample
n_unc       = float(parser.get('sample','n_unc')) # uncertainty in the refractive index of the sample
g           = float(parser.get('sample','g'))     # anisotropy factor of the sample
t           = float(parser.get('sample','t'))     # thickness of the sample in cm
#====Solver Parameters====#
mu_a_guess  = float(parser.get('minimization','mua_guess')) # initial guess for the absorption coefficient, mu_a, in 1/cm
mu_s__guess = float(parser.get('minimization','mus_guess')) # initial guess for the reduced scattering coefficient, mu_s', in 1/cm
mu_a_min    = float(parser.get('minimization','mua_min'))   # minimimum allowed mu_a value. values less than this will be discarded
mu_a_max    = float(parser.get('minimization','mua_max'))   # maximum allowed mu_a value. values greater than this will be discarded
mu_s__min   = float(parser.get('minimization','mus_min'))   # minimimum allowed mu_s' value. values less than this will be discarded
mu_s__max   = float(parser.get('minimization','mus_max'))   # maximum allowed mu_s' value. values greater than this will be discarded
step        = float(parser.get('minimization','step'))      # maximum step size for the solver
runs        = int(parser.get('minimization','runs'))      # number of minimizations to do
tolerance   = float(parser.get('minimization','tolerance')) # tolerance, bigger the better (1, 0.1, and 0.01 are good)
#====Sphere Parameters====#
R       = int(parser.get('spheres','R_sphere'))                     # boolean to determine if the reflection sphere is present or not
R_pw    = float(parser.get('spheres','R_pw'))                       # reflectance of reflection sphere
R_pw_unc= float(parser.get('spheres','R_pw_unc'))                   # uncertainty in reflectance of reflection sphere
R_fs    = float(parser.get('spheres','R_fs'))                       # sample port fraction of reflection sphere
R_fs_unc= float(parser.get('spheres','R_fs_unc'))                   # 5% uncertainty in sample port fraction of reflection sphere
R_fp    = float(parser.get('spheres','R_fp'))                       # source port fraction of reflection sphere
R_fp_unc= float(parser.get('spheres','R_fp_unc'))                   # 5% uncertainty in source port fraction of reflection sphere
R_fd    = float(parser.get('spheres','R_fd'))                       # detector fraction of reflection sphere
R_fd_unc= float(parser.get('spheres','R_fd_unc'))                   # 5% uncertainty in detector fraction of reflection sphere
R_f     = R_fs + R_fp + R_fd                                        # total port fraction of reflection sphere
R_f_unc = np.sqrt(R_fs**2 + R_fp**2 + R_fd**2)                      # uncertainty in total port fraction of reflection sphere
R_angle = float(parser.get('spheres','R_angle'))  #degrees          # angle threshold for specular reflection if no Rsphere is present
R_angle_unc = float(parser.get('spheres','R_angle_unc')) # degrees  # uncertainty in angle threshold for specular reflection
specular_included =  int(parser.get('spheres','specular_included')) # boolean to determine if specular reflection is included or not

T       = int(parser.get('spheres','T_sphere'))                     # boolean to determine if the transmission sphere is present or not
T_pw    = float(parser.get('spheres','T_pw'))                       # reflectance of transmission sphere
T_pw_unc= float(parser.get('spheres','T_pw_unc'))                   # uncertainty in reflectance of transmission sphere
T_fs    = float(parser.get('spheres','T_fs'))                       # sample port fraction of transmission sphere
T_fs_unc= float(parser.get('spheres','T_fs_unc'))                   # 5% uncertainty in sample port fraction of transmission sphere
T_fp    = float(parser.get('spheres','T_fp'))                       # optional port fraction of transmission sphere
T_fp_unc= float(parser.get('spheres','T_fp_unc'))                   # 5% uncertainty in optional port fraction of transmission sphere
T_fd    = float(parser.get('spheres','T_fd'))                       # detector fraction of transmission sphere
T_fd_unc= float(parser.get('spheres','T_fd_unc'))                   # 5% uncertainty in detector fraction of transmission sphere
T_f     = T_fs + T_fp + T_fd                                        # total port fraction of transmission sphere
T_f_unc = np.sqrt(T_fs**2 + T_fp**2 + T_fd**2)                      # uncertainty in total port fraction of transmission sphere
T_angle = float(parser.get('spheres','T_angle'))  #degrees          # angle threshold for unscattered transmission if no Tsphere is present
T_angle_unc = float(parser.get('spheres','T_angle_unc')) # degrees  # uncertainty in angle threshold for unscattered transmittance

separate = int(parser.get('spheres','separate'))    # True if R_d and T_d and (R_s and T_u) were measured separately (single-sphere experiment), False otherwise (Dual-sphere)
#========#

n_dist = np.random.normal(n,n_unc,uncertainty_runs)
R_pw_dist = np.random.normal(R_pw,R_pw_unc,uncertainty_runs)
R_fs_dist = np.random.normal(R_fs,R_fs_unc,uncertainty_runs)
R_fp_dist = np.random.normal(R_fp,R_fp_unc,uncertainty_runs)
R_fd_dist = np.random.normal(R_fd,R_fd_unc,uncertainty_runs)
T_pw_dist = np.random.normal(T_pw,T_pw_unc,uncertainty_runs)
T_fs_dist = np.random.normal(T_fs,T_fs_unc,uncertainty_runs)
T_fp_dist = np.random.normal(T_fp,T_fp_unc,uncertainty_runs)
T_fd_dist = np.random.normal(T_fd,T_fd_unc,uncertainty_runs)

R_angle_dist = np.random.normal(R_angle,R_angle_unc,uncertainty_runs)
T_angle_dist = np.random.normal(T_angle,T_angle_unc,uncertainty_runs)

if R_d_exact:
    R_d_exact_dist = np.random.normal(R_d_exact, R_d_exact_unc, uncertainty_runs)
if R_s_exact:
    R_s_exact_dist = np.random.normal(R_s_exact, R_s_exact_unc, uncertainty_runs)
if T_d_exact:
    T_d_exact_dist = np.random.normal(T_d_exact, T_d_exact_unc, uncertainty_runs)
if T_u_exact:
    T_u_exact_dist = np.random.normal(T_u_exact, T_u_exact_unc, uncertainty_runs)

mu_a_results = np.zeros(uncertainty_runs)
mu_a_std_results = np.zeros(uncertainty_runs)
mu_s__results = np.zeros(uncertainty_runs)
mu_s__std_results = np.zeros(uncertainty_runs)

for r in range(uncertainty_runs):

    print("Uncertainty Run: %d of %d" %(r+1, uncertainty_runs))

    if args.cuda:
        if not args.GPU:
            raise RuntimeError("Must supply GPU IDs if --CUDA is specified. Use -h or --help for usage.")
        if not args.source:
            raise RuntimeError("Must supply compiled CUDA source if --CUDA is specified. Use -h or --help for usage.")

        # formulate the command that will run the CUDA ISMC simulation
        cmd = "%s --GPU %s" %(args.source, args.GPU)
        cmd = cmd + " -N %d --runs %d --step %f --tol %f -n %f -g %f -t %f --Rangle %f --Tangle %f --Rsphere \"%f %f %f %f\" --Tsphere \"%f %f %f %f\" --mua %f --mus %f --muaMin %f --muaMax %f --musMin %f --musMax %f " %(N, runs, step, tolerance, n_dist[r], g, t, R_angle_dist[r], T_angle_dist[r], R_pw_dist[r], R_fs_dist[r], R_fp_dist[r], R_fd_dist[r], T_pw_dist[r], T_fs_dist[r], T_fp_dist[r], T_fd_dist[r], mu_a_guess, mu_s__guess, mu_a_min, mu_a_max, mu_s__min, mu_s__max)
        if R_d_exact is not None:
            cmd = cmd + "--Rd %f " %R_d_exact_dist[r]
        if R_s_exact is not None:
            cmd = cmd + "--Rs %f " %R_s_exact_dist[r]
        if T_d_exact is not None:
            cmd = cmd + "--Td %f " %T_d_exact_dist[r]
        if T_u_exact is not None:
            cmd = cmd + "--Tu %f " %T_u_exact_dist[r]
        if separate:
            cmd = cmd + "--separate "
        if specular_included:
            cmd = cmd + "--specularIncluded "
        cmd = cmd + "--go "

        # function to filter out garbage from the results
        f = lambda x: x!='' and x!='\n'

        res = os.popen(cmd).read()
        res = list(filter(f,res.replace("\n"," ").split(" ")))

        # pull the results from the command line
        mu_a_results[r] = res[0]
        mu_s__results[r] = res[2]
        mu_a_std_results[r] = res[1]
        mu_s__std_results[r] = res[3]


    if args.python:
        verbose = True
        def cost(coeffs, R_d_exact, R_s_exact, T_d_exact, T_u_exact, n, g, t, N, R, R_pw, R_fs, R_fp, \
            R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle, separate, verbose):
        
            mu_a, mu_s_ = coeffs[0], coeffs[1]
            if g == 1: mu_s = mu_s_         # scattering coefficient of sample in 1/cm
            else: mu_s = mu_s_/(1-g)        #
            R_d, R_s, T_d, T_u = float('Nan'), float('Nan'),float('Nan'),float('Nan')
            # run the integrating sphere Monte Carlo simulation
            if separate:
                if R_d_exact is not None:
                    _, R_d, _, _, _ = pythonISMC(N, n, g, t, mu_a, mu_s, mu_a+mu_s, R, R_pw, R_fs, R_fp, \
                        R_fd, R_f, R_angle, specular_included, False, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)
        
                # diffuse reflectance is measured separately from diffuse reflectance
                if T_d_exact is not None:
                    _, _, _, T_d, _ = pythonISMC(N, n, g, t, mu_a, mu_s, mu_a+mu_s, False, R_pw, R_fs, R_fp, \
                        R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)
        
                if R_s_exact is not None or T_u_exact is not None:
                    # T_u and R_s are measured without integrating spheres, so this config must be separate
                    _, _, R_s, _, T_u = pythonISMC(N, n, g, t, mu_a, mu_s, mu_a+mu_s, False, 0, 0, 0, \
                        0, 0, R_angle, specular_included, 0, 0, 0, 0, 0, 0, T_angle)
            else:
                _, R_d, R_s, T_d, T_u = pythonISMC(N, n, g, t, mu_a, mu_s, mu_a+mu_s, R, R_pw, R_fs, R_fp, \
                    R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle) 

            # output the current material properties and total reflectance and total transmittance
            if verbose: print("mu_a: %f, mu_s': %f     \
                R_d: %f, R_s: %f, T_d: %f, T_u: %f" %(mu_a, mu_s_, R_d, R_s, T_d, T_u))
        
            # determine what the cost should be based on what values are known and return it
            R_d_cost, R_s_cost, T_d_cost, T_u_cost = 0., 0., 0., 0.
            if R_d_exact is not None: R_d_cost = ((R_d - R_d_exact)/R_d_exact)**2
            if R_s_exact is not None: R_s_cost = (R_s - R_s_exact)**2
            if T_d_exact is not None: T_d_cost = ((T_d - T_d_exact)/T_d_exact)**2
            if T_u_exact is not None: T_u_cost = (T_u - T_u_exact)**2
        
            return np.sqrt(R_d_cost + R_s_cost + T_d_cost + T_u_cost)

        mu_a_array, mu_s__array = np.empty(runs), np.empty(runs)

        if R_d_exact is not None:
            R_d_exact_ = R_d_exact_dist[r]
        else:
            R_d_exact_ = None 
        if R_s_exact is not None:
            R_s_exact_ = R_s_exact_dist[r]
        else:
            R_s_exact_ = None
        if T_d_exact is not None:
            T_d_exact_ = T_d_exact_dist[r]
        else:
            T_d_exact_ = None
        if T_u_exact is not None:
            T_u_exact_ = T_u_exact_dist[r]
        else:
            T_u_exact_ = None

        i = 0 
        while i < runs:
            if verbose: print("RUN %d" %(i+1))
        
            # minimize the cost function, therefore solving for mu_a and mu_s' of the sample
            res = minimize(cost, [mu_a_guess, mu_s__guess], args=(R_d_exact_, R_s_exact_, T_d_exact_, \
                T_u_exact_, n_dist[r], g, t, N, R, R_pw_dist[r], R_fs_dist[r], R_fp_dist[r], R_fd_dist[r], R_fs_dist[r]+R_fp_dist[r]+R_fd_dist[r], R_angle_dist[r], specular_included, \
                T, T_pw_dist[r], T_fs_dist[r], T_fp_dist[r], T_fd_dist[r], T_fs_dist[r]+T_fp_dist[r]+T_fd_dist[r], T_angle_dist[r], separate, verbose), method='Powell', tol=tolerance)
            mu_a, mu_s_ = res.x[0], res.x[1]
        
            if mu_a_min <= mu_a <= mu_a_max and mu_s__min <= mu_s_ <= mu_s__max:
                mu_a_array[i], mu_s__array[i] = mu_a, mu_s_
                i += 1

        mu_a_results[r] = np.mean(mu_a_array)
        mu_s__results[r] = np.mean(mu_s__array)

mu_a_avg = np.mean(mu_a_results)
mu_a_unc = np.std(mu_a_results)
mu_s__avg = np.mean(mu_s__results)
mu_s__unc = np.std(mu_s__results)

# return the results
print("\nSolver Finished. Results:\nmu_a: %s +/- %s 1/cm\nmu_s': %s +/- %s 1/cm" \
    %(mu_a_avg, mu_a_unc, mu_s__avg, mu_s__unc))
