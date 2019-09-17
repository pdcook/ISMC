# ISMLMC.py

#====FOREWORD====#
"""
This software is a minimum working example of the Integrating Sphere
Monte Carlo method as described in the accompanying paper by Cook et al.

This program was written for Python 3.6.3.

This software is being provided "as is", without any express or implied warranty. In
particular, the authors do not make any representation or warranty of any kind concerning the
merchantability of this software or its fitness for any particular purpose.
"""
#========#

from math import *
from random import uniform
import MLMC

N       = 100000        # number of photons in simulation

#====Sample Definition====#
n       = [1.37,1.37,1.37]             # refractive index of sample
g       = [0.9,0,0.7]          # anisotropy factor of sample
mu_a    = [1,1,2]            # absorption coefficient of sample in 1/cm
mu_s_   = [10,10,3]          # reduced scattering coefficient of sample in 1/cm
t       = [0.1,0.1,0.2]          # thickness of sample in cm

mu_s = [0 for _ in range(len(mu_s_))]
mu_t = [0 for _ in range(len(mu_s_))]
for i in range(len(mu_s_)):
    if g[i] == 1: mu_s[i] = mu_s_[i]     # scattering coefficient of sample in 1/cm
    else: mu_s[i] = mu_s_[i]/(1-g[i])    #
    mu_t[i]    = mu_a[i] + mu_s[i]   # total interaction coefficient of sample in 1/cm
#========#

#====Sphere Parameters====#
R       = False         # boolean to determine if the reflection sphere is present or not
R_D     = 8.382          # diameter of sphere in cm
R_pw    = 0.85          # reflectance of reflection sphere
R_fs    = 0.1           # sample port fraction of reflection sphere
R_fp    = 0.05          # source port fraction of reflection sphere
R_fd    = 0.05          # detector fraction of reflection sphere
R_f     = R_fs + R_fp + R_fd    # total port fraction of reflection sphere
R_angle = pi/10         # angle threshold for specular reflection if no Rsphere is present
specular_included = False   # boolean to determine if specular reflection is included or not

T       = False         # boolean to determine if the transmission sphere is present or not
T_D     = 8.382          # diameter of sphere in cm
T_pw    = 0.85          # reflectance of transmission sphere
T_fs    = 0.1           # sample port fraction of transmission sphere
T_fp    = 0.05          # optional port fraction of transmission sphere
T_fd    = 0.05          # detector fraction of transmission sphere
T_f     = T_fs + T_fp + T_fd    # total port fraction of transmission sphere
T_angle = pi/10         # angle threshold for direct transmission if no Tsphere is present
#========#

#====Physics Functions====#

# Snell's law
def multilayer_snell(ni, nt, mu_x, mu_y, mu_z):
    ti = acos(abs(mu_z))
    tt = asin((ni*sin(ti))/nt)
    return mu_x*ni/nt, mu_y*ni/nt, (mu_z/abs(mu_z))*cos(tt)

# function used for determining reflection/transmission
def fresnel_snell(ni,nt,mu_z):
    if abs(mu_z) > 0.99999: R = ((nt-ni)/(nt+ni))**2
    else:
        ti = acos(abs(mu_z))
        # if ni*sin(ti)/nt >=1 then total internal reflection occurs, and thus R = 1
        if (ni*sin(ti))/nt >=1.: R= 1.
        else:
            tt = asin((ni*sin(ti))/nt)
            R = 0.5 * ( (sin(ti-tt)**2)/(sin(ti+tt)**2) + (tan(ti-tt)**2)/(tan(ti+tt)**2) )
    return R

# determine if a photon incident on the sample begins to propagate or is reflected
def incident_reflection(mu_z, n, mu_s, layer):
    # check if the incident layer is glass
    if mu_s[layer] == 0.0:
        n1 = 1.
        n2 = n[layer]
        n3 = n[layer+1] if mu_z > 0 else n[layer-1]

        r1 = fresnel_snell(n1, n2, mu_z)
        r2 = fresnel_snell(n2, n3, mu_z)

        R = r1 + ((1-r1)**2)*r2/(1-r1*r2)

        return uniform(0,1) < R

    else: return uniform(0,1) < fresnel_snell(1., n[layer], mu_z)
# if a photon is inside a sphere, this function determines if it is re-incident on the sample
# returns True if re-incident and False if not
def reincidence(pw, fs, f):
        return uniform(0,1) < (pw*fs)/(1-pw*(1-f))

#========#

def ISMLMC(N, n, g, t, mu_a, mu_s, mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, \
    specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle):

    A           = 0 # total number of absorbed photons
    R_diffuse   = 0 # total number of diffusely reflected/backscattered photons
    R_specular  = 0 # total number of specularly reflected photons
    T_diffuse   = 0 # total number of diffusely transmitted photons
    T_direct    = 0 # total number of directly transmitted photons

    t_tot = sum(t);

    for i in range(N):
        w                   = 1         # initial weight of photon
        x, y, z             = 0, 0, 0   # initial position of photon

        # determine inital direction for photon
        if specular_included: mu_x, mu_y, mu_z = sqrt(1-0.99**2), 0, 0.99
        else: mu_x, mu_y, mu_z = 0, 0, 1

        # keep track of what sphere the photon is in
        # r for reflection sphere, t for transmission sphere
        sample_side = 'r'

        while w > 0:
            # determine if the photon is incidently reflected from the sample
            layer = 0 if mu_z > 0 else len(n) - 1
            incident_reflect = incident_reflection(mu_z, n, mu_s, layer)

            # check if the photon was incidently reflected and where
            if incident_reflect and sample_side == 'r':
                # the photon is incidently reflected off of the 'top' surface

                # check if a reflection sphere is present
                if R:   # reflection sphere present
                    # check to see if the photon leaves directly through the source port
                    if abs(mu_z) >= sqrt(1-R_fp):
                        # score the photon as specular reflection and stop propagating
                        R_specular += w
                        w = 0
                        break

                    # if the photon doesn't leave directly through the source port,
                    # check to see if it is re-incident on the sample
                    elif reincidence(R_pw,R_fs,R_f):
                        # photon is re-incident on the sample; sample a random angle of
                        # reincidence, reset its position, and continue propagating
                        mu_z    = uniform(0,1)
                        phi     = 2*pi*uniform(0,1)
                        mu_x    = cos(phi)*sqrt(1-mu_z**2)
                        mu_y    = sin(phi)*sqrt(1-mu_z**2)
                        x, y, z = 0, 0, 0
                        continue

                    else:
                        # the photon neither leaves through the source port, nor is re-incident
                        # score the photon as diffuse reflection and stop propagating
                        R_diffuse += w
                        w = 0
                        break

                else:   # there is no reflection sphere
                    # calculate an effective source port fraction from the angle threshold
                    fp = 0.5*(1-cos(2*R_angle))

                    # check if the photon leaves in a direction within
                    # this effective source port fraction
                    if abs(mu_z) >= sqrt(1-fp): # photon is within angle threshold, score as
                        R_specular += w         # specular reflection and stop propagating
                        w = 0
                        break

                    else:   # photon is NOT within angle threshold, since there is no sphere
                            # present, score it as diffuse reflection and stop propagating
                        R_diffuse += w
                        w = 0
                        break

            elif incident_reflect and sample_side == 't':
                # the photon is incidently reflected off of the 'bottom' surface

                # check if a transmission sphere is present
                if T: # transmission sphere present
                    # check to see if the photon leaves directly through the optional port
                    if abs(mu_z) >= sqrt(1-T_fp):
                        # score the photon as direct transmission and stop propagating
                        T_direct += w
                        w = 0
                        break

                    # if the photon doesn't leave directly through the optional port,
                    # check to see if it is re-incident on the sample
                    elif reincidence(T_pw,T_fs,T_f):
                        # photon is re-incident on the sample; sample a random angle of
                        # reincidence, reset its position, and continue propagating
                        mu_z    = -1*uniform(0,1)
                        phi     = 2*pi*uniform(0,1)
                        mu_x    = cos(phi)*sqrt(1-mu_z**2)
                        mu_y    = sin(phi)*sqrt(1-mu_z**2)
                        x, y, z = 0, 0, t_tot
                        continue

                    else:
                        # the photon neither leaves through the optional port, nor is re-incident
                        # score the photon as diffuse transmission and stop propagating
                        T_diffuse += w
                        w = 0
                        break

                else:   # there is no transmission sphere
                    # calculate an effective optional port fraction from the angle threshold
                    fp = 0.5*(1-cos(2*T_angle))

                    # check if the photon leaves in a direction within
                    # this effective optional port fraction
                    if abs(mu_z) >= sqrt(1-fp): # photon is within angle threshold, score as
                        T_direct += w           # direct transmission and stop propagating
                        w = 0
                        break

                    else:   # photon is NOT within angle threshold, since there is no sphere
                            # present, score it as diffuse transmission and stop propagating
                        T_diffuse += w
                        w = 0
                        break

            else:   # the photon is not incidently reflected and may begin propagating

                # incident refraction by Snell's Law
                mu_x, mu_y, mu_z = multilayer_snell(1., n[layer], mu_x, mu_y, mu_z)

                # Monte Carlo Photon Transport
                Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z = \
                    MLMC.MC(n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z)

                # partial absorption
                A += Absorbed

                # check to see what happened to the photon
                if Reflected:   # the photon was reflected/backscattered
                    w = Reflected
                    sample_side = 'r'

                    # check if a reflection sphere is present
                    if R:   # reflection sphere present
                        # check to see if the photon leaves directly through the source port
                        if abs(mu_z) >= sqrt(1-R_fp):
                            # score the photon as specular reflection and stop propagating
                            R_specular += w
                            w = 0
                            break

                        # if the photon doesn't leave directly through the
                        # source port, check to see if it is re-incident on the sample
                        elif reincidence(R_pw,R_fs,R_f):
                            # photon is re-incident on the sample; sample a random angle of
                            # reincidence, reset its position, and continue propagating
                            mu_z    = uniform(0,1)
                            phi     = 2*pi*uniform(0,1)
                            mu_x    = cos(phi)*sqrt(1-mu_z**2)
                            mu_y    = sin(phi)*sqrt(1-mu_z**2)
                            x, y, z = 0, 0, 0
                            continue

                        else:   # the photon neither leaves through the source port, nor is
                                # re-incident, score the photon as diffuse reflection
                                # and stop propagating
                            R_diffuse += w
                            w = 0
                            break

                    else:   # there is no reflection sphere
                        # calculate an effective source port fraction from the angle threshold
                        fp = 0.5*(1-cos(2*R_angle))

                        # check if the photon leaves in a direction within
                        # this effective source port fraction
                        if abs(mu_z) >= sqrt(1-fp):
                            # photon is within angle threshold, score as
                            # specular reflection and stop propagating
                            R_specular += w
                            w = 0
                            break

                        else:   # photon is NOT within angle threshold, since there is no sphere
                                # present, score it as diffuse reflection and stop propagating
                            R_diffuse += w
                            w = 0
                            break

                elif Transmitted:   # the photon transmitted through the sample
                    w = Transmitted
                    sample_side = 't'

                    # check if a transmission sphere is present
                    if T:   # transmission sphere present
                        # check to see if the photon leaves directly through the optional port
                        if abs(mu_z) >= sqrt(1-T_fp):
                            # score the photon as direct transmission and stop propagating
                            T_direct += w
                            w = 0
                            break

                        # if the photon doesn't leave directly through the
                        # optional port, check to see if it is re-incident on the sample
                        elif reincidence(T_pw,T_fs,T_f):
                            # photon is re-incident on the sample; sample a random angle of
                            # reincidence, reset its position, and continue propagating
                            mu_z    = -1*uniform(0,1)
                            phi     = 2*pi*uniform(0,1)
                            mu_x    = cos(phi)*sqrt(1-mu_z**2)
                            mu_y    = sin(phi)*sqrt(1-mu_z**2)
                            x, y, z = 0, 0, t_tot
                            continue

                        else:   # the photon neither leaves through the optional port,
                                # nor is re-incident, score the photon as diffuse
                                # transmission and stop propagating
                            T_diffuse += w
                            w = 0
                            break

                    else:   # there is no transmission sphere
                        # calculate an effective optional port
                        # fraction from the angle threshold
                        fp = 0.5*(1-cos(2*T_angle))

                        # check if the photon leaves in a direction within this
                        # effective optional port fraction
                        if abs(mu_z) >= sqrt(1-fp):
                            # photon is within angle threshold, score as direct
                            # transmission and stop propagating
                            T_direct += w
                            w = 0
                            break

                        else:   # photon is NOT within angle threshold, since there is no sphere
                                # present, score it as diffuse transmission and stop propagating
                            T_diffuse += w
                            w = 0
                            break

                else:  # the photon was wholly absorbed
                    w = 0
                    break

    # convert values to percents and return them
    return A/N, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N

#========#

if __name__ == '__main__':

    A, R_diffuse, R_specular, T_diffuse, T_direct = ISMLMC(N, n, g, t, mu_a, mu_s, \
        mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, \
        specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)

    # after all of the photons have propagated, round values and report results
    sigfigs = len(str(N))

    Absorptance             = round(A, sigfigs)
    diffuse_Reflectance     = round(R_diffuse, sigfigs)
    specular_Reflectance    = round(R_specular, sigfigs)
    diffuse_Transmittance   = round(T_diffuse, sigfigs)
    direct_Transmittance    = round(T_direct, sigfigs)
    total_Reflectance       = diffuse_Reflectance + specular_Reflectance
    total_Transmittance     = diffuse_Transmittance + direct_Transmittance

    print("""Absorptance: %f\nDiffuse Reflectance: %f\nSpecular Reflectance: %f\nDiffuse \
Transmittance: %f\nDirect Transmittance: %f\nTotal Reflectance: %f\nTotal Transmittance: \
%f""" %(Absorptance, diffuse_Reflectance, specular_Reflectance, diffuse_Transmittance, \
    direct_Transmittance, total_Reflectance, total_Transmittance))
