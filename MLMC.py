# MLMC.py

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

N       = 10000        # number of photons in simulation

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

        return R

    else: return fresnel_snell(1., n[layer], mu_z)
#========#

#====Single Photon Monte Carlo====#

def MC(n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z):

    """
        This function propagates a photon from position x, y, z with direction cosines mu_x,
        mu_y, mu_z, until it is either absorbed, reflected, or transmitted. The entire weight
        of the photon ends up in one of these three bins, which this function returns alongside
        the final direction of the photon.
    """

    Absorbed, Reflected, Transmitted = 0, 0, 0
    threshold = 0.0001
    m = 10
    layer = 0
    bounds = [0 for _ in range(len(t)+1)]
    for i in range(1,len(t)+1):
        bounds[i] = bounds[i-1]+t[i-1]

    layer = 0 if mu_z > 0 else len(n) - 1

    while w > 0:
        # draw a stepsize if not in glass
        if mu_s[layer]!=0:
            s = -log(uniform(0,1))/mu_t[layer]     # stepsize

        if mu_z < 0:
            d = (bounds[layer] - z)/mu_z
            nextlayer = layer - 1
        elif mu_z > 0:
            d = (bounds[layer+1] - z)/mu_z
            nextlayer = layer + 1
        elif mu_z == 0:
            d = inf
            nextlayer = layer

        # move the photon directly to the next boundary if in glass
        if mu_s[layer] == 0: s = d

        # boundary conditions
        while s >= d:

            x += d*mu_x
            y += d*mu_y
            z += d*mu_z
            s -= d

            if nextlayer == len(n):                   # photon attempts to transmit
                if uniform(0,1) < fresnel_snell(n[layer], 1., mu_z):
                    # photon is internally reflected
                    mu_z *= -1
                else:                   # photon is transmitted
                    Transmitted += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

            elif nextlayer == -1:                 # photon attempts to reflect/backscatter
                if uniform(0,1) < fresnel_snell(n[layer], 1., mu_z):
                    # photon is internally reflected
                    mu_z *= -1
                else:                   # photon backscatters
                    Reflected += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

            else:
                if uniform(0,1) < fresnel_snell(n[layer], n[nextlayer], mu_z):
                    mu_z *= -1
                else:
                    mu_x, mu_y, mu_z = multilayer_snell(n[layer], n[nextlayer], mu_x, mu_y, mu_z)
                    if mu_s[nextlayer] != 0 and s != 0:
                        s *= mu_t[layer]/mu_t[nextlayer]
                    layer = nextlayer

            if mu_z < 0:
                d = (bounds[layer] - z)/mu_z
                nextlayer = layer - 1
            elif mu_z > 0:
                d = (bounds[layer+1] - z)/mu_z
                nextlayer = layer + 1
            elif mu_z == 0:
                d = inf
                nextlayer = layer

            # if the photon is in glass, move it to the boundary
            if mu_s[layer] == 0: s = d

        x += s*mu_x     #
        y += s*mu_y     # Hop
        z += s*mu_z     #

        # if not in glass
        if mu_s[layer]!=0:
            # partial absorption event
            deltaW = w*mu_a[layer]/mu_t[layer]
            w -= deltaW
            Absorbed += deltaW

        # roullette
        if w <= threshold: w = m*w if uniform(0,1) <= 1/m else 0

        # scattering event: update the photon's direction cosines only if it's weight isn't 0
        # and it isn't in glass
        ### Spin ###
        if w > 0 and mu_s[layer]!=0:
            if g[layer] == 0.: cos_theta = 2*uniform(0,1) - 1
            else: cos_theta = (1/(2*g[layer]))*(1+g[layer]*g[layer]-((1-g[layer]*g[layer])/(1-g[layer]+2*g[layer]*uniform(0,1)))**2)

            phi = 2 * pi * uniform(0,1)
            cos_phi, sin_phi = cos(phi), sin(phi)
            sin_theta = sqrt(1. - cos_theta**2)

            if abs(mu_z) > 0.99999:
                mu_x_ = sin_theta*cos_phi
                mu_y_ = sin_theta*sin_phi
                mu_z_ = (mu_z/abs(mu_z))*cos_theta
            else:
                z_sqrt = sqrt(1 - mu_z*mu_z)
                mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta
                mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta
                mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta

            mu_x, mu_y, mu_z = mu_x_, mu_y_, mu_z_

    return Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z

#========#

def MLMC(N, n, g, t, mu_a, mu_s, mu_t):

    A           = 0 # total number of absorbed photons
    R_diffuse   = 0 # total number of diffusely reflected/backscattered photons
    R_specular  = 0 # total number of specularly reflected photons
    T_diffuse   = 0 # total number of diffusely transmitted photons
    T_direct    = 0 # total number of directly transmitted photons

    for i in range(N):
        w                   = 1         # initial weight of photon
        x, y, z             = 0, 0, 0   # initial position of photon
        mu_x, mu_y, mu_z = 0, 0, 1      # initial direction of photon

        while w > 0:
            # determine if the photon is incidently reflected from the sample
            layer = 0 if mu_z > 0 else len(n) - 1
            incident_reflect = incident_reflection(mu_z, n, mu_s, layer)
            R_specular += incident_reflect
            w -= incident_reflect
            Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z = MC(n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z)

            A += Absorbed
            if abs(mu_z)==1:
                R_specular += Reflected
                T_direct += Transmitted
                w = 0
                break

            else:
                R_diffuse += Reflected
                T_diffuse += Transmitted
                w = 0
                break

    # convert values to percents and return them
    return A/N, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N

#========#

if __name__ == '__main__':

    A, R_diffuse, R_specular, T_diffuse, T_direct = MLMC(N, n, g, t, mu_a, mu_s, mu_t)

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
