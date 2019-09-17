# ISMC.py

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

N       = 100000        # number of photons in simulation

#====Sample Definition====#
n       = 1.4           # refractive index of sample
g       = 0.5           # anisotropy factor of sample
mu_a    = 1.78          # absorption coefficient of sample in 1/cm
mu_s_   = 31.2          # reduced scattering coefficient of sample in 1/cm
t       = 0.135         # thickness of sample in cm

if g == 1: mu_s = mu_s_     # scattering coefficient of sample in 1/cm
else: mu_s = mu_s_/(1-g)    #

mu_t    = mu_a + mu_s   # total interaction coefficient of sample in 1/cm
#========#

#====Sphere Parameters====#
R       = True          # boolean to determine if the reflection sphere is present or not
R_pw    = 0.99          # reflectance of reflection sphere
R_fs    = 0.02351       # sample port fraction of reflection sphere
R_fp    = 0.054638      # source port fraction of reflection sphere
R_fd    = 0.02351       # detector fraction of reflection sphere
R_f     = R_fs + R_fp + R_fd    # total port fraction of reflection sphere
R_angle = pi/10         # angle threshold for specular reflection if no Rsphere is present
specular_included = False   # boolean to determine if specular reflection is included or not

T       = False         # boolean to determine if the transmission sphere is present or not
T_pw    = 0.99          # reflectance of transmission sphere
T_fs    = 0.054638      # sample port fraction of transmission sphere
T_fp    = 0.02351       # optional port fraction of transmission sphere
T_fd    = 0.02351       # detector fraction of transmission sphere
T_f     = T_fs + T_fp + T_fd    # total port fraction of transmission sphere
T_angle = 0.22*pi/180   # angle threshold for direct transmission if no Tsphere is present
#========#

#====Physics Functions====#

# Snell's law
def snell(ni, nt, mu_x, mu_y, mu_z):
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
def incident_reflection(mu_z, n):
    if uniform(0,1) > fresnel_snell(1., n, mu_z): return False
    else: return True

# if a photon is inside a sphere, this function determines if it is re-incident on the sample
# returns True if re-incident and False if not
def reincidence(pw, fs, f):
        return uniform(0,1) < (pw*fs)/(1-pw*(1-f))

#========#

#====Single Photon Monte Carlo====#

def MC(n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z):

    """
        This function propagates a photon from position x, y, z with direction cosines mu_x,
        mu_y, mu_z, until it is either absorbed, reflected, or transmitted. These bins are
        returned by this function alongside the final direction cosines of the photon.
    """

    Absorbed, Reflected, Transmitted = 0, 0, 0

    while w > 0:
        s = -log(uniform(0,1))/mu_t     # stepsize

        x += s*mu_x     #
        y += s*mu_y     # Hop
        z += s*mu_z     #

        # boundary conditions
        while z > t or z < 0:           # photon is outside of sample
            if z > t:                   # photon attempts to transmit
                if uniform(0,1) < fresnel_snell(n, 1., mu_z):
                    # photon is internally reflected
                    z = 2*t - z
                    mu_z *= -1
                else:                   # photon is transmitted
                    Transmitted += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = snell(n, 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

            elif z < 0:                 # photon attempts to reflect/backscatter
                if uniform(0,1) < fresnel_snell(n, 1., mu_z):
                    # photon is internally reflected
                    z *= -1
                    mu_z *= -1
                else:                   # photon backscatters
                    Reflected += w
                    # refraction via Snell's Law
                    mu_x, mu_y, mu_z = snell(n, 1., mu_x, mu_y, mu_z)
                    w = 0
                    break

        # partial absorption event
        deltaW = w*mu_a/mu_t
        w -= deltaW
        Absorbed += deltaW

        # roullette
        if w <= 0.001: w = 10*w if uniform(0,1) <= 1/10 else 0

        # scattering event: update the photon's direction cosines only if it's weight isn't 0
        ### Spin ###
        if w > 0:
            if g == 0.: cos_theta = 2*uniform(0,1) - 1
            else: cos_theta = (1/(2*g))*(1+g*g-((1-g*g)/(1-g+2*g*uniform(0,1)))**2)

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

def ISMC(N, n, g, t, mu_a, mu_s, mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, \
    specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle):

    A           = 0 # total number of absorbed photons
    R_diffuse   = 0 # total number of diffusely reflected/backscattered photons
    R_specular  = 0 # total number of specularly reflected photons
    T_diffuse   = 0 # total number of diffusely transmitted photons
    T_direct    = 0 # total number of directly transmitted photons

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
            incident_reflect = incident_reflection(mu_z, n)

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
                        x, y, z = 0, 0, t
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
                mu_x, mu_y, mu_z = snell(1., n, mu_x, mu_y, mu_z)

                # Monte Carlo Photon Transport
                Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z = \
                    MC(n, mu_a, mu_s, mu_t, g, t, w, x, y, z, mu_x, mu_y, mu_z)

                A += Absorbed   # part of the photon was absorbed, score it

                # check to see what happened to the photon
                if Reflected:   # the photon was reflected/backscattered
                    sample_side = 'r'
                    w = Reflected   # update the photon's weight

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
                    sample_side = 't'
                    w = Transmitted # update the photon's weight

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
                            x, y, z = 0, 0, t
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

                else:   # the photon was wholly absorbed
                    w = 0
                    break

    # convert values to percents and return them
    return A/N, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N

#========#

if __name__ == '__main__':

    A, R_diffuse, R_specular, T_diffuse, T_direct = ISMC(N, n, g, t, mu_a, mu_s, \
        mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, \
        specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle)

    # after all of the photons have propagated, round values and report results
    sigfigs = len(str(N))

    Absorptance, A_unc              = round(A, sigfigs), \
                                        round(sqrt(A/N), sigfigs)
    diffuse_Reflectance, R_d_unc    = round(R_diffuse, sigfigs), \
                                        round(sqrt(R_diffuse/N), sigfigs)
    specular_Reflectance, R_s_unc   = round(R_specular, sigfigs), \
                                        round(sqrt(R_specular/N), sigfigs)
    diffuse_Transmittance, T_d_unc  = round(T_diffuse, sigfigs), \
                                        round(sqrt(T_diffuse/N), sigfigs)
    direct_Transmittance, T_u_unc   = round(T_direct, sigfigs), \
                                        round(sqrt(T_direct/N), sigfigs)
    total_Reflectance, R_t_unc      = diffuse_Reflectance + specular_Reflectance, \
                                round(sqrt(sqrt(R_diffuse/N)**2+sqrt(R_specular/N)**2), sigfigs)
    total_Transmittance, T_t_unc    = diffuse_Transmittance + direct_Transmittance, \
                                round(sqrt(sqrt(T_diffuse/N)**2+sqrt(T_direct/N)**2), sigfigs)

    print("""Absorptance: %f +/- %f\nDiffuse Reflectance: %f +/- %f\nSpecular Reflectance: \
%f +/- %f\nDiffuse Transmittance: %f +/- %f\nDirect Transmittance: %f +/- %f\nTotal Reflectance: \
%f +/- %f\nTotal Transmittance: %f +/- %f""" %(Absorptance, A_unc, diffuse_Reflectance, R_d_unc, \
specular_Reflectance, R_s_unc, diffuse_Transmittance, T_d_unc, direct_Transmittance, T_u_unc, \
total_Reflectance, R_t_unc, total_Transmittance, T_t_unc))
