// nvcc -x cu -arch=sm_60 -std=c++11 ISMLMC.cu -o ISMLMC.o -ccbin /usr/bin/g++-6

#include "cmdlineparse.h"
#include "CUDAISMLMC.h"
#include <iostream>
#include <fstream>
#include <thrust/tuple.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <future>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

int main(int argc, char* argv[])
{

    if (cmdOptionExists(argv, argv+argc, "--help") || cmdOptionExists(argv, argv+argc, "-h"))
    {
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nIntegrating Sphere Multilayer Monte Carlo Photon Transport\n";
        cout << "\nWritten by Patrick Cook | Fort Hays State University | 4 May 2019\n";
        cout << "pdcook@mail.fhsu.edu or qphysicscook@gmail.com\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nUseful Flags:\n";
        cout << "\n--help -    Shows this help text.\n";
        cout << "\n-h     -    Same as --help.\n";
        cout << "\n--specularIncluded\n       -    Changes the incident beam to 8deg from the normal to include\n            specular reflection in any reflection sphere that may be present.\n";
        cout << "\n--example\n       -    Show an example configuration and run it.\n            Makes all required parameters except --GPU optional.\n            Useful for ensuring your installation works.\n";
        cout << "\n--go   -    Just run the simulation. Don't ask to confirm.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nRequired Parameters:\n";
        cout << "\n--GPU  -    Device ID for GPU to run on.\n            Use the 'nvidia-smi' command to list available GPUs.\n            Mutliple GPUs are NOT supported.\n            Example: --GPU 0\n";
        cout << "\n--unc  -    Uncertainty threshold for all measured parameters. The number of photons will be\n            calculated such that this uncertainty should be reached.\n";
        cout << "\n--layers\n       -    Number of layers in the sample.\n";
        cout << "\n-n     -    Relative refractive index (relative to the surrounding medium) for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -n \"1.33 1.4 1.33\"\n";
        cout << "\n-g     -    Anisotropy for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -g \"0.0 -0.2 1\"\n";
        cout << "\n-t     -    Thickness for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -t \"0.1 0.1 0.2\"\n";
        cout << "\n--mua -     Absorption coefficient for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -mua \"1.1 0.5 1\"\n";
        cout << "\n--mus -     REDUCED scattering coefficient for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -mus \"30 10 1\"\n";
        cout << "            To convert from the reduced scattering coefficient, mus', to the scattering coefficient:\n";
        cout << "            mus = mus'/(1-g) for g!=1 and mus = mus' if g = 1.\n";
        cout << "\n\nOptional Parameters:\n";
        cout << "\n--est  -    Estimate of largest value to be measured. Will reduce number of photons\n            necessary to reach certain error.\n            Example: --est 0.75\n";
        cout << "\n--Rsphere\n       -    Parameters of the sphere measuring reflectance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample port fractional area\n            - fp is the source port fractional area.\n            - fd is the detector fractional area\n            Example: --Rsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Rsphere is not specified then Rangle MUST be. <<<\n";
        cout << "\n--Rangle\n       -    Angle threshold in degrees for what counts as specular\n            reflectance when there is no reflection sphere present.\n";
        cout << "\n--Tsphere\n       -    Parameters of the sphere measuring transmittance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample/source port fractional area\n            - fp is the optional port fractional area.\n            - fd is the detector fractional area\n            Example: --Tsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Tsphere is not specified then Tangle MUST be. <<<\n";
        cout << "\n--Tangle\n       -    Angle threshold in degrees for what counts as unscattered\n            transmittance when there is no transmission sphere present.\n";
        cout << "\n-N     -    Can be used to override --unc. Number of photons to use per simulation.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nNotes:\n";
        cout << "If you are getting 'out of memory' errors, reduce N or change the RNGS variable in the source\ncode to something smaller.\n\n";

        return 1;
    }

    if (!cmdOptionExists(argv, argv+argc, "--GPU"))
    {
        cout << "Must specify device ID with the --GPU flag.\nUse --help to see available options.\n";
        return 2;
    }

    srand(time(NULL));

    int GPU = readIntOption(argv, argv+argc, "--GPU");
    int nGPU = 1;   // only single GPU supported at this time

    bool go = cmdOptionExists(argv, argv+argc, "--go");

    if (!go)
    {
        cout << "\n";
        CUDABasicProperties(GPU);
    }

    //// Declare global variables for CURAND
    rand_set = new bool[nGPU]();
    globalDeviceStates = new curandState*[nGPU];
    ////

    // initial parameters that will persist if the user doesn't specify them

    //====Sphere Parameters====//
    bool R = false;                 // boolean to determine if the reflection sphere is present or not
    float R_pw = nanf("0");         // reflectance of reflection sphere
    float R_fs = nanf("0");         // sample port fraction of reflection sphere
    float R_fp = nanf("0");         // source port fraction of reflection sphere
    float R_fd = nanf("0");         // detector fraction of reflection sphere
    float R_f = nanf("0");          // total port fraction of reflection sphere
    float R_angle = nanf("0");      // angle threshold for specular reflection if no Rsphere is present
    bool specular_included = false; // boolean to determine if specular reflection is included or not

    bool T = false;                 // boolean to determine if the transmission sphere is present or not
    float T_pw = nanf("0");         // reflectance of transmission sphere
    float T_fs = nanf("0");         // sample port fraction of transmission sphere
    float T_fp = nanf("0");         // optional port fraction of transmission sphere
    float T_fd = nanf("0");         // detector fraction of transmission sphere
    float T_f = nanf("0");          // total port fraction of transmission sphere
    float T_angle = nanf("0");      // angle threshold for direct transmission if no Tsphere is present
    //========//

    int N = -1;
    int layers = -1;
    float uncertainty = 0;
    float estimate = 1;

    float* n;
    float* g;
    float* t;

    float* mu_a;
    float* mu_s_;

    // use example parameters
    if (cmdOptionExists(argv, argv+argc, "--example"))
    {

        //====Sphere Parameters====//
        R       = true;                 // boolean to determine if the reflection sphere is present or not
        R_pw    = 0.99;                 // reflectance of reflection sphere
        R_fs    = 0.023510;             // sample port fraction of reflection sphere
        R_fp    = 0.054638;             // source port fraction of reflection sphere
        R_fd    = 0.023510;             // detector fraction of reflection sphere
        R_f     = R_fs + R_fp + R_fd;   // total port fraction of reflection sphere
        R_angle = 0;                    // angle threshold for specular reflection if no Rsphere is present
        specular_included = false;      // boolean to determine if specular reflection is included or not

        T       = true;                 // boolean to determine if the transmission sphere is present or not
        T_pw    = 0.99;                 // reflectance of transmission sphere
        T_fs    = 0.054638;             // sample port fraction of transmission sphere
        T_fp    = 0.023510;             // optional port fraction of transmission sphere
        T_fd    = 0.023510;             // detector fraction of transmission sphere
        T_f     = T_fs + T_fp + T_fd;   // total port fraction of transmission sphere
        T_angle = 0.22*M_PI/180;        // angle threshold for direct transmission if no Tsphere is present
        //========//

        N = 100000;
        layers = 3;

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        n[0] = 1.37; n[1] = 1.37; n[2] = 1.37;
        g[0] = 0.9; g[1] = 0.; g[2] = 0.7;
        t[0] = 0.1; t[1] = 0.1; t[2] = 0.2;

        mu_a = new float[layers];
        mu_s_ = new float[layers];

        mu_a[0] = 1; mu_a[1] = 1; mu_a[2] = 2;
        mu_s_[0] = 10; mu_s_[1] = 10; mu_s_[2] = 3;

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            cout << "\nEXAMPLE INTEGRATING SPHERE MLMC SIMULATION\n";
            printf("\nParameters (Sample as found in Wang et al 1995 | Spheres as found in Cook et al 2019):\n\n-N %d\n--layers %d\n-n", N, layers);
            for (int i = 0; i < layers; i++)
            {
                cout << " " << n[i];
            }
            cout << "\n-g";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << g[i];
            }
            cout <<"\n-t";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << t[i];
            }
            cout << "\n--mua";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_a[i];
            }
            cout << "\n--mus";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_s_[i];
            }
            printf("\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n", R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI);
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // user specified parameters parsed from command line
    else
    {

        if ((cmdOptionExists(argv, argv+argc, "-N")) && (cmdOptionExists(argv, argv+argc, "--unc") || cmdOptionExists(argv, argv+argc, "--est")))
        {
            cout << "-N cannot be used with --unc or --est\n";
            return 2;
        }

        //====Sphere Parameters====//
        R       = cmdOptionExists(argv, argv+argc, "--Rsphere");         // boolean to determine if the reflection sphere is present or not
        float* R_params = new float[4];
        for (int i = 0; i < 4; i++)
        {
            R_params[i] = nanf("1");
        }
        if (R)
        {
            R_params = readArrOption(argv, argv+argc, "--Rsphere", 4);
        }
        R_pw    = R_params[0];          // reflectance of reflection sphere
        R_fs    = R_params[1];          // sample port fraction of reflection sphere
        R_fp    = R_params[2];          // source port fraction of reflection sphere
        R_fd    = R_params[3];          // detector fraction of reflection sphere
        R_f     = R_fs + R_fp + R_fd;   // total port fraction of reflection sphere
        if (cmdOptionExists(argv, argv+argc, "--Rangle")){ R_angle = readFloatOption(argv, argv+argc, "--Rangle")*M_PI/180; }         // angle threshold for specular reflection if no Rsphere is present
        else { R_angle = 0; }
        specular_included = cmdOptionExists(argv, argv+argc, "--specularIncluded");   // boolean to determine if specular reflection is included or not

        T       = cmdOptionExists(argv, argv+argc, "--Tsphere");         // boolean to determine if the transmission sphere is present or not
        float* T_params = new float[4];
        for (int i = 0; i < 4; i++)
        {
            T_params[i] = nanf("1");
        }
        if (T)
        {
            T_params = readArrOption(argv, argv+argc, "--Tsphere", 4);
        }
        T_pw    = T_params[0];          // reflectance of transmission sphere
        T_fs    = T_params[1];          // sample port fraction of transmission sphere
        T_fp    = T_params[2];          // optional port fraction of transmission sphere
        T_fd    = T_params[3];          // detector fraction of transmission sphere
        T_f     = T_fs + T_fp + T_fd;   // total port fraction of transmission sphere
        if (cmdOptionExists(argv, argv+argc, "--Tangle")){ T_angle = readFloatOption(argv, argv+argc, "--Tangle")*M_PI/180; }        // angle threshold for direct transmission if no Tsphere is present
        else { T_angle = 0; }
        //========//

        if (cmdOptionExists(argv, argv+argc, "-N"))
        {
            N = readIntOption(argv, argv+argc, "-N");
        }
        else if (cmdOptionExists(argv, argv+argc, "--unc"))
        {
            if (cmdOptionExists(argv, argv+argc, "--est"))
            {
                estimate = readFloatOption(argv, argv+argc, "--est");
            }
            uncertainty = readFloatOption(argv, argv+argc, "--unc");
            N = (int)ceil(estimate/(uncertainty*uncertainty));
        }

        layers = readIntOption(argv, argv+argc, "--layers");

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        n = readArrOption(argv, argv+argc, "-n", layers);
        g = readArrOption(argv, argv+argc, "-g", layers);
        t = readArrOption(argv, argv+argc, "-t", layers);

        mu_a = new float[layers];
        mu_s_ = new float[layers];

        mu_a = readArrOption(argv, argv+argc, "--mua", layers);
        mu_s_ = readArrOption(argv, argv+argc, "--mus", layers);

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            printf("\nParameters:\n");
            if ( cmdOptionExists(argv, argc+argv, "--unc") )
            {
                printf("\n--unc %f", uncertainty);
            }
            if ( cmdOptionExists(argv, argc+argv, "--est") )
            {
                printf("\n--est %f", estimate);
            }
            if ( cmdOptionExists(argv, argc+argv, "-N") )
            {
                printf("\n-N %d", N);
            }
            else
            {
                printf("\n-N %d (calculated)", N);
            }
            printf("\n--layers %d\n-n", layers);
            for (int i = 0; i < layers; i++)
            {
                cout << " " << n[i];
            }
            cout << "\n-g";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << g[i];
            }
            cout <<"\n-t";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << t[i];
            }
            cout << "\n--mua";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_a[i];
            }
            cout << "\n--mus";
            for (int i = 0; i < layers; i++)
            {
                cout << " " << mu_s_[i];
            }
            printf("\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n", R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI);
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // calculate mus and mut from mua and mus'
    float* mu_s = new float[layers];
    float* mu_t = new float[layers];
    for (int i = 0; i < layers; i++)
    {
        if (g[i] == 1) { mu_s[i] = mu_s_[i]; }
        else { mu_s[i] = mu_s_[i]/(1-g[i]); }
        mu_t[i] = mu_a[i] + mu_s[i];

    }

    // define layer boundaries
    float* bounds = new float[layers+1];
    bounds[0] = 0;

    for( int i = 1; i < layers+1; i++ )
    {
        bounds[i] = bounds[i-1] + t[i-1];
    }

    // declare result variables
    float A,R_d,R_s,T_d,T_u;
    float A_unc,R_d_unc,R_s_unc,T_d_unc,T_u_unc;

    // run the simulation
    thrust::tie(A,R_d,R_s,T_d,T_u) = ISMLMC(GPU, N, layers, n, g, t, bounds, mu_a, mu_s, mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);

    // find uncertainties given by the poisson counting distribution

    A_unc = sqrt(A/N);
    R_d_unc = sqrt(R_d/N);
    R_s_unc = sqrt(R_s/N);
    T_d_unc = sqrt(T_d/N);
    T_u_unc = sqrt(T_u/N);

    // return the results
    printf("\nA: %f +/- %f\nR_d: %f +/- %f\nR_s %f +/- %f\nT_d %f +/- %f\nT_u %f +/- %f\n", A, A_unc, R_d, R_d_unc, R_s, R_s_unc, T_d, T_d_unc, T_u, T_u_unc);

    return 0;


}
