// nvcc -x cu -arch=sm_60 -std=c++11 ISMLMC-minimization.cu -o minimize.o -ccbin /usr/bin/g++-6

#include "praxis.cpp"
#include "../CUDA/cmdlineparse.h"
#include "../CUDA/CUDAISMLMC.h"
#include <iostream>
#include <thrust/tuple.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <fstream>
#include <future>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

//// Global variables which the cost function needs

extern int GPU;
int GPU;

extern int N;
int N;

extern bool separate;
bool separate;

extern float* n;
extern float* g;
extern float* t;
extern float* bounds;
float* n;
float* g;
float* t;
float* bounds;

extern bool R;
extern float R_pw;
extern float R_fs;
extern float R_fp;
extern float R_fd;
extern float R_f;
extern float R_angle;
extern bool specular_included;
bool R;
float R_pw;
float R_fs;
float R_fp;
float R_fd;
float R_f;
float R_angle;
bool specular_included;

extern bool T;
extern float T_pw;
extern float T_fs;
extern float T_fp;
extern float T_fd;
extern float T_f;
extern float T_angle;
bool T;
float T_pw;
float T_fs;
float T_fp;
float T_fd;
float T_f;
float T_angle;

extern float R_d_exact;
float R_d_exact;
extern float R_s_exact;
float R_s_exact;
extern float T_d_exact;
float T_d_exact;
extern float T_u_exact;
float T_u_exact;

////


// cost function to be minimized
// takes an array of {mua, mus'} and an integer which doesn't matter
// returns the Euclidean distance from a measured R_d, R_s, T_d, and T_u
// to the simulated one
double cost(double* coeffs, int ncoeffs)
{

    // coeffs is an array of {mua, mus'}
    float* mu_a = new float[1];
    mu_a[0] = (float)coeffs[0];

    float* mu_s = new float[1];
    if (g[0] == 1){ mu_s[0] = (float)coeffs[1]; }
    else { mu_s[0] = (float)coeffs[1]/(1-g[0]); }

    float* mu_t = new float[1];
    mu_t[0] = mu_a[0] + mu_s[0];

    float A_, R_d_, R_s_, T_d_, T_u_;
    float R_d, R_s, T_d, T_u;

    if (!separate)
    {
        thrust::tie(A_,R_d,R_s,T_d,T_u) = ISMLMC(GPU, N, 1, n, g, t, bounds, mu_a, mu_s, mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);
    }
    else
    {
        if (!isnan(R_d_exact)) { thrust::tie(A_,R_d,R_s_,T_d_,T_u_) = ISMLMC(GPU, N, 1, n, g, t, bounds, mu_a, mu_s, mu_t, true, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, false, T_pw, T_fs, T_fp, T_fd, T_f, T_angle); }
        if (!isnan(T_d_exact)) { thrust::tie(A_,R_d_,R_s_,T_d,T_u_) = ISMLMC(GPU, N, 1, n, g, t, bounds, mu_a, mu_s, mu_t, false, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, true, T_pw, T_fs, T_fp, T_fd, T_f, T_angle); }
        if (!isnan(R_s_exact) || !isnan(T_u_exact)) { thrust::tie(A_,R_d_,R_s,T_d_,T_u) = ISMLMC(GPU, N, 1, n, g, t, bounds, mu_a, mu_s, mu_t, false, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, false, T_pw, T_fs, T_fp, T_fd, T_f, T_angle); }
    }

    //printf("mua: %8.4f mus': %8.4f mus: %8.4f mut: %8.4f g: %8.4f | R_d %1.6f T_d %1.6f T_u %1.6f\n", coeffs[0], coeffs[1], mu_s[0], mu_t[0], g[0], R_d, T_d, T_u);

    float R_d_cost = 0;
    float R_s_cost = 0;
    float T_d_cost = 0;
    float T_u_cost = 0;

    if (!isnan(R_d_exact)) { R_d_cost = pow((R_d - R_d_exact)/R_d_exact,2); }
    if (!isnan(R_s_exact)) { R_s_cost = pow((R_s - R_s_exact),2); }
    if (!isnan(T_d_exact)) { T_d_cost = pow((T_d - T_d_exact)/T_d_exact,2); }
    if (!isnan(T_u_exact)) { T_u_cost = pow((T_u - T_u_exact),2); }

    return (double)sqrt(R_d_cost+R_s_cost+T_d_cost+T_u_cost);

}

int main(int argc, char* argv[])
{

    if (cmdOptionExists(argv, argv+argc, "--help") || cmdOptionExists(argv, argv+argc, "-h"))
    {
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nIntegrating Sphere Monte Carlo Photon Transport Minimization\nTo recover optical properties of experimental samples.\n";
        cout << "\nWritten by Patrick Cook | Fort Hays State University | 4 May 2019\n";
        cout << "pdcook@mail.fhsu.edu or qphysicscook@gmail.com\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nUseful Flags:\n";
        cout << "\n--help -    Shows this help text.\n";
        cout << "\n-h     -    Same as --help.\n";
        cout << "\n--specularIncluded\n       -    Changes the incident beam to 8deg from the normal to include\n            specular reflection in any reflection sphere that may be present.\n";
        cout << "\n--separate\n       -    Run simulations separately to measure R_d, T_d, R_s, and T_u.\n            Usually enabled when using a single sphere to measure R_d and T_d.\n            Disable for dual-sphere experiments.\n            >>> If separate is enabled then ALL of --Rsphere\n            --Rangle --Tsphere and --Tangle must be specified for accurate results. <<<";
        cout << "\n--example\n       -    Show an example configuration and run it.\n            Makes all required parameters except --GPU optional.\n            Useful for ensuring your installation works.\n";
        cout << "\n--verbose\n       -    Output minimization at every step.\n";
        cout << "\n--go   -    Just run the simulation. Don't ask to confirm.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nRequired Parameters:\n";
        cout << "\n--GPU  -    Device ID for GPU to run on.\n            Use the 'nvidia-smi' command to list available GPUs.\n            Mutliple GPUs are NOT supported.\n            Example: --GPU 0\n";
        cout << "\n-N     -    Number of photons to use per simulation.\n";
        cout << "\n--runs -    Number of times to minimize. The average and standard deviation will be reported.\n";
        cout << "\n--step -    Maximum step size for minimization. Should be set to approximately the\n            Euclidean distance from the initial guess to the minimum.\n            >>>If you are getting a lot of minimizations that don't move from the\n            initial guess, change this number.<<<\n";
        cout << "\n-n     -    Relative refractive index (relative to the surrounding medium) for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -n \"1.33 1.4 1.33\"\n";
        cout << "\n-g     -    Anisotropy for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -g \"0.0 -0.2 1\"\n";
        cout << "\n-t     -    Thickness for each layer.\n            In order from incident surface to exiting surface. Must be in quotes.\n            Example: -t \"0.1 0.1 0.2\"\n";
        cout << "\n--mua  -    Initial guess for the absorption coefficient in 1/cm.\n";
        cout << "\n--mus  -    Initial guess for the REDUCED scattering coefficient in 1/cm.\n";
        cout << "            To convert from the reduced scattering coefficient, mus', to the scattering coefficient:\n";
        cout << "            mus = mus'/(1-g) for g!=1 and mus = mus' if g = 1.\n";
        cout << "\n--muaMin\n       -    Minimum acceptable value for mua. Results lower than this will be discarded.\n";
        cout << "\n--muaMax\n       -    Maximum acceptable value for mua. Results greater than this will be discarded.\n";
        cout << "\n--musMin\n       -    Minimum acceptable value for mus'. Results lower than this will be discarded.\n";
        cout << "\n--musMax\n       -    Maximum acceptable value for mus'. Results greater than this will be discarded.\n";
        cout << "\n--tol  -    Tolerance of the minimization. 1, 0.1, and 0.01 are good values.\n";
        cout << "\n--Rd   -    Measured value of diffuse reflectance to minimize to. Do not use if unmeasured.\n";
        cout << "\n--Rs   -    Measured value of specular reflectance to minimize to. Do not use if unmeasured.\n";
        cout << "\n--Td   -    Measured value of diffuse transmittance to minimize to. Do not use if unmeasured.\n";
        cout << "\n--Tu   -    Measured value of unscattered transmittance to minimize to. Do not use if unmeasured.\n";
        cout << "            >>>At least one of Rd, Rs, Td, Tu must be specified.<<<\n";
        cout << "\n\nOptional Parameters:\n";
        cout << "\n--Rsphere\n       -    Parameters of the sphere measuring reflectance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample port fractional area\n            - fp is the source port fractional area.\n            - fd is the detector fractional area\n            Example: --Rsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Rsphere is not specified then Rangle MUST be. <<<\n";
        cout << "\n--Rangle\n       -    Angle threshold in degrees for what counts as specular\n            reflectance when there is no reflection sphere present.\n";
        cout << "\n--Tsphere\n       -    Parameters of the sphere measuring transmittance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample/source port fractional area\n            - fp is the optional port fractional area.\n            - fd is the detector fractional area\n            Example: --Tsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Tsphere is not specified then Tangle MUST be. <<<\n";
        cout << "\n--Tangle\n       -    Angle threshold in degrees for what counts as unscattered\n            transmittance when there is no transmission sphere present.\n";
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

    // declare variables regarding CURAND for the total number of devices
    rand_set = new bool[nGPU]();
    globalDeviceStates = new curandState*[nGPU];


    // placeholder parameters that will persist if the user forgets to specify them

    //====Sphere Parameters====//
    R = false;                  // boolean to determine if the reflection sphere is present or not
    R_pw = nanf("0");           // reflectance of reflection sphere
    R_fs = nanf("0");           // sample port fraction of reflection sphere
    R_fp = nanf("0");           // source port fraction of reflection sphere
    R_fd = nanf("0");           // detector fraction of reflection sphere
    R_f = nanf("0");            // total port fraction of reflection sphere
    R_angle = nanf("0");        // angle threshold for specular reflection if no Rsphere is present
    specular_included = false;  // boolean to determine if specular reflection is included or not

    T = false;                  // boolean to determine if the transmission sphere is present or not
    T_pw = nanf("0");           // reflectance of transmission sphere
    T_fs = nanf("0");           // sample port fraction of transmission sphere
    T_fp = nanf("0");           // optional port fraction of transmission sphere
    T_fd = nanf("0");           // detector fraction of transmission sphere
    T_f = nanf("0");            // total port fraction of transmission sphere
    T_angle = nanf("0");        // angle threshold for direct transmission if no Tsphere is present
    //========//

    N = -1;
    int runs = -1;
    int layers = 1;             // only single layer minimizations supported at this time
    double step = nan("0");
    separate = false;           // measure R_d, T_d, and (R_s T_u) separately
    bool verbose = false;

    float* mu_a = new float[layers];    // mua and mus' must be arrays since CUDAMLMC expects arrays
    float* mu_s_ = new float[layers];

    float mu_a_min = 0;
    float mu_a_max = 10000;
    float mu_s__min = 0;
    float mu_s__max = 10000;

    double tolerance = nan("0");

    // measured values
    R_d_exact = nanf("0");
    R_s_exact = nanf("0");
    T_d_exact = nanf("0");
    T_u_exact = nanf("0");

    // use the example config
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

        N = 10000;
        runs = 5;
        step = 5;
        separate = true;
        verbose = true;

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        n[0] = 1.4;
        g[0] = 0.5;
        t[0] = 0.135;

        mu_a[0] = 5;
        mu_s_[0] = 35;

        mu_a_min = 1;
        mu_a_max = 3;
        mu_s__min = 20;
        mu_s__max = 40;

        tolerance = 1;

        R_d_exact = 0.2716;
        T_d_exact = 0.0853;
        T_u_exact = 0.0000;

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            cout << "\nEXAMPLE MINIMIZATION\n";
            printf("\nParameters (as found in Cook et al 2019):\n\n-N %d\n--runs %d\n--step %f\n-n", N, runs, (float)step);
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
            printf("\n--muaMin %f\n--muaMax %f\n--musMin %f\n--musMax %f\n--tol %f", mu_a_min, mu_a_max, mu_s__min, mu_s__max, (float)tolerance);
            printf("\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n--Rd %f\n--Rs %f\n--Td %f\n--Tu %f\n--separate\n--verbose\n", R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI, R_d_exact, R_s_exact, T_d_exact, T_u_exact);
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // user config parsed from command line
    else
    {

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

        N = readIntOption(argv, argv+argc, "-N");
        runs = readIntOption(argv, argv+argc, "--runs");
        step = (double)readFloatOption(argv, argv+argc, "--step");
        separate = cmdOptionExists(argv, argv+argc, "--separate");
        verbose = cmdOptionExists(argv, argv+argc, "--verbose");

        n = new float[layers];
        g = new float[layers];
        t = new float[layers];

        n = readArrOption(argv, argv+argc, "-n", layers);
        g = readArrOption(argv, argv+argc, "-g", layers);
        t = readArrOption(argv, argv+argc, "-t", layers);

        R_d_exact = readFloatOption(argv, argv+argc, "--Rd");
        R_s_exact = readFloatOption(argv, argv+argc, "--Rs");
        T_d_exact = readFloatOption(argv, argv+argc, "--Td");
        T_u_exact = readFloatOption(argv, argv+argc, "--Tu");

        mu_a = readArrOption(argv, argv+argc, "--mua", layers);
        mu_s_ = readArrOption(argv, argv+argc, "--mus", layers);

        mu_a_min = readFloatOption(argv, argv+argc, "--muaMin");
        mu_a_max = readFloatOption(argv, argv+argc, "--muaMax");
        mu_s__min = readFloatOption(argv, argv+argc, "--musMin");
        mu_s__max = readFloatOption(argv, argv+argc, "--musMax");

        tolerance = (double)readFloatOption(argv, argv+argc, "--tol");

        if (!go)
        {
            cout << "\n---------------------------------------------------------------------------------------\n";
            printf("\nParameters:\n\n-N %d\n--runs %d\n--step %f\n-n", N, runs, (float)step);
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
            printf("\n--muaMin %f\n--muaMax %f\n--musMin %f\n--musMax %f\n--tol %f", mu_a_min, mu_a_max, mu_s__min, mu_s__max, (float)tolerance);
            printf("\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n--Rd %f\n--Rs %f\n--Td %f\n--Tu %f\n", R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI, R_d_exact, R_s_exact, T_d_exact, T_u_exact);
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // define layer boundaries
    bounds = new float[layers+1];
    bounds[0] = 0;

    for( int i = 1; i < layers+1; i++ )
    {
        bounds[i] = bounds[i-1] + t[i-1];
    }

    // declare array to give the cost function
    double* coeffs = new double[2];

    // declare the array to store minimized values in
    double* mu_a_array = new double[runs];
    double* mu_s__array = new double[runs];

    int verbosity = 0;
    if (verbose) { verbosity = 1; }

    for (int i = 0; i < runs; i++)
    {
        coeffs[0] = (double)mu_a[0];
        coeffs[1] = (double)mu_s_[0];

        if (verbose) { printf("----------------\nRUN %d of %d\n----------------\n", i+1, runs); }

        // minimize the cost function using PRAXIS
        // which is a modified Powell's method
        praxis(tolerance, step, 2, verbosity, coeffs, cost);

        // only save the result if it's in the acceptable range
        if ( coeffs[0] >= mu_a_min && coeffs[0] <= mu_a_max && coeffs[1] >= mu_s__min && coeffs[1] <= mu_s__max)
        {
            mu_a_array[i] = coeffs[0];
            mu_s__array[i] = coeffs[1];
        }
        else
        {
            i -= 1;
        }
    }

    // calulate the average and standard error of the runs
    double mu_a_average = 0;
    double mu_s__average = 0;

    float Nruns = (float)runs;

    for (int i = 0; i < runs; i++)
    {
        mu_a_average += (1/Nruns)*mu_a_array[i];
        mu_s__average += (1/Nruns)*mu_s__array[i];
    }

    double mu_a_temp_sum = 0;
    double mu_s__temp_sum = 0;

    for (int i = 0; i<runs; i++)
    {
        mu_a_temp_sum += pow((mu_a_array[i] - mu_a_average),2);
        mu_s__temp_sum += pow((mu_s__array[i] - mu_s__average),2);
    }

    double mu_a_stderr = sqrt((1/Nruns)*mu_a_temp_sum)/sqrt(Nruns);
    double mu_s__stderr = sqrt((1/Nruns)*mu_s__temp_sum)/sqrt(Nruns);

    // return the results
    if (verbose) { printf("\n\nMinimization Finished. Recovered Optical Properties:\n\nmua: %f +/- %f\nmus': %f +/- %f\n\n", mu_a_average, mu_a_stderr, mu_s__average, mu_s__stderr); }
    else { printf("%f\n%f\n%f\n%f\n", mu_a_average, mu_a_stderr, mu_s__average, mu_s__stderr); }

    return 0;
}
