// nvcc -x cu -arch=sm_60 -std=c++11 ISMLMC-lookup.cu -o Lookup.o -ccbin /usr/bin/g++-4.8

#include "cmdlineparse.h"
#include "CUDAISMLMC.h"
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

using namespace std;

// compute the elementwise average of a bunch of 2d arrays
float** array_avg(float*** meta_array, int n_arrays, int rows, int cols)
{
    float** avg_array = new float*[rows];
    for (int i = 0; i < rows; i++)
    {
        avg_array[i] = new float[cols]();
    }

    for (int a = 0; a < n_arrays; a++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                avg_array[i][j] += meta_array[a][i][j]/n_arrays;
            }
        }
    }

    return avg_array;
}

// compute the elementwise standard error of a bunch of 2d arrays
float** array_stderr(float*** meta_array, int n_arrays, int rows, int cols)
{
    float** avg_array = new float*[rows];
    float** stderr_array = new float*[rows];
    for (int i = 0; i < rows; i++)
    {
        avg_array[i] = new float[cols]();
        stderr_array[i] = new float[cols]();
    }

    avg_array = array_avg(meta_array, n_arrays, rows, cols);

    for (int a = 0; a < n_arrays; a++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                stderr_array[i][j] += pow(meta_array[a][i][j]-avg_array[i][j],2);
            }
        }
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // standard error is standard deviation divided by number of data points
            stderr_array[i][j] = sqrt(stderr_array[i][j]/n_arrays)/sqrt(n_arrays);
        }
    }


    return stderr_array;
}

// compute the uncertainty of a run based on the poisson counting distribution
float** array_poisson(float*** meta_array, int n_arrays, int rows, int cols, int N)
{

    // in Poisson counting distribution, R_unc = sqrt(R)/N where R is the number of counts
    //  and N is the number of trials
    // In our case, R = r*N where r is the measured value and N is the number of photons
    //  so r_unc = sqrt(r/N)

    float** unc_array = new float*[rows];
    for (int i = 0; i < rows; i++)
    {
        unc_array[i] = new float[cols];
    }

    for (int a = 0; a < n_arrays; a++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                unc_array[i][j] = sqrt(meta_array[a][i][j]/N);
            }
        }
    }

    return unc_array;

}

// write the data in a neat way so Python (and humans) can easily read it
void write_data(string filename, string header, int N_mu_a, int N_mu_s, float n, float g, float t, float* mu_a, float* mu_s_, float** data, bool first)
{
    ofstream file;

    if (first)
    {
        file.open(filename);
    }
    else
    {
        file.open(filename, ios_base::app);
    }
    file << "\"" << header << "\",\"mua\"";
    for (int i = 0; i < N_mu_s; i++)
    {
        file << ",\"\"";
    }
    file << "\n";
    file << "\"mus'\",\"\"";
    for (int i = 0; i < N_mu_s; i++)
    {
        file << "," << mu_s_[i];
    }
    file << "\n";

    for (int i = 0; i < N_mu_a; i++)
    {
        file << "\"\"," << mu_a[i];
        for (int j = 0; j < N_mu_s; j++)
        {
            file << "," << data[i][j];
        }
        file << "\n";
    }

    file.close();

}

// sample the whole space of optical coefficients provided in order to generate data for a lookup table
void singleLayerSampleSpace(int* GPUs, int nGPU, int N, int runs, int N_mu_a, int N_mu_s, float* n, float* g, float* t, float* bounds, float* mu_a, float* mu_s_, bool R, float R_pw, float R_fs, float R_fp, float R_fd, float R_f, float R_angle, bool specular_included, bool T, float T_pw, float T_fs, float T_fp, float T_fd, float T_f, float T_angle, bool separate)
{
    // future types for values that will be filled by threads
    future<thrust::tuple<float,float,float,float,float>>* thread_vals = new future<thrust::tuple<float,float,float,float,float>>[nGPU];

    // determine if the poisson distribution should be used to find the error with just one run
    bool poisson = false;
    if (runs == 0)
    {
        runs = 1;
        poisson = true;
    }



    // initialization of 2D arrays that will hold measurement values for the whole sample space
    // we only care about R_d and T_d, but I have left the necessary code in to deal with the other values
    // T_u would need to be simulated separately if we wanted to find all the values with this code
    // just add thread3_vals just like the other two and set it up with R=false and T=false
    // then rip T_u from thread3_vals and Bob's your uncle.

    //float*** meta_A = new float**[runs];
    float*** meta_R_d = new float**[runs];
    //float*** meta_R_s = new float**[runs];
    float*** meta_T_d = new float**[runs];
    //float*** meta_T_u = new float**[runs];

    for (int i = 0; i < runs; i++)
    {
        //meta_A[i] = new float*[N_mu_a];
        meta_R_d[i] = new float*[N_mu_a];
        //meta_R_s[i] = new float*[N_mu_a];
        meta_T_d[i] = new float*[N_mu_a];
        //meta_T_u[i] = new float*[N_mu_a];
    }
    for (int i = 0; i < runs; i++)
    {
        for (int j = 0; j < N_mu_a; j++)
        {
            //meta_A[i][j] = new float[N_mu_s];
            meta_R_d[i][j] = new float[N_mu_s];
            //meta_R_s[i][j] = new float[N_mu_s];
            meta_T_d[i][j] = new float[N_mu_s];
            //meta_T_u[i][j] = new float[N_mu_s];
        }
    }

    //float** A = new float*[N_mu_a];
    float** R_d = new float*[N_mu_a];
    //float** R_s = new float*[N_mu_a];
    float** T_d = new float*[N_mu_a];
    //float** T_u = new float*[N_mu_a];
    //float** A_unc = new float*[N_mu_a];
    float** R_d_unc = new float*[N_mu_a];
    //float** R_s_unc = new float*[N_mu_a];
    float** T_d_unc = new float*[N_mu_a];
    //float** T_u_unc = new float*[N_mu_a];
    for (int i = 0; i < N_mu_a; i++)
    {
        //A[i] = new float[N_mu_s];
        R_d[i] = new float[N_mu_s];
        //R_s[i] = new float[N_mu_s];
        T_d[i] = new float[N_mu_s];
        //T_u[i] = new float[N_mu_s];
        //A_unc[i] = new float[N_mu_s];
        R_d_unc[i] = new float[N_mu_s];
        //R_s_unc[i] = new float[N_mu_s];
        T_d_unc[i] = new float[N_mu_s];
        //T_u_unc[i] = new float[N_mu_s];
    }

    // temp variables to hold each measurement value from each GPU
    float* A_ = new float[nGPU];
    float* R_d_ = new float[nGPU];
    float* R_s_ = new float[nGPU];
    float* T_d_ = new float[nGPU];
    float* T_u_ = new float[nGPU];

    // temp variables to store the currect optical properties
    float** _mu_a = new float*[nGPU];
    float** _mu_s = new float*[nGPU];
    float** _mu_s_ = new float*[nGPU];
    float** _mu_t = new float*[nGPU];
    for (int i = 0; i < nGPU; i++)
    {
        _mu_a[i] = new float[1];
        _mu_s[i] = new float[1];
        _mu_s_[i] = new float[1];
        _mu_t[i] = new float[1];
    }

    // calculating mu_s from mu_s'
    float* mu_s = new float[N_mu_s];
    for (int i = 0; i < N_mu_s; i++)
    {
        if (g[0] == 0) { mu_s[i] = mu_s_[i]; }
        else { mu_s[i] = mu_s_[i]/(1-g[0]); }
    }

    // current device id
    int dev;

    // run as many times as requested
    for (int r = 0; r < runs; r++)
    {
        // sample all mu_s
        for (int j = 0; j < N_mu_s; j++)
        {
            for (int d = 0; d < nGPU; d++)
            {
                _mu_s[d][0] = mu_s[j];
                _mu_s_[d][0] = mu_s_[j];
            }

            // sample all mu_a
            for (int i = 0; i < N_mu_a; i+=nGPU)
            {
                // if T_d and R_d were measured separately, do both configurations at once, if not, then two two samples at once
                if (!separate)
                {
                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        _mu_a[d][0] = mu_a[i+d];
                        _mu_t[d][0] = _mu_s[d][0] + _mu_a[d][0];
                        thread_vals[d] = async(launch::async, &ISMLMC, dev, N, 1, n, g, t, bounds, _mu_a[d], _mu_s[d], _mu_t[d], R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);
                    }
                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        thrust::tie(A_[d],R_d_[d],R_s_[d],T_d_[d],T_u_[d]) = thread_vals[d].get();
                    }
                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        printf("RUN: %2d of %2d | GPU %2d | mua: %8.4f mus: %8.4f | A: %1.6f R_d %1.6f T_d %1.6f\n", r+1, runs, dev, mu_a[i+d], mu_s_[j], A_[d], R_d_[d], T_d_[d]);

                        // save values to arrays

                        if (R || T)
                        {
                            //A[i+d][j] = A_[d];
                            R_d[i+d][j] = R_d_[d];
                            //R_s[i+d][j] = R_s_[d];
                            T_d[i+d][j] = T_d_[d];
                            //T_u[i+d][j] = T_u_[d];
                        }
                        else
                        {
                            //A[i+d][j] = A_[d];
                            R_d[i+d][j] = R_d_[d] + R_s_[d];
                            //R_s[i+d][j] = R_s_[d];
                            T_d[i+d][j] = T_d_[d] + T_u_[d];
                            //T_u[i+d][j] = T_u_[d];
                        }
                    }
                }

                else if (separate)
                {
                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        _mu_a[d][0] = mu_a[i+d];
                        _mu_t[d][0] = _mu_s[d][0] + _mu_a[d][0];
                        thread_vals[d] = async(launch::async, &ISMLMC, dev, N, 1, n, g, t, bounds, _mu_a[d], _mu_s[d], _mu_t[d], true, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, false, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);
                    }
                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        thrust::tie(A_[d],R_d_[d],R_s_[d],T_d_[d],T_u_[d]) = thread_vals[d].get();
                    }

                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        printf("RUN: %2d of %2d | GPU %2d | R Sphere | mua: %8.4f mus: %8.4f | A: %1.6f R_d %1.6f T_d %1.6f\n", r+1, runs, dev, mu_a[i+d], mu_s_[j], A_[d], R_d_[d], T_d_[d]);

                        // save R_d to array
                        R_d[i+d][j] = R_d_[d];
                    }

                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        _mu_a[d][0] = mu_a[i+d];
                        _mu_t[d][0] = _mu_s[d][0] + _mu_a[d][0];
                        thread_vals[d] = async(launch::async, &ISMLMC, dev, N, 1, n, g, t, bounds, _mu_a[d], _mu_s[d], _mu_t[d], false, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, true, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);
                    }
                    for (int d = 0; d < nGPU&& i+d < N_mu_a; d++)
                    {
                        thrust::tie(A_[d],R_d_[d],R_s_[d],T_d_[d],T_u_[d]) = thread_vals[d].get();
                    }

                    for (int d = 0; d < nGPU && i+d < N_mu_a; d++)
                    {
                        dev = GPUs[d];
                        printf("RUN: %2d of %2d | GPU %2d | T Sphere | mua: %8.4f mus: %8.4f | A: %1.6f R_d %1.6f T_d %1.6f\n", r+1, runs, dev, mu_a[i+d], mu_s_[j], A_[d], R_d_[d], T_d_[d]);

                        // save T_d to array
                        T_d[i+d][j] = T_d_[d];
                    }


                    // save values to arrays

                    //A[i+d][j] = something
                    //R_s[i+d][j] = something
                    //T_u[i+d][j] = something


                }
            }

            // save this run's data


            for (int i = 0; i < N_mu_a; i++)
            {
                for (int j = 0; j < N_mu_s; j++)
                {
                    //meta_A[r][i][j] = A[i][j];
                    meta_R_d[r][i][j] = R_d[i][j];
                    //meta_R_s[r][i][j] = R_s[i][j];
                    meta_T_d[r][i][j] = T_d[i][j];
                    //meta_T_u[r][i][j] = T_u[i][j];
                }
            }
        }

    }

    // find averages and standard deviations

    //A = array_avg(meta_A, runs, N_mu_a, N_mu_s);
    R_d = array_avg(meta_R_d, runs, N_mu_a, N_mu_s);
    //R_s = array_avg(meta_R_s, runs, N_mu_a, N_mu_s);
    T_d = array_avg(meta_T_d, runs, N_mu_a, N_mu_s);
    //T_u = array_avg(meta_T_u, runs, N_mu_a, N_mu_s);

    if (poisson)
    {
        //A_unc = array_poisson(meta_A, runs, N_mu_a, N_mu_s, N);
        R_d_unc = array_poisson(meta_R_d, runs, N_mu_a, N_mu_s, N);
        //R_s_unc = array_poisson(meta_R_s, runs, N_mu_a, N_mu_s, N);
        T_d_unc = array_poisson(meta_T_d, runs, N_mu_a, N_mu_s, N);
        //T_u_unc = array_poisson(meta_T_u, runs, N_mu_a, N_mu_s, N);
    }
    else
    {
        //A_unc = array_stderr(meta_A, runs, N_mu_a, N_mu_s);
        R_d_unc = array_stderr(meta_R_d, runs, N_mu_a, N_mu_s);
        //R_s_unc = array_stderr(meta_R_s, runs, N_mu_a, N_mu_s);
        T_d_unc = array_stderr(meta_T_d, runs, N_mu_a, N_mu_s);
        //T_u_unc = array_stderr(meta_T_u, runs, N_mu_a, N_mu_s);
    }
    // write data in such a way that python and numpy can easily read it and generate a lookup table

    //write_data("Absorptance.csv", "Average Absorptance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, A, true);
    write_data("Diffuse_Reflectance.csv", "Average Diffuse Reflectance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, R_d, true);
    //write_data("Specular_Reflectance.csv", "Average Specular Reflectance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, R_s, true);
    write_data("Diffuse_Transmittance.csv", "Average Diffuse Transmittance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, T_d, true);
    //write_data("Unscattered_Transmittance.csv", "Average Unscattered Transmittance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, T_u, true);

    //write_data("Absorptance.csv", "Standard Error in Absorptance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, A_unc, false);
    write_data("Diffuse_Reflectance.csv", "Standard Error in Diffuse Reflectance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, R_d_unc, false);
    //write_data("Specular_Reflectance.csv", "Standard Error in Specular Reflectance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, R_s_unc, false);
    write_data("Diffuse_Transmittance.csv", "Standard Error in Diffuse Transmittance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, T_d_unc, false);
    //write_data("Unscattered_Transmittance.csv", "Standard Error in Unscattered Transmittance", N_mu_a, N_mu_s, n[0], g[0], t[0], mu_a, mu_s_, T_u_unc, false);

}

int main(int argc, char* argv[])
{

    if (cmdOptionExists(argv, argv+argc, "--help") || cmdOptionExists(argv, argv+argc, "-h"))
    {
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nIntegrating Sphere Monte Carlo Lookup Table Data Generator\n";
        cout << "\nWritten by Patrick Cook | Fort Hays State University | 4 May 2019\n";
        cout << "pdcook@mail.fhsu.edu or qphysicscook@gmail.com\n";
        cout << "\nTo be used in conjuction with the provided python script to create lookup tables.\nThis code currently only produces data for diffuse reflectance and diffuse transmittance.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nUseful Flags:\n";
        cout << "\n--help -    Shows this help text.\n";
        cout << "\n-h     -    Same as --help.\n";
        cout << "\n--specularIncluded\n       -    Changes the incident beam to 8deg from the normal to include\n            specular reflection in any reflection sphere that may be present.\n";
        cout << "\n--separate\n       -    Run simulations separately to measure diffuse reflectance and diffuse transmittance.\n            Usually enabled when using a single sphere to measure R_d and T_d.\n            Disable for dual-sphere experiments.\n            >>> If separate is enabled then ALL of --Rsphere\n            --Rangle --Tsphere and --Tangle must be specified for accurate results. <<<";
        cout << "\n--example\n       -    Show an example configuration and run it.\n            Makes all required parameters except --GPU optional.\n            Useful for ensuring your installation works.\n";
        cout << "\n--go   -    Just run the simulation. Don't ask to confirm.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nRequired Parameters:\n";
        cout << "\n--GPU  -    Device IDs for GPUs to run on.\n            Use the 'nvidia-smi' command to list available GPUs.\n            Mutliple GPU IDs must be in quotes.\n            Example: --GPU \"0 2\"\n";
        cout << "\n--unc  -    Uncertainty threshold for all measured parameters. The number of photons will be\n            calculated such that this uncertainty should be reached.\n";
        cout << "\n-n     -    Relative refractive index (relative to the surrounding medium) to use for all samples.\n";
        cout << "\n-g     -    Anisotropy of all samples.\n";
        cout << "\n-t     -    Thickness of all samples in centimeters.\n";
        cout << "\n--Nmua -    Total number of absorption coefficients.\n";
        cout << "\n--muaS -    Starting value for the absorption coefficients in 1/cm.\n";
        cout << "\n--muaB -    Base of the geometric range for the absorption coefficients.\n";
        cout << "            Absorption coefficients will be generated with mua[i] = muaS*muaB^i for 0 <= i < Nmua.\n";
        cout << "\n--Nmus -    Total number of REDUCED scattering coefficients.\n";
        cout << "\n--musS -    Starting value for the REDUCED scattering coefficients in 1/cm.\n";
        cout << "\n--musB -    Base of the geometric range for the REDUCED scattering coefficients.\n";
        cout << "            REDUCED scattering coeffs will be generated with mus[i] = musS*musB^i for 0 <= i < Nmus.\n";
        cout << "            To convert from the reduced scattering coefficient, mus', to the scattering coefficient:\n";
        cout << "            mus = mus'/(1-g) for g!=1 and mus = mus' if g = 1.\n";
        cout << "\n\nOptional Parameters:\n";
        cout << "\n--est  -    Estimate of largest value to be measured. Will reduce number of photons\n            necessary to reach certain error.\n            Example: --est 0.75\n";
        cout << "\n--Rsphere\n       -    Parameters of the sphere measuring reflectance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample port fractional area\n            - fp is the source port fractional area.\n            - fd is the detector fractional area\n            Example: --Rsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Rsphere is not specified then Rangle MUST be. <<<\n";
        cout << "\n--Rangle\n       -    Angle threshold in degrees for what counts as specular\n            reflectance when there is no reflection sphere present.\n";
        cout << "\n--Tsphere\n       -    Parameters of the sphere measuring transmittance. Must be in quotes and\n            in the following order: pw fs fp fd\n            - pw is the reflectance of the inner wall\n            - fs is the sample/source port fractional area\n            - fp is the optional port fractional area.\n            - fd is the detector fractional area\n            Example: --Tsphere \"0.99 0.1 0.1 0.2\"\n            >>> If --Tsphere is not specified then Tangle MUST be. <<<\n";
        cout << "\n--Tangle\n       -    Angle threshold in degrees for what counts as unscattered\n            transmittance when there is no transmission sphere present.\n";
        cout << "\n-N     -    Can be used to override --unc. Must be used in conjuction with --runs.\n            Number of photons to use per simulation.\n";
        cout << "\n--runs -    Can be used to override --unc. Must be used in conjuction with -N.\n            Number of times to run each simulation.\n            Results will report the average and standard error of these runs.\n";
        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nNotes:\n";
        cout << "If you are getting 'out of memory' errors, reduce N or change the RNGS variable in the source\ncode to something smaller.\n\n";

        return 1;
    }

    if (!cmdOptionExists(argv, argv+argc, "--GPU"))
    {
        cout << "Must specify device IDs with the --GPU flag.\nUse --help to see available options.\n";
        return 2;
    }

    srand(time(NULL));

    //// This block finds the number of GPUs and declares variables for CURAND for each of them ////
    int nGPU = 0;
    char* GPU = getCmdOption(argv, argv+argc, "--GPU");
    string GPUstr(GPU);
    stringstream GPUss1(GPUstr);
    int temp1;
    while (GPUss1 >> temp1)
    {
        nGPU++;
    }
    stringstream GPUss2(GPUstr);
    int temp2;
    int* GPUs = new int[nGPU];
    int fGPU = 0;
    for (int i = 0; GPUss2 >> temp2; i++)
    {
        GPUs[i] = temp2;
        if (GPUs[i] > fGPU)
        {
            fGPU = GPUs[i];
        }

        cout << "\n";

        CUDABasicProperties(GPUs[i]);
    }
    cout << "\n";
    fGPU++;
    rand_set = new bool[fGPU]();
    globalDeviceStates = new curandState*[fGPU];
    //// //// //// //// //// //// //// //// //// ////

    bool go = cmdOptionExists(argv, argv+argc, "--go");


    // placeholder parameters that will persist if the user doesn't set them

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
    int runs = -1;
    float uncertainty = 0;
    float estimate = 1;
    bool separate = false;          // measure R_d and T_d separately if true

    float n[1] = {nanf("0")};
    float g[1] = {nanf("0")};
    float t[1] = {nanf("0")};

    int N_mu_a = -1;
    float mu_a_start = nanf("0");
    float mu_a_base = nanf("0");

    int N_mu_s = -1;
    float mu_s__start = nanf("0");
    float mu_s__base = nanf("0");

    // example parameters
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
        runs = 10;
        separate = true;

        n[0] = 1.4;
        g[0] = 0.5;
        t[0] = 0.135;

        N_mu_a = 13;
        mu_a_start = 0.5;
        mu_a_base = 1.2;

        N_mu_s = 11;
        mu_s__start = 10;
        mu_s__base = 1.2;

        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nEXAMPLE LOOKUP TABLE SIMULATION\n";
        printf("\nParameters (as found in Cook et al 2019):\n\n-N %d\n--runs %d\n-n %f\n-g %f\n-t %f\n--Nmua %d\n--muaS %f\n--muaB %f\n--Nmus %d\n--musS %f\n--musB %f\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n--separate\n", N, runs, n[0], g[0], t[0], N_mu_a, mu_a_start, mu_a_base, N_mu_s, mu_s__start, mu_s__base, R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI);
        if (!go)
        {
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.";
            getchar();
        }
    }

    // user specified parameters parsed from command line
    else
    {

        if ((cmdOptionExists(argv, argv+argc, "-N") != cmdOptionExists(argv, argv+argc, "--runs")) && !(cmdOptionExists(argv, argv+argc, "--unc")))
        {
            cout << "-N and --runs must be used in conjuction, not by themselves.\n";
            return 2;
        }

        if ((cmdOptionExists(argv, argv+argc, "-N") || cmdOptionExists(argv, argv+argc, "--runs")) && (cmdOptionExists(argv, argv+argc, "--unc") || cmdOptionExists(argv, argv+argc, "--est")))
        {
            cout << "-N and --runs cannot be used with --unc or --est\n";
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

        if (cmdOptionExists(argv, argv+argc, "-N") && cmdOptionExists(argv, argv+argc, "--runs"))
        {
            N = readIntOption(argv, argv+argc, "-N");
            runs = readIntOption(argv, argv+argc, "--runs");
        }
        else if (cmdOptionExists(argv, argv+argc, "--unc"))
        {
            runs = 0;
            if (cmdOptionExists(argv, argv+argc, "--est"))
            {
                estimate = readFloatOption(argv, argv+argc, "--est");
            }
            uncertainty = readFloatOption(argv, argv+argc, "--unc");
            N = (int)ceil(estimate/(uncertainty*uncertainty));
        }
        separate = cmdOptionExists(argv, argv+argc, "--separate");

        n[0] = readFloatOption(argv, argv+argc, "-n");
        g[0] = readFloatOption(argv, argv+argc, "-g");
        t[0] = readFloatOption(argv, argv+argc, "-t");

        N_mu_a = readIntOption(argv, argv+argc, "--Nmua");
        mu_a_start = readFloatOption(argv, argv+argc, "--muaS");
        mu_a_base = readFloatOption(argv, argv+argc, "--muaB");

        N_mu_s = readIntOption(argv, argv+argc, "--Nmus");
        mu_s__start = readFloatOption(argv, argv+argc, "--musS");
        mu_s__base = readFloatOption(argv, argv+argc, "--musB");

        cout << "\n---------------------------------------------------------------------------------------\n";
        cout << "\nParameters:\n";
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
            printf("\n-N %d\n--runs %d", N, runs);
        }
        else
        {
            printf("\n-N %d (calculated)", N);
        }
        printf("\n-n %f\n-g %f\n-t %f\n--Nmua %d\n--muaS %f\n--muaB %f\n--Nmus %d\n--musS %f\n--musB %f\n--Rsphere \"%f %f %f %f\"\n--Rangle %f\n--Tsphere \"%f %f %f %f\"\n--Tangle %f\n", n[0], g[0], t[0], N_mu_a, mu_a_start, mu_a_base, N_mu_s, mu_s__start, mu_s__base, R_pw, R_fs, R_fp, R_fd, R_angle*180/M_PI, T_pw, T_fs, T_fp, T_fd, T_angle*180/M_PI);
        if (!go)
        {
            cout <<"\nPress [enter] to start or Ctrl+C to cancel.\n";
            getchar();
        }
    }


    // calculate the arrays that store the entire space of mua and mus'
    float* mu_a = new float[N_mu_a];
    float* mu_s_= new float[N_mu_s];

    for (int i = 0; i < N_mu_a; i++)
    {
        mu_a[i] = mu_a_start*pow(mu_a_base,i);
    }
    for (int i = 0; i < N_mu_s; i++)
    {
        mu_s_[i] = mu_s__start*pow(mu_s__base,i);
    }

    float* mu_s = new float[N_mu_a];
    for (int i = 0; i < N_mu_s; i++)
    {
        if (g[0] == 1) { mu_s[i] = mu_s_[i]; }
        else { mu_s[i] = mu_s_[i]/(1-g[0]); }
    }

    int layers = 1;

    // define layer boundaries
    float* bounds = new float[layers+1];
    bounds[0] = 0;

    for( int i = 1; i < layers+1; i++ )
    {
        bounds[i] = bounds[i-1] + t[i-1];
    }

    // generate the data for the lookup table
    singleLayerSampleSpace(GPUs, nGPU, N, runs, N_mu_a, N_mu_s, n, g, t, bounds, mu_a, mu_s_, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle, separate);

    return 0;


}
