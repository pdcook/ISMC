#ifndef __ISMLMC__
#define __ISMLMC__

#include <thrust/tuple.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
//#include <stdio.h>
#include <thread>

#define RNGS 20000000 // total number of RNGS before repeat, make big, but small enough to fit on each device
#define _USE_MATH_DEFINES
#define INF 2000000000 // a really big number ~2 billion is the max for int
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }    // inline function used for returning errors that occur in __device__ code since they cannot be seen otherwise

using namespace std;

// global boolean to see if the random number generator has been initialized on each device
extern bool* rand_set;
bool* rand_set;

// global random number generator states for each device
extern curandState** globalDeviceStates;
curandState** globalDeviceStates;

// prints the basic properties of the currently selected gpu
void CUDABasicProperties(int device_id)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    fprintf(stderr, "GPU %d: %s\n", device_id, prop.name);
    double mem = (double)prop.totalGlobalMem/1000000000;
    fprintf(stderr, "MEM: %f GB\n", mem);
    double freq = (double)prop.clockRate/1000000;
    fprintf(stderr, "CLOCK: %f GHZ\n", freq);
    fprintf(stderr, "Compute: %d.%d\n", prop.major, prop.minor);
}

// inline function used for returning errors that occur in __device__ code since they cannot be seen otherwise
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// function for initializing CUDA random number generator for each thread on the device
__global__ void initialize_curand_on_kernels(curandState * state, unsigned long seed)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// function for generating random numbers in CUDA device code
__device__ float RND(curandState* globalState, int ind)
{
    //copy state to local mem
    curandState localState = globalState[ind];
    //apply uniform distribution with calculated random
    float rndval = curand_uniform( &localState );
    //update state
    globalState[ind] = localState;
    //return value
    return rndval;
}

// Snell's law
__device__ thrust::tuple<float,float,float> multilayer_snell(float ni, float nt, float mu_x, float mu_y, float mu_z)
{
    float ti = acosf(abs(mu_z));
    float tt = asinf((ni*sinf(ti))/nt);
    return thrust::make_tuple(mu_x*ni/nt, mu_y*ni/nt, (mu_z/abs(mu_z))*cosf(tt));
}

// function used for determining reflection/transmission
__device__ float fresnel_snell(float ni, float nt, float mu_z)
{
    float ti, tt, R;

    if (abs(mu_z) > 0.99999){ R = pow(((nt-ni)/(nt+ni)),2); }
    else
    {
        ti = acosf(abs(mu_z));
        // if ni*sinf(ti)/nt >=1 then total internal reflection occurs, and thus R = 1
        if ( (ni*sinf(ti))/nt >=1. ) { R= 1.; }
        else
        {
            tt = asinf((ni*sinf(ti))/nt);
            R = 0.5 * ( (pow(sinf(ti-tt),2))/(pow(sinf(ti+tt),2)) + (pow(tan(ti-tt),2))/(pow(tan(ti+tt),2)) );
        }
    }
    return R;
}

// determine if a photon incident on the sample begins to propagate or is reflected
__device__ bool incident_reflection(int idx, curandState* globalState, float mu_z, float* n, float* mu_s, int layer)
{
    // check if the incident layer is glass
    if (mu_s[layer] == 0.0)
    {
        float n1,n2,n3,r1,r2;

        n1 = 1.;
        n2 = n[layer];
        if (mu_z > 0) { n3 = n[layer+1]; }
        else { n3 = n[layer-1]; }

        r1 = fresnel_snell(n1, n2, mu_z);
        r2 = fresnel_snell(n2, n3, mu_z);

        return (RND(globalState, idx) < r1 + (pow((1-r1),2))*r2/(1-r1*r2));

    }

    else
    {
        return (RND(globalState, idx) < fresnel_snell(1., n[layer], mu_z));
    }
}

// if a photon is inside a sphere, this function determines if it is re-incident on the sample
// returns True if re-incident and False if not
__device__ bool reincidence(int idx, curandState* globalState, float pw, float fs, float f)
{
        return (RND(globalState, idx) < (pw*fs)/(1-pw*(1-f)));
}

// single photon propagator
__device__ thrust::tuple<float,float,float,float,float,float> MLMC(int idx, curandState* globalState, int layers, float* n, float* mu_a, float* mu_s, float* mu_t, float* g, float* t, float* bounds, float w, float x, float y, float z, float mu_x, float mu_y, float mu_z)
{

    // set up initial values

    float Absorbed = 0;
    float Reflected = 0;
    float Transmitted = 0;
    float threshold = 0.0001;
    int m = 10;
    int layer;
    int nextlayer;
    float s, d, deltaW, cos_theta, sin_theta, phi, cos_phi, sin_phi, mu_x_, mu_y_, mu_z_, z_sqrt;

    // define starting layer
    if (mu_z > 0){ layer = 0; }
    else { layer = layers - 1; }

    // propagate while w > 0

    while (w > 0)
    {
        // draw a stepsize if the photon isn't in glass
        if (mu_s[layer] != 0)
        {
            s = -logf(RND(globalState, idx))/mu_t[layer];
        }

        if (mu_z < 0)
        {
            d = (bounds[layer] - z)/mu_z;
            nextlayer = layer - 1;
        }
        else if (mu_z > 0)
        {
            d = (bounds[layer+1] - z)/mu_z;
            nextlayer = layer + 1;
        }
        else if (mu_z == 0)
        {
            d = INF;
            nextlayer = layer;
        }

        // move the photon directly to the next boundary if in glass
        if (mu_s[layer] == 0){s = d;}

        // boundary conditions
        while ( s >= d )
        {

            x += d*mu_x;
            y += d*mu_y;
            z += d*mu_z;
            s -= d;

            if (nextlayer == layers)
            {
                if ( RND(globalState, idx) < fresnel_snell(n[layer],1.,mu_z) )
                {
                    // internal reflection
                    mu_z *= -1;
                }
                else                   // photon is transmitted
                {
                    Transmitted += w;
                    // refraction via Snell's Law
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z);
                    w = 0;
                    break;
                }
            }

            else if (nextlayer == -1)                 // photon attempts to reflect/backscatter
            {
                if ( RND(globalState, idx) < fresnel_snell(n[layer], 1., mu_z))
                {
                    // photon is internally reflected
                    mu_z *= -1;
                }
                else                   // photon backscatters
                {
                    Reflected += w;
                    // refraction via Snell's Law
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z);
                    w = 0;
                    break;
                }
            }
            else
            {
                if (RND(globalState, idx) < fresnel_snell(n[layer], n[nextlayer], mu_z))
                {
                    mu_z *= -1;
                }
                else
                {
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], n[nextlayer], mu_x, mu_y, mu_z);
                    if (mu_s[nextlayer] != 0 && s != 0)
                    {
                        s *= mu_t[layer]/mu_t[nextlayer];
                    }
                    layer = nextlayer;
                }
            }

            if (mu_z < 0)
            {
                d = (bounds[layer] - z)/mu_z;
                nextlayer = layer - 1;
            }
            else if (mu_z > 0)
            {
                d = (bounds[layer+1] - z)/mu_z;
                nextlayer = layer + 1;
            }
            else if (mu_z == 0)
            {
                d = INF;
                nextlayer = layer;
            }

            // if the photon is in glass, move it to the boundary
            if (mu_s[layer] == 0){ s = d; }

        }

        x += s*mu_x;     //
        y += s*mu_y;     // Hop
        z += s*mu_z;     //

        // if not in glass
        if ( mu_s[layer] != 0 )
        {
            // partial absorption event
            deltaW = w*mu_a[layer]/mu_t[layer];
            w -= deltaW;
            Absorbed += deltaW;
        }

        // roullette
        if (w <= threshold)
        {
            if (RND(globalState, idx) <= 1/m){ w*=m; }
            else { w = 0; }
        }

        // scattering event: update the photon's direction cosines only if it's weight isn't 0
        // and it isn't in glass
        /// Spin ///
        if ( w > 0 && mu_s[layer]!=0 )
        {
            if (g[layer] == 0.){ cos_theta = 2*RND(globalState, idx) - 1;}
            else { cos_theta = (1/(2*g[layer]))*(1+g[layer]*g[layer]-pow(((1-g[layer]*g[layer])/(1-g[layer]+2*g[layer]*RND(globalState, idx))),2)); }

            phi = 2 * M_PI * RND(globalState, idx);
            cos_phi = cosf(phi);
            sin_phi = sinf(phi);
            sin_theta = sqrt(1. - pow(cos_theta,2));

            if (abs(mu_z) > 0.99999)
            {
                mu_x_ = sin_theta*cos_phi;
                mu_y_ = sin_theta*sin_phi;
                mu_z_ = (mu_z/abs(mu_z))*cos_theta;
            }
            else
            {
                z_sqrt = sqrt(1 - mu_z*mu_z);
                mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta;
                mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta;
                mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta;
            }

            mu_x = mu_x_;
            mu_y = mu_y_;
            mu_z = mu_z_;
        }
    }

    return thrust::make_tuple(Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z);

}

__global__ void CUDAISMLMC(curandState* globalState, float* A, float* R_diffuse, float* R_specular, float* T_diffuse, float* T_direct, int N, int layers, float* n, float* g, float* t, float* bounds, float* mu_a, float* mu_s, float* mu_t, bool R, float R_pw, float R_fs, float R_fp, float R_fd, float R_f, float R_angle, bool specular_included, bool T, float T_pw, float T_fs, float T_fp, float T_fd, float T_f, float T_angle)
{

    int idx = blockIdx.x*blockDim.x+threadIdx.x;        // unique thread identifier used in the CUDA random number generator

    int step = blockDim.x * gridDim.x;                  // step size so each thread knows which photons it is responsible for

    float Absorbed  = 0;
    float Reflected = 0;
    float Transmitted = 0;

    float w, x, y, z, mu_x, mu_y, mu_z, incident_reflect, phi, fp;
    int layer;
    bool sample_side;


    float t_tot = 0;
    for (int i = 0; i < layers; i++)
    {
        t_tot += t[i];
    }

    for (int i = idx; i < N; i += step)
    {
        w = 1;      // initial weight of photon

        x = 0;
        y = 0;      // initial position of photon
        z = 0;

        // determine inital direction for photon
        if (specular_included)
        {
            mu_x = sqrt(1 - 0.99*0.99);
            mu_y = 0;
            mu_z = 0.99;
        }
        else
        {
            mu_x = 0;
            mu_y = 0;
            mu_z = 1;
        }

        // keep track of what sphere the photon is in
        // true for reflection sphere, false for transmission sphere
        sample_side = true;

        while (w > 0)
        {
            // determine if the photon is incidently reflected from the sample
            if ( mu_z > 0 ){ layer = 0; }
            else { layer = layers - 1; }
            incident_reflect = incident_reflection(idx, globalState, mu_z, n, mu_s, layer);

            // check if the photon was incidently reflected and where
            if (incident_reflect && sample_side == true)
            {
                // the photon is incidently reflected off of the 'top' surface

                // check if a reflection sphere is present
                if (R)   // reflection sphere present
                {
                    // check to see if the photon leaves directly through the source port
                    if ( abs(mu_z) >= sqrt(1-R_fp) )
                    {
                        // score the photon as specular reflection and stop propagating
                        atomicAdd(R_specular, w);
                        w = 0;
                        break;
                    }
                    // if the photon doesn't leave directly through the source port,
                    // check to see if it is re-incident on the sample
                    else if (reincidence(idx, globalState, R_pw,R_fs,R_f))
                    {
                        // photon is re-incident on the sample; sample a random angle of
                        // reincidence, reset its position, and continue propagating
                        mu_z    = RND(globalState, idx);
                        phi     = 2*M_PI*RND(globalState, idx);
                        mu_x    = cosf(phi)*sqrt(1-mu_z*mu_z);
                        mu_y    = sinf(phi)*sqrt(1-mu_z*mu_z);
                        x = 0;
                        y = 0;
                        z = 0;
                        continue;
                    }

                    else
                    {
                        // the photon neither leaves through the source port, nor is re-incident
                        // score the photon as diffuse reflection and stop propagating
                        atomicAdd(R_diffuse, w);
                        w = 0;
                        break;
                    }
                }

                else
                {   // there is no reflection sphere
                    // calculate an effective source port fraction from the angle threshold
                    fp = 0.5*(1-cosf(2*R_angle));

                    // check if the photon leaves in a direction within
                    // this effective source port fraction
                    if ( abs(mu_z) >= sqrt(1-fp) ) // photon is within angle threshold, score as
                    {
                        atomicAdd(R_specular, w);         // specular reflection and stop propagating
                        w = 0;
                        break;
                    }

                    else
                    {       // photon is NOT within angle threshold, since there is no sphere
                            // present, score it as diffuse reflection and stop propagating
                        atomicAdd(R_diffuse, w);
                        w = 0;
                        break;
                    }
                }
            }

            else if (incident_reflect && sample_side == false)
            {
                // the photon is incidently reflected off of the 'bottom' surface

                // check if a transmission sphere is present
                if (T)
                {   // transmission sphere present
                    // check to see if the photon leaves directly through the optional port
                    if (abs(mu_z) >= sqrt(1-T_fp))
                    {
                        // score the photon as direct transmission and stop propagating
                        atomicAdd(T_direct, w);
                        w = 0;
                        break;
                    }
                    // if the photon doesn't leave directly through the optional port,
                    // check to see if it is re-incident on the sample
                    else if (reincidence(idx, globalState, T_pw,T_fs,T_f))
                    {
                        // photon is re-incident on the sample; sample a random angle of
                        // reincidence, reset its position, and continue propagating
                        mu_z    = -1*RND(globalState, idx);
                        phi     = 2*M_PI*RND(globalState, idx);
                        mu_x    = cosf(phi)*sqrt(1-mu_z*mu_z);
                        mu_y    = sinf(phi)*sqrt(1-mu_z*mu_z);
                        x = 0;
                        y = 0;
                        z = t_tot;
                        continue;
                    }
                    else
                    {
                        // the photon neither leaves through the optional port, nor is re-incident
                        // score the photon as diffuse transmission and stop propagating
                        atomicAdd(T_diffuse, w);
                        w = 0;
                        break;
                    }
                }
                else
                {   // there is no transmission sphere
                    // calculate an effective optional port fraction from the angle threshold
                    fp = 0.5*(1-cosf(2*T_angle));

                    // check if the photon leaves in a direction within
                    // this effective optional port fraction
                    if (abs(mu_z) >= sqrt(1-fp)) // photon is within angle threshold, score as
                    {
                        atomicAdd(T_direct, w);           // direct transmission and stop propagating
                        w = 0;
                        break;
                    }
                    else
                    {       // photon is NOT within angle threshold, since there is no sphere
                            // present, score it as diffuse transmission and stop propagating
                        atomicAdd(T_diffuse, w);
                        w = 0;
                        break;
                    }
                }
            }

            else
            {   // the photon is not incidently reflected and may begin propagating

                // incident refraction by Snell's Law
                thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(1., n[layer], mu_x, mu_y, mu_z);

                // Monte Carlo Photon Transport
                thrust::tie(Absorbed, Reflected, Transmitted, mu_x, mu_y, mu_z) = MLMC(idx, globalState, layers, n, mu_a, mu_s, mu_t, g, t, bounds, w, x, y, z, mu_x, mu_y, mu_z);

                // partial absorption
                atomicAdd(A, Absorbed);

                // check to see what happened to the photon
                if (Reflected)   // the photon was reflected/backscattered
                {
                    w = Reflected;
                    sample_side = true;

                    // check if a reflection sphere is present
                    if (R)
                    {   // reflection sphere present
                        // check to see if the photon leaves directly through the source port
                        if (abs(mu_z) >= sqrt(1-R_fp))
                        {
                            // score the photon as specular reflection and stop propagating
                            atomicAdd(R_specular, w);
                            w = 0;
                            break;
                        }
                        // if the photon doesn't leave directly through the
                        // source port, check to see if it is re-incident on the sample
                        else if (reincidence(idx, globalState, R_pw,R_fs,R_f))
                        {
                            // photon is re-incident on the sample; sample a random angle of
                            // reincidence, reset its position, and continue propagating
                            mu_z    = RND(globalState, idx);
                            phi     = 2*M_PI*RND(globalState, idx);
                            mu_x    = cosf(phi)*sqrt(1-mu_z*mu_z);
                            mu_y    = sinf(phi)*sqrt(1-mu_z*mu_z);
                            x = 0;
                            y = 0;
                            z = 0;
                            continue;
                        }
                        else
                        {       // the photon neither leaves through the source port, nor is
                                // re-incident, score the photon as diffuse reflection
                                // and stop propagating
                            atomicAdd(R_diffuse, w);
                            w = 0;
                            break;
                        }
                    }
                    else
                    {   // there is no reflection sphere
                        // calculate an effective source port fraction from the angle threshold
                        fp = 0.5*(1-cosf(2*R_angle));

                        // check if the photon leaves in a direction within
                        // this effective source port fraction
                        if (abs(mu_z) >= sqrt(1-fp))
                        {
                            // photon is within angle threshold, score as
                            // specular reflection and stop propagating
                            atomicAdd(R_specular, w);
                            w = 0;
                            break;
                        }
                        else
                        {       // photon is NOT within angle threshold, since there is no sphere
                                // present, score it as diffuse reflection and stop propagating
                            atomicAdd(R_diffuse, w);
                            w = 0;
                            break;
                        }
                    }
                }

                else if (Transmitted)
                {   // the photon transmitted through the sample
                    w = Transmitted;
                    sample_side = false;

                    // check if a transmission sphere is present
                    if (T)
                    {   // transmission sphere present
                        // check to see if the photon leaves directly through the optional port
                        if (abs(mu_z) >= sqrt(1-T_fp))
                        {
                            // score the photon as direct transmission and stop propagating
                            atomicAdd(T_direct, w);
                            w = 0;
                            break;
                        }
                        // if the photon doesn't leave directly through the
                        // optional port, check to see if it is re-incident on the sample
                        else if (reincidence(idx, globalState, T_pw,T_fs,T_f))
                        {
                            // photon is re-incident on the sample; sample a random angle of
                            // reincidence, reset its position, and continue propagating
                            mu_z    = -1*RND(globalState, idx);
                            phi     = 2*M_PI*RND(globalState, idx);
                            mu_x    = cosf(phi)*sqrt(1-mu_z*mu_z);
                            mu_y    = sinf(phi)*sqrt(1-mu_z*mu_z);
                            x = 0;
                            y = 0;
                            z = t_tot;
                            continue;
                        }
                        else
                        {       // the photon neither leaves through the optional port,
                                // nor is re-incident, score the photon as diffuse
                                // transmission and stop propagating
                            atomicAdd(T_diffuse, w);
                            w = 0;
                            break;
                        }
                    }
                    else
                    {   // there is no transmission sphere
                        // calculate an effective optional port
                        // fraction from the angle threshold
                        fp = 0.5*(1-cosf(2*T_angle));

                        // check if the photon leaves in a direction within this
                        // effective optional port fraction
                        if (abs(mu_z) >= sqrt(1-fp))
                        {
                            // photon is within angle threshold, score as direct
                            // transmission and stop propagating
                            atomicAdd(T_direct, w);
                            w = 0;
                            break;
                        }
                        else
                        {       // photon is NOT within angle threshold, since there is no sphere
                                // present, score it as diffuse transmission and stop propagating
                            atomicAdd(T_diffuse, w);
                            w = 0;
                            break;
                        }
                    }
                }

                else  // the photon was wholly absorbed
                {
                    w = 0;
                    break;
                }
            }
        }
    }

    // convert values to percents and return them
    //return thrust::make_tuple(A/N, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N);
}

thrust::tuple<float,float,float,float,float> ISMLMC(int dev, int N, int layers, float* n, float* g, float* t, float* bounds, float* mu_a, float* mu_s, float* mu_t, bool R, float R_pw, float R_fs, float R_fp, float R_fd, float R_f, float R_angle, bool specular_included, bool T, float T_pw, float T_fs, float T_fp, float T_fd, float T_f, float T_angle)
{

    // dictate which GPU to run on
    cudaSetDevice(dev);

    int threadsPerBlock = 256;              // number of threads per block, internet claims that 256 gives best performance...
    int nBlocks = N/threadsPerBlock + 1;    // number of blocks is always the number of photons divided by the number of threads per block then rounded up (+1). so that each GPU thread gets an equal number of photons while still covering every photon

    // only seed the random number generator once per device
    // rand_set must be initialized to false for all devices in the main function
    // of whatever file uses this function
    // Moreover, globalDeviceStates must be initialized in that same main function
    // for all devices
    if (!rand_set[dev])
    {
        //alocate space for each kernels curandState which is used in the CUDA random number generator
        cudaMalloc(&globalDeviceStates[dev], RNGS*sizeof(curandState));

        //call curand_init on each kernel with the same random seed
        //and init the rng states
        initialize_curand_on_kernels<<<nBlocks,threadsPerBlock>>>(globalDeviceStates[dev], unsigned(time(NULL)));

        gpuErrchk( cudaPeekAtLastError() );     // this prints out the last error encountered in the __device__ code (if there was one)
        gpuErrchk( cudaDeviceSynchronize() );   // this waits for the GPU to finish before continuing while also printing out any errors that are encountered in __device__ code

        rand_set[dev] = true;
    }

    curandState* deviceStates = globalDeviceStates[dev];

    // initialize measurement values
    float A = 0;
    float R_d = 0;
    float R_s = 0;
    float T_d = 0;
    float T_u = 0;

    // copy measurement values to GPU
    float* dev_A;
    cudaMalloc(&dev_A, sizeof(float));
    cudaMemcpy(dev_A, &A, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_R_d;
    cudaMalloc(&dev_R_d, sizeof(float));
    cudaMemcpy(dev_R_d, &R_d, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_R_s;
    cudaMalloc(&dev_R_s, sizeof(float));
    cudaMemcpy(dev_R_s, &R_s, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_T_d;
    cudaMalloc(&dev_T_d, sizeof(float));
    cudaMemcpy(dev_T_d, &T_d, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_T_u;
    cudaMalloc(&dev_T_u, sizeof(float));
    cudaMemcpy(dev_T_u, &T_u, sizeof(float), cudaMemcpyHostToDevice);

    // copy sample properties to GPU
    float* dev_n;
    cudaMalloc(&dev_n, layers*sizeof(float));
    cudaMemcpy(dev_n, n, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_g;
    cudaMalloc(&dev_g, layers*sizeof(float));
    cudaMemcpy(dev_g, g, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_t;
    cudaMalloc(&dev_t, layers*sizeof(float));
    cudaMemcpy(dev_t, t, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_bounds;
    cudaMalloc(&dev_bounds, (layers+1)*sizeof(float));
    cudaMemcpy(dev_bounds, bounds, (layers+1)*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_a;
    cudaMalloc(&dev_mu_a, layers*sizeof(float));
    cudaMemcpy(dev_mu_a, mu_a, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_s;
    cudaMalloc(&dev_mu_s, layers*sizeof(float));
    cudaMemcpy(dev_mu_s, mu_s, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_t;
    cudaMalloc(&dev_mu_t, layers*sizeof(float));
    cudaMemcpy(dev_mu_t, mu_t, layers*sizeof(float), cudaMemcpyHostToDevice);

    // run the simulation on the GPU
    CUDAISMLMC<<<nBlocks,threadsPerBlock>>>(deviceStates, dev_A, dev_R_d, dev_R_s, dev_T_d, dev_T_u, N, layers, dev_n, dev_g, dev_t, dev_bounds, dev_mu_a, dev_mu_s, dev_mu_t, R, R_pw, R_fs, R_fp, R_fd, R_f, R_angle, specular_included, T, T_pw, T_fs, T_fp, T_fd, T_f, T_angle);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // copy the results back
    cudaMemcpy(&A, dev_A, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&R_d, dev_R_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&R_s, dev_R_s, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T_d, dev_T_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T_u, dev_T_u, sizeof(float), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // free GPU memory
    cudaFree(dev_A); cudaFree(dev_R_d); cudaFree(dev_R_s); cudaFree(dev_T_d); cudaFree(dev_T_u); cudaFree(dev_n); cudaFree(dev_g); cudaFree(dev_t); cudaFree(dev_bounds); cudaFree(dev_mu_a); cudaFree(dev_mu_s); cudaFree(dev_mu_t);


    return thrust::make_tuple(A/N, R_d/N, R_s/N, T_d/N, T_u/N);
}

#endif
