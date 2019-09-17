#ifndef __cmdlineparse__
#define __cmdlineparse__

#include <iostream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>

using namespace std;

// blatantly copied from stack overflow

// each of these functions takes a pointer to argv for begin, a pointer to the end of argv (argv+argc) for the end, and the option to parse, for example: "--flag"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char** itr = find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return find(begin, end, option) != end;
}

float readFloatOption(char** begin, char** end, const string& option)
{
    if (cmdOptionExists(begin, end, option))
    {
        float val;
        char* char_val = getCmdOption(begin, end, option);
        val = (float)atof(char_val);
        return val;
    }
    else
    {
        return nanf("0");
    }
}

int readIntOption(char** begin, char** end, const string& option)
{
    if (cmdOptionExists(begin, end, option))
    {
        int val;
        char* char_val = getCmdOption(begin, end, option);
        val = (int)atoi(char_val);
        return val;
    }
    else
    {
        return -1;
    }
}

string readStrOption(char** begin, char** end, const string& option)
{
    char* char_vals = getCmdOption(begin, end, option);
    string str_vals(char_vals);

    return str_vals;
}

float* readArrOption(char** begin, char** end, const string& option, int len)
{
    float* vals = new float[len];
    char* char_vals = getCmdOption(begin, end, option);
    string str_vals(char_vals);
    stringstream stream_vals(str_vals);
    float temp;
    for (int i = 0;  stream_vals >> temp; i++)
    {
        vals[i] = temp;
    }
    return vals;
}

#endif
