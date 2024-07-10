// (c) 2024 Tim Teichmann

#include <math.h>

double f(int n, double *x, void *user_data) {
    double t = x[0];
    double *c = (double *)user_data;
    double c1 = c[0]; // s0
    return pow(cos(t), 3.0)*(1.0 + erf(c1*cos(t)))*exp(pow(c1*cos(t), 2.0));
}
