#include <paradiseo/eo.h>
#include <paradiseo/eo/utils/eoRNG.h>

#include <sstream>  // std::stringstream
#include <assert.h> // assertions are used by serialization tests
#include <vector> 
#include <array> 
#include <time.h> // in order to reseed generator 
#include <Python.h> // statistical tests using Python
#include <numpy/arrayobject.h> // statistical tests using Python 
#include <limits>

void basic_tests()
{
    std::cout << "basic_tests" << std::endl;

    std::cout << "rand():" << rng.rand() << std::endl;
    std::cout << "rand_max():" << rng.rand_max() << std::endl;

    uint_t s = static_cast<uint_t>(time(0));
    rng.reseed(s);

    std::cout << "uniform():" << rng.uniform() << std::endl;
    std::cout << "uniform(2.4):" << rng.uniform(2.4) << std::endl;
    std::cout << "uniform(0.0, 20.0):" << rng.uniform(0.0, 20.0) << std::endl;

    std::cout << "flip:" << rng.flip() << std::endl;
    
    std::cout << "normal():" << rng.normal() << std::endl;
    std::cout << "normal(6.7):" << rng.normal(6.7) << std::endl;
    std::cout << "normal(1000, 0.1):" << rng.normal(1000, 0.1) << std::endl;
    
    std::cout << "negexp:" << rng.negexp(4.7) << std::endl;
    
    int tab[] = {3, 46, 6, 89, 50, 78};
    std::vector<int> v (tab, tab + sizeof(tab) / sizeof(int));
    std::cout << "roulette_wheel(v):" << rng.roulette_wheel(v) << std::endl;
    std::cout << "choice(v):" << rng.choice(v) << std::endl;
}

void test_function(const unsigned N, uint_t(eoRng::*ptr)())
{
    std::stringstream ss;
    rng.rand(); 
    for (unsigned i = 0; i < N; i++)
        (rng.*ptr)();
    ss << rng; // Print eoRNG on stream
    uint_t r1 = rng.rand();
    ss >> rng; // Read eoRNG from stream
    uint_t r2 = rng.rand();    

    assert(r1 == r2);
}

void test_function(const unsigned N, double(eoRng::*ptr)(double), const double m)
{
    std::stringstream ss;
    rng.rand(); 
    for (unsigned i = 0; i < N; i++)
        (rng.*ptr)(m);
    ss << rng; // Print eoRNG on stream
    uint_t r1 = rng.rand();
    ss >> rng; // Read eoRNG from stream
    uint_t r2 = rng.rand();    

    assert(r1 == r2);
}

/** Test the serialization
 *  The serialization allows the user to stop the pseudo-random generator
 *  and save its state in a stream in order to continue the generation 
 *  from this exact state later.
 */
void serialization_tests(const unsigned N, const double d)
{
    std::cout << "serialization_test with N=" << N << " and d=" << d << ":" << std::endl;

    uint_t(eoRng::*ptr_rand)() = &eoRng::rand;
    double(eoRng::*ptr_uniform)(double) = &eoRng::uniform;
    double(eoRng::*ptr_normal)(double) = &eoRng::normal;
    double(eoRng::*ptr_negexp)(double) = &eoRng::negexp;

    uint_t s = static_cast<uint_t>(time(0));

    rng.reseed(s);
    test_function(N, ptr_rand); 
   
    rng.reseed(s); 
    test_function(N, ptr_uniform, double(1.0)); 

    rng.reseed(s);
    test_function(N, ptr_normal, d); 
    
    rng.reseed(s);
    test_function(N, ptr_negexp, d); 

    std::cout << "ok" << std::endl;
}

/** Statistical tests using Python
 * 
 * scipy.stats.shapiro:
 * "Perform the Shapiro-Wilk test for normality".
 * "The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution."
 * See documentation for more details : http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
 * This test is based on the 'normal()' eoRng-implemented method.
 *
 * scipy.stats.chisquare:
 * "Calculates a one-way chi square test."
 * "The chi square test tests the null hypothesis that the categorical data has the given frequencies."
 * See documentation for more details : http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
 * This test is based on the 'uniform()' eoRng-implemented method.
 */
void stat_tests(const unsigned N)
{
    // Initialize the Python interpreter
    Py_Initialize();
    // Create Python objects which will be used later
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs_shapi, *pArgs_csquare, *f_obs, *f_expect, *pX, *pResult = NULL;
    // Convert the module name into a Python string
    pName = PyString_FromString("scipy.stats");
    // Import the Python module
    pModule = PyImport_Import(pName);
    // Import Numpty too 
    pName = PyString_FromString("numpy");
    PyImport_Import(pName);
    // Create a dictionary for the contents of the module
    pDict = PyModule_GetDict(pModule);
    // In order to reseed generator
    uint_t s; 
    // For Python N-arrays constructions 
    import_array();
    npy_intp dims[1] = {N};  

    std::cout << "Python Shapiro-Wilk test based on normal() distribution " << N << " values:" << std::endl;

    // Memory allocation requested (if N too big) 
    double* x = NULL;
    x = (double*) malloc(N*sizeof(double));
    if (x == NULL)
    {
        std::cout << "Memory allocation failed" << std::endl;
        exit(-1);
    }
    
    for (unsigned i = 0; i < N; i++)
    {
        // Important : don't forget to reseed the generator
        s = static_cast<uint_t>(time(0) + i-1); // Chosen method to reseed generator (contestable):
                                                // Add i-1 seconds to the current time based on the current system
        rng.reseed(s);

        x[i] = rng.normal();
    }

    // Create a Python tuple to hold the method arguments
    pArgs_shapi = PyTuple_New(1); 

    pX = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, x);
    PyTuple_SetItem(pArgs_shapi, 0, pX);

    // Get the method from the dictionary
    pFunc = PyDict_GetItemString(pDict, "shapiro"); // scipy.stats.shapiro(x, a=None, reta=False)

    // Call the Python method
    pResult = PyObject_CallObject(pFunc, pArgs_shapi);

    // Print a message if the method call failed
    if (pResult == NULL) 
    {
        std::cout << "Python Shapiro-Wilk test method call has failed. See shapirowilk_test()." << std::endl;
        PyErr_Print();
        exit(-1);
    }
    else
    {
        double W = PyFloat_AsDouble(PyTuple_GetItem(pResult, 0));
        double p_value = PyFloat_AsDouble(PyTuple_GetItem(pResult, 1));
        std::cout << "Shapiro-Wilk test statistic:" << W << std::endl;
        std::cout << "p-value of the Shapiro-Wilk test (strong if < 0.05):" << p_value << std::endl;
        Py_DECREF(pResult);
    }

    // Free allocated memory
    free(x);

    std::cout << "Python chi square test based on uniform() distribution with " << N << " values:" << std::endl;

    uint_t max = rng.rand_max();
    double v_exp = 1.0/max;  

    // Memory allocation requested (if N too big) 
    double* expect_freq = NULL;
    double* obs_freq = NULL;
    
    expect_freq = (double*) malloc(N*sizeof(double));
    if (expect_freq == NULL)
    {
        std::cout << "First memory allocation failed" << std::endl;
        exit(-1);
    }
    obs_freq = (double*) malloc(N*sizeof(double)); 
    if (obs_freq == NULL)
    {
        std::cout << "Second memory allocation failed" << std::endl;
        exit(-1);
    }
    
    for (unsigned i = 0; i < N; i++)
    {
        // // Important : don't forget to reseed the generator
        // s = static_cast<uint_t>(time(0) + i-1); // Chosen method to reseed generator (contestable):
        //                                         // Add i-1 seconds to the current time based on the current system
        // rng.reseed(s);

        // Observed frequencies
        obs_freq[i] = rng.uniform();

        // Expected frequencies
        expect_freq[i] = v_exp;
    }

    f_obs = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, obs_freq);
    f_expect = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, expect_freq);

    // Create Python tuple to hold the method arguments
    pArgs_csquare = PyTuple_New(2); 

    PyTuple_SetItem(pArgs_csquare, 0, f_obs);
    PyTuple_SetItem(pArgs_csquare, 1, f_expect);

    // Get the method from the dictionary
    pFunc = PyDict_GetItemString(pDict, "chisquare"); // scipy.stats.chisquare(f_obs, f_exp=None, ddof=0, axis=0)

    // Call the Python method
    pResult = PyObject_CallObject(pFunc, pArgs_csquare);

    // Print a message if the method call failed
    if (pResult == NULL) // it should be two float numbers (chisq := the chi square test statistic, p := the p-value of the test)
    {
        std::cout << "Python chi square test method call has failed. See chisquare_test()." << std::endl;
        PyErr_Print();
    }
    else
    {
        double chisq = PyFloat_AsDouble(PyTuple_GetItem(pResult, 0));
        double p_value = PyFloat_AsDouble(PyTuple_GetItem(pResult, 1));
        std::cout << "chi square test statistic:" << chisq << std::endl;
        std::cout << "p-value of the chi square test (strong if < 0.05):" << p_value << std::endl;
        Py_DECREF(pResult);
    }

    // Free allocated memory
    free(expect_freq);
    free(obs_freq);

    // Clean up 
    // Desallocation Python function call 
    // Python objects are : PyObject *pName, *pModule, *pDict, *pFunc, *pArgs_shapi, *pArgs_csquare, *f_obs, *f_expect, *pX, *pResult;
    Py_DECREF(pName);
    Py_DECREF(pModule);
    //Py_DECREF(pDict); // Causes a 'Fatal Python error: deletion of interned string failed. Aborted'
    Py_DECREF(pFunc);
    Py_DECREF(pArgs_shapi); // Causes a 'Segmentation fault' error. 
    Py_DECREF(pArgs_csquare);
    //Py_DECREF(pX); // Causes a 'Segmentation fault' error. 
    //Py_DECREF(f_obs); // Causes a 'Segmentation fault' error. 
    //Py_DECREF(f_expect); // Causes a 'Segmentation fault' error. 

    // Destroy the Python interpreter
    Py_Finalize(); 
}

int main()
{
    basic_tests();

    rng.reseed(static_cast<uint_t>(time(0)));
    constexpr double min = std::numeric_limits<double>::min();
    constexpr double max = std::numeric_limits<double>::max();
    double d = rng.uniform(min, max);
    serialization_tests(635, d); // Reminder:
                                 // Mersenne Twister pseudo-random generator of 32-bits numbers with a state of 19937 bits
                                 // 624 elements in the state sequence
                                 // Mersenne Twister pseudo-random generator of 64-bits numbers with a state of 19937 bits
                                 // 312 elements in the state sequence

    stat_tests(2500); // Warning: limit for N (Shapiro-Wilk test)
                      // if N > 5000: warnings.warn("p-value may not be accurate for N > 5000.")
                      // See https://github.com/scipy/scipy/blob/master/scipy/stats/morestats.py for more details.
                      // Personal warning with API C-Python : causes valgrind errors 

    return 0;
}