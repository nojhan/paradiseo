
namespace multi_function {
    
double plus(arg_ptr args) {
    return *args[0] + *args[1];
}

double mult(arg_ptr args) {
    return *args[0] * *args[1];
}

double min(arg_ptr args) {
    return -**args;
}

double inv(arg_ptr args) {
    return 1 / **args;
}   

//template <typename f> class F { public: double operator()(double a) { return f(a); } };

double exp(arg_ptr args) {
    return ::exp(**args);
}

} // namespace
