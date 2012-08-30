#include <stdio.h>
#include <libtcc.h>
#include <math.h>

static TCCState* s = 0;

extern void symc_init() {
    if (s != 0) {
	tcc_delete(s);
    }
    s = tcc_new();
    if (s == 0) {
	fprintf(stderr, "Tiny cc doesn't function properly");
	exit(1);
    }
    
    tcc_set_output_type(s, TCC_OUTPUT_MEMORY);
}

extern int symc_compile(const char* func_str) {
    //printf("Compiling %s\n", func_str);
    int err = tcc_compile_string(s, func_str);

    if (err) {
	fprintf(stderr,"Compile failed");
    }
    return err;
}

extern int symc_link() {
    int err = tcc_relocate(s);
    if (err) {
	fprintf(stderr,"Compile failed");
	exit(1);
    }
    return err;
}

extern void* symc_get_fun(const char* func_name) {
    unsigned long val;
    tcc_get_symbol(s, &val, func_name);
    
    if (val == 0) {
	fprintf(stderr,"getfun failed");
	exit(1);
    }
    
    return (void*) val;
}

extern void* symc_make(const char* func_str, const char* func_name) {
    symc_init();
    symc_compile(func_str);
    symc_link();
    return symc_get_fun(func_name);
}
    
