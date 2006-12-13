AM_CXXFLAGS =\
	 -Wall 

SUBDIRS=share

INCLUDES = -I${EO_DIR}/src/ -I../../src/ 

bin_PROGRAMS = hill_climb simul_anneal tabu_search gen_algo mimetic

# Simple Hill Climbing
hill_climb_SOURCES = \
	hill_climb.cpp

hill_climb_LDFLAGS = 

hill_climb_LDADD = share/libtsp_share.a ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a 

# Simple Simulated Annealing
simul_anneal_SOURCES = \
	simul_anneal.cpp

simul_anneal_LDFLAGS = 

simul_anneal_LDADD = share/libtsp_share.a ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a 

# Simple Tabu Search
tabu_search_SOURCES = \
	tabu_search.cpp

tabu_search_LDFLAGS = 

tabu_search_LDADD = share/libtsp_share.a ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a 

# Simple Genetic Algorithm
gen_algo_SOURCES = \
	gen_algo.cpp

gen_algo_LDFLAGS = 

gen_algo_LDADD = share/libtsp_share.a ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a 

# Simple Mimetic Algorithm
mimetic_SOURCES = \
	mimetic.cpp

mimetic_LDFLAGS = 

mimetic_LDADD = share/libtsp_share.a ${EO_DIR}/src/libeo.a ${EO_DIR}/src/utils/libeoutils.a 

