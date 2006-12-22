CXX=mpicxx

INCLUDES = \
	 -I$(EO_DIR)/src/ \
	 -I$(MO_DIR)/src/ \
	 -I$(MOEO_DIR)/src/ \
	 -I../../src/ 

AM_CXXFLAGS = \
	 -Wall -g

lib_LIBRARIES = \
	libtsp_shared.a

libtsp_shared_a_SOURCES = \
	data.h \
	data.cpp \
	opt_route.h \
	opt_route.cpp \
	param.h \
	param.cpp \
	node.h \
	node.cpp \
	route.h \
	route.cpp \
	route_init.h \
	route_init.cpp \
	route_eval.h \
	route_eval.cpp \
	two_opt.h \
	two_opt.cpp \
	two_opt_init.h \
	two_opt_init.cpp \
	two_opt_incr_eval.h \
	two_opt_incr_eval.cpp \
	two_opt_next.h \
	two_opt_next.cpp \
	order_xover.h \
	order_xover.cpp \
	partial_mapped_xover.h \
	partial_mapped_xover.cpp \
	edge_xover.h \
	edge_xover.cpp \
	city_swap.h \
	city_swap.cpp \
	part_route_eval.h \
	part_route_eval.cpp \
	merge_route_eval.h \
	merge_route_eval.cpp 
##

#tspExample_LDFLAGS = \
#	-L$(EO_DIR)/src/ \
#	-L$(EO_DIR)/src/utils \
#	-L../src/ \
#	-L../src/rmc/mpi/ \
#	`xml2-config --libs` 

#tspExample_LDADD = -lparadiseo -lrmc_mpi -leoutils -leo