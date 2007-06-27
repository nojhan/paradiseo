INCLUDES = -I ${EO_DIR}/src/ -I ../../../../src/


AM_CXXFLAGS =\
	-Wall\
	-Werror\
	-Wno-deprecated\
	-ansi\
	-pedantic

lib_LIBRARIES  = libtsp.a

libtsp_a_SOURCES =\
	graph.cpp\
        route_init.cpp\
        route_eval.cpp\
        part_route_eval.cpp\
        edge_xover.cpp\
	order_xover.cpp\
        route_valid.cpp\
        partial_mapped_xover.cpp\
        city_swap.cpp\
	two_opt.cpp\
	two_opt_init.cpp\
	two_opt_next.cpp\
	two_opt_incr_eval.cpp\
	two_opt_tabu_list.cpp\
	two_opt_rand.cpp



