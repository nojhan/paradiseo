CXX=mpicxx

INCLUDES = \
	`xml2-config --cflags` \
	-I$(EO_DIR)/src 

AM_CXXFLAGS =\
	-g \
	-Wall

lib_LIBRARIES = librmc_mpi.a

librmc_mpi_a_SOURCES = \
	node.h \
	node.cpp \
	param.h \
	param.cpp \
	comm.h \
	comm.cpp \
	coop.h \
	coop.cpp \
	mess.h \
	mess.cpp \
	rmc.cpp \
	scheduler.h \
	scheduler.cpp \
	worker.h \
	worker.cpp \
	send.h \
	send.cpp \
	recv.h \
	recv.cpp \
	xml_parser.h \
	xml_parser.cpp \
	schema.h \
	schema.cpp \
	runner.cpp \
	service.h \
	service.cpp
