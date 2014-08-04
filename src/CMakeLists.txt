#####################################################################################
### Define paths to sources
#####################################################################################

set(      EO_SRC_DIR "${CMAKE_SOURCE_DIR}/src/eo"         CACHE INTERNAL "ParadisEO-EO       source directory" FORCE)
set(     EDO_SRC_DIR "${CMAKE_SOURCE_DIR}/src/edo"        CACHE INTERNAL "ParadisEO-EDO      source directory" FORCE)
set(      MO_SRC_DIR "${CMAKE_SOURCE_DIR}/src/mo"         CACHE INTERNAL "ParadisEO-MO       source directory" FORCE)
set(PROBLEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/src/problems"   CACHE INTERNAL "Problems dependant source directory" FORCE)
set(    MOEO_SRC_DIR "${CMAKE_SOURCE_DIR}/src/moeo"       CACHE INTERNAL "ParadisEO-MOEO     source directory" FORCE)
set(     SMP_SRC_DIR "${CMAKE_SOURCE_DIR}/src/smp"        CACHE INTERNAL "ParadisEO-SMP      source directory" FORCE)
set(   EOMPI_SRC_DIR "${CMAKE_SOURCE_DIR}/src/eompi"      CACHE INTERNAL "ParadisEO-EOMPI    source directory" FORCE)
set(EOSERIAL_SRC_DIR "${CMAKE_SOURCE_DIR}/src/eoserial"   CACHE INTERNAL "ParadisEO-EOSERIAL source directory" FORCE)

#####################################################################################
### Include required modules
#####################################################################################

###############
## EO Module ##
###############

# For eo::mpi
enable_language(C)

# For openmp parallel
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  else()
      message( "ERROR: You asked for OpenMP but it has not been found." )
      set(IS_FATAL 1)
  endif(OPENMP_FOUND)
endif(ENABLE_OPENMP)

if(ENABLE_GNUPLOT)
    include(FindGnuplot)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_GNUPLOT -DGNUPLOT_PROGRAM=\\\"${GNUPLOT_EXECUTABLE}\\\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_GNUPLOT -DGNUPLOT_PROGRAM=\\\"${GNUPLOT_EXECUTABLE}\\\"")
endif(ENABLE_GNUPLOT)

# set a special flag if the environment is windows (should do the same in a config.g file)
if (WIN32)
  add_definitions(-D_WINDOWS=1)
endif (WIN32)

################
## EDO Module ##
################

if (EDO)
    if(EDO_USE_LIB STREQUAL "uBLAS")
        find_package(Boost)
        if(Boost_FOUND)
            include_directories( ${Boost_INCLUDE_DIRS} )
            add_definitions( -DWITH_BOOST )
        else()
            message(FATAL_ERROR "\n\nERROR: You asked for Boost::uBLAS but it has not been found.\n" )
        endif()
    elseif(EDO_USE_LIB STREQUAL "Eigen3")
        find_package(Eigen3)
        if(EIGEN3_FOUND)
            include_directories( ${EIGEN3_INCLUDE_DIR} )
            add_definitions( -DWITH_EIGEN )
        else()
            message(FATAL_ERROR  "\n\nERROR: You asked for Eigen3 but it has not been found.\n" )
        endif()

    else()
        # FIXME ideally, we would have a minimal implementation with STL vectors…
        message(FATAL_ERROR  "\n\nYou must set EDO_USE_LIB to either 'uBLAS' or 'Eigen3'.\n" )
    endif()
endif(EDO)

################
## MO Module ##
################



#################
## MOEO Module ##
#################



################
## SMP Module ##
################



#####################
## EOSERIAL Module ##
#####################



##################
## EOMPI Module ##
##################



#####################################################################################
### Include sources
#####################################################################################

## EO Module
set(EO_MODULE_NAME "Evolving Object")
set(CMAKE_SOURCE_DIR ${EO_SRC_DIR})
add_subdirectory(${EO_SRC_DIR})

if(NOT EO_ONLY)

    ## MO Module
    if(MO)
        set(MO_MODULE_NAME "Moving objects")
        #set(CMAKE_SOURCE_DIR ${MO_SRC_DIR})
        add_subdirectory(${MO_SRC_DIR})
        add_subdirectory(${PROBLEMS_SRC_DIR}) # related to mo module
    endif(MO)

    ## EDO Module
    if(EDO)
        set(EDO_MODULE_NAME "Evolving Distribution Objects")
        #set(CMAKE_SOURCE_DIR ${EDO_SRC_DIR})
        add_subdirectory(${EDO_SRC_DIR})
    endif(EDO)

    ## MOEO Module
    if(MOEO)
        set(MOEO_MODULE_NAME "Multi-Objectives EO")
        #set(CMAKE_SOURCE_DIR ${MOEO_SRC_DIR})
        add_subdirectory(${MOEO_SRC_DIR})
    endif(MOEO)

    ## SMP Module
    if(SMP)
        set(SMP_MODULE_NAME "Symmetric Multi-Processing")
        #set(CMAKE_SOURCE_DIR ${SMP_SRC_DIR})
        add_subdirectory(${SMP_SRC_DIR})
    endif(SMP)

    ## EOMPI Module
    if(EOMPI)
        find_package(MPI REQUIRED)
        add_definitions(-DWITH_MPI)
        set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
        set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
        set(EOMPI_MODULE_NAME "EOMPI Module")
        #set(CMAKE_SOURCE_DIR ${EOMPI_SRC_DIR})
        add_subdirectory(${EOMPI_SRC_DIR})
    endif(EOMPI)

    ## EOSERIAL Module
    if(EOSERIAL)
        set(EOSERIAL_MODULE_NAME "Serialization Module")
        #set(CMAKE_SOURCE_DIR ${EOSERIAL_SRC_DIR})
        add_subdirectory(${EOSERIAL_SRC_DIR})
    endif(EOSERIAL)

endif()


