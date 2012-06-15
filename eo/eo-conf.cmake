
# Current version
SET(PROJECT_VERSION_MAJOR 1)
SET(PROJECT_VERSION_MINOR 3)
SET(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION_MISC "-edge")

# If you plan to use MPI, precise here where are the static libraries from
# openmpi and boost::mpi.

SET(WITH_MPI FALSE CACHE BOOL "Use mpi ?" FORCE)
SET(MPI_DIR "put root directory of openmpi here" CACHE PATH "OpenMPI directory" FORCE)
SET(BOOST_DIR "put root directory of boost here" CACHE PATH "Boost directory" FORCE)

