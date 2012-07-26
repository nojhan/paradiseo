
# Current version
SET(PROJECT_VERSION_MAJOR 1)
SET(PROJECT_VERSION_MINOR 3)
SET(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION_MISC "-edge")

# ADD_DEFINITIONS(-DDEPRECATED_MESSAGES) # disable warning deprecated function messages
# If you plan to use OpenMP, put the following boolean to true :
SET(WITH_OMP FALSE CACHE BOOL "Use OpenMP ?" FORCE)

# If you plan to use MPI, precise here where are the static libraries from
# openmpi and boost::mpi.

SET(WITH_MPI FALSE CACHE BOOL "Use mpi ?" FORCE)
SET(MPI_DIR "/mpi/directory" CACHE PATH "OpenMPI directory" FORCE)

