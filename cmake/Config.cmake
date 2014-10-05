######################################################################################

# Inspired by Boost and SFML CMake files
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(MACOSX 1)

    # detect OS X version. (use '/usr/bin/sw_vers -productVersion' to extract V from '10.V.x'.)
    execute_process (COMMAND /usr/bin/sw_vers -productVersion OUTPUT_VARIABLE MACOSX_VERSION_RAW)
    string(REGEX REPLACE "10\\.([0-9]).*" "\\1" MACOSX_VERSION "${MACOSX_VERSION_RAW}")
    if(${MACOSX_VERSION} LESS 5)
        message(FATAL_ERROR "Unsupported version of OS X : ${MACOSX_VERSION_RAW}")
        return()
    endif()
endif()

# Determine architecture
include(CheckTypeSize)
check_type_size(void* SIZEOF_VOID_PTR)
if("${SIZEOF_VOID_PTR}" STREQUAL "4")
    set(ARCH x86)
    set(LIB lib32)
elseif("${SIZEOF_VOID_PTR}" STREQUAL "8")
    set(ARCH x86_64)
    set(LIB lib64)
else()
    message(FATAL_ERROR "Unsupported architecture")
    return()
endif()

######################################################################################
### 0) Define general CXX flags for DEBUG and RELEASE
######################################################################################

add_definitions(-DDEPRECATED_MESSAGES)
set(CMAKE_CXX_FLAGS_DEBUG  "-Wunknown-pragmas -O0 -g -Wall -Wextra -ansi -pedantic" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE  "-Wunknown-pragmas -O2" CACHE STRING "" FORCE)

if(SMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -pthread" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -pthread" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -pthread" CACHE STRING "" FORCE)
    add_definitions(-D_GLIBCXX_USE_NANOSLEEP)
endif(SMP)

######################################################################################
### 1) Define installation type
######################################################################################

if(INSTALL_TYPE STREQUAL full)
    set(ENABLE_CMAKE_EXAMPLE "true" CACHE BOOL "ParadisEO examples")
    set(ENABLE_CMAKE_TESTING "true" CACHE BOOL "ParadisEO tests")
elseif(INSTALL_TYPE STREQUAL min OR NOT DEFINED INSTALL_TYPE)
    set(ENABLE_CMAKE_EXAMPLE "false" CACHE BOOL "ParadisEO examples")
    set(ENABLE_CMAKE_TESTING "false" CACHE BOOL "ParadisEO tests")
endif()

######################################################################################
### 2) Define profiling flags
######################################################################################

if(PROFILING)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg --coverage" CACHE STRING "" FORCE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
        set(ENABLE_CMAKE_TESTING "true" CACHE BOOL "" FORCE)
endif(PROFILING)

######################################################################################
### 3) Testing part
######################################################################################
 
if(ENABLE_CMAKE_TESTING)
    enable_testing()
    include(CTest REQUIRED)
endif(ENABLE_CMAKE_TESTING)

# required by t-eoParserBoost.cpp
find_package( Boost 1.36.0 )
if(Boost_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -L ${Boost_INCLUDE_DIRS} -lboost_program_options -Wreorder")

    if (UNIX OR CMAKE_COMPILER_IS_GNUCXX) 
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11") 
    endif (UNIX OR CMAKE_COMPILER_IS_GNUCXX)
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7")
    endif (APPLE) 
endif(Boost_FOUND)

######################################################################################
### 4) Build examples ?
######################################################################################

set(ENABLE_CMAKE_EXAMPLE "true" CACHE BOOL "ParadisEO examples")

######################################################################################
### 5) Random numbers
######################################################################################

set(ENABLE_CXX11_RANDOM "false" CACHE BOOL "For C++11 random numbers")
set(ENABLE_64_BIT_RNG_NUMBERS "false" CACHE BOOL "For 64-bit random numbers")

# For C++11 random numbers
if(ENABLE_CXX11_RANDOM)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_RANDOM") 

    if (UNIX OR CMAKE_COMPILER_IS_GNUCXX) 
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11") 
    endif (UNIX OR CMAKE_COMPILER_IS_GNUCXX)
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.7")
    endif (APPLE)

    # For 64-bit random numbers
    if(ENABLE_64_BIT_RNG_NUMBERS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_64_BIT_RNG_NUMBERS")
    endif(ENABLE_64_BIT_RNG_NUMBERS)

endif(ENABLE_CXX11_RANDOM)

######################################################################################
### 6) Determine prefix for installation
######################################################################################

if(UNIX)
    set(PROJECT_TAG "paradiseo" CACHE STRING "Project name as installation prefix")

    # Define the path where the project headers have to be installed and where they can be found (linker) 
    set(PROJECT_HDRS_INSTALL_SUBPATH /include CACHE PATH "Install subpath towards the project header")
    set(PROJECT_HDRS_INSTALL_SUBPATH ${CMAKE_INSTALL_PREFIX}${PROJECT_HDRS_INSTALL_SUBPATH})
endif()

if (ENABLE_CMAKE_TESTING OR ENABLE_CMAKE_EXAMPLE)
    message("Warning: Please ensure that you have executed the 'sudo make install' command before asking the compilation of the tests or examples.")
endif ()