# File: FindParadiseo.cmake
# Version: 0.0.1
#
# The following variables are filled out:
# - PARADISEO_INCLUDE_DIRS
# - PARADISEO_LIBRARY_DIRS
# - PARADISEO_LIBRARIES
# - PARADISEO_FOUND
#
# Here are the components:
# - PyEO
# - es
# - ga
# - cma
#
# You can use FIND_PACKAGE( EO COMPONENTS ... ) to enable one or several components.
#

# Default enabled components
set(PARADISEO_LIBRARIES_TO_FIND eo eoutils cma es flowshop ga moeo)

# Use FIND_PACKAGE( Paradiseo COMPONENTS ... ) to enable modules
if(PARADISEO_FIND_COMPONENTS)
  foreach(component ${PARADISEO_FIND_COMPONENTS})
    string(TOUPPER ${component} _COMPONENT)
    set(PARADISEO_USE_${_COMPONENT} 1)
  endforeach(component)
endif(PARADISEO_FIND_COMPONENTS)

# Path
set(PARADISEO_PATHS
        ${PARADISEO_ROOT}
        $ENV{PARADISEO_ROOT}
        /usr/local/
        /usr/
        /sw # Fink
        /opt/local/ # DarwinPorts
        /opt/csw/ # Blastwave
        /opt/
        HKEY_CURRENT_USER\Software
        HKEY_LOCAL_MACHINE\Software
)

# Set lib path
if(NOT PARADISEO_INCLUDE_DIRS)
    find_path(
        PARADISEO_INCLUDE_DIRS
        PATH_SUFFIXES include
        PATHS ${PARADISEO_PATHS}
    )
endif(NOT PARADISEO_INCLUDE_DIRS)

# Set include path
if(NOT PARADISEO_LIBRARY_DIRS)
    find_path(
        PARADISEO_LIBRARY_DIRS
        PATH_SUFFIXES lib
        PATHS ${PARADISEO_PATHS}
    )
endif(NOT PARADISEO_LIBRARY_DIRS)

if(NOT PARADISEO_LIBRARIES)
  set(PARADISEO_LIBRARIES)
  foreach(component ${PARADISEO_LIBRARIES_TO_FIND})
    find_library(
      PARADISEO_${component}_LIBRARY
      NAMES ${component}
      PATH_SUFFIXES lib64 lib
      PATHS ${PARADISEO_PATHS}
      )

      
    if(PARADISEO_${component}_LIBRARY)
      set(PARADISEO_LIBRARIES ${PARADISEO_LIBRARIES} ${PARADISEO_${component}_LIBRARY})
    else(PARADISEO_${component}_LIBRARY)
      message(FATAL_ERROR "${component} component not found.")
    endif(PARADISEO_${component}_LIBRARY)
  endforeach(component)
endif(NOT PARADISEO_LIBRARIES)

if(PARADISEO_INCLUDE_DIRS AND PARADISEO_LIBRARY_DIRS AND PARADISEO_LIBRARIES)
  set(PARADISEO_FOUND 1)
  mark_as_advanced(PARADISEO_FOUND)
  mark_as_advanced(PARADISEO_INCLUDE_DIRS)
  mark_as_advanced(PARADISEO_LIBRARY_DIRS)
  mark_as_advanced(PARADISEO_LIBRARIES)
endif(PARADISEO_INCLUDE_DIRS AND PARADISEO_LIBRARY_DIRS AND PARADISEO_LIBRARIES)
