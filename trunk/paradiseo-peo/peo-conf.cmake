
######################################################################################
######################################################################################
###  In this file, you can specify many CMake variables used to build paradisEO-PEO.
###  For example, if you don't want ot give the EO path each time on the command line,
###  uncomment the line the "SET(PROJECT_NAME...)" and set your favorite name.
###  The section numbers are the same as those used in the CMakeLists.txt file.
######################################################################################
######################################################################################


######################################################################################
### 0) OPTIONNAL - Overwrite project default config 
######################################################################################
  
# SET(PROJECT_NAME "ParadisEO-PEO")

######################################################################################


######################################################################################
### 3) OPTIONNAL - Overwrite default paths
######################################################################################

# SET(MOEO_DIR "<your path>" CACHE PATH "ParadisEO-PEO main directory") 
# SET(EO_DIR "<path to ParadisEO-EO>" CACHE PATH "ParadisEO-EO main directory")

# SET(EO_SRC_DIR "<path to ParadisEO-EO src dir>")
# SET(MOEO_SRC_DIR "<path to ParadisEO-MO src dir>")
# SET(MOEO_DOC_DIR "<path to ParadisEO-MO doc dir>")

######################################################################################


#####################################################################################
### 5) OPTIONNAL - Overwrite subdirs
######################################################################################

# SUBDIRS(doc src tutorial)

######################################################################################

