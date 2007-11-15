:: genParadisEOSol.bat
:: Generates ParadisEO's Visual Studio solutions
::
@ECHO OFF


:: Keep the current directory
set _cdir=%CD%



:-------------------------- How ParadisEO will be compiled ? ----------------------------------------------:

:: Preliminary checking
IF NOT EXIST paradiseo-eo GOTO errorNoEO
IF NOT EXIST paradiseo-mo GOTO errorNoMO
IF NOT EXIST paradiseo-moeo GOTO errorNoMOEO

CLS 
 
echo.
echo.
echo =====================================================================
echo 	   Generate configuration files for ParadisEO
echo =====================================================================
echo.
echo.

:: Install type
echo CMake can generate many configuration files for ParadisEO.
echo.

echo 1)  Visual Studio 8 2005        
echo 2)  Visual Studio 8 2005 Win64
echo 3)  Visual Studio 7 .NET 2003
echo 4)  Visual Studio 7
echo 5)  Visual Studio 6
echo 6)  NMake Makefiles
echo 7)  MinGW Makefiles
echo 8)  MSYS Makefiles
echo 9)  Borland Makefiles
echo 10) Watcom WMake
echo 11) Unix Makefiles


:: Read the user's choice
echo.
set /p install=Please choose your generator: 

:: Adapt the CMake generator
if "%install%"=="1" set _cmake_generator="Visual Studio 8 2005"
if "%install%"=="2" set _cmake_generator="Visual Studio 8 2005 Win64"
if "%install%"=="3" set _cmake_generator="Visual Studio 7 .NET 2003" 
if "%install%"=="4" set _cmake_generator="Visual Studio 7"
if "%install%"=="5" set _cmake_generator="Visual Studio 6"
if "%install%"=="6" set _cmake_generator="NMake Makefiles"
if "%install%"=="7" set _cmake_generator="MinGW Makefiles"
if "%install%"=="8" set _cmake_generator="MSYS Makefiles"
if "%install%"=="9" set _cmake_generator="Borland Makefiles"
if "%install%"=="10" set _cmake_generator="Watcom WMake"
if "%install%"=="11" set _cmake_generator="Unix Makefiles"

if %_cmake_generator%=="" goto invalidInstall

:----------------------------------------------------------------------------------------------------------:



:-------------------------- EO INSTALL ----------------------------------------------:
:: Go to ParadisEO-EO directory
cd %_cdir%\paradiseo-eo\build

:: Run CMake
cmake "%_cdir%\paradiseo-eo" -G %_cmake_generator%

echo.
if "%errorlevel%"=="0" echo ----- Generate ParadisEO-EO configuration files DONE
if "%errorlevel%"=="-1" goto errorInstall
echo.
echo.

:: Go back to the initial directory
cd %_cdir%

:------------------------------------------------------------------------------------:




:-------------------------- MO INSTALL ----------------------------------------------:

:: Go to ParadisEO-MO directory
cd %_cdir%\paradiseo-mo\build

:: Run CMake
cmake "%_cdir%\paradiseo-mo" -G %_cmake_generator% "-Dconfig=%_cdir%\install.cmake"

echo.
if %errorlevel%==0 echo ----- Generate ParadisEO-MO configuration files DONE
if %errorlevel%==-1 goto errorInstall
echo.
echo.
:------------------------------------------------------------------------------------:




:-------------------------- MOEO INSTALL ----------------------------------------------:

:: Go back to the initial directory
cd %_cdir%

:: Go to ParadisEO-MOEO directory
cd %_cdir%\paradiseo-moeo\build

:: Run CMake
cmake "%_cdir%\paradiseo-moeo" -G %_cmake_generator% "-Dconfig=%_cdir%\install.cmake"


echo.
if %errorlevel%==0 echo ----- Generate ParadisEO-MOEO configuration files DONE
if not %errorlevel%==0 goto errorInstall
echo.
echo.
:------------------------------------------------------------------------------------:




:-------------------------- Error management ----------------------------------------:

:: Go back to the initial directory
cd %_cdir%

goto end

:: Error management
:errorNoEO
echo.
echo. Cound not find  %_cdir%\paradiseo-eo. Abort.
echo.
goto end

:errorNoMO
echo.
echo. Cound not find  %_cdir%\paradiseo-mo. Abort.
echo.
goto end

:errorNoMOEO
echo.
echo. Cound not find  %_cdir%\paradiseo-moeo. Abort.
echo.
goto end

:invalidInstall
echo.
echo. No CMake generator read, Abort.
echo.
goto end


:errorInstall
echo.
echo. An error has occurend, end.
echo.
goto end
:------------------------------------------------------------------------------------:


:: End
:end
PAUSE 
