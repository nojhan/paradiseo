#! /bin/tcsh -f

if ($#argv < 2) then
    echo Usage $0 SourceName TargetName [Additional lib]
    echo Example: $0 t-eoGA t_eoga ga
    echo "    will create t_eoga.dsp that in turn is the Win project file"
    echo "    that will create the executable t_eoga using source"
    echo "    t-eoga.cpp in test dir"
    exit
endif
echo "Creating $2.dsp"
echo s/DIRNAME/$2/g  > toto.sed
echo s/SOURCENAME/$1/g >> toto.sed
# additional libraries: ADDLIBR = name of "Release" library
#                       ADDLIBD = name of "Debug" library
if ($#argv == 3) then
    echo s/ADDLIBR/$3.lib/g >> toto.sed
    echo s/ADDLIBD/$3d.lib/g >> toto.sed
else   # remove the ADDLIBs !!!!
    echo s/ADDLIBR//g >> toto.sed
    echo s/ADDLIBD//g >> toto.sed
endif

sed -f toto.sed test_dsp.tmpl > $2.dsp
/bin/rm toto.sed

echo "Adding $2.dsp in the main eo.dsw project file"

echo "" >> eo.dsw
echo Project: \"$2\"=.\\\\$2.dsp - Package Owner=\<4\> >> eo.dsw
echo "" >> eo.dsw
echo Package=\<5\> >> eo.dsw
echo '{{{' >> eo.dsw
echo '}}}' >> eo.dsw
echo "" >> eo.dsw
echo Package=\<4\> >> eo.dsw
echo '{{{' >> eo.dsw
echo '}}}' >> eo.dsw
echo "" >> eo.dsw
echo '###############################################################################' >> eo.dsw

