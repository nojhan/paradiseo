#! /bin/tcsh -f

if ($#argv < 3) then
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
if ($#argv == 3) then
    echo s/ADDLIB/$3/g >> toto.sed
endif

sed -f toto.sed test_dsp.tmpl > $2.dsp
/bin/rm toto.sed
