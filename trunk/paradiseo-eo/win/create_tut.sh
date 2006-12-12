#! /bin/tcsh -f

if ($#argv < 2) then
    echo "Usage $0 LessonNumber NickName SourceName (no .cpp)"
    echo Example: $0 1 bit FirstBitGA
    echo "    will create the Win project file tut_1_bit.dsp"
    echo "    that will create the executable tut_1_bit.exe using source"
    echo "    FirstBitGA.cpp in tutorial/Lesson1 dir"
    exit
endif
echo "Creating tut_{$1}_{$2}.dsp"
echo s/tut_N_XXX/tut_{$1}_{$2}/g  > toto.sed
echo s/LessonN/Lesson$1/g >> toto.sed
echo s/SRCXXX/$3/g >> toto.sed

sed -f toto.sed tut_N_XXX.tmpl > tut_{$1}_{$2}.dsp
/bin/rm toto.sed

echo "Adding tut_{$1}_{$2}.dsp in the main eo.dsw project file"

echo "" >> eo.dsw
echo Project: \"tut_{$1}_{$2}\"=.\\\\tut_{$1}_{$2}.dsp - Package Owner=\<4\> >> eo.dsw
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

echo "AND DON'T FORGET to add the 3 lines in eo.dsw"
echo "  (too lazy to do it automatically)"
echo "    Begin Project Dependency"
echo "   " Project_Dep_Name tut_{$1}_{$2}
echo "    End Project Dependency"
