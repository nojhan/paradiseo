plotOne()
{
    OUTPUT=$1
    BENCH=$2
    METRIC=$3
#     echo "set term postscript eps enhanced" >> temp
    echo "set term epslatex colour" >> temp
    echo "set output \"$OUTPUT\"" >> temp
    echo "set nokey" >> temp
    echo "set size 0.6,0.6" >> temp
    echo "set bmargin 1" >> temp
    echo "set tmargin 1" >> temp
    echo "set lmargin 2" >> temp
    echo "set rmargin 1" >> temp
    echo "set xrange [0:31]" >> temp
    echo "set yrange [0:0.15]" >> temp
    echo "plot 'fsp2.results/${BENCH}_OneOne_${METRIC}' title 'DMLS \$(1 \cdot 1)\$' with lines lt 1 lw 2, 'fsp2.results/${BENCH}_OneND_${METRIC}' title 'DMLS \$(1 \cdot 1_{\not\prec})\$' with lines lt 2 lw 2, 'fsp2.results/${BENCH}_OneFirst_${METRIC}' title 'DMLS \$(1 \cdot 1_{\succ})\$' with lines lt 3 lw 2, 'fsp2.results/${BENCH}_OneAll_${METRIC}' title 'DMLS \$(1 \cdot \star)\$' with lines lt 4 lw 2, 'fsp2.results/${BENCH}_AllOne_${METRIC}' title 'DMLS \$(\star \cdot 1)\$' with lines lt 5 lw 2, 'fsp2.results/${BENCH}_AllND_${METRIC}' title 'DMLS \$(\star \cdot 1_{\not\prec})\$' with lines lt 9 lw 2,'fsp2.results/${BENCH}_AllFirst_${METRIC}' title 'DMLS \$(\star \cdot 1_{\succ})\$' with lines lt 6 lw 2,'fsp2.results/${BENCH}_AllAll_${METRIC}' title 'DMLS \$(\star \cdot \star)\$' with lines lt 8 lw 2" >> temp
    echo "set out" >> temp
    gnuplot temp
    rm temp
}


plottest()
{
    OUTPUT=$1
#     echo "set term postscript eps enhanced" >> temp
    echo "set term epslatex" >> temp
    echo "set output \"$OUTPUT\"" >> temp
    echo "set key reverse left center" >> temp
    echo "set noborder" >> temp
echo "set noxtics" >> temp
echo "set noytics" >> temp
    echo "set size 0.5,0.5" >> temp
  #  echo "set bmargin 1" >> temp
   # echo "set tmargin 1" >> temp
   # echo "set lmargin 2" >> temp
   #s echo "set rmargin 1" >> temp
    echo "set xrange [0:31]" >> temp
    echo "set yrange [0:1]" >> temp
    echo "plot 2 title 'DMLS \$(1 \cdot 1)\$' with lines lt 1 lw 2, 2 title 'DMLS \$(1 \cdot 1_{\not\prec})\$' with lines lt 2 lw 2, 2 title 'DMLS \$(1 \cdot 1_{\succ})\$' with lines lt 3 lw 2, 2 title 'DMLS \$(1 \cdot \star)\$' with lines lt 4 lw 2, 2 title 'DMLS \$(\star \cdot 1)\$' with lines lt 5 lw 2, 2 title 'DMLS \$(\star \cdot 1_{\not\prec})\$' with lines lt 9 lw 2, 2 title 'DMLS \$(\star \cdot 1_{\succ})\$' with lines lt 6 lw 2, 2 title 'DMLS \$(\star \cdot \star)\$' with lines lt 8 lw 2" >> temp
    echo "set out" >> temp
    gnuplot temp
    rm temp
}

plotOneBis()
{
 OUTPUT=$1
    BENCH=$2
    METRIC=$3
    echo "set term postscript eps enhanced" >> temp
    echo "set output \"$OUTPUT\"" >> temp
    echo "set size 0.7,0.7" >> temp
    echo "set xrange [0:21]" >> temp
    echo "plot 'fsp2.results/${BENCH}_OneAll_${METRIC}' title 'DMLS (1 . *)' with lines lt 2 lw 5, 'fsp2.results/${BENCH}_AllOne_${METRIC}' title 'DMLS (* . 1)' with lines lt 3 lw 5, 'fsp2.results/${BENCH}_AllAll_${METRIC}' title 'DMLS (* . *)' with lines lt 5 lw 5" >> temp
    gnuplot temp
    rm temp
}

plot()
{
    BENCH=$1
    plotOne fsp2.fig.test/$BENCH\_eps.eps $BENCH eps
    plotOne fsp2.fig.test/$BENCH\_hyp.eps $BENCH hyp
    #plotOneBis fsp2.fig/$BENCH\_eps_bis.eps $BENCH eps
    #plotOneBis fsp2.fig/$BENCH\_hyp_bis.eps $BENCH hyp
}

#plot 0200501
#plot 0201001
#plot 0202001
#plot 0500501
#plot 0501001
#plot 0502001
plot 1001001
plot 1002001

#plottest fsp2.fig/legende.eps

cd fsp2.fig
sed -i -e 's/-1287/2450/g' *.tex
