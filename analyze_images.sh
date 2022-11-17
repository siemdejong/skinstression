#!/bin/bash
maxdir=62
for ((dir = 0; dir <= $maxdir ; dir++)); do

    # Show progress bar.
    echo -n "[ "
    for ((i = 0 ; i <= $dir; i++)); do echo -n "###"; done
    for ((j = $i ; j <= $maxdir ; j++)); do echo -n "   "; done
    # v=`echo "scale=2 ; $dir / $maxdir * 100" | bc`
    v=`printf "%.0f\n" $(bc -l <<< "$dir / $maxdir * 100")`
    echo -n " ] "
    echo -n "$v %" $'\r'

    # Run pyimq to analyze the images in the directorie.
    pyimq.main \
        --mode=directory \
        --mode=analyze \
        --working-directory=/scistor/guest/sjg203/projects/shg-strain-stress/data/grayscale/z-stacks/$dir \
        --rgb-channel=0 \
        > outputs/pyimq/image_analysis_stdout.txt \
        2> outputs/pyimq/image_analysis_stderr.txt
done
echo