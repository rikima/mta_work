#!/bin/sh
unit=10
for i in `seq 0 $unit` ; do
    echo $i
    python3 ./flat_svmdata_maker.py $unit $i &
done