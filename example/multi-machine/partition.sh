#!/bin/bash
if [ $# -lt 2 ]; then
    echo "usage: $0 train.lst im2bin"
    exit -1;
fi

# set -x
for i in {0..7}; do
    start=$(( $i * 2500 + 1))
    end=$(( $start + 2500 -1 ))
    sed -n "${start}, ${end}p" $1 >tr_${i}.lst
    $2 tr_${i}.lst ./ tr_${i}.bin
done

end=$(($end + 1))
sed -n "${end}, 40000p" $1 > va.lst
$2 va.lst ./ va.bin
