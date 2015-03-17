#! /bin/bash
echo "Fetch mshadow..."
git clone https://github.com/tqchen/mshadow.git -b master
make $1
