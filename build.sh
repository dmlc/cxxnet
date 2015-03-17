#! /bin/bash
echo "Fetch mshadow..."
git clone https://github.com/dmlc/mshadow.git -b master
make $1
