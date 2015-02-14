#! /bin/bash
echo "Fetch mshadow..."
git clone https://github.com/tqchen/mshadow.git -b v1.1
make $1

