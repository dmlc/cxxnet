echo "Fetch mshadow..."
cd ..
git clone https://github.com/tqchen/mshadow.git
cd make
make -f ./Cblas.cpu.mk

