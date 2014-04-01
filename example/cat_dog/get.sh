wget antinucleon.com/train.zip
wget antinucleon.com/test1.zip

unzip train.zip
unzip test1.zip

mkdir resized_train
python resize.py
python mklst.py
