1. Resize all image to 48 X 48
```
mkdir /home/cxxnet/example/kaggle_bowl/data
python gen_train.py /home/data/bowl/train/ /home/cxxnet/example/kaggle_bowl/data/train/
python gen_test.py /home/data/bowl/test/ /home/cxxnet/example/kaggle_bowl/data/test/
```

2. Generate img list
```
python gen_img_list.py train /home/data/bowl/sampleSubmission.csv data/train/ train.lst
python gen_img_list.py test /home/data/bowl/sampleSubmission.csv data/test/ test.lst
```

3. Generate binary image file
```
../../tools/im2bin train.lst ./ train.bin
../../tools/im2bin test.lst ./ test.bin
```

4. Run CXXNET
```
mkdir models
../../bin/cxxnet bowl.conf
```
It take about 5 minute to train a deep conv net model on Geforece 780

5. Run Prediction
```
../../bin/cxxnet pred.conf
```
It will write softmax result in test.txt

6. Make a submission file

```
python make_submission.py /home/data/bowl/sampleSubmission.csv test.lst test.txt out.csv
```

7. Submit out.csv, you will get a result of 1.382285
