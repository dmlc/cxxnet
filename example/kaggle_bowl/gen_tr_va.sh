sed -n '1,20000p' $1 > tr.lst
sed -n '20000, 40000p' $1 > va.lst
