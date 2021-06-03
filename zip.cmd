cd new

mkdir M G V

python ..\cocoprune.py --filter-size *.json

copy M_train.json M\
copy M_val.json M\
copy M_test.json M\
copy G_train.json G\
copy G_val.json G\
copy G_test.json G\
copy V_train.json V\
copy V_val.json V\
copy V_test.json V\

7z a -tzip M.zip M
7z a -tzip G.zip G
7z a -tzip V.zip V

move M.zip ..\zips\
move G.zip ..\zips\
move V.zip ..\zips\

cd ..