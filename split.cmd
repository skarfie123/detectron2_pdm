:: split M
python cocosplit.py annotations\instances_M.json new\M_train.json new\M_valtest.json -s 0.7
python cocosplit.py new\M_valtest.json new\M_val.json new\M_test.json -s 0.5
python cococount.py annotations\instances_M.json new\M_train.json new\M_val.json new\M_test.json

:: copy G,V
copy annotations\instances_G.json new\G_train.json
copy annotations\instances_G.json new\G_val.json
copy annotations\instances_G.json new\G_test.json
copy annotations\instances_V.json new\V_train.json
copy annotations\instances_V.json new\V_val.json
copy annotations\instances_V.json new\V_test.json

:: cocosame M to G,V
python cocosame.py new\M_train.json new\G_train.json
python cocosame.py new\M_train.json new\V_train.json
python cocosame.py new\M_val.json new\G_val.json
python cocosame.py new\M_val.json new\V_val.json
python cocosame.py new\M_test.json new\G_test.json
python cocosame.py new\M_test.json new\V_test.json

:: sort M,G,V
python cocosort.py new\*.json
