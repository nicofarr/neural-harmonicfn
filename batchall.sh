python 1_prepareroi.py

## Single subjects
for CURSUBJ in $(cat $1)
do
    python 2_GLM_univariate.py $CURSUBJ
    python 3_GLM_singletrial.py $CURSUBJ
    python 7_decoding_wholebrain.py $CURSUBJ
    python 8_RSA_crossnobis.py $CURSUBJ
done

## Group level
python 4_GLM_secondlevel.py
python 5_GLM_secondlevel_covar.py
python 6_decoding_allsubjects.py
python 9_RSA_inference.py