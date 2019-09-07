OUTPUT_DIR=../squad_info_output
python2 eval_squad.py sample1k-HCVerifyAll.json $OUTPUT_DIR/predictions.json
python2 eval_squad.py sample1k-HCVerifySample.json $OUTPUT_DIR/predictions.json

