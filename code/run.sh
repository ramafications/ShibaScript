#!/bin/bash
#!/bin/bash
source  /opt/anaconda3/bin/activate Shiba

DATA_DIR=/Users/ramadas/ShibaScript/code/Finn
NAME=Finn.mp4
ORIGIN_DIR=`pwd`
# 1. raw panns to get "sentence"
echo "step 1. raw panns to get sentence"
cd ~/ShibaScript/code/AudioTagging/audoset_tagging_cnn
LOG_PATH=${DATA_DIR}/log.txt


python3 pytorch/process_shiba.py sound_event_detection --log_file ${LOG_PATH} --model_type Cnn14_DecisionLevelMax --checkpoint_path ~/panns_data/Cnn14_DecisionLevelMax.pth --file_path ~/${NAME}/ --wav_path ~/${NAME}/ --cuda

# 2. process the result of panns
echo "step 2. process the result of panns"
cd $ORIGIN_DIR

SAVE_PATH=${DATA_DIR}/sentences/

if [ ! -d ${SAVE_PATH} ];then
    mkdir ${SAVE_PATH}
fi

python3 process_pannsresult.py --log_path ${LOG_PATH} --save_path ${SAVE_PATH} --audio_path ~/${NAME}/

# 4. remove those noise
echo "step 3. remove noise"
cd ~/ShibaScript/code/AudioTagging/audoset_tagging_cnn
python pytorch/process_noiseremover.py sound_event_detection --model_type Cnn14_DecisionLevelMax --checkpoint_path /home/jieyi/panns_data/Cnn14_DecisionLevelMax.pth --file_path ${SAVE_PATH} --cuda


# 3. automatically generate sentence->word
echo "step 4. automatically generate sentence to word"
cd ~/ShibaScript/code/CutRawClips/
OUT_DIR_AUTO=~/Ani/code/CutRawClips/result_${NAME}.txt
SAVE_PATH_WORD=${DATA_DIR}/words/
if [ ! -d ${SAVE_PATH_WORD} ];then
    mkdir ${SAVE_PATH_WORD}
fi 

conda activate maskrcnn_benchmark
python3 inference.py --out_dir ${OUT_DIR_AUTO} --audio_path ${SAVE_PATH}
python3 processresult.py --audio_path ${SAVE_PATH} --save_path ${SAVE_PATH_WORD} --result_file ${OUT_DIR_AUTO}
conda deactivate


# # 5. generate syllables
echo "step 5. generate syllables"
SAVE_PATH_SYLLABLES=${DATA_DIR}/syllables/
if [ ! -d ${SAVE_PATH_SYLLABLES} ];then
    mkdir ${SAVE_PATH_SYLLABLES}
fi
cd ${ORIGIN_DIR}
python3 cutSyllables.py --wav_path ${SAVE_PATH_WORD} --res_out_root ${SAVE_PATH_SYLLABLES} --segment_info_path ${DATA_DIR}/log_segment.txt

# 6. feature extraction
echo "step 6. feature extraction"
cd ~/Ani/processdata
FEATURE_PATH=${DATA_DIR}/fb/

if [ ! -d ${FEATURE_PATH} ];then
    mkdir ${FEATURE_PATH}
fi

conda activate shennong
python3 generate_filterbank.py --data_path ${SAVE_PATH_SYLLABLES} --ref_path ${SAVE_PATH_WORD} --save_path ${FEATURE_PATH}
conda deactivate
