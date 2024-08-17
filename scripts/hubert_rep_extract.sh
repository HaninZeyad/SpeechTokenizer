CONFIG="config/spt_base_cfg.json"
AUDIO_DIR="/ocean/projects/cis220031p/sbaek/sbaek_temp/LibriSpeech"
REP_DIR="/ocean/projects/cis220031p/hatwany/SpeechTokenizer/hubert_base/LibriSpeech"
EXTS="flac"
SPLIT_SEED=0
VALID_SET_SIZE=2000



CUDA_VISIBLE_DEVICES=0 python scripts/hubert_rep_extract.py\
    --config ${CONFIG}\
    --audio_dir ${AUDIO_DIR}\
    --rep_dir ${REP_DIR}\
    --exts ${EXTS}\
    --split_seed ${SPLIT_SEED}\
    --valid_set_size ${VALID_SET_SIZE}