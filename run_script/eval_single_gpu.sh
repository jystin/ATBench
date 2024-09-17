export PYTHONPATH=$PYTHONPATH:xdecoder_data/eval_utils/coco
export PYTHONPATH=$PYTHONPATH:xdecoder_data/eval_utils/vizwiz
export DATASET=xdecoder_data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
cd ..

mask=True
captioning=False
vqa=False
depth=False
ocr=False

save_dir="logs/eval_mask"${mask}"_cap"${captioning}"_vqa"${vqa}"_depth"${depth}"_ocr"${ocr}
log_file="logs.log"

CUDA_VISIBLE_DEVICES=0 python entry.py evaluate \
            --conf_files configs/xdecoder/super_tiny_alltasks.yaml \
            --overrides \
            MODEL.DECODER.MASK ${mask} \
            MODEL.DECODER.CAPTIONING.ENABLED ${captioning} \
            MODEL.DECODER.DEPTH.ENABLED ${depth} \
            MODEL.DECODER.OCR.ENABLED ${ocr} \
            MODEL.DECODER.VQA.ENABLED ${vqa} \
            ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
            MODEL.DECODER.TEST.SEMANTIC_ON False \
            MODEL.DECODER.TEST.INSTANCE_ON False \
            MODEL.DECODER.TEST.PANOPTIC_ON True \
            FP16 True \
            WEIGHT True \
            RESUME_FROM logs/epochs30_lr0.0001_keyDatasetseg_maskTrue_captioningTrue_vqaFalse_depthTrue_ocrTrue20231019101759/00101040/default/model_state_dict.pth \
            SAVE_DIR ${save_dir} \
            LOG_FILE ${log_file}