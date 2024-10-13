export PYTHONPATH=$PYTHONPATH:atmodel_data/eval_utils/coco
export PYTHONPATH=$PYTHONPATH:atmodel_data/eval_utils/vizwiz
export DATASET=atmodel_data
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
            --conf_files configs/atmodel_single_task.yaml \
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
            RESUME_FROM checkpoints/best_SEG.pt \
            SAVE_DIR ${save_dir} \
            LOG_FILE ${log_file}