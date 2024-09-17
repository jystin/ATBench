export PYTHONPATH=$PYTHONPATH:atmodel_data/eval_utils/coco
export PYTHONPATH=$PYTHONPATH:atmodel_data/eval_utils/vizwiz
export DATASET=atmodel_data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
cd ..

all_tasks=True
mask=True
captioning=True
vqa=True
depth=True
ocr=True
num_classes=150

max_epoch=50
lr=0.0001
key_dataset="seg"
if [ ${all_tasks} = true ]; then
    save_dir="logs"/"epochs"${max_epoch}"_alltasks"
else
    save_dir="logs"/"epochs"${max_epoch}"_keyDataset"${key_dataset}"_mask"${mask}"_cap"${captioning}"_vqa"${vqa}"_depth"${depth}"_ocr"${ocr}
fi

log_file="logs.log"

CUDA_VISIBLE_DEVICES=0 python entry.py train \
            --conf_files configs/atmodel_all_task.yaml \
            --overrides \
            GRADIENT_ACCUMULATE_STEP 1 \
            LOADER.KEY_DATASET ${key_dataset} \
            MODEL.ALL_TASKS ${all_tasks} \
            MODEL.ENCODER.NUM_CLASSES ${num_classes}\
            MODEL.DECODER.MASK ${mask} \
            MODEL.DECODER.CAPTIONING.ENABLED ${captioning} \
            MODEL.DECODER.VQA.ENABLED ${vqa} \
            MODEL.DECODER.DEPTH.ENABLED ${depth} \
            MODEL.DECODER.OCR.ENABLED ${ocr} \
            ADE20K.TEST.BATCH_SIZE_TOTAL 1 \
            ADE20K.TRAIN.BATCH_SIZE_TOTAL 1 \
            ADE20K.TRAIN.BATCH_SIZE_PER_GPU 1 \
            OCR.TEST.BATCH_SIZE_TOTAL 2 \
            OCR.TRAIN.BATCH_SIZE_TOTAL 1 \
            OCR.TRAIN.BATCH_SIZE_PER_GPU 1 \
            NYU_V2.TEST.BATCH_SIZE_TOTAL 2 \
            NYU_V2.TRAIN.BATCH_SIZE_TOTAL 1 \
            NYU_V2.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VIZWIZ_CAPTIONING.TEST.BATCH_SIZE_TOTAL 2 \
            VIZWIZ_CAPTIONING.TRAIN.BATCH_SIZE_TOTAL 1 \
            VIZWIZ_CAPTIONING.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VIZWIZ_VQA.TEST.BATCH_SIZE_TOTAL 2 \
            VIZWIZ_VQA.TRAIN.BATCH_SIZE_TOTAL 1 \
            VIZWIZ_VQA.TRAIN.BATCH_SIZE_PER_GPU 1 \
            FP16 True \
            WEIGHT False \
            RESUME_FROM checkpoints/focalt_unicl_pretrain.pt \
            SOLVER.MAX_NUM_EPOCHS ${max_epoch} \
            SOLVER.BASE_LR ${lr} \
            SAVE_DIR ${save_dir} \
            LOG_FILE ${log_file}