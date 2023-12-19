export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/cs/labs/Academic/dataset/Download/adiyoss/AudioTokenDataset/vggsound/"
export OUTPUT_DIR="output/"
export LEARNED_EMBEDS="output/embedder_learned_embeds.bin"

accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --learned_embeds=$LEARNED_EMBEDS