# Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation
This repo contains the official PyTorch implementation of  *Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation*

# Abstract
In recent years, image generation has shown a great leap in per-
formance, where diffusion models play a central role. Although
generating high-quality images, such models are mainly con-
ditioned on textual descriptions. This begs the question: how
can we adopt such models to be conditioned on other modal-
ities?. In this paper, we propose a novel method utilizing la-
tent diffusion models, trained for text-to-image-generation, to
generate images, conditioned on audio recordings. Using a pre-
trained audio encoding model, the proposed method encodes
audio into a new token which can be considered as an adap-
tation layer between the audio and text representations. Such a
modeling paradigm requires a small number of trainable param-
eters making the proposed approach appealing for lightweight
optimization. Results suggest the proposed method is superior
to the evaluated baseline methods considering both objective
and subjective metrics.

# News
### May 17, 2023
Weights for [Lura](https://huggingface.co/blog/lora) were added. With only 3.3 MB of weights, the performance is pushed to higher level. The weights are stored in ```output/weights/lora_layers_learned_embeds.bin```.

# Installation
```
git clone git@github.com:guyyariv/AudioToken.git
cd AudioToken
pip install -r requirements.txt
```
And initialize an Accelerate environment with:
```angular2html
accelerate config
```
Download BEATs pre-trained model 
```
mkdir -p models/BEATs/ && wget "https://msranlcmtteamdrive.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2022-12-18T10%3A41%3A16Z&se=3022-12-19T10%3A41%3A00Z&sr=b&sp=r&sig=gSSExKP0otwVBgKwdV8FoMWL2VppARFq%2B26xKin5rKw%3D" -P "models/BEATs/"
```

# Pre-Trained Embedder

![alt text](https://github.com/guyyariv/AudioToken/blob/master/figs/fig1.png)

The embedder's weights, which we pre-trained and on which the article is based, may be found at:
```output/embedder_learned_embeds.bin```

# Training

First, download our data set. [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)

```angular2html
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="./vggsound/"
export OUTPUT_DIR="output/"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR 
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=30000 \
  --learning_rate=1.0e-05 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 
```
Note: Change the resolution to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.
# Inference

After you've trained a model with the above command, you can simply generate images using the following script:
```angular2html
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="./vggsound/"
export OUTPUT_DIR="output/"
export LEARNED_EMBEDS="output/embedder_learned_embeds.bin"

accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --test_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \ 
  --learned_embeds=$LEARNED_EMBEDS
```

# Cite
If you use our work in your research, please cite the following paper:
@article{yariv2023adaptation,
    title    =   {Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation},
    author   =   {Yariv, Guy and Gat, Itai and Lior, Wolf and Adi, Yossi and Schwartz, Idan},
    journal  =   {arXiv preprint arXiv:XXXXXX},
    year     =   {2023}
}

# License
This repository is released under the MIT license as found in the [LICENSE](LICENSE) file. 

