# Leapfusion Hunyuan Image-to-Video

## Installation

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. Verified to work with version 2.5.1.

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create -n musubi-tuner python=3.10
conda activate musubi-tuner
pip install ipykernel
python -m ipykernel install --user --name musubi-tuner --display-name "musubi-tuner"

pip install torch torchvision

#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command:

```bash
git clone https://github.com/svjack/leapfusion-hunyuan-image2video && cd leapfusion-hunyuan-image2video
pip install -r requirements.txt
```

Optionally, you can use FlashAttention and SageAttention (see [SageAttention Installation](#sageattention-installation) for installation instructions).

Additionally, install `ascii-magic` (used for dataset verification), `matplotlib` (used for timestep visualization), and `tensorboard` (used for logging training progress) as needed:

```bash
pip install ascii-magic matplotlib tensorboard huggingface_hub datasets
pip install moviepy==1.0.3
pip install sageattention==1.0.6
```

### Model Download

Download the model following the [official README](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md) and place it in your chosen directory with the following structure:

```bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
wget https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py
python preprocess_text_encoder_tokenizer_utils.py --input_dir llava-llama-3-8b-v1_1-transformers --output_dir text_encoder
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

```bash
wget https://huggingface.co/leapfusion-image2vid-test/image2vid-512x320/resolve/main/img2vid.safetensors -O img2vid.safetensors
huggingface-cli download Comfy-Org/HunyuanVideo_repackaged --include "split_files/text_encoders/*" --local-dir text_encoders
```

**Show your support!** You can try HunyuanVideo free with some of our custom spice [here](https://leapfusion.ai/). Supporting LeapFusion enables us to do more open source releases like this in the future!

Training code can be found [Here](https://github.com/AeroScripts/musubi-tuner-img2video).

# Usage

- gen target_latent.pt
```bash
sudo chmod 777 /root
python encode_image.py --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling --image "老佛爷1.jpg"
```


![老佛爷1](https://github.com/user-attachments/assets/ebadf762-f8ef-4699-a2a8-84e341ec55c7)

- gen video
```bash
python generate.py --fp8 --video_size 320 512 --infer_steps 30 --save_path ./samples/ --output_type both \
 --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt \
 --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 --text_encoder1 text_encoders/split_files/text_encoders/llava_llama3_fp16.safetensors  \
 --text_encoder2 text_encoders/split_files/text_encoders/clip_l.safetensors --lora_multiplier 1.0 --lora_weight img2vid.safetensors --video_length 129 --prompt "" --seed 123
```




https://github.com/user-attachments/assets/35e4b5ce-340a-4abe-be7a-bfcbe39830ad


First, Download the hunyuan weights as explained [here](https://github.com/AeroScripts/musubi-tuner-img2video/tree/main?tab=readme-ov-file#use-the-official-hunyuanvideo-model) and get the image2video lora weights from [here](https://huggingface.co/leapfusion-image2vid-test/image2vid-512x320/blob/main/img2vid.safetensors). Then run the following command to encode an image: (ex. input_image.png)
```bash
wget https://huggingface.co/leapfusion-image2vid-test/image2vid-512x320/resolve/main/img2vid.safetensors -O img2vid.safetensors
```

```
python encode_image.py --vae hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling --image ./input_image.png
```

Then, you can launch generate a video with something like:
```
python generate.py --fp8 --video_size 320 512 --infer_steps 30 --save_path ./samples/ --output_type both --dit mp_rank_00_model_states.pt --attn_mode sdpa --split_attn --vae hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 --text_encoder1 llava_llama3_fp16.safetensors --text_encoder2 clip_l.safetensors --lora_multiplier 1.0 --lora_weight img2vid.safetensors --video_length 129 --prompt "" --seed 123 
```
Leaving the prompt blank, the model will infer based on the image alone. If you prompt changes, make sure to describe some baseline details about the image too or you might get bad results.

**Note**: The current model is trained at 512x320, as our research budget is quite small. If anyone would like to help train a higher res chekpoint and has some spare compute, please reach out!

# Samples
https://github.com/user-attachments/assets/1410ede0-9d88-4c29-b785-bc934525a0da

https://github.com/user-attachments/assets/ef6fcf10-8cdf-42b5-b1fc-d9f8673c894a

https://github.com/user-attachments/assets/30038397-c67c-49f3-9707-db0deb110268

https://github.com/user-attachments/assets/4c53f88e-f44d-45df-81c7-ec3bafc77ec2



## License

Much of the code is based on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.
Other code is under the Apache License 2.0. Some code is copied and modified from musubi-tuner, k-diffusion and Diffusers.
