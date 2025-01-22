# Leapfusion Hunyuan Image-to-Video
**Show your support!** You can try HunyuanVideo free with some of our custom spice [here](https://leapfusion.ai/). Supporting LeapFusion enables us to do more open source releases like this in the future!

Training code can be found [Here](https://github.com/AeroScripts/musubi-tuner-img2video).

# Usage
First, Download the hunyuan weights as explained [here](https://github.com/AeroScripts/musubi-tuner-img2video/tree/main?tab=readme-ov-file#use-the-official-hunyuanvideo-model) and get the image2video lora weights from [here](https://huggingface.co/leapfusion-image2vid-test/image2vid-512x320/blob/main/img2vid.safetensors). Then run the following command to encode an image: (ex. input_image.png)
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

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.
Other code is under the Apache License 2.0. Some code is copied and modified from musubi-tuner, k-diffusion and Diffusers.
