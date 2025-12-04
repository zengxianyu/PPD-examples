## Phase-Preserving Diffusion
Example adaptations of SD1.5, FLUX.1-dev and Wan2.2-14b with phase-preserving diffusion. 

This repo is largely borrowed from https://github.com/modelscope/DiffSynth-Studio. Please refer to the original repo for training script and more use cases. 

### Usage

1. Install dependency:

```
pip install -r requirements.txt
pip install git+https://github.com/zengxianyu/structured-noise
```

2. Download [model weights](https://huggingface.co/zengxianyu/ppd/tree/main) and put them in `models/ppd/`

3. Inference: 

Example input images can be found [here](https://huggingface.co/zengxianyu/ppd/tree/main).

SD 1.5: 

```
PYTHONPATH=. python examples/image_synthesis/sd_text_to_image_ppd.py --input_image dog.jpg --radius 15 --prompt "A high quality picture captured by a professional camera. Picture of a cute border collie" --output output.png
```

FLUX1.1-dev

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python examples/flux/model_inference/FLUX.1-dev_ppd.py --input_image test2.jpg --prompt "$(cat test2.txt)" --output output.png --radius 30
```

Wan2.2-14b

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python examples/wanvideo/model_inference/Wan2.2-I2V-A14B_ppd.py --input_image output.png --input_video test2.mp4 --prompt  "$(cat test2.txt)" --radius 30 --output output.mp4
```

4. For training please refer to https://github.com/modelscope/DiffSynth-Studio 

If you find this work useful, please cite:

```
@article{zeng2025neuralremaster,
  title   = {{NeuralRemaster}: Phase-Preserving Diffusion for Structure-Aligned Generation},
  author  = {Zeng, Yu and Ochoa, Charles and Zhou, Mingyuan and Patel, Vishal M and
             Guizilini, Vitor and McAllister, Rowan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```
