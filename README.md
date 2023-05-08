# Eye of Segmento

## Demo
https://user-images.githubusercontent.com/93939472/236849036-576b8050-cd3e-481c-b9a8-7cfb6dc30ba0.mp4

For demo purposes, camera input is replaced with video.

## Install
```shell
git clone https://github.com/star-bits/eye-of-segmento-mac.git
cd eye-of-segmento-mac

pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Download models
```shell
ls ./models

__init__.py			layers.py
convnext.py			sam_vit_b_01ec64.pth
convnext_base_1k_224_ema.pth	sam_vit_h_4b8939.pth
convnext_small_1k_224_ema.pth	sam_vit_l_0b3195.pth
convnext_tiny_1k_224_ema.pth
```

- [<code>convnext_tiny_1k_224_ema.pth</code>](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth)
- [<code>convnext_small_1k_224_ema.pth</code>](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth)
- [<code>convnext_base_1k_224_ema.pth</code>](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth)
- [<code>sam_vit_b_01ec64.pth</code>](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [<code>sam_vit_l_0b3195.pth</code>](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [<code>sam_vit_h_4b8939.pth</code>](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## Run
Ensure `hud_font` config points to the path of your font.
```shell
python run.py
```
