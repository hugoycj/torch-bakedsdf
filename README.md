# torch-bakedsdf
An unofficial pytorch implementation of [Meshing Neural SDFs for Real-Time View Synthesis](https://bakedsdf.github.io/).

We support exporting baked assets for **real-time rendering on WebGL, Unity and Unreal**

# Install 
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -r requirements.txt
```
For COLMAP, alternative installation options are also available on the [COLMAP website](https://colmap.github.io/)

# Data preparation
To get COLMAP data from custom images, you should have COLMAP installed (see here for installation instructions). Then put your images in the images/ folder, and run scripts/imgs2poses.py specifying the path containing the images/ folder. For example:
```
python scripts/imgs2poses.py ./load/bmvs_dog # images are in ./load/bmvs_dog/images
```
Existing data following this file structure also works as long as images are store in images/ and there is a sparse/ folder for the COLMAP output, for example the data provided by [MipNeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

# Run BakedSDF!
```
python launch.py --config configs/neus-colmap.yaml --gpu 0 --train     dataset.root_dir=$1
python launch.py --config configs/bakedsdf-colmap.yaml --gpu 0 --train     dataset.root_dir=$1 \
                --resume_weights_only --resume latest
```

# Export BakedSDF!
```
python export.py --exp_dir ./exp/${exp_name}/${trail-name}
```
for example, when we want to export neus-colmap data, we could run
```
python export.py --exp_dir ./exp/neus-colmap-stump/@20230907-133647
```

# Bring Bakedsdf into your APP!
## On Unity and Unreal
You can use BakedSDF2FBX to convert the exported glb and import them into the sample projects of Unity and Unreal
* BakedSDF2FBX:
http://github.com/AyoubKhammassi/BakedSDF2FBX
* UnityBakedSDF:
http://github.com/AyoubKhammassi/UnityBakedSDF
* UnrealBakedSDF:
http://github.com/AyoubKhammassi/UnrealBakedSDF

## On Web
The web plugin is comming soon. You could currently view you asset by running
```
python viewer.py --data results
```

# Acknowledgement
The code is based on
```
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```
The origin paper:
```
@article{yariv2023bakedsdf,
  title={BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis},
  author={Yariv, Lior and Hedman, Peter and Reiser, Christian and Verbin, Dor and Srinivasan, Pratul P and Szeliski, Richard and Barron, Jonathan T and Mildenhall, Ben},
  journal={arXiv preprint arXiv:2302.14859},
  year={2023}
}
```
