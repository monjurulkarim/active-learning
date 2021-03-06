# Bridge element segmentation from inspection videos
This is the implementation code for the paper,<a href="https://arxiv.org/pdf/2109.05078.pdf"> "A semi-supervised self-training method to develop assistive intelligence for segmenting multiclass bridge elements from inspection videos"</a>.</p> 


<!-- ![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/40798690/121276061-f15fb700-c89b-11eb-85c0-5e86f19784b8.gif) ![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/40798690/121276072-f6bd0180-c89b-11eb-9b96-7188097ba531.gif) -->

![ezgif com-gif-maker (4)](https://user-images.githubusercontent.com/40798690/121276306-6e8b2c00-c89c-11eb-8fc1-87f2b411011c.gif) ![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/40798690/121360023-0a9b4e80-c902-11eb-9804-d9a6c13f0485.gif)




<!-- ![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/40798690/121276307-6fbc5900-c89c-11eb-89f8-959d9b0c738a.gif) -->

<!-- ## Contents
* Installation
* Dataset Preparation
* Pre-trained models
* Training and Testing
* Citation --->

## Getting Started
* Install the required dependencies: (for reference see [how_to_install.pdf](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/how_to_install.pdf) )
* [Dataset](https://drive.google.com/drive/folders/1vTNgPi2SSefO9fzxHxCa2Sgmec_B8MkM?usp=sharing): Download the initial dataset from here.
* [weights](https://drive.google.com/drive/folders/1wYTNf4nf_79OgqTSOcVc39XVr6s4d9Z-?usp=sharing): Download pre-trained resnet_50 coco weights and trained weights for bridge element segmentation from here.
*  [custom.py](https://github.com/monjurulkarim/active_learning/blob/main/custom.py) : this code is used for loading data and training the model
*  [Training.ipynb](https://github.com/monjurulkarim/active_learning/blob/main/Training.ipynb): loading the weight and calling the training function
*  [inference.ipynb](https://github.com/monjurulkarim/active_learning/blob/main/inference.ipynb): this code is used for inferencing. 
*  [mrcnn/visualize.py](https://github.com/monjurulkarim/active_learning/blob/main/mrcnn/visualize.py) : this code is used for visualizing the segmented bridge elements with mask.


## Citation
If you use this repository, please cite the following paper:

~~~~
@article{karim2021semi,
  title={A semi-supervised self-training method to develop assistive intelligence for segmenting multiclass bridge elements from inspection videos},
  author={Karim, Muhammad Monjurul and Qin, Ruwen and Chen, Genda and Yin, Zhaozheng},
  journal={Structural Health Monitoring},
  pages={14759217211010422},
  year={2021},
  publisher={SAGE Publications Sage UK: London, England}
}
~~~~

Note that part of the codes are referred from <a href="https://github.com/matterport/Mask_RCNN">Mask RCNN</a> project.
