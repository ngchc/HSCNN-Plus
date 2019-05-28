HSCNN+: Advanced HyperSpectral CNN
====
Zhan Shi, Chang Chen, Zhiwei Xiong, Dong Liu, Feng Wu. [HSCNN+: Advanced CNN-Based Hyperspectral Recovery from RGB Images](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Shi_HSCNN_Advanced_CNN-Based_CVPR_2018_paper.pdf). In CVPR Workshop 2018. (Winner of NTIRE Challenge on Spectral Reconstruction from RGB Images) <br/>

## Test the pre-trained models
Usage example to test HSCNN-D model for clean RGB images <br/>
```
cd hscnn-d_clean
cd inference/models && unzip *.zip && cd ../
/bin/bash demo.sh
```
Usage example to test HSCNN-D model for real-world RGB images <br/>
```
cd hscnn-d_real
cd inference/models && unzip *.zip
cat *.tar.gz.* | tar -xzv && cd ../
/bin/bash demo.sh
```
To access the validation dataset and the reconstructed results of HSCNN-D <br/>
Download the NTIRE2018_Validate folder from <br/>
[http://pan.bitahub.com/index.php?mod=shares& <br/>
sid=eTJ2bFFQR3BzTm5FTGxjcC1WUWk3TXRsbGo3YTBjYi05SWVvSlE](http://pan.bitahub.com/index.php?mod=shares&sid=eTJ2bFFQR3BzTm5FTGxjcC1WUWk3TXRsbGo3YTBjYi05SWVvSlE)

## Train the model
Usage example to train a new model for clean RGB images <br/>
```
cd hscnn-d_clean/train python train.py
```
Usage example to train a new model for real-world RGB images <br/>
```
cd hscnn-d_real/train python train.py
```
