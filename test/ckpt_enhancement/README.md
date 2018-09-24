## File specification

``iphone_trackA.ckpt`` and ``iphone_trackBC.ckpt`` are final submitted checkpoint files in the [PIRM 2018](http://ai-benchmark.com/challenge.html). **(trained with L1 and VGG losses)**

``student.ckpt`` is the better checkpoint that trained without L1 and VGG losses, which is mentioned in the paper.

``teacher.ckpt`` is the teacher model with more parameters and get the higher performance in terms of PSNR and MS-SSIM.

| <sub>Method</sub> | <sub>PSNR</sub> | <sub>MS-SSIM</sub> | <sub>CPU, ms</sub> |
|:---:|:---:|:---:|:---:|
| <sub>Teacher</sub> | <sub>**22.96**</sub> | <sub>**0.9299**</sub> | <sub>1,182</sub> |
| <sub>Student</sub> | <sub>22.54</sub> | <sub>0.9244</sub> | <sub>392</sub> |
| <sub>Student w/o knowledge transfer</sub> | <sub>22.34</sub> | <sub>0.9229</sub> | <sub>445</sub> |
| <sub>Student with L1 and VGG losses</sub> | <sub>22.37</sub> | <sub>0.9231</sub> | <sub>478</sub> |
| <sub>SRCNN-Baseline</sub> | <sub>21.32</sub> | <sub>0.9030</sub> | <sub>1,832</sub> |
| <sub>DPED-Baseline</sub> | <sub>22.17</sub> | <sub>0.9204</sub> | <sub>10,701</sub> |
| <sub>ResNet_8_32</sub> | <sub>22.38</sub> | <sub>0.9156</sub> | <sub>3,226</sub> |
