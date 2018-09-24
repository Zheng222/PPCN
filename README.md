![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
# PPCN (Team: Rainbow)

## Testing

### Super-Resolution Task
First, download the [SR_Test_Datasets](https://drive.google.com/open?id=1_K6mchwDGOQMIXuBIGrlDA4EAYgbtdmU) and put them in ``test/SR_test_data`` folder.

Run the following command to super-resolve low-resolution images

```
python evaluate_super_resolution.py
```
### Enhancement Task

Run the following command to enhance low-quality images
```
python evaluate_enhancement.py
```

## Code References
[1]https://github.com/aiff22/ai-challenge

[2]https://github.com/aiff22/DPED

[3]https://github.com/roimehrez/contextualLoss
