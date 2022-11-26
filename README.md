# Hypernetwork-MonkeyPatch-Extension
Extension that patches Hypernetwork structures and training
![image](https://user-images.githubusercontent.com/35677394/203494809-9874c123-fca7-4d14-9995-63dc8772c920.png)

For Hypernetwork structure, see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4334

For Variable Dropout, see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4288


### Train_Beta tab allows some more options with improved training.
![image](https://user-images.githubusercontent.com/35677394/203494907-68e0ef39-4d8c-42de-ba2e-65590375c435.png)

### Features

**No-Crop Training**
![image](https://user-images.githubusercontent.com/35677394/203495373-cef04677-cdd6-43b0-ba42-d7c0f3d5a78f.png)
You can train without cropping images. 

**Fix OSError while training**

**Unload Optimizer while generating Previews**




### Create_Beta_hypernetwork allows creating beta hypernetworks.


This extension also overrides how webui loads and finds hypernetworks, to use variable dropout rates, and etc.
Thus, hypernetwork created with variable dropout rate might not work correctly in original webui.

### Train_Gamma tab uses Gradient Accumulation from separate PR.(as well as beta functionalities)
![image](https://user-images.githubusercontent.com/35677394/204087550-94b8e7fb-70cb-4157-96bc-e022340901c9.png)



### CosineAnnealingWarmupRestarts
![image](https://user-images.githubusercontent.com/35677394/204087530-b7938e7e-ebe5-4326-b5cd-25480645a11b.png)

This also fixes some CUDA memory issues. Currently both Beta and Gamma Training is working very well, as far as I could say.
