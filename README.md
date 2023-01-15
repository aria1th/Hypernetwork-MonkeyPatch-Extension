# Hypernetwork-MonkeyPatch-Extension
Extension that patches Hypernetwork structures and training
![image](https://user-images.githubusercontent.com/35677394/210898033-44da3cdb-a501-4cb3-a176-07ff8548d699.png)

![image](https://user-images.githubusercontent.com/35677394/203494809-9874c123-fca7-4d14-9995-63dc8772c920.png)

For Hypernetwork structure, see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4334

For Variable Dropout, see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4288


### Train_Beta(now, train_gamma) tab allows some more options with improved training.
![image](https://user-images.githubusercontent.com/35677394/203494907-68e0ef39-4d8c-42de-ba2e-65590375c435.png)

### Features

**No-Crop Training**
![image](https://user-images.githubusercontent.com/35677394/203495373-cef04677-cdd6-43b0-ba42-d7c0f3d5a78f.png)
You can train without cropping images. 

**Fix OSError while training**

**Unload Optimizer while generating Previews**




### Create_Beta_hypernetwork allows creating beta hypernetworks.

Beta hypernetworks* can contain additional informations and specified dropout structures. It will be loaded without extension too, but it won't load dropout structure, so training won't work as original. Generating images should work identically.

This extension also overrides how webui loads and finds hypernetworks, to use variable dropout rates, and etc.
Thus, hypernetwork created with variable dropout rate might not work correctly in original webui.

Well, at least now it should work, without having any problem except you cannot use variable dropout rate in original webui. If you have problem with loading hypernetworks, please create an issue. I can submit pr to original branch to load these beta typed hypernetworks correctly.

### Training features are in train_gamma tab.
![image](https://user-images.githubusercontent.com/35677394/204087550-94b8e7fb-70cb-4157-96bc-e022340901c9.png)

If you're unsure about options, just enable every checkbox, and don't change default value. 


### CosineAnnealingWarmupRestarts
![image](https://user-images.githubusercontent.com/35677394/204087530-b7938e7e-ebe5-4326-b5cd-25480645a11b.png)

This also fixes some CUDA memory issues. Currently both Beta and Gamma Training is working very well, as far as I could say.


### Hyperparameter Tuning
![image](https://user-images.githubusercontent.com/35677394/212574147-22a32b03-6544-4aee-9ac7-fdefd2b7ee56.png)
Now you can save hypernetwork generation / training setting, and load it in train_tuning tab. This will allow combination of hypernetwork structures, and training setups, to find best way for stuff. 

### CLIP change test tab
![image](https://user-images.githubusercontent.com/35677394/212574217-3dd08007-e33f-4179-96e9-5a90bccd4907.png)
Now you can select CLIP model, its difference is significant but whether its better or not is unknown.



### Planned features
Allow using general models with .safetensor save /loading


### Some personal researches

We cannot apply convolution for attention, it does do something, but hypernetwork here, only affects attention, and its different from 'attention map' which is already a decoded form(image BW vectors) of attention(latent space). Same goes to SENet, unfortunately.
