# ComfyUI_FL-Trainer

![image](https://github.com/user-attachments/assets/896b2885-0b04-48a7-a9fe-763d9e2ec3c7)


Train Image Loras on both sd1.5 and SDXL. This repo git clones the pieces needed to train. It pops open a second terminal window do do the training.
It will also display the inference samples in the node itself so you can track the results.

# Easy Trainer
![image](https://github.com/user-attachments/assets/8a6d8eac-1517-440c-83be-38c70faf1574)



Stripped-down version of the trainer for people who want settings that just work.
recommended settings in the screenshot! Load in your images and captions and let it rip.

# Some things to know.

Iv tested the nodes on a cloud instance and it doesn't work as fully intended. When you connect everything and hit Queue for the first time it will create a folder in the comfy outputs folder named "FL_train_workspaces". In this folder, you will see a few things.
- Folder Named after your Lora input
- Output folder with your epochs
- Sample_image folder where your samples are stored
- Train images folder which is your data and captions
- config for the train
- and a bat and sh file.

If your on a cloud instance and you setup the train in comfy and hit queue, it will run through the nodes and stop the queue like nothing happened. But the folder structure will be created. 

If you navigate to the bat or the sh file in the workspace folder, you can manually run the SH and BAT file to start the train effectively in a new terminal window without comfy front end needed. You can also do this locally as well! 

# Things I want to add.

- More simplified nodes are similar to the easy train, with some focus on style and characters. Super easy 1 click trainers for people who just want a good result and dont wanna mess with anything or waste time.
- Chainable training. If you notice on the main Kohya nodes, you can start training from an epoch or even a random Lora. With this in mind im going to build out chainable instances of training that allow you to efficiently modulate the training settings through the training process with different settings, learning rates, and even training data!

