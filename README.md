# ComfyUI_FL-Trainer

![image](https://github.com/user-attachments/assets/896b2885-0b04-48a7-a9fe-763d9e2ec3c7)


Train Image Loras on both sd1.5 and SDXL. This repo git clones the pieces needed to train. It pops open a second terminal window do do the training.
It will also display the inference samples in the node itself so you can track the results.

# Easy Trainer
![image](https://github.com/user-attachments/assets/8a6d8eac-1517-440c-83be-38c70faf1574)



Stripped-down version of the trainer for people who want settings that just work.
recommended settings in the screenshot! Load in your images and captions and let it rip.


# Things I want to add.

- More simplified nodes are similar to the easy train, with some focus on style and characters. Super easy 1 click trainers for people who just want a good result and dont wanna mess with anything or waste time.
- Chainable training. If you notice on the main Kohya nodes, you can start training from an epoch or even a random Lora. With this in mind im going to build out chainable instances of training that allow you to efficiently modulate the training settings through the training process with different settings, learning rates, and even training data!

