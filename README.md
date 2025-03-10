# CILv3D
This repository contains the implementation described in the paper: CILv3D - Scaling Multi-Frame Transformers for End-to-End Driving.

**CILv3D** is a state-of-the-art vision-based end-to-end controller for autonomous driving, designed to improve generalization, stability, and driving performance using multi-view video inputs and advanced transformer architectures. Built on the CARLA simulator, it achieves superior results in complex driving scenarios while reducing training time and computational costs.

## Key Features
- **Multi-View Processing**: Utilizes synchronized left, front, and right camera views for 180° horizontal field-of-view (HFOV).
- **3D Convolutions & UniFormer Backbone**: Captures spatio-temporal dependencies with 3D convolutions and the UniFormer Transformer model for efficient feature extraction.
- **Enhanced Dataset**: Trained on a diverse CARLA 0.9.15 dataset with dynamic weather, traffic, and pedestrian behavior.
- **Leaderboard Performance**: Outperforms prior methods (CIL++, CILRL++) in CARLA Leaderboard metrics, including driving score, route completion, and collision reduction.

**This work was presented at [ICAART 2025](https://icaart.scitevents.org/). Read the full paper [here](https://github.com/kochlisGit/CILv3D/blob/main/Scaling_Multi_Frame_Transformers_for_End_to_End_Driving.pdf).**

## Requirements
- numpy>=1.21  
- tensorflow>=2.12  
- tensorflow-addons>=0.22  
- keras-cv-attention-models>=1.3  
- torch>=2.0  
- torchvision>=0.15  
- opencv-python  
- scikit-learn>=1.3  
- pandas>=2.0  
- matplotlib>=3.7  
- tqdm>=4.66   
- carla==0.9.15                

## Data Collection
This guide explains how to collect driving datasets using CARLA simulations.

**1. Prerequisites**
- CARLA simulator running (./CarlaUE4.sh or CarlaUE4.exe).
- Python dependencies installed (see Requirements).

**2. Basic Usage**
Run the dataset builder script:

```python scripts/build_dataset.py```

**Key Configuration (Modify in build_carla_dataset.py)**

```
# Dataset settings  
root_directory = "storage/datasets/carla"  # Where to save data  
town_name = "Town10HD"                     # CARLA map name  
vehicle_model = "vehicle.tesla.model3"     # Vehicle blueprint  
num_episodes = 10                          # Number of driving episodes  
steps_per_episode = 1000                   # Steps per episode  
skips_per_step = 10                        # Save data every N steps  

# Simulation settings  
spectate = True                            # Render simulation window  
enable_dynamic_weather_after_episode = 3   # Randomize weather after episode 3
```
For more advanced changes you should modify the: **WeatherSettings**, **SensorBuilderSettings**, **SimulationSettings** or **TrafficSettings** according do your needs.
These files are located in the ```src/simulators/carla``` directory

## Implemented Models

- CILv2 (The original CIL++ architecture, described in [CIL++](https://github.com/yixiao1/CILv2_multiview)) 
- CILv2 with Video Sequences instead of still images
- CILv2 with UniFormer backbone for enhanced feature extraction
- CILv3D (Our proposed model)

## Training the Models
The training process for the models described in the paper is implemented across several Python scripts, each corresponding to a specific model. Below is an overview of the training pipeline and how to execute it:

**1. Training Scripts** 
The training scripts are organized as follows:

- train_cilv2.py: Trains the CILv2 model.
- train_cilv23d.py: Trains the CILv2 with Video Sequences instead of still images
- train_cilv3.py: Trains theCILv2 with UniFormer backbone for enhanced feature extraction.
- train_cilv3d.py: Trains the CILv3D model.

**2. Training Process**

The training process involves the following steps:

1. Seed Initialization:
   - Random seeds are set for reproducibility using random.seed, np.random.seed, and tf.random.set_seed.
Dataset Loading:
3. Dataset Loading:
   - Training data includes image augmentations and control noise.
   - Validation data is loaded without augmentations.
3. Model Construction:
   - The model is built using the build_model function for the specific architecture.
   - A model summary is printed if verbose=True.
4. Training Execution:
   - The TFTrainer class manages training, including learning rate scheduling, early stopping, and checkpointing.
   - The model is trained using the fit method.

**3. Running the Scripts**

To train a model, run the corresponding script from the command line. For example:

```
python train_cilv2.py
```

Each script will:

- Load the dataset.
- Build the model.
- Train the model using the specified hyperparameters.
- Save checkpoints and logs to the directory specified in config.checkpoint_dir.

**4. Post-Training**

After training:

- The trained model and training history are returned by the TFTrainer.fit method.
- The control normalizer (if used) is saved as a pickle file in the checkpoint directory for later use during inference.

**5. Notes**

- Ensure TensorFlow and other dependencies are installed.
- Adjust the train_town_list and test_town_list variables to customize the dataset used for training and validation.
- Modify the configuration parameters in the respective Config classes to experiment with different training setups.
  
This training pipeline provides a framework for training and evaluating the various CIL architectures described in the paper. For further details, refer to the individual training scripts and the configuration files.

## Video Demonstration

Watch our model in action:

[![CILv3D Demo](https://img.youtube.com/vi/65k9P3mIkcY/0.jpg)](https://www.youtube.com/watch?v=65k9P3mIkcY)

## Future Work 🚀

While CILv3D has demonstrated significant improvements over previous approaches, there are several directions for future research that could further enhance its performance:

- **360° Environmental Awareness**: Extending the current multi-view setup to include full panoramic vision by integrating additional cameras or LiDAR-like depth estimations.
- **Reinforcement Learning Fine-Tuning**: Investigating hybrid learning strategies that combine imitation learning with reinforcement learning to further refine decision-making in dynamic scenarios.

We welcome contributions and collaborations to push the boundaries of end-to-end autonomous driving! 🚗💨  

## Contact 📩

If you have any questions, feedback, or collaboration ideas, feel free to reach out:  

👤 **Vasileios Kochliaridis**  
📧 vkochlia@csd.auth.gr  

👤 **Filippos Moumtzidellis**  
📧 philipm124@live.com
