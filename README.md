# CILv3D
This repository contains the implementation described in the paper: CILv3D - Scaling Multi-Frame Transformers for End-to-End Driving

**CILv3D** is a state-of-the-art vision-based end-to-end controller for autonomous driving, designed to improve generalization, stability, and driving performance using multi-view video inputs and advanced transformer architectures. Built on the CARLA simulator, it achieves superior results in complex driving scenarios while reducing training time and computational costs.

## Key Features
- **Multi-View Processing**: Utilizes synchronized left, front, and right camera views for 180° horizontal field-of-view (HFOV).
- **3D Convolutions & UniFormer Backbone**: Captures spatio-temporal dependencies with 3D convolutions and the UniFormer Transformer model for efficient feature extraction.
- **Enhanced Dataset**: Trained on a diverse CARLA 0.9.15 dataset with dynamic weather, traffic, and pedestrian behavior.
- **Leaderboard Performance**: Outperforms prior methods (CIL++, CILRL++) in CARLA Leaderboard metrics, including driving score, route completion, and collision reduction.

**This work was presented at [ICAART 2025](https://icaart.scitevents.org/). Read the full paper [here](...).**

TODOs
1. Project Description
2. Requirements
3. Collect Data
4. Models
5. Train.py files Description
6. Video Demo somewhere
7. Future Work
