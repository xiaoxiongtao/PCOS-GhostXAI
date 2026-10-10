# PCOS-GhostXAI

Research scripts for ovarian ultrasound image classification with a compact GhostNet backbone, model comparisons, and Grad-CAM visualizations. Some filenames were made descriptive.

## Script map

| Stage | File | Purpose |
| --- | --- | --- |
| Raw data review | `analyze_raw_dataset.py` | Inspect class counts, resolutions, MD5 duplicate groups, and image blur scores. It reports findings; it does not remove duplicates. |
| Dataset preparation | `prepare_dataset.py` | Remove exact MD5 duplicates from the input lists and copy an 80/20 split into `train` and `val` folders. |
| Dataset check | `count_cleaned_images.py` | Count images in each class and split. |
| Preprocessing figure | `plot_preprocessing_steps.py` | Show grayscale input, CLAHE enhancement, and Gaussian smoothing. |
| Model definition | `ghostnet.py` | Define the local GhostNet classifier. |
| Training | `train_models.py` | Select a model, preprocess images, train, save the best validation checkpoint, and plot smoothed accuracy/loss curves. |
| Evaluation | `evaluate_models.py` | Load a checkpoint and plot a confusion matrix, ROC curve, and precision–recall curve. |
| Efficiency | `profile_model_efficiency.py` | Estimate parameters, FLOPs, latency, and GPU memory for the model candidates. |
| Architecture table | `ghostnet_architecture_table.py` | Print a layer-level GhostNet parameter and output-shape table. |
| Explanation | `gradcam_triple_panel.py` | Render the original image, overlaid activation heatmap, and red contour view. |

## Running notes

- The manuscript compares **10 models**. `train_models.py` currently selects `ghostnet` and runs `range(1, 2)` (one run), while the manuscript describes 10 independent runs of 50 epochs per model. These defaults do not directly reproduce the reported table.
- Some absolute AutoDL paths differ between scripts (`/root/autodl-tmp/demo/PCOS_Final_Ready` and `/root/autodl-tmp/PCOS_Final_Ready`). Check the paths and model selection in each script before running. `train_models.py` may remove unreadable images from its configured training directory.
- `evaluate_models.py` generates a confusion matrix and ROC/PR plots for a selected checkpoint; it does not aggregate precision, recall, F1, and all metrics across 10 runs.
- The 3,996 ultrasound images and trained checkpoints are not included in this code repository. The dataset is cited in the manuscript.
- Dependencies used across the scripts include PyTorch, torchvision, timm, OpenCV, Pillow, NumPy, SciPy, scikit-learn, Matplotlib, seaborn, pandas, and fvcore. No verified dependency lockfile or runnable environment was supplied.

The Python source has been checked for syntax. Model training and inference were not run in this workspace.
