# Detectron2 PDM

## Usage

- Apply black masks using `black_mask.py`
  - eg. `python folder/to/images mask1.png mask2.png`
- Prepare dataset using `split.cmd` and `zip.cmd`
- Train the Mask R-CNN model using [`Detectron-PDM.ipynb`](https://gist.github.com/skarfie123/1874f1deacf9aaeb83ca9199543ba9ff) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/skarfie123/1874f1deacf9aaeb83ca9199543ba9ff/detectron2-pdm.ipynb)
- Validation data is in the `validation` folder
  - plot these using [TensorBoard](https://github.com/tensorflow/tensorboard)
  - eg. `tensorboard --logdir validation --port 6008`
- Plot graphs from the evaluation results using [`Detectron-PDM-Results.ipynb`](https://gist.github.com/skarfie123/9b08b5041f56526bf7916f20c7177ac6) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/skarfie123/9b08b5041f56526bf7916f20c7177ac6/detectron2-pdm-results.ipynb)
- Evaluation plots are in the `graphs` folder
- [Dataset](https://github.com/skarfie123/detectron2_pdm/releases/tag/d1)

## Acknowledgements

- [Detectron2 Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
- [Original cocosplit](https://github.com/akarazniewicz/cocosplit)
