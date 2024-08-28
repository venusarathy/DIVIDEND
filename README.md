
## **DIVIDEND 🤑**
 *Means* - Dynamic Integration for Video Improvement and Digital Eradication of Non-Desired Data


### FUNTIONALITY

Fr ? Read Description for functionality



### Features
- **Automated Watermark Detection**
- **High-Resolution Output**
- **Temporal Consistency** 
- **Plug-and-Play Integration**
- **Scalable and Robust**

---

### Project Structure (ofcourse i'd use gpt what do you think am i ? a legacy coder uhh ??)
```
DIVIDEND/
│
├── datasets/                # Training and testing datasets
│   ├── train/
│   ├── test/
│
├── models/                  # Model architectures
│   ├── unet_attention.py     # U-Net with attention mechanism
│   ├── discriminator.py      # GAN-based discriminator
│   └── temporal_network.py   # Temporal consistency model (3D CNN)
│
├── scripts/                 # Training and utility scripts
│   ├── train.py              # Training script
│   ├── test.py               # Testing script
│   └── utils.py              # Utility functions (data loading, pre-processing)
│
├── checkpoints/             # Saved model weights
├── results/                 # Output video frames
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

### Getting Started

#### 1. **Clone the repository:**
```bash
git clone https://github.com/venusarathy/dividend.git
```

#### 2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

#### 3. **Prepare your dataset:**
- Place your watermarked and non-watermarked video frames in the `datasets/` directory. 
- Ensure the dataset is properly structured into `train/` and `test/` directories.

#### 4. **Train the model:**
```bash
python scripts/train.py
```

#### 5. **Test the model:**
```bash
python scripts/test.py
```

---

### Models

1. **U-Net with Attention** (`unet_attention.py`):
   - Detects and removes watermarks using an attention mechanism to focus on watermark regions.
   
2. **Temporal Consistency Network** (`temporal_network.py`):
   - Ensures smooth transitions between video frames using 3D convolutions.

3. **Discriminator** (`discriminator.py`):
   - Used in GAN architecture to enhance the realism of generated frames during adversarial training.

---

### Contributions
- Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.
- See you on the pull request tab :)
---

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### Author
**Venu Sarathy**  
[GitHub Profile](https://github.com/venusarathy)

---

### Acknowledgments
 **Special thanks to all contributors and supporters.**
 
![](https://github.com/venusarathy/nord-dotfiles/blob/main/templates/not_yet.gif)
