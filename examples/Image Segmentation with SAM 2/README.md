# Text-Driven Image Segmentation with SAM 2

This example demonstrates **text-prompted image segmentation** using the **Segment Anything Model 2 (SAM 2)**.
You can specify an object in the image via a **text description**, and the model automatically segments that region.

---

## 🧠 Overview

Text-driven segmentation allows you to extract a specific object or region from an image by providing a natural language prompt.

This implementation integrates SAM 2 with a grounding model (like GroundingDINO/GLIP) to link text to image regions.

---

## ⚙️ Requirements

Install dependencies before running the script:

```bash
pip install opencv-python-headless matplotlib pillow tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git@main
pip install git+https://github.com/IDEA-Research/GroundingDINO.git@main
pip install --upgrade roboflow albumentations
```

---

## 🚀 How to Run

Run the segmentation example script:

```bash
python inference_example.py --image-path path/to/image.jpg --text-prompt "segment the person"
```

You can also modify the script to test different input images or prompts.

---

## 🖼️ Example Output

Upload your segmented image result below:

```
![Segmented Output](data/segmented_result.jpg)
```

<img width="794" height="536" alt="download" src="https://github.com/user-attachments/assets/f0ecb469-f330-4a4c-ac70-8f18105501e7" />

---

## 📁 File Structure

```
examples/
└── text_driven_segmentation/
    ├── README.md
    ├── requirements.txt
    ├── inference_example.py
    └── setup.sh
```

---

## 💡 Notes

* The accuracy depends on the grounding model and SAM’s segmentation mask quality.
* For better results, ensure images are clear and objects are well-separated.


This example builds on:

* [Segment Anything Model (SAM 2)](https://github.com/facebookresearch/segment-anything)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
* [Supervision](https://github.com/roboflow/supervision)
