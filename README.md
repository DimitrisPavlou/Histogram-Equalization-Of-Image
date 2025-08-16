## 🌈 Project Overview

This project implements two fundamental **image enhancement algorithms** based on histogram equalization:

1. **Global Histogram Equalization (HE)**  
   - Adjusts the contrast of the entire image based on its global histogram.  

2. **Adaptive Histogram Equalization (AHE)**  
   - Enhances contrast locally by applying histogram equalization to contextual regions of the image.  

---

## ⚙️ Implementation Notes

- The **AHE algorithm** does not yet handle border cases, which may lead to visible discontinuities between inner contextual regions and border contextual regions.  
- The algorithm assumes that the image dimensions are **perfectly divisible by the contextual region size**.  
  - If this condition is not met, an **AssertionError** will be raised.  

---

## 📂 Usage

- A demo script (`demo.py`) is provided.  
- By default, it uses the `input_img.png` file. You can replace this with your own images.  
- **Important:** Ensure that the chosen contextual region size divides the image dimensions evenly.  

---

## 🛠️ Implementation Details

- Written in **Python** using **NumPy**.  
- Includes two helper functions written from scratch:  
  - `image_histogram` → counts the frequency of pixel values.  
  - `cumsum` → computes the cumulative sum needed for histogram equalization.  

---

## 🖼️ Test Images

A set of **test images** is included to validate and demonstrate the algorithms.  
