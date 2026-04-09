# Cima Geometry AI (GATP)

A computer vision pipeline that detects geometric shapes from images and converts them into structured training data.

---

## Overview

This project implements a lightweight vision system using classical computer vision techniques.
It processes raw images, identifies geometric shapes, and exports structured metadata for AI training workflows.

---

## Features

* Contour-based shape detection
* Polygon approximation for geometric classification
* Detection of basic shapes (triangle, rectangle, circle)
* Automatic JSON dataset generation

---

## How It Works

1. Image is converted to grayscale
2. Noise is reduced using Gaussian blur
3. Edges are detected using Canny edge detection
4. Contours are extracted from the image
5. Each contour is approximated into a polygon
6. Shape is classified based on number of vertices
7. Results are saved as structured JSON data

---

## Output Example

```json
[
  {
    "shape": "rectangle",
    "x": 120,
    "y": 80,
    "width": 60,
    "height": 40
  }
]
```

---

## Tech Stack

* Python
* OpenCV

---

## Purpose

Designed to simulate data generation for AI training by converting raw visual input into labeled geometric data.

---

## Run Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python shape_detector.py
```

---

## Architecture

Image → Edge Detection → Contours → Shape Classification → JSON Output

---
