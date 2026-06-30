![Uploading image.png…]()


https://github.com/user-attachments/assets/8231ecef-d571-4111-98dc-4e79d81ca76b

# FindTrack-R³

**FindTrack-R³: Enhancing Decoupled Referring Video Object Segmentation with an R³-Loop and Soft Semantic Alignment**

This is the official repository for the FindTrack-R³ framework. It extends the original decoupled RVOS pipeline by integrating an innovative **R³-Loop (Refine, Requery, Reinforce)** and **Soft Semantic Alignment (SSA)** to guarantee high-quality starting reference masks and maintain strict temporal consistency across the video.

## Abstract
Existing referring VOS methods typically fuse visual and textual features in a highly entangled manner, which leads to challenges in resolving ambiguous target identification and maintaining consistent mask propagation across frames.
To address these issues, we propose **FindTrack-R³**, an enhanced decoupled framework that explicitly separates object identification from mask propagation.
- **Refine & Requery**: We apply Entropy-based Refinement to filter uncertain boundary pixels from the initial EVF-SAM mask, extract a tight bounding box, and requery SAM to produce a perfectly sharp starting anchor.
- **Reinforce (SSA)**: During tracking (via Cutie), we extract L2-normalized frame embeddings and apply Soft Semantic Alignment (SSA) loss against a lightweight rolling FIFO queue. This prevents the tracking memory from drifting to distractors during partial or full occlusions.

## Setup
1. Download the datasets: [Ref-YouTube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282), [Ref-DAVIS17](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions), [MeViS](https://codalab.lisn.upsaclay.fr/competitions/15094).
2. Download [Alpha-CLIP](https://drive.google.com/file/d/1dG_j98hh7AFvhSADlhp9CpoNY-9rBHoc/view?usp=drive_link) weights and place it in the `weights/` directory.

## Running

### Testing
For Ref-YouTube-VOS dataset:
```bash
python run_ytvos.py
```
For MeViS dataset:
```bash
python run_mevis.py
```
For Ref-DAVIS17 dataset:
```bash
python run_davis.py
```

### Modern Web Application (FastAPI + Next.js)
This repository includes a production-ready, modern AI SaaS web application that provides a beautiful UI for segmenting objects using the FindTrack-R³ pipeline.

#### 1. Start the Backend (FastAPI)
The backend manages the Cutie and EVF-SAM models, serving the Next.js static files concurrently.
```bash
python api.py
```
This will start Uvicorn at `http://0.0.0.0:8000`.

#### 2. Start the Frontend (Next.js)
If you need to make changes to the frontend UI, navigate to the `rvos-frontend` directory and start the development server:
```bash
cd rvos-frontend
npm install
npm run dev
```

To build a static version to serve with FastAPI:
```bash
cd rvos-frontend
npm run build
```

## Contact
Code and models are only available for non-commercial research purposes 
ijazu4412@gmail.com
