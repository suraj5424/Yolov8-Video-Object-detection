# 🎞📸 Video Frame Processing and 🎯🕵️ Object Detection
-----
This project delivers a complete pipeline for **video frame extraction and object detection** using the **YOLOv8** model. It's packaged as a single Jupyter Notebook (`.ipynb`) for easy execution in environments like Kaggle, demonstrating a full end-to-end computer vision solution.

-----

## 🚀 Key Methodologies & Approaches (Skills Highlighted)

  * **Automated Video Processing**: Efficiently extracts and pre-processes (resize, rotate) frames from multiple videos using `OpenCV`, ensuring consistent data input.
  * **Custom Data Annotation**: Leverages **MakeSense.ai** for precise manual bounding box annotation, demonstrating expertise in preparing high-quality, YOLO-formatted datasets for specialized detection tasks.
  * **Advanced Object Detection (YOLOv8)**: Implements and fine-tunes **YOLOv8m** for custom object recognition. Demonstrates skills in model training, hyperparameter tuning, and rigorous performance evaluation using industry-standard metrics (P, R, mAP@50, mAP@50-95).
  * **Inference & Visualization**: Applies the trained model for real-time object detection on new videos. Visualizes results with overlaid bounding boxes and generates animated video outputs, showcasing practical application and interpretability.
  * **Structured Output Management**: Organizes all generated data (frames, models, results) and provides a single, downloadable archive, highlighting strong project management and deployment readiness.

-----

## 🛠️ Technologies Used

  * **Python**: Core scripting and automation.
  * **OpenCV**: Video/image processing.
  * **`ultralytics`**: YOLOv8 framework.
  * **`matplotlib`**: Data visualization.
  * **MakeSense.ai**: (External) Data annotation.

-----

## ⚙️ Project Setup & Run

### Dataset Collection Process

To ensure a diverse and comprehensive dataset for robust model training, the following collection process was employed:

1.  **Object Selection**: Two distinct toys were chosen as the target objects for detection.
2.  **Video Recording**: Separate video recordings were conducted for:
      * Each toy individually.
      * Both toys together in a single frame.
3.  **Angle and Pose Diversity**: Care was taken to capture the toys from multiple angles and various poses within the frame.
4.  **Lighting Variation**: Videos were recorded under a range of lighting conditions, including both bright and dim environments, to enhance the model's ability to generalize across different real-world scenarios.
5.  **Frame Extraction**: Each video was recorded at 30 frames per second (FPS). A Python script was then used to systematically extract individual frames from these videos, converting the raw video data into static image samples.
6.  **Manual Annotation**: The extracted images were subsequently annotated using the **MakeSense.ai** tool. This involved drawing accurate bounding boxes around each instance of the target objects within the frames and assigning the corresponding class labels.

This meticulous process resulted in a complete, well-labeled dataset with diverse image samples, crucial for effective machine learning model training and achieving high detection accuracy in varied conditions.

### Input Data Structure & Dataset Description

The `.ipynb` notebook expects data in these paths (typical for Kaggle):

```
.
├── /kaggle/input/input_video/        # Source videos
└── /kaggle/input/object-detection/   # Annotated dataset (from MakeSense.ai) & test video
    ├── data.yaml
    ├── images/                       # Directory containing all training/validation images
    └── labels/                       # Directory containing YOLO format annotation files
    └── testing_video.mp4             # Video for inference
```

**About the Object Detection Dataset (YOLO Format):**
The dataset within `/kaggle/input/object-detection/` adheres to the standard **YOLO format**. This means for every image (`.jpg`, `.png`, etc.) in the `images/` folder, there is a corresponding text file (`.txt`) with the exact same name in the `labels/` folder. Each line in these `.txt` annotation files represents one detected object and follows the format:

`class_id center_x center_y width height`

Where:

  * `class_id`: An integer representing the object's class (e.g., 0 for `tiger_toy`, 1 for `superman_toy`).
  * `center_x`, `center_y`: Normalized (0 to 1) coordinates of the bounding box center.
  * `width`, `height`: Normalized (0 to 1) dimensions of the bounding box.

A `data.yaml` file further defines the dataset by listing the paths to training/validation images, the number of classes, and the human-readable names for each `class_id`. This structured format is essential for training YOLO models effectively.

### How to Run

1.  **Upload the `.ipynb` file** to your Jupyter environment (Kaggle, Google Colab, or your local Jupyter setup).
2.  **Ensure Input Data is Configured** at the expected `/kaggle/input/` paths (or their local equivalents). For local setups, create `input_video/` and `object-detection/` directories at the same level as your notebook and place the data inside.
3.  **Run All Cells** sequentially within the notebook. The entire pipeline, from frame extraction to model training, inference, and result generation, is automated.

-----

## 📊 Results and Output

Upon successful execution of the notebook cells, the project yields comprehensive results:

  * **`created_frames/`**: Contains systematically extracted, resized, and rotated `.png` frames, organized into video-specific subfolders.
  * **YOLOv8 Model Training Performance**:
      * **Training Duration**: 100 epochs completed in approximately $0.228$ hours.
      * **Resource Utilization**: Peak GPU Memory usage was $7.28\\text{G}$.
      * **Validation Metrics (Demonstrating High Accuracy)**:
          * **Overall Dataset Performance**:
              * Precision (P): $0.997$
              * Recall (R): $1.0$
              * mAP@50: $0.995$
              * mAP@50-95: $0.98$
          * **Class-specific Performance**:
              * `tiger_toy`: P: $0.998$, R: $1.0$, mAP@50: $0.995$, mAP@50-95: $0.979$
              * `superman_toy`: P: $0.997$, R: $1.0$, mAP@50: $0.995$, mAP@50-95: $0.981$
      * The `best.pt` model weights, optimized during training, are saved within the notebook's output directory.
  * **Object Detection Inference Output**: The `testing_video.avi` with overlaid bounding box detections is saved to `/kaggle/working/runs/detect/yolov8m_custom2/`.

-----
