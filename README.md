# ESC_STM32F411RE

CNN-based ESC model on the STM32F411RE using UrbanSound8K.

## Overview

This project implements a Convolutional Neural Network (CNN)-based Environmental Sound Classification (ESC) model on the STM32F411RE ARM. The model is trained using the UrbanSound8K dataset, which contains labeled sound from urban environments. The goal is to classify environmental sounds in real-time on a resource-constrained embedded system.

## Features

- **Pre-trained CNN Model**: A lightweight CNN model optimized for the STM32F411RE.
- **Real-Time Inference**: Classifies audio input in real-time.
- **Dataset**: Utilizes the UrbanSound8K dataset for training and evaluation.
- **Embedded Deployment**: Model quantized and deployed on STM32F411RE using STM32Cube.AI.

## Requirements

- STM32F411RE Nucleo Board
- STM32CubeIDE
- STM32Cube.AI
- Python 3.x (for training and preprocessing)
- TensorFlow/Keras (for model development)
- UrbanSound8K dataset

## Project Structure

```
ESC_STM32F411RE/
├── Dataset/
│   ├── UrbanSound8K/
│   └── Preprocessed/
├── Model/
│   ├── train_model.py
│   ├── model.h5
│   └── quantized_model.tflite
├── STM32/
│   ├── Core/
│   ├── Drivers/
│   └── Inc/
├── README.md
└── LICENSE
```

## Getting Started

1. **Dataset Preparation**:
    - Download the UrbanSound8K dataset from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html).
    - Preprocess the dataset using `train_model.py`.

2. **Model Training**:
    - Train the CNN model using the preprocessed dataset.
    - Save the trained model as `model.h5`.

3. **Model Quantization**:
    - Convert the trained model to TensorFlow Lite format using `tflite_convert`.
    - Quantize the model for embedded deployment.

4. **STM32 Deployment**:
    - Import the quantized model into STM32Cube.AI.
    - Generate the C code and integrate it into the STM32 project.

5. **Build and Flash**:
    - Use STM32CubeIDE to build the project.
    - Flash the firmware onto the STM32F411RE board.

## Usage

- Connect a microphone to the STM32F411RE board.
- Run the firmware to classify real-time audio input.
- The classification results will be displayed via UART or an attached display.

## Future Work

- Improve model accuracy with advanced preprocessing techniques.
- Optimize inference speed for real-time applications.
- Add support for additional datasets and sound classes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UrbanSound8K dataset authors for providing the dataset.
- STM32Cube.AI for enabling embedded AI deployment.
- TensorFlow/Keras for model development tools.
- Open-source contributors for their valuable tools and libraries.
