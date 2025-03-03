### Sign-O-Voice

## Overview
Sign-O-Voice is an advanced system designed to bridge the communication gap between individuals using Indian Sign Language (ISL) and those unfamiliar with it. Utilizing a combination of computer vision, deep learning, and natural language processing, Sign-O-Voice translates ISL signs into English sentences and further converts them into speech for seamless communication.

## Methodology

# A. Data Preprocessing
The collected key-points of ISL were pre-processed before being input into the neural network. The Pose Contours were resized, and key-points below the shoulders were removed, as the lower body does not contribute to sign language. This step reduced unnecessary data and improved efficiency.

To account for variations in user positioning within the camera frame, key-points were normalized using the nose as a central reference:

pose_keypoint = pose_keypoint - nose_keypoint
pose_keypoint = pose_keypoint / shoulder_distance_euclidean

Similarly, hand landmarks were normalized with respect to the wrist:

left_hand_keypoint = left_hand_keypoint - left_wrist_keypoint
right_hand_keypoint = right_hand_keypoint - right_wrist_keypoint

# B. Bi-LSTM Networks
Sign-O-Voice employs Bi-LSTM (Bidirectional Long Short-Term Memory) networks to classify ISL signs. Unlike standard LSTMs, Bi-LSTMs process data in both forward and backward directions, enabling the model to capture long-term dependencies in sequential data.

# C. Training
The model was implemented using TensorFlow's Sequential API. The architecture consists of:
1. CNN input layer for feature extraction.
2. Three Bi-LSTM layers with 128, 128, and 64 neurons, respectively.
3. Batch Normalization and Dropout layers (40% and 30%) to prevent overfitting.
4. L2 regularization to enhance generalization.
5. The loss function with L2 regularization:
   
The model was compiled using the Adam optimizer and trained for 100 epochs, achieving:
Training Accuracy: 99.94%
Testing Accuracy: 99.69%

# D. English Sentence Generation
While Bi-LSTM classifies individual ISL words, full communication requires structured English sentences. Since ISL syntax differs from English, traditional NLP techniques like syntax restructuring were ineffective. Instead, we utilized:
Llama 3 (8B parameters) with Few-Shot Prompting to map raw ISL sentences to structured English.

# E. Text-to-Speech and Reversed Translation
Once ISL signs are translated into English, Google Text-to-Speech (gTTS) converts the text into audio. Additionally, spoken English can be converted back into ISL signs through:
1. Speech Recognition API to transcribe speech.
2. Llama 3 to generate raw ISL sentences from English.
3. OpenCV to visualize sign key-points.

## Installation

  pip install tensorflow numpy mediapipe opencv-python gtts sklearn matplotlib

## Usage
1. Capture ISL signs using a webcam.
2. Translate signs into English sentences.
3. Convert translated text into speech.
4. Reverse process for speech-to-sign translation.

## Results
1. Real-time ISL to speech conversion
2. 99%+ accuracy in sign classification
3. Seamless integration of Generative AI for sentence structuring
