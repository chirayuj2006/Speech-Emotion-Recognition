# Speech-Emotion-Recognition
A modular real-time Speech Emotion Recognition pipeline with completed audio capture, feature extraction, and preprocessing, ready for TensorFlow model integration.
The system captures live microphone audio, preprocesses it via a low-latency streaming pipeline, and extracts emotion-relevant acoustic features to generate model-ready inputs for a deep learning classifier. The architecture is proposed to be modular, extensible, and model-agnostic, allowing the easy integration of a trained TensorFlow/Keras model in the final stage.

Currently, the whole audio pipeline and preprocessing framework is ready, with only the training and integration of the neural network model to be completed.

