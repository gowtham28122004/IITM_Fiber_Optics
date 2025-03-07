# IITM Fiber Optics DAS: Machine Learning for Distributed Acoustic Sensing

## Project Overview

This repository contains a machine learning solution for classifying activities using Distributed Acoustic Sensing (DAS) data from fiber optic cables. DAS technology transforms standard optical fibers into thousands of virtual microphones, detecting vibrations and acoustic signals along the entire fiber length.

### Key Features
- Processing and classification of massive DAS datasets (2 billion+ data points)
- High-performance CNN+LSTM hybrid model optimized for time-series signal classification
- Multi-class classification of activities: Bike Throttle, Jackhammer, Jumping, and Walking
- Advanced data augmentation techniques for improved model generalization
- Memory-efficient implementation for handling large datasets

## Dataset

The original dataset consisted of 8 raw files, each containing approximately 250 million data points of fiber optic sensing data, for a total of ~2 billion points. These represent acoustic signals captured from the fiber optic sensing system during various activities.

### Data Processing Pipeline

1. **Standardization**: Raw signals were standardized independently to normalize the amplitude ranges.
2. **Visualization**: Line plots were created to identify patterns and signal characteristics for each activity.
3. **Labeling**: Each file was associated with specific activity labels (Bike Throttle, Jackhammer, Jumping, Walking).
4. **Batching & Reshaping**: Data was converted from a (2B,1) vector to (80K,25K) format, creating 80,000 samples of 25,000 time points each.
5. **Downsampling**: For model training efficiency, signals were further downsampled to 5,000 points per sample.

## Model Architecture

The model uses a hybrid CNN+LSTM architecture specifically designed to balance complexity and memory efficiency:

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_layer (InputLayer)    [(None, 5000, 1)]           0         []                            
                                                                                                  
 conv1d (Conv1D)             (None, 2500, 16)            128       ['input_layer[0][0]']         
                                                                                                  
 batch_normalization (Batch  (None, 2500, 16)            64        ['conv1d[0][0]']              
 Normalization)                                                                                   
                                                                                                  
 activation (Activation)     (None, 2500, 16)            0         ['batch_normalization[0][0]'] 
                                                                                                  
 conv1d_1 (Conv1D)           (None, 2500, 16)            784       ['activation[0][0]']          
                                                                                                  
 batch_normalization_1 (Bat  (None, 2500, 16)            64        ['conv1d_1[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 conv1d_2 (Conv1D)           (None, 2500, 16)            32        ['input_layer[0][0]']         
                                                                                                  
 batch_normalization_2 (Bat  (None, 2500, 16)            64        ['conv1d_2[0][0]']            
 chNormalization)                                                                                 
                                                                                                  
 add (Add)                   (None, 2500, 16)            0         ['batch_normalization_1[0][0]'
                                                                   , 'batch_normalization_2[0][0]
                                                                   ']                             
                                                                                                  
 activation_1 (Activation)   (None, 2500, 16)            0         ['add[0][0]']                 
                                                                                                  
 spatial_dropout1d (Spatial  (None, 2500, 16)            0         ['activation_1[0][0]']        
 Dropout1D)                                                                                       
                                                                                                  
 max_pooling1d (MaxPooling1  (None, 1250, 16)            0         ['spatial_dropout1d[0][0]']   
 D)                                                                                               
                                                                                                  
 conv1d_3 (Conv1D)           (None, 625, 32)             2592      ['max_pooling1d[0][0]']       
                                                                                                  
 batch_normalization_3 (Bat  (None, 625, 32)             128       ['conv1d_3[0][0]']            
 chNormalization)                                                                                 
...
```

### Key Architectural Components:
- **Residual Blocks**: Enabling deeper network training with skip connections
- **Convolutional Layers**: For feature extraction from time-series data
- **Bidirectional LSTM**: Capturing temporal dependencies in both directions
- **Multiple Feature Extraction Paths**: Combining global average pooling, max pooling, and sequential features
- **Regularization**: Carefully tuned dropout, batch normalization, and L1/L2 regularization

## Results

The model achieved excellent performance metrics:

- **Overall Accuracy**: 96.60%
- **Class-wise Metrics**:
  - **Bike Throttle**: Precision 99.81%, Recall 90.62%, F1 Score 94.99%
  - **Jackhammer**: Precision 98.85%, Recall 98.62%, F1 Score 98.74%
  - **Jumping**: Precision 89.64%, Recall 99.30%, F1 Score 94.22%
  - **Walking**: Precision 99.19%, Recall 97.85%, F1 Score 98.51%
- **ROC-AUC**: 99.84% (macro-averaged)
- **Cross-validation**: 96.69% ± 0.28% (5-fold)

### Robustness Testing

The model showed strong resilience to noise:
- Maintained 96.23% accuracy even with 50% noise level (only 0.37% drop)

However, it showed vulnerability to:
- Adversarial attacks: 76.60% accuracy drop with epsilon = 0.01
- Time-series shifts: Performance varies with shift direction and magnitude

## Usage

### Prerequisites
```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

### Model Training
```python
# Example usage
X_TRAIN_PATH = 'path/to/X_train.npy'
Y_TRAIN_PATH = 'path/to/y_train.npy'
X_TEST_PATH = 'path/to/X_test.npy'
Y_TEST_PATH = 'path/to/y_test.npy'

# Create output directory with timestamp
output_dir = f'das_model_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Train the model
models, history, accuracy = train_improved_das_model(
    X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH,
    output_dir=output_dir
)
```

## Future Work
- Implement real-time inference pipeline for live DAS data
- Extend the model to detect additional activity classes
- Develop anomaly detection capabilities for unknown events
- Improve robustness against time-series shifts and adversarial attacks

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Indian Institute of Technology Madras (IITM) for providing the DAS system and data
- Thanks to all contributors and research advisors
