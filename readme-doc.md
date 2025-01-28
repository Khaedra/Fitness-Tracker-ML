# Barbell Exercise Classification and Rep Counter

## Overview
This project implements a machine learning model that classifies barbell exercises and counts repetitions using accelerometer and gyroscope data. The system processes data from a metaMotion wrist sensor to identify five different exercises and accurately count repetitions.

## Features
- Exercise classification for 5 barbell movements:
  - Bench Press
  - Squat
  - Row
  - Deadlift
  - Overhead Press
- Automated repetition counting
- High accuracy classification (>99%)
- Robust data preprocessing pipeline
- Feature engineering optimization

## Dataset
- 70,000 rows of sensor data
- 5 participants
- Data collected using metaMotion wrist sensor
- Accelerometer and gyroscope measurements

## Technical Implementation

### Data Preprocessing
- Outlier detection methods:
  - Interquartile Range (IQR)
  - Chauvenet's criterion
  - Local Outlier Factor

### Feature Engineering
- Butterworth's Low Pass Filter
- Principal Component Analysis
- Sum of Square attributes
- Temporal Abstraction
- Frequency Abstraction
- Cluster features

### Model Development
- Forward feature selection
- Grid Search hyperparameter optimization
- Models evaluated:
  - Neural Networks
  - Random Forest
  - Naive Bayes
  - Decision Trees
- Final model: Decision Tree with 4 selected features

### Rep Counting Algorithm
- Low-pass filter for signal smoothing
- Extrema detection for rep counting

## Results
- Classification accuracy: >99%
- Classification precision: >99%
- Successful repetition counting implementation




