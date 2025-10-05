"""
DGA Domain Classifier

A high-performance CPU-optimized classifier for detecting Domain Generation Algorithm (DGA) domains
using lexical and semantic features with incremental learning capabilities.

Project Structure:
- data_loader.py: Data loading and chunked processing utilities
- feature_extractor.py: Lexical and semantic feature extraction
- trainer.py: Incremental model training with LightGBM
- predictor.py: Fast inference interface
- main.py: Main training script with monitoring
- utils.py: Logging and performance utilities
"""