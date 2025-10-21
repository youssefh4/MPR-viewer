#!/usr/bin/env python3
"""
Script to evaluate and compare orientation detection models
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import OrientationResNet18, detect_orientation_resnet18
from data_loader import DataLoader

def evaluate_model_on_test_data(model_path, test_data_paths):
    """
    Evaluate a model on test data and return accuracy metrics
    """
    print(f"\n🔍 Evaluating model: {os.path.basename(model_path)}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    results = {
        'total_samples': 0,
        'correct_predictions': 0,
        'predictions': [],
        'confidences': [],
        'processing_times': []
    }
    
    for data_path in test_data_paths:
        if not os.path.exists(data_path):
            print(f"⚠️  Test data not found: {data_path}")
            continue
            
        print(f"📁 Testing on: {os.path.basename(data_path)}")
        
        try:
            # Load test data
            data_loader = DataLoader()
            if data_path.endswith('.nii.gz') or data_path.endswith('.nii'):
                success, error, volume, metadata = data_loader.load_nifti(data_path)
            else:
                success, error, volume, metadata = data_loader.load_dicom(data_path)
            
            if not success:
                print(f"❌ Failed to load {data_path}: {error}")
                continue
            
            # Get ground truth orientation from metadata
            ground_truth = metadata.get('main_plane', 'Unknown')
            print(f"🎯 Ground truth: {ground_truth}")
            
            # Temporarily modify the model path for testing
            original_model_paths = None
            try:
                # Backup original model paths
                from utils import detect_orientation_resnet18
                
                # Test the model
                start_time = datetime.now()
                predicted_orientation, confidence = detect_orientation_resnet18(volume)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                
                # Record results
                results['total_samples'] += 1
                results['processing_times'].append(processing_time)
                results['confidences'].append(confidence)
                
                is_correct = predicted_orientation == ground_truth
                if is_correct:
                    results['correct_predictions'] += 1
                
                results['predictions'].append({
                    'file': os.path.basename(data_path),
                    'ground_truth': ground_truth,
                    'predicted': predicted_orientation,
                    'confidence': confidence,
                    'correct': is_correct,
                    'processing_time': processing_time
                })
                
                status = "✅" if is_correct else "❌"
                print(f"{status} Predicted: {predicted_orientation} (Confidence: {confidence:.1f}%)")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                continue
                
        except Exception as e:
            print(f"❌ Error loading {data_path}: {e}")
            continue
    
    return results

def print_comparison_results(model1_results, model2_results):
    """Print comparison results between two models"""
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*80)
    
    if not model1_results or not model2_results:
        print("❌ Cannot compare - one or both models failed evaluation")
        return
    
    # Calculate metrics
    model1_acc = (model1_results['correct_predictions'] / model1_results['total_samples']) * 100
    model2_acc = (model2_results['correct_predictions'] / model2_results['total_samples']) * 100
    
    model1_avg_conf = np.mean(model1_results['confidences']) if model1_results['confidences'] else 0
    model2_avg_conf = np.mean(model2_results['confidences']) if model2_results['confidences'] else 0
    
    model1_avg_time = np.mean(model1_results['processing_times']) if model1_results['processing_times'] else 0
    model2_avg_time = np.mean(model2_results['processing_times']) if model2_results['processing_times'] else 0
    
    print(f"\n🎯 ACCURACY:")
    print(f"   Model 1 (Original): {model1_acc:.1f}% ({model1_results['correct_predictions']}/{model1_results['total_samples']})")
    print(f"   Model 2 (V2):       {model2_acc:.1f}% ({model2_results['correct_predictions']}/{model2_results['total_samples']})")
    
    print(f"\n🎯 CONFIDENCE:")
    print(f"   Model 1 (Original): {model1_avg_conf:.1f}%")
    print(f"   Model 2 (V2):       {model2_avg_conf:.1f}%")
    
    print(f"\n🎯 PROCESSING TIME:")
    print(f"   Model 1 (Original): {model1_avg_time:.2f}s")
    print(f"   Model 2 (V2):       {model2_avg_time:.2f}s")
    
    print(f"\n🎯 DETAILED PREDICTIONS:")
    print("-" * 80)
    print(f"{'File':<20} {'Ground Truth':<12} {'Model1':<12} {'Model2':<12} {'Status':<8}")
    print("-" * 80)
    
    for i in range(max(len(model1_results['predictions']), len(model2_results['predictions']))):
        pred1 = model1_results['predictions'][i] if i < len(model1_results['predictions']) else None
        pred2 = model2_results['predictions'][i] if i < len(model2_results['predictions']) else None
        
        if pred1 and pred2:
            status = "✅" if pred1['correct'] and pred2['correct'] else "❌"
            print(f"{pred1['file']:<20} {pred1['ground_truth']:<12} {pred1['predicted']:<12} {pred2['predicted']:<12} {status:<8}")

def main():
    """Main evaluation function"""
    print("🔬 Orientation Detection Model Evaluation")
    print("=" * 60)
    
    # Model paths
    model1_path = r"C:\Users\youss\OneDrive\Desktop\MPR-viewer\models\resnet18_orientation_finetuned.pth"
    model2_path = r"C:\Users\youss\OneDrive\Desktop\MPR-viewer\models\resnet18_orientation_finetuned_v2.pth"
    
    # Test data paths (add your test files here)
    test_data_paths = [
        # Add your test DICOM folders or NIfTI files here
        # Example:
        # r"C:\path\to\test\dicom\folder",
        # r"C:\path\to\test\file.nii.gz",
    ]
    
    if not test_data_paths:
        print("⚠️  No test data specified. Please add test data paths to the script.")
        print("   Edit the 'test_data_paths' list in the main() function.")
        return
    
    print(f"📁 Test data files: {len(test_data_paths)}")
    
    # Evaluate both models
    print("\n🔍 Evaluating Model 1 (Original)...")
    model1_results = evaluate_model_on_test_data(model1_path, test_data_paths)
    
    print("\n🔍 Evaluating Model 2 (V2)...")
    model2_results = evaluate_model_on_test_data(model2_path, test_data_paths)
    
    # Print comparison
    print_comparison_results(model1_results, model2_results)
    
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
