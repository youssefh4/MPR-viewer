#!/usr/bin/env python3
"""
Simple script to inspect model properties and compare architectures
"""

import os
import torch
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def inspect_model(model_path):
    """Inspect a model file and print its properties"""
    print(f"\nInspecting: {os.path.basename(model_path)}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    try:
        # Load the model
        device = torch.device('cpu')  # Use CPU for inspection
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Check if it's a state_dict or full model
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("Model type: Full checkpoint (with state_dict)")
                state_dict = checkpoint['state_dict']
                if 'epoch' in checkpoint:
                    print(f"Training epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"Training loss: {checkpoint['loss']:.4f}")
            else:
                print("Model type: State dictionary only")
                state_dict = checkpoint
        else:
            print("Model type: Full model object")
            state_dict = checkpoint.state_dict()
        
        # Analyze the state_dict
        print(f"\nModel Architecture Analysis:")
        print(f"   Total parameters: {len(state_dict)}")
        
        # Count parameters by layer type
        layer_counts = {}
        total_params = 0
        
        for key, tensor in state_dict.items():
            layer_type = key.split('.')[0] if '.' in key else key
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            
            if isinstance(tensor, torch.Tensor):
                param_count = tensor.numel()
                total_params += param_count
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Layer distribution:")
        for layer_type, count in sorted(layer_counts.items()):
            print(f"     {layer_type}: {count} layers")
        
        # Check for common issues
        print(f"\nPotential Issues:")
        
        # Check for NaN values
        has_nan = False
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                has_nan = True
                break
        
        if has_nan:
            print("   ERROR: Contains NaN values (training issue)")
        else:
            print("   OK: No NaN values")
        
        # Check for extreme values
        extreme_values = False
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                if torch.abs(tensor).max() > 100 or torch.abs(tensor).min() < -100:
                    extreme_values = True
                    break
        
        if extreme_values:
            print("   WARNING: Contains extreme values (possible overfitting)")
        else:
            print("   OK: Values in normal range")
        
        # Check final layer (classification head)
        final_layer_keys = [k for k in state_dict.keys() if 'fc' in k or 'classifier' in k or 'head' in k]
        if final_layer_keys:
            print(f"   Final layer keys: {final_layer_keys}")
            for key in final_layer_keys:
                if key in state_dict:
                    tensor = state_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        print(f"     {key}: shape {tensor.shape}, mean {tensor.mean():.4f}, std {tensor.std():.4f}")
        
        return state_dict
        
    except Exception as e:
        print(f"ERROR inspecting model: {e}")
        return None

def compare_models(model1_path, model2_path):
    """Compare two models"""
    print("MODEL COMPARISON")
    print("=" * 80)
    
    model1_info = inspect_model(model1_path)
    model2_info = inspect_model(model2_path)
    
    if model1_info and model2_info:
        print(f"\nCOMPARISON SUMMARY:")
        print("-" * 40)
        
        # Compare parameter counts
        model1_params = sum(tensor.numel() for tensor in model1_info.values() if isinstance(tensor, torch.Tensor))
        model2_params = sum(tensor.numel() for tensor in model2_info.values() if isinstance(tensor, torch.Tensor))
        
        print(f"Model 1 parameters: {model1_params:,}")
        print(f"Model 2 parameters: {model2_params:,}")
        
        if model1_params != model2_params:
            print("WARNING: Different parameter counts - architectures may differ")
        else:
            print("OK: Same parameter counts - architectures match")
        
        # Compare layer structures
        model1_keys = set(model1_info.keys())
        model2_keys = set(model2_info.keys())
        
        if model1_keys == model2_keys:
            print("OK: Same layer structure")
        else:
            print("WARNING: Different layer structures:")
            only_in_1 = model1_keys - model2_keys
            only_in_2 = model2_keys - model1_keys
            if only_in_1:
                print(f"   Only in Model 1: {only_in_1}")
            if only_in_2:
                print(f"   Only in Model 2: {only_in_2}")

def main():
    """Main inspection function"""
    print("Model Inspection Tool")
    print("=" * 60)
    
    # Model paths
    model1_path = r"C:\Users\youss\OneDrive\Desktop\MPR-viewer\models\resnet18_orientation_finetuned.pth"
    model2_path = r"C:\Users\youss\OneDrive\Desktop\MPR-viewer\models\resnet18_orientation_finetuned_v2.pth"
    
    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"ERROR: Model 1 not found: {model1_path}")
        return
    
    if not os.path.exists(model2_path):
        print(f"ERROR: Model 2 not found: {model2_path}")
        return
    
    # Compare models
    compare_models(model1_path, model2_path)
    
    print(f"\nRECOMMENDATIONS:")
    print("=" * 40)
    print("1. If Model 2 has NaN values -> Retrain with lower learning rate")
    print("2. If Model 2 has extreme values -> Add regularization or early stopping")
    print("3. If architectures differ -> Check training script for changes")
    print("4. If same architecture but worse performance -> Check training data quality")
    print("5. Run evaluation script with test data to get accuracy metrics")

if __name__ == "__main__":
    main()