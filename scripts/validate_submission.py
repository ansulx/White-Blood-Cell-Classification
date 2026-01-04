"""
Comprehensive submission validation script
Checks all edge cases and ensures submission is ready
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

def comprehensive_validation(submission_path, test_csv_path=None):
    """
    Comprehensive validation of submission file.
    
    Args:
        submission_path: Path to submission CSV
        test_csv_path: Path to test CSV (optional)
    
    Returns:
        (is_valid, report_dict)
    """
    report = []
    errors = []
    warnings = []
    
    print(f"\n{'='*60}")
    print("Comprehensive Submission Validation")
    print(f"{'='*60}")
    
    # Load submission
    try:
        submission = pd.read_csv(submission_path)
        report.append(f"✓ Submission file loaded: {len(submission)} rows")
    except Exception as e:
        errors.append(f"Failed to load submission: {e}")
        return False, {'errors': errors, 'warnings': warnings, 'report': report}
    
    # Check columns
    required_cols = ['ID', 'labels']
    if not all(col in submission.columns for col in required_cols):
        errors.append(f"Missing columns. Found: {list(submission.columns)}, Required: {required_cols}")
    else:
        report.append("✓ Required columns present")
    
    # Check for duplicates
    duplicates = submission['ID'].duplicated().sum()
    if duplicates > 0:
        dup_ids = submission[submission['ID'].duplicated()]['ID'].tolist()[:10]
        errors.append(f"Found {duplicates} duplicate IDs: {dup_ids}")
    else:
        report.append("✓ No duplicate IDs")
    
    # Check valid classes
    valid_classes = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
    invalid = submission[~submission['labels'].isin(valid_classes)]
    if len(invalid) > 0:
        errors.append(f"Found {len(invalid)} invalid class labels: {invalid['labels'].unique().tolist()}")
    else:
        report.append("✓ All class labels are valid")
    
    # Check for missing values
    null_counts = submission.isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Found missing values: {null_counts[null_counts > 0].to_dict()}")
    else:
        report.append("✓ No missing values")
    
    # Check class distribution
    class_counts = submission['labels'].value_counts()
    report.append(f"\nClass distribution:")
    for cls in valid_classes:
        count = class_counts.get(cls, 0)
        report.append(f"  {cls}: {count}")
        if count == 0:
            warnings.append(f"Class {cls} has no predictions")
    
    # Check against test set if provided
    if test_csv_path and Path(test_csv_path).exists():
        try:
            test_df = pd.read_csv(test_csv_path)
            test_ids = set(test_df['ID'].values)
            submission_ids = set(submission['ID'].values)
            
            missing = test_ids - submission_ids
            extra = submission_ids - test_ids
            
            if missing:
                errors.append(f"Missing {len(missing)} IDs from test set")
                if len(missing) <= 10:
                    errors.append(f"  Missing IDs: {list(missing)}")
            else:
                report.append("✓ All test IDs present")
            
            if extra:
                warnings.append(f"Found {len(extra)} extra IDs not in test set")
            
            if len(submission) != len(test_df):
                warnings.append(f"Row count mismatch: submission={len(submission)}, test={len(test_df)}")
        except Exception as e:
            warnings.append(f"Could not validate against test set: {e}")
    else:
        warnings.append("Test CSV not provided or not found - skipping ID validation")
    
    # Check ID format
    id_format_issues = submission[~submission['ID'].astype(str).str.match(r'.*\.(jpg|jpeg|png|JPG|JPEG|PNG)$')]
    if len(id_format_issues) > 0:
        warnings.append(f"Found {len(id_format_issues)} IDs with unusual format (should be image filenames)")
    
    # Check for reasonable class distribution (not all one class)
    if submission['labels'].nunique() < 5:
        warnings.append(f"Only {submission['labels'].nunique()} unique classes predicted (expected 13)")
    
    # Summary
    is_valid = len(errors) == 0
    
    print("\nValidation Report:")
    for item in report:
        print(f"  {item}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ All validations passed!")
    
    return is_valid, {'errors': errors, 'warnings': warnings, 'report': report}

def main():
    """Main validation function"""
    config = Config()
    
    submission_path = config.PRED_DIR / getattr(config, 'SUBMISSION_FILENAME', 'final_submission.csv')
    test_csv_path = config.PHASE2_TEST_CSV
    
    if not submission_path.exists():
        print(f"ERROR: Submission file not found: {submission_path}")
        print("Generate submission first: python scripts/final_submission.py")
        return 1
    
    is_valid, results = comprehensive_validation(submission_path, test_csv_path)
    
    if is_valid:
        print(f"\n{'='*60}")
        print("Submission is VALID and ready for upload!")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("Submission has ERRORS. Please fix before submitting.")
        print(f"{'='*60}")
        return 1

if __name__ == '__main__':
    exit(main())

