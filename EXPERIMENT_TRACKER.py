"""
Experiment Tracker for WBC-Bench-2026
Track all your experiments to see what works
"""

import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, log_file='experiments.json'):
        self.log_file = Path(log_file)
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = []
    
    def log_experiment(self, name, config, val_acc, test_acc=None, notes=""):
        """Log an experiment"""
        experiment = {
            'id': len(self.experiments) + 1,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'notes': notes
        }
        self.experiments.append(experiment)
        self.save()
        return experiment['id']
    
    def save(self):
        """Save experiments to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_best(self, metric='val_acc'):
        """Get best experiment"""
        if not self.experiments:
            return None
        return max(self.experiments, key=lambda x: x.get(metric, 0))
    
    def print_summary(self):
        """Print summary of all experiments"""
        print("=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"{'ID':<5} {'Name':<30} {'Val Acc':<10} {'Test Acc':<10} {'Notes':<20}")
        print("-" * 80)
        
        for exp in self.experiments:
            val_acc = f"{exp.get('val_acc', 0):.2f}%" if exp.get('val_acc') else "N/A"
            test_acc = f"{exp.get('test_acc', 0):.2f}%" if exp.get('test_acc') else "N/A"
            notes = exp.get('notes', '')[:20]
            print(f"{exp['id']:<5} {exp['name']:<30} {val_acc:<10} {test_acc:<10} {notes:<20}")
        
        best = self.get_best()
        if best:
            print("-" * 80)
            print(f"Best: {best['name']} - Val Acc: {best.get('val_acc', 0):.2f}%")
        print("=" * 80)

# Example usage
if __name__ == '__main__':
    tracker = ExperimentTracker()
    
    # Example: Log baseline
    tracker.log_experiment(
        name="Baseline - EfficientNet-B0",
        config={
            'model': 'efficientnet_b0',
            'img_size': 224,
            'batch_size': 64
        },
        val_acc=87.2,
        notes="Initial baseline model"
    )
    
    # Print summary
    tracker.print_summary()

