from pathlib import Path

from orchestrator import MLOrchestrator
from sklearn.datasets import load_iris, load_diabetes

# Initialize orchestrator
orchestrator = MLOrchestrator()

# Test with sample data
print("🧪 Creating test datasets...")

Path('data/samples').mkdir(parents=True, exist_ok=True)

# 1. Classification dataset (Iris)
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_df.to_csv('data/samples/iris.csv', index=False)
print("✅ Created: data/samples/iris.csv")

# 2. Regression dataset (Diabetes)
diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.frame
diabetes_df.to_csv('data/samples/diabetes.csv', index=False)
print("✅ Created: data/samples/diabetes.csv")

print("\n" + "="*80)
print("TEST FILES CREATED!".center(80))
print("="*80)
print("\nNow you can run:")
print("1. Classification test:")
print("   python -c \"from orchestrator import MLOrchestrator; MLOrchestrator().run('data/samples/iris.csv', 'target', 'supervised')\"")
print("\n2. Regression test:")
print("   python -c \"from orchestrator import MLOrchestrator; MLOrchestrator().run('data/samples/diabetes.csv', 'target', 'supervised')\"")
print("\n3. Unsupervised test:")
print("   python -c \"from orchestrator import MLOrchestrator; MLOrchestrator().run('data/samples/iris.csv', pipeline_type='unsupervised')\"")
