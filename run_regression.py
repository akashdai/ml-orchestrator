from orchestrator import MLOrchestrator

print("🚀 Starting Regression Test...\n")

orchestrator = MLOrchestrator()

# Run supervised regression
results = orchestrator.run(
    file_path='data/samples/diabetes.csv',
    target_column='target',
    pipeline_type='supervised'
)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"✅ Best Model: {results['best_model']}")
print(f"✅ Task Type: {results['task_type']}")
print(f"✅ Metrics: {results['metrics']}")
print(f"✅ Model saved at: {results['model_path']}")
print("="*80)
