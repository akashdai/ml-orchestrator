from orchestrator import MLOrchestrator

print("🚀 Starting Classification Test...\n")

orchestrator = MLOrchestrator()

# Run supervised classification
results = orchestrator.run(
    file_path='data/samples/iris.csv',
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
