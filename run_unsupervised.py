from orchestrator import MLOrchestrator

print("🚀 Starting Unsupervised Learning Test...\n")

orchestrator = MLOrchestrator()

# Run unsupervised learning
results = orchestrator.run(
    file_path='data/samples/iris.csv',
    pipeline_type='unsupervised'
)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"✅ Pipeline: {results['pipeline']}")
print(f"✅ Dimensionality Reduction: {results['dimensionality_reduction']}")
print(f"✅ Clustering Results: {results['clustering']}")
print(f"✅ Results saved at: {results['results_path']}")
print("="*80)
