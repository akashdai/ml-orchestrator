from orchestrator import MLOrchestrator

print("🚀 ML ORCHESTRATOR - Interactive Mode\n")

orchestrator = MLOrchestrator()

# This will ask you to:
# 1. Choose supervised or unsupervised
# 2. Select target column (if supervised)

results = orchestrator.run('data/samples/iris.csv')

print("\n✅ Process Complete!")
print(f"Results: {results}")
