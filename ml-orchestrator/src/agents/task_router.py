import sys

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class TaskRouter:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = ColoredLogger()

    def route(self, default_pipeline="supervised"):
        self.logger.section("TASK SELECTION")

        print("\n" + "=" * 60)
        print("SELECT LEARNING TYPE".center(60))
        print("=" * 60)
        print("\n1. 🎯 Supervised Learning")
        print("2. 🔍 Unsupervised Learning")
        print("=" * 60 + "\n")

        while True:
            try:
                choice = input("Enter choice (1 or 2): ").strip()
            except (EOFError, KeyboardInterrupt):
                self.logger.warning("No input received; defaulting to supervised learning")
                return default_pipeline

            if choice in ['1', '2']:
                pipeline = 'supervised' if choice == '1' else 'unsupervised'
                self.logger.success(f"Selected: {pipeline.upper()}")
                return pipeline

            if choice == "":
                self.logger.warning("No choice provided; defaulting to supervised learning")
                return default_pipeline

            print("❌ Invalid choice!")


if __name__ == "__main__":
    router = TaskRouter(config={})
    router.route()