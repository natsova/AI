# scripts/main.py

from core.config import Config
from core.dataset_manager import DatasetManager
from core.model_manager import ModelManager
from core.dataloader import create_dataloader

def main():
    config = Config(categories=["sky", "ocean", "umbrella", "dog", "book"])

    # Dataset workflow
    dataset = DatasetManager(config)
    dataset.prepare_full_dataset()

    # Dataloader
    dls = create_dataloader(config)

    # Model workflow
    model = ModelManager(dls)
    model.build()
    model.train(epochs=10)
    model.save("imageclassifier.pkl")

    # Example inference
    pred, probs = model.predict_url(
        "https://example.com/dog.jpg"
    )
    print(pred, probs)

if __name__ == "__main__":
    main()
