class ModelManager:
    def __init__(self, dls, arch=resnet18, metrics=error_rate):
        self.dls = dls
        self.arch = arch
        self.metrics = metrics
        self.learn = None

    def build(self):
        self.learn = vision_learner(self.dls, self.arch, metrics=self.metrics)
        return self.learn

    def train(self, epochs=10):
        if self.learn is None:
            self.build()
        self.learn.fine_tune(epochs)

    def save(self, path="model.pkl"):
        if self.learn is None:
            raise RuntimeError("Model not built")
        self.learn.export(path)

    def load(self, path="model.pkl"):
        self.learn = load_learner(path)
        return self.learn

    def predict_url(self, url):
        if self.learn is None:
            raise RuntimeError("Model not loaded or trained")

        img = PILImage.create(BytesIO(requests.get(url).content))
        pred, _, probs = self.learn.predict(img)
        vocab = list(self.learn.dls.vocab)

        return pred, dict(zip(vocab, map(float, probs)))

    def predict_image(self, img):
        if self.learn is None:
            raise RuntimeError("Model not loaded or trained")

        pred, _, probs = self.learn.predict(img)
        vocab = list(self.learn.dls.vocab)

        return pred, dict(zip(vocab, map(float, probs)))

