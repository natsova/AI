class DatasetManager:
    def __init__(self, config: Config):
        self.config = config

    def setup(self):
        create_folders_for_categories(self.config)
        download_images(self.config)
        select_img_for_deletion(self.config)
        refill_categories(self.config)
        display_images(self.config)
        create_dataloader(self.config)
