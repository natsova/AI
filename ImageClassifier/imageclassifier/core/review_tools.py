# core/review_tools.py

from pathlib import Path
from PIL import Image
import io
from IPython.display import display, clear_output
import ipywidgets as widgets

# Global mapping: str(path) -> Checkbox widget
checkboxes = {}


def select_img_for_deletion(config):
    """
    Displays all images with checkboxes for manual review.
    Uncheck images you want deleted, then run delete_unchecked_images().
    """
    global checkboxes
    checkboxes.clear()

    dataset_path = Path(config.dataset_path)

    for category in config.categories:
        category_path = dataset_path / category

        if not category_path.exists():
            print(f"No folder: {category_path}")
            continue

        all_images = sorted(
            [p for p in category_path.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        if not all_images:
            print(f"No images in '{category}'")
            continue

        print(f"\nCategory: {category}")

        rows = []

        for img_path in all_images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((150, 150))
                    bio = io.BytesIO()
                    img.save(bio, format="JPEG")
                    bio.seek(0)
                    img_widget = widgets.Image(
                        value=bio.read(),
                        format="jpeg",
                        width=150,
                        height=150
                    )
            except Exception as e:
                print(f"Could not load {img_path.name}: {e}")
                continue

            cb = widgets.Checkbox(value=True, description=img_path.name)
            checkboxes[str(img_path)] = cb

            rows.append(widgets.HBox([img_widget, cb]))

        display(widgets.VBox(rows))

    print("\nUncheck images to delete, then run delete_unchecked_images().")


def delete_unchecked_images(clear_ui: bool = True):
    """
    Deletes images that have been unchecked in the review UI.
    """
    global checkboxes

    if not checkboxes:
        print("No checkboxes found.")
        return

    deleted = 0
    failed = 0

    for path_str, cb in list(checkboxes.items()):
        try:
            if not cb.value:
                p = Path(path_str)
                p.unlink(missing_ok=True)
                deleted += 1
                print(f"Deleted: {p}")
        except Exception as e:
            failed += 1
            print(f"Failed to delete {path_str}: {e}")

    print(f"Done. Deleted: {deleted}. Failed: {failed}.")

    if clear_ui:
        clear_output(wait=True)
        print(f"Deleted: {deleted}. Failed: {failed}.")
        checkboxes.clear()
