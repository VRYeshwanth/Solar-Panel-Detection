import os
import random
import shutil


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = 'dataset'

IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, 'train')
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, 'val')

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, 'train')
VAL_LABELS_DIR = os.path.join(LABELS_DIR, 'val')

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


# ============================================================
# Create validation directories
# ============================================================

os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)


# ============================================================
# Check whether validation split already exists
# ============================================================

existing_val_images = os.listdir(VAL_IMAGES_DIR)
existing_val_labels = os.listdir(VAL_LABELS_DIR)

if len(existing_val_images) > 0 or len(existing_val_labels) > 0:
    raise RuntimeError(
        "Validation directory is not empty. "
        "The dataset may have already been split."
    )


# ============================================================
# Group training images according to number of panels
#
# 0, 1, 2, 3, 4, and 5+ panels
# ============================================================

panel_groups = {}

train_labels = os.listdir(TRAIN_LABELS_DIR)

for label_file in train_labels:

    label_path = os.path.join(
        TRAIN_LABELS_DIR,
        label_file
    )

    with open(label_path, 'r') as f:
        lines = f.readlines()

    number_of_panels = len(lines)

    if number_of_panels >= 5:
        group = '5+'
    else:
        group = str(number_of_panels)

    if group not in panel_groups:
        panel_groups[group] = []

    image_name = os.path.splitext(label_file)[0]

    panel_groups[group].append(image_name)


# ============================================================
# Display original distribution
# ============================================================

print("Original panel distribution:")
print("--------------------------------")

for group in sorted(panel_groups.keys(), key=lambda x: int(x.rstrip('+'))):
    print(
        f"{group} panels: "
        f"{len(panel_groups[group])} images"
    )

print()


# ============================================================
# Select validation images
# ============================================================

random.seed(RANDOM_SEED)

validation_images = []

for group, images in panel_groups.items():

    images = images.copy()

    random.shuffle(images)

    validation_count = round(
        len(images) * VALIDATION_RATIO
    )

    selected_images = images[:validation_count]

    validation_images.extend(selected_images)


# ============================================================
# Move image-label pairs to validation
# ============================================================

print("Moving validation images...")
print("--------------------------------")

for image_name in validation_images:

    # Find the image extension
    image_file = None

    for extension in ['.jpg', '.jpeg', '.png']:

        candidate = image_name + extension

        candidate_path = os.path.join(
            TRAIN_IMAGES_DIR,
            candidate
        )

        if os.path.exists(candidate_path):
            image_file = candidate
            break

    # Make sure the image exists
    if image_file is None:
        raise FileNotFoundError(
            f"Image not found for: {image_name}"
        )

    # Corresponding label
    label_file = image_name + '.txt'

    label_path = os.path.join(
        TRAIN_LABELS_DIR,
        label_file
    )

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Label not found for: {image_name}"
        )

    # Move image
    shutil.move(
        os.path.join(
            TRAIN_IMAGES_DIR,
            image_file
        ),
        os.path.join(
            VAL_IMAGES_DIR,
            image_file
        )
    )

    # Move label
    shutil.move(
        label_path,
        os.path.join(
            VAL_LABELS_DIR,
            label_file
        )
    )


# ============================================================
# Count final dataset
# ============================================================

train_images = os.listdir(TRAIN_IMAGES_DIR)
val_images = os.listdir(VAL_IMAGES_DIR)

train_labels = os.listdir(TRAIN_LABELS_DIR)
val_labels = os.listdir(VAL_LABELS_DIR)

print()
print("Final dataset:")
print("--------------------------------")

print(f"Train images: {len(train_images)}")
print(f"Train labels: {len(train_labels)}")

print(f"Validation images: {len(val_images)}")
print(f"Validation labels: {len(val_labels)}")

print()


# ============================================================
# Check image-label correspondence
# ============================================================

train_image_names = {
    os.path.splitext(file)[0]
    for file in train_images
}

train_label_names = {
    os.path.splitext(file)[0]
    for file in train_labels
}

val_image_names = {
    os.path.splitext(file)[0]
    for file in val_images
}

val_label_names = {
    os.path.splitext(file)[0]
    for file in val_labels
}


train_images_without_labels = (
    train_image_names - train_label_names
)

train_labels_without_images = (
    train_label_names - train_image_names
)

val_images_without_labels = (
    val_image_names - val_label_names
)

val_labels_without_images = (
    val_label_names - val_image_names
)


print("Image-label consistency:")
print("--------------------------------")

print(
    f"Train images without labels: "
    f"{len(train_images_without_labels)}"
)

print(
    f"Train labels without images: "
    f"{len(train_labels_without_images)}"
)

print(
    f"Validation images without labels: "
    f"{len(val_images_without_labels)}"
)

print(
    f"Validation labels without images: "
    f"{len(val_labels_without_images)}"
)

print()


# ============================================================
# Check for train/validation overlap
# ============================================================

train_val_overlap = (
    train_image_names & val_image_names
)

print("Train/validation overlap:")
print("--------------------------------")

print(
    f"Overlapping images: "
    f"{len(train_val_overlap)}"
)

print()


# ============================================================
# Final result
# ============================================================

if (
    len(train_images_without_labels) == 0
    and len(train_labels_without_images) == 0
    and len(val_images_without_labels) == 0
    and len(val_labels_without_images) == 0
    and len(train_val_overlap) == 0
):

    print("Validation split completed successfully.")

else:

    print(
        "WARNING: Some dataset consistency checks failed."
    )