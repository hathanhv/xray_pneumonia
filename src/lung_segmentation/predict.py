import cv2
import numpy as np

try:
    from .preprocess import image_to_tensor, read_image
except ImportError:
    from preprocess import image_to_tensor, read_image


def predict_mask_array(model, image, img_size, device, threshold=0.5):
    try:
        import torch
    except ImportError as error:
        raise ImportError("Missing dependency: torch is required for prediction.") from error

    tensor = image_to_tensor(image, img_size=img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= threshold).float()

    mask_np = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
    original_height, original_width = image.shape[:2]
    return cv2.resize(
        mask_np,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )


def predict_lung_mask(model, image_path, img_size, device, threshold=0.5):
    """
    Predict lung mask for one image.

    Returns:
        original_image: Original grayscale or BGR image from cv2.
        original_size_mask: Binary uint8 mask resized to original image size.
    """
    try:
        import torch
    except ImportError as error:
        raise ImportError("Missing dependency: torch is required for prediction.") from error

    _, original_image = read_image(image_path, img_size=img_size)
    original_size_mask = predict_mask_array(
        model,
        original_image,
        img_size,
        device,
        threshold=threshold,
    )
    return original_image, original_size_mask
