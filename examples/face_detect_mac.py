import argparse
from pathlib import Path

import PIL.Image
import torch

from nanoowl.owl_drawing import draw_owl_output
from nanoowl.owl_predictor import OwlPredictor


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NanoOWL face detection on macOS without TensorRT."
    )
    parser.add_argument("--image", type=str, default="../assets/class.jpg")
    parser.add_argument("--output", type=str, default="../data/face_detect_out.jpg")
    parser.add_argument("--prompt", type=str, default="[a face]")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    prompt = [part.strip() for part in args.prompt.strip("[]()").split(",") if part.strip()]

    predictor = OwlPredictor(
        model_name=args.model,
        device=device,
        image_encoder_engine=None,
    )

    image = PIL.Image.open(args.image).convert("RGB")
    text_encodings = predictor.encode_text(prompt)
    output = predictor.predict(
        image=image,
        text=prompt,
        text_encodings=text_encodings,
        threshold=args.threshold,
        pad_square=False,
    )

    print(f"device={device}")
    print(f"detections={len(output.labels)}")
    for index, (label, score, box) in enumerate(zip(output.labels, output.scores, output.boxes), start=1):
        name = prompt[int(label)]
        coords = [round(float(value.detach()), 1) for value in box]
        print(f"{index}. {name} score={float(score.detach()):.3f} box={coords}")

    rendered = draw_owl_output(image, output, text=prompt, draw_text=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    print(f"saved={output_path.resolve()}")
