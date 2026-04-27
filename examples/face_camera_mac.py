import argparse
import queue
import sys
import threading
import time

import cv2
import PIL.Image
import torch

from nanoowl.owl_predictor import OwlPredictor
from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def color_for_label(index: int):
    colors = [
        (80, 220, 80),
        (255, 180, 40),
        (255, 80, 220),
        (80, 220, 255),
        (255, 110, 110),
        (170, 120, 255),
        (80, 255, 180),
        (255, 220, 80),
    ]
    return colors[index % len(colors)]


def normalize_prompt(prompt_text: str):
    stripped = prompt_text.strip()
    if not stripped:
        return "[a face]"
    if stripped[0] in "[(":
        return stripped
    return f"[{stripped}]"


def prompt_reader(prompt_queue: "queue.Queue[str]"):
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        prompt_text = line.strip()
        if prompt_text:
            prompt_queue.put(prompt_text)


def safe_display_overlay(window_name: str, message: str, duration_ms: int):
    try:
        cv2.displayOverlay(window_name, message, duration_ms)
    except cv2.error:
        print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live face detection from a webcam on macOS using NanoOWL."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--prompt", type=str, default="[a face]")
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--skip-frames", type=int, default=1, help="Run detection every Nth frame.")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    prompt_text = normalize_prompt(args.prompt)

    owl_predictor = OwlPredictor(
        model_name=args.model,
        device=device,
        image_encoder_engine=None,
    )
    predictor = TreePredictor(
        owl_predictor=owl_predictor,
        clip_predictor=None,
        device=device,
    )
    tree = Tree.from_prompt(prompt_text)
    clip_text_encodings = predictor.encode_clip_text(tree)
    owl_text_encodings = predictor.encode_owl_text(tree)

    prompt_queue: "queue.Queue[str]" = queue.Queue()
    input_thread = threading.Thread(target=prompt_reader, args=(prompt_queue,), daemon=True)
    input_thread.start()

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        raise RuntimeError(
            f"Nie udalo sie otworzyc kamery {args.camera}. Sprawdz uprawnienia do kamery w macOS."
        )

    print(f"device={device}")
    print(f"aktywny_prompt={prompt_text}")
    print("Sterowanie: wpisz nowy prompt w terminalu i nacisnij Enter, np. [a cell phone]")
    print("Sterowanie: q albo ESC zamyka okno.")

    frame_index = 0
    detections = []
    last_inference_ms = 0.0
    last_loop_time = time.perf_counter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Nie udalo sie odczytac klatki z kamery.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            while not prompt_queue.empty():
                next_prompt_text = prompt_queue.get_nowait()
                if next_prompt_text.strip():
                    prompt_text = normalize_prompt(next_prompt_text)
                    try:
                        tree = Tree.from_prompt(prompt_text)
                        clip_text_encodings = predictor.encode_clip_text(tree)
                        owl_text_encodings = predictor.encode_owl_text(tree)
                        detections = []
                        print(f"aktywny_prompt={prompt_text}")
                        safe_display_overlay(
                            "NanoOWL Face Camera",
                            f"Prompt zmieniony na: {prompt_text}",
                            1500,
                        )
                    except Exception as exc:
                        print(f"Nie udalo sie ustawic promptu {prompt_text}: {exc}")

            run_detection = frame_index % max(args.skip_frames, 1) == 0

            if run_detection:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = PIL.Image.fromarray(image_rgb)

                t0 = time.perf_counter()
                output = predictor.predict(
                    image=image_pil,
                    tree=tree,
                    clip_text_encodings=clip_text_encodings,
                    owl_text_encodings=owl_text_encodings,
                    threshold=args.threshold,
                )
                last_inference_ms = (time.perf_counter() - t0) * 1000.0

                detections = []
                label_map = tree.get_label_map()
                label_depths = tree.get_label_depth_map()
                for detection in output.detections:
                    x1, y1, x2, y2 = [int(float(value)) for value in detection.box]
                    label_indices = [label_index for label_index in detection.labels if label_index != 0]
                    label_names = [label_map[label_index] for label_index in label_indices]
                    best_score = max(detection.scores) if detection.scores else 0.0
                    primary_label_index = max(
                        label_indices,
                        key=lambda label_index: (label_depths.get(label_index, 0), label_index),
                        default=0,
                    )
                    detections.append(
                        {
                            "label": " | ".join(label_names) if label_names else "object",
                            "score": float(best_score),
                            "box": (x1, y1, x2, y2),
                            "color": color_for_label(primary_label_index),
                        }
                    )

            now = time.perf_counter()
            fps = 1.0 / max(now - last_loop_time, 1e-6)
            last_loop_time = now

            display = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                color = detection["color"]
                label = f'{detection["label"]} {detection["score"]:.2f}'
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                text_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    2,
                )
                text_x = x1
                text_y = max(28, y1 - 10)
                cv2.rectangle(
                    display,
                    (text_x - 4, text_y - text_size[1] - 8),
                    (text_x + text_size[0] + 6, text_y + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    display,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                    )

            overlay = (
                f"device={device}  fps={fps:.1f}  infer={last_inference_ms:.0f}ms  "
                f"detections={len(detections)}  prompt={prompt_text}"
            )
            cv2.putText(
                display,
                overlay,
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("NanoOWL Face Camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
