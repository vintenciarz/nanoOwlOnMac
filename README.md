<h1 align="center">nanoOwlOnMac</h1>

<p align="center">
Repozytorium do uruchamiania NanoOWL na macOS / Apple Silicon.
</p>

> To repozytorium powstało na bazie projektu [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl).

## O Projekcie

To repo adaptuje [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) do działania na MacBookach z Apple Silicon, bez TensorRT i bez Jetsona.  
Zamiast ścieżki `CUDA + TensorRT`, używany jest PyTorch na `mps` lub `cpu`.

W repo są przygotowane dwa najważniejsze scenariusze:

- detekcja obiektów ze zdjęcia,
- live detekcja z kamerki z możliwością zmiany promptu w runtime.

Obsługiwane są też prompty hierarchiczne, na przykład:

```text
[a face [a nose, an eye, a mouth]]
```

czyli:

1. znajdź twarz,
2. potem szukaj nosa, oka i ust wewnątrz twarzy.

## Wymagania

- macOS na Apple Silicon
- Python 3.11
- dostęp do internetu przy pierwszym uruchomieniu modelu z Hugging Face
- dostęp do kamery w macOS, jeśli chcesz używać trybu live

## Szybki Start Na Macu

```bash
git clone https://github.com/vintenciarz/nanoOwlOnMac
cd nanoOwlOnMac
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-build-isolation
```

Jeśli nie masz jeszcze zainstalowanych podstawowych zależności, doinstaluj je:

```bash
.venv/bin/pip install transformers pillow matplotlib
```

Uwaga:
- repo korzysta też z `torch`, `torchvision` i `opencv-python`,
- w moim środowisku były już dostępne globalnie, dlatego `venv` zostało utworzone z `--system-site-packages`.

## Uruchomienie Ze Zdjęcia

Przykład detekcji twarzy:

```bash
cd /sciezka/do/nanoOwlOnMac
.venv/bin/python examples/face_detect_mac.py \
  --image assets/class.jpg \
  --output data/face_detect_out.jpg
```

Własny prompt:

```bash
.venv/bin/python examples/face_detect_mac.py \
  --image /sciezka/do/obrazu.jpg \
  --output data/moj_wynik.jpg \
  --prompt "[a cell phone]"
```

## Uruchomienie Live Z Kamerki

Najprostszy start:

```bash
cd /sciezka/do/nanoOwlOnMac
.venv/bin/python examples/face_camera_mac.py --mirror
```

Po uruchomieniu możesz wpisywać nowy prompt w tym samym terminalu i nacisnąć `Enter`, na przykład:

```text
[a face]
[a cell phone]
[a face [a nose, an eye, a mouth]]
```

Zamykanie:

```text
q
```

albo `Esc` w oknie OpenCV.

## Uprawnienia Do Kamery W macOS

Jeśli kamera się nie otwiera:

1. otwórz `System Settings`,
2. wejdź w `Privacy & Security`,
3. kliknij `Camera`,
4. włącz dostęp dla aplikacji, z której uruchamiasz skrypt:
   `Terminal`, `iTerm`, `Codex` albo podobnej.

Jeśli aplikacji nie ma jeszcze na liście, uruchom skrypt jeszcze raz i kliknij `Allow` w systemowym popupie.

## Przykładowe Prompty

Duże obiekty:

```text
[a face]
[a cell phone]
[a laptop]
[a coffee mug]
```

Kilka obiektów naraz:

```text
[a laptop, a keyboard]
```

Prompty hierarchiczne:

```text
[a face [a nose]]
[a face [a nose, an eye, a mouth]]
```

Ważne:
- `[a nose]` próbuje znaleźć nos w całym obrazie i zwykle działa gorzej,
- `[a face [a nose]]` najpierw znajduje twarz, a dopiero potem nos, więc jest dużo sensowniejsze.

## Jak Zwiększyć FPS

Największą różnicę robią:

- mniejsza rozdzielczość wejścia,
- `--skip-frames 2` albo `--skip-frames 3`,
- prostszy prompt,
- mniej obiektów naraz.

Przykłady:

```bash
.venv/bin/python examples/face_camera_mac.py \
  --mirror \
  --width 960 \
  --height 540 \
  --skip-frames 2 \
  --prompt "[a face]"
```

```bash
.venv/bin/python examples/face_camera_mac.py \
  --mirror \
  --width 640 \
  --height 360 \
  --skip-frames 3 \
  --prompt "[a face]"
```

## Co Zostało Zmienione Względem Oryginału

Najważniejsze zmiany względem oryginalnego `NVIDIA-AI-IOT/nanoowl`:

- uruchamianie bez TensorRT,
- automatyczny wybór `mps` / `cpu`,
- skrypt do detekcji na pojedynczym obrazie:
  `examples/face_detect_mac.py`,
- skrypt live pod macOS:
  `examples/face_camera_mac.py`,
- zmiana promptu w runtime z terminala,
- wsparcie promptów hierarchicznych w trybie live,
- poprawki rysowania ramek dla obrazów z PIL / OpenCV na macOS.

## Oryginalny Projekt

Oryginał znajduje się tutaj:

- [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl)

Jeśli potrzebujesz wersji zoptymalizowanej pod Jetsona i TensorRT, użyj oryginalnego repo NVIDIA.
