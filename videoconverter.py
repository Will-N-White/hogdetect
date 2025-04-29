#!/usr/bin/env python3
import cv2
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 on video and overlay pig alerts")
    p.add_argument("--weights", "-w", required=True, help="Path to your YOLOv8 weights (e.g. best.pt)")
    p.add_argument("--source",  "-s", required=True, help="Input MP4 video path")
    p.add_argument("--output",  "-o", required=True, help="Output MP4 video path")
    p.add_argument("--conf",    "-c", type=float, default=0.25,
                   help="Confidence threshold for detection")
    return p.parse_args()

def main():
    args = parse_args()
    # Load model
    model = YOLO(args.weights)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {args.source}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing '{args.source}' → '{args.output}' at {fps:.1f} FPS…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on this frame :contentReference[oaicite:0]{index=0}
        results = model(frame, conf=args.conf)[0]

        # Draw boxes & labels on the frame :contentReference[oaicite:1]{index=1}
        annotated = results.plot()

        # Count how many pigs (class 0)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        pig_count = int((classes == 0).sum())

        # Choose overlay text
        if pig_count >= 2:
            text = "ready for action"
            color = (0, 255, 0)  # green
        else:
            text = "Waiting for confirmation"
            color = (0, 0, 255)  # red

        # Overlay text at bottom-left
        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale      = 1.0
        thickness  = 2
        margin     = 10
        ((tw, th), _) = cv2.getTextSize(text, font, scale, thickness)
        pos = (margin, height - margin)

        cv2.putText(
            annotated, text, pos, font, scale, color, thickness, cv2.LINE_AA
        )

        # Write out
        out.write(annotated)

    cap.release()
    out.release()
    print("✅ Done.")

if __name__ == "__main__":
    main()
