import cv2
import argparse


def capture_rtsp_stream(rtsp_url: str) -> None:
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            cv2.imshow('RTSP Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(rtsp_url: str) -> None:
    capture_rtsp_stream(rtsp_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture and display RTSP stream."
    )
    parser.add_argument(
        "--rtsp_url", type=str, required=True,
        help="URL of the RTSP video stream."
    )
    arguments = parser.parse_args()

    main(rtsp_url=arguments.rtsp_url)
