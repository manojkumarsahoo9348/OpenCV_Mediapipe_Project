import cv2
import mediapipe as mp

# Load MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video
video_path = r"C:\Users\hp\Downloads\AA Dance.mp4"
cap = cv2.VideoCapture(video_path)

# Create VideoWriter object to save output video
output_path = r'D:\DS NOTE(JANUARY 2024)\DT-12-01-2024 (Mediapipe Resume Project)\dancetracking detection\output.mp4'  # Include the filename and extension
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Render the pose landmarks on the frame
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    # Write the frame with landmarks to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Dance Tracking', frame)
    if cv2.waitKey(12) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

