import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
    else:
        cv2.imshow('Captured Frame', frame)
        cv2.waitKey(0)
        # Save the captured frame to a file
        output_path = 'captured_frame_test.png'
        cv2.imwrite(output_path, frame)
        print(f"Captured frame saved as {output_path}")
    cap.release()
cv2.destroyAllWindows()
