import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2

class GestureRecognition():
    def __init__(self):
        self.recognizer = mp.tasks.vision.GestureRecognizer
        self.result = mp.tasks.vision.GestureRecognizerResult
        self.create_recognizer()

    def create_recognizer(self):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        def update_result(result:  mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='./gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=update_result)
        
        self.recognizer = GestureRecognizer.create_from_options(options)
        
    def display_batch_of_images_with_gestures_and_hand_landmarks(self, frame, results):
        image = frame
        gestures = [top_gesture for (top_gesture, _) in results]
        multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

        # Display gestures and hand landmarks.
        label = f"{gestures[0].category_name} ({gestures[0].score:.2f})"
        annotated_image = image.copy()

        for multihand_landmarks in multi_hand_landmarks_list:
            for hand_landmarks in multihand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        return annotated_image, label
    
    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.recognizer.recognize_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        
    def close(self):
        self.recognizer.close()
        
        
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    recognizer = GestureRecognition()
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        recognizer.detect_async(frame)
        time.sleep(0.01)
        title = "Gesture Recognition"
        label = "No gesture"
        
        try:
            if len(recognizer.result.gestures) != 0:
                top_gesture = recognizer.result.gestures[0][0]
                hand_landmarks = recognizer.result.hand_landmarks
                results = [(top_gesture, hand_landmarks)]
                frame, label = recognizer.display_batch_of_images_with_gestures_and_hand_landmarks(frame, results)
        except:
            pass
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(title, frame)


    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()
    