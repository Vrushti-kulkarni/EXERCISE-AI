import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None

def calc_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0] - b[0])  #np.arctan2(y value of a - y value of b, x value of a - x value of b)
    angle = np.abs(radians*180.0/np.pi) 

    if angle > 180.0:
        angle -= 360

    return angle   

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():

        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # set flag
        image.flags.writeable = False

            #result
        result = pose.process(image)

            #set flag
        image.flags.writeable = True

            #convert back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = result.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y ]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y ]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y ]

            ang = calc_angle(shoulder, elbow, wrist)
            
            cv2.putText(image, str(ang), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            if ang>160:
                 stage = 'down'
            elif ang<20 and stage =='down':
                 stage = 'up'
                 counter+=1 
                 print(counter)
                 
            cv2.putText(image, str(counter), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5 , (255, 255, 255), 2, cv2.LINE_AA)

            
            # print(landmarks)
        except:
             pass
        
            #RENDERING
        # if result.multi_hand_landmarks: #to check if we have got any result, if not skip rendering
        #     for num, hand in enumerate(result.multi_hand_landmarks): #loop through each result
        mp_drawing.draw_landmarks(image, result.pose_landmarks,mp_pose.POSE_CONNECTIONS , mp_drawing.DrawingSpec(color = (121, 22, 85), thickness = 2, circle_radius = 4), mp_drawing.DrawingSpec(color = (250, 44, 76), thickness = 2, circle_radius = 2))        



        cv2.imshow('Frame',image)
        if cv2.waitKey(10)==ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# print(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
# print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])




#LANDMARK COORDS


print(ang)
