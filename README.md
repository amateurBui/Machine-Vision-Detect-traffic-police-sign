# Machine-Vision-Detect-traffic-police-sign
1) make_data.py: create data yourself using landmarks in the mediapipe library
   - Collect 600 frames for each label (total 3600 images for 6 commands) and save into 6 different csv files
   - Stand in front of the camera webcam and pose. Mediapipe will recognize, draw, and save the coordinates of the keypoints (landmarks).
   - Labels included: ALL_STOP (all vehicles must stop); Front_and_Back_STOP (vehicles in front and behind the traffic police must stop); Left_STOP (vehicles on the left of traffic police must stop); Right_STOP  (vehicles on the right of traffic police must stop); Left_Faster (vehicles on the left of traffic police go faster); Right_Faster (vehicles on the right of traffic police go faster)
2) Train_police_sign: Training and evaluation of traffic police signal recognition model
   - With previously collected data as input, train and save the LSTM model
   - Evaluate the model through precision, f1-score, recall.
3) Test_police_sign_v2: Use pre-trained model to evaluate real-world quality via web cam
   - Recognize and save landmark coordinates in each frame
   - Every 50 saved frames will be entered into the model and give a prediction of the traffic police signal. However, the self-generated data is not good enough so the model is still Overfitting.
