
import sys

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

import logging.config

class FaceTorward:
    def __init__(
        self, 
        logger,
        frame,
        conf_path = "config/", 
        model_path = 'models',
        scene = 'non-mask'
    ) -> None:
        self.scene = scene
        self.model_path = model_path
        self.logger = logger
        self.stander_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
            
        ])

        # 相机内参
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
                )
        self.dist_coeffs = np.zeros((4, 1))  # 假设没有发生镜头畸变

        with open('config/model_conf.yaml') as f:
            self.model_conf = yaml.safe_load(f)\

        self.__ini_detect()
        self.__ini_alignment()
        
    def __ini_detect(self) -> None:
        """
        初始化人脸探测模型 
        """
        # model setting, modified along with model
        model_name =  self.model_conf[self.scene]['face_detection']

        self.logger.info('Start to load the face detection model...')
        # load model
        try:
            faceDetModelLoader = FaceDetModelLoader(self.model_path, 'face_detection', model_name)
        except Exception as e:
            self.logger.error('Failed to parse model configuration file!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successfully parsed the model configuration file model_meta.json!')

        try:
            model, cfg = faceDetModelLoader.load_model()
        except Exception as e:
            self.logger.error('Model loading failed!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successfully loaded the face detection model!')
        
        self.faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)


    def __ini_alignment(self) -> None:
        """
        初始化关键点检测模型 
        """
        model_name =  self.model_conf[self.scene]['face_alignment']

        self.logger.info('Start to load the face landmark model...')
        # load model
        try:
            faceAlignModelLoader = FaceAlignModelLoader(self.model_path, 'face_alignment', model_name)
        except Exception as e:
            self.logger.error('Failed to parse model configuration file!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successfully parsed the model configuration file model_meta.json!')

        try:
            model, cfg = faceAlignModelLoader.load_model()
        except Exception as e:
            self.logger.error('Model loading failed!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successfully loaded the face landmark model!')

        self.faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)

    def detect(self, img):
        """ 
        Detect the face on the img
        """
        try:
            self.logger.info('Trying detcet the face in the img!')
            dets = self.faceDetModelHandler.inference_on_image(img)
        except Exception as e:
            self.logger.error('Face detection failed!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successful face detection!')

        try:
            for box in dets:
                box = list(map(int, box[0:4]))
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                box = np.array(box, np.int32)
                landmarks = self.faceAlignModelHandler.inference_on_image(img, box)
                for (index, (x, y) ) in enumerate(landmarks.astype(np.int32)):
                    cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                    # cv2.putText(img, str(index), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
                    
                euler, imgpts = self.getAngles(landmarks)  # (yaw, pitch, roll)
                cv2.putText(img, 'yaw: ' + str(euler[0]), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, 'pitch: ' + str(euler[1]), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, 'roll: ' + str(euler[2]), (0, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                # cv2.line(img, (50, 50), tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
                # cv2.line(img, (50, 50), tuple(imgpts[0].ravel()), (255, 0, 0), 3)  # BLUE
                # cv2.line(img, (50, 50), tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED
        except Exception as e:
            self.logger.error('Face landmark failed!')
            self.logger.error(e)
            sys.exit(-1)
        else:
            self.logger.info('Successful face landmark!')



    def getAngles(self, landmarks):
        img_points = np.array([
            landmarks[54],              # Nose tip
            landmarks[16],              # Chin
            landmarks[66],              # Left eye left corner
            landmarks[79],              # Right eye right corne
            landmarks[84],              # Left Mouth corner
            landmarks[90]               # Right mouth corner
        ], dtype='double')
        
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.stander_points, 
            img_points, 
            self.camera_matrix, 
            self.dist_coeffs,  
            flags=cv2.SOLVEPNP_ITERATIVE
        )  # dist_coeffs为摄像机的畸变系数  
        axis = np.float32([[500, 0, 0],
                          [0, 500, 0],
                          [0, 0, 500]])
 
        imgpts, jac = cv2.projectPoints(
            axis, 
            rotation_vector, 
            translation_vector, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        

        return self.getEuler(rotation_vector, translation_vector), imgpts                                                                                                                          

    def getEuler(self, rotation_vector, translation_vector):
        """
        此函数用于从旋转向量计算欧拉角
        :param rotation_vector: 输入为旋转向量
        :return: 返回欧拉角在三个轴上的值
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]
        rot_params = np.array([yaw, pitch, roll])
        return rot_params




