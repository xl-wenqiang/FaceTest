import sys
sys.path.insert(0, '/path/to/dlib')

import cv2
import dlib


# 初始化人脸检测模型及特征点检测模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 检测人脸质量
def check_face_quality(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = detector(gray)

    # 检测到多张人脸或者没有人脸
    if len(faces) != 1:
        return False

    # 检测美颜
    landmarks = predictor(gray, faces[0])
    # 检测特定的面部特征点，比如嘴巴、眉毛等
    # 根据特征点的位置判断是否使用了美颜
    # if is_beauty(face_landmarks):
    #     return False

    return True

# 判断是否使用了美颜
def is_beauty(face_landmarks):
    # 根据特征点的位置判断是否使用了美颜的逻辑
    return False

# 注册人脸
def register_face(image_path, name):
    if check_face_quality(image_path):
        # 人脸质量检测通过，进行人脸注册操作
        # 将人脸信息保存到人脸库中
        
        print(f"{name}的人脸注册成功！")
    else:
        print(f"{name}的人脸质量不符合要求，注册失败！")


image_path = "81800_1986-06-13_2011.jpg"
name = "Alice"
register_face(image_path, name)