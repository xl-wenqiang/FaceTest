from fastapi import FastAPI  # 导入FastAPI库，用于构建API服务  
from pydantic import BaseModel  # 导入pydantic库中的BaseModel，用于定义数据模型  
import cv2 as cv  # 导入OpenCV库，用于图像处理  
import dlib  # 导入dlib库，用于人脸检测和特征点检测  
import numpy as np  # 导入numpy库，用于处理数值计算  
import skimage.io as io  # 导入skimage库的io模块，用于读取图像
  
app = FastAPI()  

# 初始化人脸检测模型及特征点检测模型  
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 加载预训练的68个面部特征点检测模型    
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # 加载人脸识别模型  
  
  
class ImagePath(BaseModel):  
    image_path: str  
  
  
@app.post("/check_face_quality")  
async def check_face_quality(data: ImagePath):  
    image = cv.imread(data.image_path)  
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   # 将图像转换为灰度图，以便于后续的人脸检测
    faces = detector(gray)  
    if len(faces) != 1:  
        return {"status": False}  
    landmarks = predictor(gray, faces[0])  
    # 检测特定的面部特征点，比如嘴巴、眉毛等，并判断是否使用了美颜  
    # if is_beauty(face_landmarks):  
    #     return {"status": False}  
    return {"status": True}  
  
  
@app.post("/register_face")  
async def register_face(data: ImagePath, name: str):  
    face_quality = await check_face_quality(data)  
    if face_quality["status"]:  
        # 人脸质量检测通过，进行人脸注册操作（这里可以添加具体的注册逻辑）  
        # 将人脸信息保存到人脸库中（这里可以添加具体的保存逻辑）  
        return {"message": f"{name}的人脸注册成功！"}  
    else:  
        return {"message": f"{name}的人脸质量不符合要求，注册失败！"}  
  
  
@app.post("/extract_face_features")  
async def extract_face_features(data: ImagePath):  
    image = io.imread(data.image_path)  
    dets = detector(image, 1)  
    if len(dets) == 0:  
        return {"status": "Failed to detect face."}  
    shape = predictor(image, dets[0])  
    face_descriptor = np.array(face_recognition_model.compute_face_descriptor(image, shape))  
    return {"face_descriptor": face_descriptor.tolist()}  
  
  
@app.post("/compare_faces")  
async def compare_faces(data1: ImagePath, data2: ImagePath):  
    face1_descriptor = await extract_face_features(data1)  
    face2_descriptor = await extract_face_features(data2)  
    if face1_descriptor["status"] == "Failed to detect face." or face2_descriptor["status"] == "Failed to detect face.":  
        return {"message": "Failed to extract face features."}  
    distance = np.linalg.norm(np.array(face1_descriptor["face_descriptor"]) - np.array(face2_descriptor["face_descriptor"]))  
    return {"distance": distance}