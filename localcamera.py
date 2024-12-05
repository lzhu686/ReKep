import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json

class LocalRealSenseD455:
    def __init__(self):
        # 创建RealSense管线
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置深度和彩色流
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # 启动相机流
        self.profile = self.pipeline.start(self.config)
        
        # 获取深度传感器
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        
        # 设置深度单位为米
        self.depth_sensor.set_option(rs.option.depth_units, 0.001)
        
        # 创建对齐对象（将深度帧对齐到彩色帧）
        self.align = rs.align(rs.stream.color)
        
        # 创建点云对象
        self.pc = rs.pointcloud()

    def get_cam_obs(self, save_dir="./scene_data"):
        """获取相机原始观察数据并保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 等待稳定的帧
        for _ in range(30):
            self.pipeline.wait_for_frames()
            
        # 获取对齐的帧
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        # 获取深度帧和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RuntimeError("无法获取相机帧")
            
        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 获取相机内参
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        
        # 生成点云
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices())
        vertices = np.array([[v[0], v[1], v[2]] for v in vertices]).reshape(-1, 3)
        
        # 保存原始数据
        rgb_path = os.path.join(save_dir, "rgb.png")
        depth_path = os.path.join(save_dir, "depth.png")
        points_path = os.path.join(save_dir, "points.npy")
        
        cv2.imwrite(rgb_path, color_image)
        cv2.imwrite(depth_path, depth_image)
        np.save(points_path, vertices)
        
        # 创建相机数据格式
        cam_obs = {
            "vlm_camera": {
                "rgb": rgb_path,
                "points": points_path,
                "depth": depth_path,
                "intrinsics": {
                    "width": color_intrin.width,
                    "height": color_intrin.height,
                    "fx": color_intrin.fx,
                    "fy": color_intrin.fy,
                    "cx": color_intrin.ppx,
                    "cy": color_intrin.ppy
                }
            }
        }
        
        # 保存相机观察数据配置
        with open(os.path.join(save_dir, "cam_obs.json"), "w") as f:
            json.dump(cam_obs, f, indent=2)
            
        return cam_obs

    def __del__(self):
        """清理资源"""
        self.pipeline.stop()

def main():
    try:
        # 创建相机对象
        camera = LocalRealSenseD455()
        
        # 获取并保存场景数据
        cam_obs = camera.get_cam_obs()
        print("原始相机数据已保存：")
        print(json.dumps(cam_obs, indent=2, ensure_ascii=False))
        
        # 验证数据格式
        rgb = cv2.imread(cam_obs["vlm_camera"]["rgb"])
        points = np.load(cam_obs["vlm_camera"]["points"])
        
        print(f"RGB shape: {rgb.shape}")      # 应该是(H, W, 3)
        print(f"Points shape: {points.shape}") # 应该是(N, 3)
        
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main()
