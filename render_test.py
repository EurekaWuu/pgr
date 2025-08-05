#!/usr/bin/env python
# 在导入任何MuJoCo相关库之前先创建虚拟显示
from pyvirtualdisplay import Display
import os
import numpy as np
import cv2

print("启动虚拟显示...")
display = Display(visible=0, size=(1400, 900))
display.start()
print(f"虚拟显示已启动，DISPLAY={os.environ.get('DISPLAY')}")

# 现在可以导入MuJoCo相关库
import gym
import dmcgym

# 测试不同的渲染器
renderers = ['glfw', 'egl', 'osmesa']
for renderer in renderers:
    try:
        print(f"\n尝试使用 {renderer} 渲染器...")
        os.environ['MUJOCO_GL'] = renderer
        
        # 创建环境
        env = gym.make('quadruped-walk-v0')
        obs = env.reset()
        
        # 渲染一帧
        frame = env.render(mode='rgb_array', width=640, height=480)
        
        # 显示帧信息并保存
        if frame is not None:
            print(f"使用 {renderer} 渲染成功！帧大小: {frame.shape}")
            # 保存图像
            cv2.imwrite(f"quadruped_{renderer}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"图像已保存为 quadruped_{renderer}.png")
            # 成功后退出循环
            break
        else:
            print(f"{renderer} 渲染返回空帧")
    except Exception as e:
        print(f"{renderer} 渲染器失败: {e}")
        import traceback
        traceback.print_exc()

# 关闭虚拟显示
display.stop()
print("虚拟显示已关闭") 