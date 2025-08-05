import argparse
import os
import sys
import signal
import torch
import gym
import dmcgym
import numpy as np
from gym.wrappers.flatten_observation import FlattenObservation
import cv2
from tqdm import tqdm

# 完全禁用MuJoCo的渲染功能
os.environ['MUJOCO_GL'] = 'disabled'
print("已禁用MuJoCo渲染，将使用纯信息可视化")

# 信号处理器，确保程序能够干净地退出
def signal_handler(sig, frame):
    print('接收到退出信号，正在清理资源...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 打印redq.algos.core模块路径，用于调试
try:
    import inspect
    import redq.algos.core
    print(f"使用redq.algos.core模块: {inspect.getfile(redq.algos.core)}")
except Exception as e:
    print(f"导入模块时出错: {e}")

def wrap_gym(env, rescale_actions=True):
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    return env

def list_available_models(base_dir='/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/models'):
    """列出可用的模型，按环境和时间戳组织"""
    if not os.path.exists(base_dir):
        print(f"模型基础目录不存在: {base_dir}")
        return
    
    print("\n可用的模型:")
    for env_name in sorted(os.listdir(base_dir)):
        env_dir = os.path.join(base_dir, env_name)
        if os.path.isdir(env_dir):
            print(f"\n环境: {env_name}")
            for timestamp in sorted(os.listdir(env_dir)):
                timestamp_dir = os.path.join(env_dir, timestamp)
                if os.path.isdir(timestamp_dir):
                    models = [f for f in os.listdir(timestamp_dir) if f.endswith('.pt')]
                    if models:
                        print(f"  └─ 时间戳: {timestamp}")
                        for model in sorted(models):
                            model_path = os.path.join(timestamp_dir, model)
                            print(f"      └─ 模型: {model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='quadruped-walk-v0', help='环境名称')
    parser.add_argument('--model_path', type=str, help='策略网络模型路径')
    parser.add_argument('--list_models', action='store_true', help='列出所有可用模型')
    parser.add_argument('--output_path', type=str, help='输出视频路径')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--episode_length', type=int, default=1000, help='每个episode的最大长度')
    parser.add_argument('--num_episodes', type=int, default=1, help='渲染的episode数量')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    args = parser.parse_args()
    
    # 列出所有模型并退出
    if args.list_models:
        list_available_models()
        return
    
    # 如果指定了环境名称但没有指定模型路径，则自动选择该环境下最新的模型
    if args.env and not args.model_path:
        base_dir = '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/models'
        env_dir = os.path.join(base_dir, args.env)
        
        if not os.path.exists(env_dir):
            print(f"找不到环境 {args.env} 的模型目录")
            return
        
        # 获取最新的时间戳文件夹
        timestamps = sorted(os.listdir(env_dir))
        if not timestamps:
            print(f"环境 {args.env} 下没有模型")
            return
        
        latest_timestamp = timestamps[-1]
        timestamp_dir = os.path.join(env_dir, latest_timestamp)
        
        # 获取最新的模型文件
        models = sorted([f for f in os.listdir(timestamp_dir) if f.endswith('.pt')])
        if not models:
            print(f"在 {timestamp_dir} 中没有找到模型文件")
            return
        
        latest_model = models[-1]
        args.model_path = os.path.join(timestamp_dir, latest_model)
        print(f"自动选择最新模型: {args.model_path}")
    
    # 检查是否指定了模型路径
    if not args.model_path:
        print("错误: 必须指定 --model_path 或 --env")
        return
    
    # 如果没有指定输出路径，则根据模型路径和环境名称生成默认输出路径
    if not args.output_path:
        model_filename = os.path.basename(args.model_path).replace('.pt', '')
        args.output_path = f"{args.env}_{model_filename}.mp4"
        print(f"输出视频将保存为: {args.output_path}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (args.width, args.height))
    
    env = None
    try:
        # 创建环境
        env = wrap_gym(gym.make(args.env))
        env.seed(args.seed)
        
        # 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 打印模型内容
        print(f"加载的模型文件包含以下键: {checkpoint.keys()}")
        
        # 创建策略网络
        from synther.online.redq_rlpd_agent import REDQRLPDCondAgent
        
        # 获取观察空间和动作空间维度
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0].item()
        
        # 创建智能体并加载模型
        agent = REDQRLPDCondAgent(
            cond_hidden_size=128,  # 这里使用默认值，如果有需要请修改
            env_name=args.env,
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=act_limit,
            device=device,
            hidden_sizes=(256, 256),  # 与训练时保持一致
            replay_size=int(1e6),     # 与训练时保持一致
            batch_size=256,           # 与训练时保持一致
            lr=3e-4,                  # 与训练时保持一致
            target_entropy=-act_dim,  # 直接使用动作维度的负值，避免使用mbpo_target_entropy_dict
        )
        
        # 加载策略网络权重
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        # 如果需要，也可以加载Q网络和条件网络
        
        # 运行并渲染环境
        total_rewards = []
        for ep in range(args.num_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            step = 0
            
            print(f"渲染第 {ep+1}/{args.num_episodes} 个episode...")
            
            with tqdm(total=args.episode_length) as pbar:
                while not done and step < args.episode_length:
                    # 使用策略网络获取动作
                    action = agent.get_deterministic_action(obs)
                    
                    # 执行动作
                    next_obs, reward, done, _ = env.step(action)
                    ep_reward += reward
                    
                    # 完全避免使用env.render()，直接创建信息面板
                    # 创建黑色背景
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    
                    # 添加环境和状态信息
                    lines = [
                        f"环境: {args.env}",
                        f"模型: {os.path.basename(args.model_path)}",
                        f"步数: {step}/{args.episode_length}",
                        f"奖励: {reward:.4f}",
                        f"累计奖励: {ep_reward:.4f}"
                    ]
                    
                    # 添加观察值信息（最多显示8个维度）
                    obs_dims = min(8, len(obs))
                    for i in range(obs_dims):
                        lines.append(f"观察值[{i}]: {obs[i]:.4f}")
                        
                    # 添加动作信息
                    act_dims = min(8, len(action))
                    for i in range(act_dims):
                        lines.append(f"动作[{i}]: {action[i]:.4f}")
                    
                    # 绘制所有信息
                    for i, line in enumerate(lines):
                        y = 30 + i * 25
                        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    # 添加进度条
                    progress = step / args.episode_length
                    bar_width = int(args.width * 0.8)
                    bar_height = 20
                    bar_x = int((args.width - bar_width) / 2)
                    bar_y = args.height - 50
                    
                    # 绘制进度条背景
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                (50, 50, 50), -1)
                    
                    # 绘制进度条前景
                    filled_width = int(bar_width * progress)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                                (0, 200, 0), -1)
                    
                    # 保存帧
                    video_writer.write(frame)
                    
                    obs = next_obs
                    step += 1
                    pbar.update(1)
                    pbar.set_description(f"Reward: {ep_reward:.2f}")
            
            total_rewards.append(ep_reward)
            print(f"Episode {ep+1} 完成，总奖励: {ep_reward:.2f}")
        
        print(f"所有episode完成，平均奖励: {np.mean(total_rewards):.2f}")
        print(f"视频已保存到 {args.output_path}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保资源被正确释放
        if video_writer is not None:
            video_writer.release()
        if env is not None:
            try:
                env.close()
            except:
                pass
        print("清理完成")

if __name__ == "__main__":
    main()


'''
使用方法：

重新训练模型（模型将自动保存到指定目录）：
# 训练quadruped-walk-v0环境
python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac.gin

# 训练cheetah-run-v0环境
CUDA_VISIBLE_DEVICE=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --log_dir '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/logs'

# 训练reacher-hard-v0环境
python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25' --log_dir '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/logs'


列出所有可用的模型：
python video.py --list_models

使用特定环境的最新模型渲染视频：
python video.py --env quadruped-walk-v0

指定具体模型渲染视频：
python video.py --model_path /mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/models/quadruped-walk-v0/20250715_021339/model_ep299.pt



python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac.gin --gin_params 'redq_sac.save_video_freq=20'

CUDA_VISIBLE_DEVICE=6 python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac.gin --gin_params 'redq_sac.save_video_freq=20'

CUDA_VISIBLE_DEVICE=7 python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac.gin --gin_params 'redq_sac.save_video_freq=20'
python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac.gin --gin_params 'redq_sac.save_video_freq=5' --log_dir '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/video'  

'''