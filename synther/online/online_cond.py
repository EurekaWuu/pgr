import sys
sys.path.insert(0, '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw')

import warnings

warnings.filterwarnings("ignore")

# 启动虚拟显示环境
print("初始化虚拟显示环境...")
try:
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    import os
    print(f"虚拟显示已启动，DISPLAY={os.environ.get('DISPLAY')}")
    # 设置渲染器为GLFW (通常与虚拟显示配合效果最好)
    os.environ['MUJOCO_GL'] = 'glfw'
    has_virtual_display = True
except ImportError:
    print("警告: pyvirtualdisplay未安装，真实环境渲染可能会失败")
    print("尝试安装: pip install pyvirtualdisplay")
    print("并确保安装了系统依赖: apt-get install -y xvfb")
    has_virtual_display = False
except Exception as e:
    print(f"启动虚拟显示时出错: {e}")
    has_virtual_display = False

import time
import os
import os.path as osp
import numpy as np

import dmcgym
import gin
import gym
import torch
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger
from redq.utils.run_utils import setup_logger_kwargs
from synther.diffusion.elucidated_diffusion import REDQCondTrainer
from synther.diffusion.diffusion_generator import CondDiffusionGenerator
from synther.diffusion.utils import construct_diffusion_model
from synther.online.redq_rlpd_agent import REDQRLPDCondAgent


@gin.configurable
def redq_sac(
        env_name,
        seed=3,
        epochs=-1,
        steps_per_epoch=1000,
        max_ep_len=1000,
        n_evals_per_epoch=1,
        logger_kwargs=dict(),
        # following are agent related hyperparameters
        hidden_sizes=(256, 256),
        replay_size=int(1e6),
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        auto_alpha=True,
        target_entropy='mbpo',
        start_steps=5000,
        delay_update_steps='auto',
        utd_ratio=20,
        num_Q=10,
        num_min=2,
        q_target_mode='min',
        policy_update_delay=20,
        diffusion_buffer_size=int(1e6),
        diffusion_sample_ratio=0.5,
        # diffusion hyperparameters
        retrain_diffusion_every=10_000,
        num_samples=100_000,
        diffusion_start=0,
        disable_diffusion=True,
        print_buffer_stats=True,
        skip_reward_norm=True,
        model_terminals=False,
        # conditional generation hyperparameters
        cfg_dropout=0.25,
        cond_top_frac=0.05,
        cfg_scale=1.0,
        cond_hidden_size=128,
        # following are bias evaluation related
        evaluate_bias=True,
        n_mc_eval=1000,
        n_mc_cutoff=350,
        reseed_each_epoch=True,
        # model saving parameters
        save_freq=20,  # 保存频率，每隔多少个epoch保存一次
        # 视频保存参数
        save_video=True,  # 是否保存视频
        save_video_freq=20,  # 保存视频的频率，每隔多少个epoch保存一次
        video_episodes=1,  # 每次保存多少个episode的视频
        video_width=640,  # 视频宽度
        video_height=480,  # 视频高度
        video_fps=30,  # 视频帧率
        preferred_renderers=['egl', 'glfw', 'osmesa'],  # 尝试的渲染器列表，按优先级排序
        force_hardware_acceleration=True,  # 是否强制使用硬件加速
):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches.get(env_name, 300)
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: wrap_gym(gym.make(env_name))
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    env_time_limit = get_time_limit(env)
    max_ep_len = env_time_limit if max_ep_len > env_time_limit else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """init agent + buffer and start training"""
    agent_config = {
        'env_name': env_name,
        'cond_hidden_size': cond_hidden_size,
        'hidden_sizes': hidden_sizes,
        'replay_size': replay_size,
        'batch_size': batch_size,
        'lr': lr,
        'gamma': gamma,
        'polyak': polyak,
        'alpha': alpha,
        'auto_alpha': auto_alpha,
        'target_entropy': target_entropy,
        'start_steps': start_steps,
        'delay_update_steps': delay_update_steps,
        'utd_ratio': utd_ratio,
        'num_Q': num_Q,
        'num_min': num_min,
        'q_target_mode': q_target_mode,
        'policy_update_delay': policy_update_delay,
    }
    agent = REDQRLPDCondAgent(cond_hidden_size, diffusion_buffer_size, diffusion_sample_ratio, env_name, obs_dim, act_dim, act_limit, device,
                              hidden_sizes, replay_size, batch_size,lr, gamma, polyak,
                              alpha, auto_alpha, target_entropy,
                              start_steps, delay_update_steps,
                              utd_ratio, num_Q, num_min, q_target_mode,
                              policy_update_delay)

    # set up diffusion model
    diff_dims = obs_dim + act_dim + 1 + obs_dim
    if model_terminals:
        diff_dims += 1
    inputs = torch.zeros((128, diff_dims)).float()
    if skip_reward_norm:
        skip_dims = [obs_dim + act_dim]
    else:
        skip_dims = []

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        

        # give new data to replay buffer
        agent.store_data(o, a, r, o2, d)
        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if not disable_diffusion and (t + 1) % retrain_diffusion_every == 0 and (t + 1) >= diffusion_start:
            print(f'Retraining diffusion model at step {t + 1}')

            # import ipdb; ipdb.set_trace()

            # Train new diffusion model
            diffusion_trainer = REDQCondTrainer(
                construct_diffusion_model(
                    inputs=inputs,
                    skip_dims=skip_dims,
                    disable_terminal_norm=model_terminals,
                    cond_dim=1,
                    cfg_dropout=cfg_dropout,
                ),
                results_folder=args.results_folder,
                model_terminals=model_terminals,
            )
            diffusion_trainer.update_normalizer(agent.replay_buffer, device=device)
            cond_distri = diffusion_trainer.train_from_redq_buffer(agent.replay_buffer, agent.cond_net, top_frac=cond_top_frac,
                                                                   curr_epoch=(t // steps_per_epoch) + 1)
            agent.reset_diffusion_buffer()

            # Add samples to agent replay buffer
            generator = CondDiffusionGenerator(env=env, ema_model=diffusion_trainer.ema.ema_model, cond_distri=cond_distri)
            observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=num_samples,
                                                                                            cfg_scale=cfg_scale)

            print(f'Adding {num_samples} samples to replay buffer.')
            for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                agent.diffusion_buffer.store(o, a, r, o2, term)

            if print_buffer_stats:
                ptr_location = agent.replay_buffer.ptr
                real_observations = agent.replay_buffer.obs1_buf[:ptr_location]
                real_actions = agent.replay_buffer.acts_buf[:ptr_location]
                real_next_observations = agent.replay_buffer.obs2_buf[:ptr_location]
                real_rewards = agent.replay_buffer.rews_buf[:ptr_location]
                # Print min, max, mean, std of each dimension in the obs, rew and action
                print('Buffer stats:')
                for i in range(observations.shape[1]):
                    print(f'Diffusion Obs {i}: {np.mean(observations[:, i]):.2f} {np.std(observations[:, i]):.2f}')
                    print(
                        f'     Real Obs {i}: {np.mean(real_observations[:, i]):.2f} {np.std(real_observations[:, i]):.2f}')
                for i in range(actions.shape[1]):
                    print(f'Diffusion Action {i}: {np.mean(actions[:, i]):.2f} {np.std(actions[:, i]):.2f}')
                    print(f'     Real Action {i}: {np.mean(real_actions[:, i]):.2f} {np.std(real_actions[:, i]):.2f}')
                print(f'Diffusion Reward: {np.mean(rewards):.2f} {np.std(rewards):.2f}')
                print(f'     Real Reward: {np.mean(real_rewards):.2f} {np.std(real_rewards):.2f}')
                print(f'Replay buffer size: {ptr_location}')
                print(f'Diffusion buffer size: {agent.diffusion_buffer.ptr}')

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            returns = test_agent(agent, test_env, max_ep_len, logger, n_evals_per_epoch)  # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
            
            # 保存策略网络模型（每save_freq个epoch保存一次）
            if epoch % save_freq == 0 or epoch == epochs - 1:
                # 创建基础保存目录
                base_save_dir = '/mnt/lustre/GPU4/home/wuhanpeng/pgr_raw/models'
                
                # 获取当前时间戳
                import datetime
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 创建按环境名称和时间戳组织的目录
                model_save_dir = osp.join(base_save_dir, env_name, timestamp)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                
                # 同时在原始日志目录中也保存一份
                if not os.path.exists(osp.join(logger.output_dir, 'pyt_save')):
                    os.makedirs(osp.join(logger.output_dir, 'pyt_save'))
                
                # 保存策略网络、Q网络和条件网络
                model_filename = f'model_ep{epoch}.pt'
                save_path = osp.join(model_save_dir, model_filename)
                logger_save_path = osp.join(logger.output_dir, 'pyt_save', model_filename)
                
                model_data = {
                    'epoch': epoch,
                    'env_name': env_name,
                    'timestamp': timestamp,
                    'policy_state_dict': agent.policy_net.state_dict(),
                    'q_state_dict': [q.state_dict() for q in agent.q_net_list],
                    'cond_net_state_dict': agent.cond_net.state_dict() if hasattr(agent, 'cond_net') else None
                }
                
                # 保存到两个位置
                torch.save(model_data, save_path)
                torch.save(model_data, logger_save_path)
                
                print(f"模型已保存至: \n1. {save_path}\n2. {logger_save_path}")
            
            # 保存视频（每save_video_freq个epoch保存一次）
            if save_video and (epoch % save_video_freq == 0 or epoch == epochs - 1):
                try:
                    import cv2
                    print(f"正在记录第{epoch}个epoch的视频...")
                    
                    # 创建视频目录
                    video_dir = osp.join(logger.output_dir, 'videos')
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir)
                    
                    # 设置视频文件名和写入器
                    video_path = osp.join(video_dir, f"{env_name}_ep{epoch}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, 
                                                 (video_width, video_height))
                    
                    # 备份当前环境变量(无论哪种情况都需要)
                    original_env_vars = os.environ.copy()
                    
                    # 尝试不同的渲染器配置
                    if has_virtual_display:
                        print("使用已启动的虚拟显示环境进行渲染")
                        render_success = True
                    else:
                        renderers_to_try = preferred_renderers
                        render_success = False
                        
                        # 尝试多种渲染器配置
                        for renderer in renderers_to_try:
                            try:
                                # 设置渲染器
                                print(f"尝试使用 {renderer} 渲染器...")
                                os.environ['MUJOCO_GL'] = renderer
                                
                                # 如果强制硬件加速，设置相关环境变量
                                if force_hardware_acceleration and renderer == 'egl':
                                    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
                                    print("已启用NVIDIA EGL硬件加速")
                                
                                # 创建临时环境测试渲染
                                test_frame = None
                                test_env_copy = wrap_gym(gym.make(env_name))
                                test_env_copy.reset()
                                
                                # 尝试渲染一帧
                                test_frame = test_env_copy.render(mode='rgb_array', width=video_width, height=video_height)
                                test_env_copy.close()
                                
                                if test_frame is not None and test_frame.size > 0:
                                    print(f"成功使用 {renderer} 渲染器! 帧大小: {test_frame.shape}")
                                    render_success = True
                                    break
                            except Exception as e:
                                print(f"{renderer} 渲染器失败: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        if not render_success:
                            print("警告: 无法使用任何渲染器。请确保服务器环境支持OpenGL渲染。")
                            print("尝试安装必要的包: apt-get install -y xvfb mesa-utils libosmesa6-dev libgl1-mesa-glx")
                            
                            # 显示系统信息以帮助调试
                            try:
                                import subprocess
                                print("\n系统信息:")
                                subprocess.run("lspci | grep -i vga", shell=True)
                                print("\nOpenGL信息:")
                                if 'DISPLAY' in os.environ:
                                    subprocess.run("glxinfo | grep OpenGL", shell=True)
                                else:
                                    print("未设置DISPLAY环境变量，无法获取OpenGL信息")
                            except:
                                print("无法获取系统信息")
                            
                            print("如果在Docker中运行，请确保以--gpus参数启动容器")
                            print("将尝试继续渲染，但可能会失败")
                    
                    # 记录视频
                    for ep in range(video_episodes):
                        obs = test_env.reset()
                        done = False
                        ep_len = 0
                        ep_ret = 0
                        
                        print(f"渲染episode {ep+1}/{video_episodes}...")
                        
                        while not done and ep_len < max_ep_len:
                            # 获取决定性动作
                            action = agent.get_test_action(obs)
                            
                            # 先执行动作获取下一状态和奖励
                            next_obs, reward, done, _ = test_env.step(action)
                            ep_ret += reward
                            
                            # 尝试渲染并保存帧
                            try:
                                frame = test_env.render(mode='rgb_array', width=video_width, height=video_height)
                                
                                if frame is not None and frame.size > 0:
                                    # 添加状态信息到帧上
                                    frame = frame.copy()  # 创建副本以避免修改原始帧
                                    
                                    # 添加文本信息
                                    info_text = f"Epoch: {epoch}, Step: {ep_len}, Reward: {ep_ret:.2f}"
                                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.7, (255, 255, 255), 1, cv2.LINE_AA)
                                    
                                    # 转换颜色通道并写入视频
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    video_writer.write(frame_bgr)
                                else:
                                    raise Exception("渲染帧为空")
                            except Exception as e:
                                if ep_len == 0:  # 只在第一步打印错误
                                    print(f"渲染错误: {e}")
                                    print("由于无法渲染真实环境，视频将显示空白帧")
                                
                                # 创建信息面板作为最后的降级选项
                                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                                
                                # 添加错误消息
                                cv2.putText(frame, "渲染失败 - 请检查环境配置", 
                                          (video_width//2-150, video_height//2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                                cv2.putText(frame, f"Epoch: {epoch}, Step: {ep_len}, Reward: {ep_ret:.2f}", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                                
                                # 写入帧
                                video_writer.write(frame)
                            
                            ep_len += 1
                            obs = next_obs
                    
                    # 恢复原始环境变量
                    os.environ.clear()
                    os.environ.update(original_env_vars)
                    
                    # 关闭视频写入器
                    video_writer.release()
                    print(f"视频已保存至: {video_path}")
                
                except Exception as e:
                    print(f"保存视频时出错: {e}")
                    import traceback
                    traceback.print_exc()

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('LossCond', with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()

def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


def get_time_limit(env: gym.Env):
    if hasattr(env, 'spec'):
        if hasattr(env.spec, 'max_episode_steps'):
            return env.spec.max_episode_steps
    if hasattr(env, 'env'):
        return get_time_limit(env.env)
    if hasattr(env, 'unwrapped'):
        return get_time_limit(env.unwrapped)
    else:
        raise ValueError("Cannot find time limit for env")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--log_dir', type=str, default='online_logs')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--gin_config_files', nargs='*', type=str,
                        default=['config/online/sac_synther_dmc.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.env, data_dir=args.log_dir, datestamp=True)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    try:
        redq_sac(args.env, target_entropy='auto', logger_kwargs=logger_kwargs)
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放虚拟显示资源
        if 'virtual_display' in globals() and has_virtual_display:
            try:
                print("正在关闭虚拟显示...")
                virtual_display.stop()
                print("虚拟显示已关闭")
            except:
                pass
