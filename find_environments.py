import os
import sys
import importlib
import inspect

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath('.'))

def scan_imports():
    """扫描代码中导入的环境库"""
    env_libraries = set()
    
    # 检查是否导入了常见的环境库
    try:
        import gym
        env_libraries.add("gym")
        print("发现 OpenAI Gym 支持")
    except ImportError:
        print("未找到 OpenAI Gym")
    
    try:
        import dmcgym
        env_libraries.add("dmcgym")
        print("发现 DeepMind Control Suite (dmcgym) 支持")
    except ImportError:
        print("未找到 DeepMind Control 环境")
    
    try:
        import mujoco_py
        env_libraries.add("mujoco_py")
        print("发现 MuJoCo 支持")
    except ImportError:
        print("未找到 MuJoCo")
    
    return env_libraries

def find_env_names_in_code():
    """在代码中查找硬编码的环境名称"""
    env_names = set()
    mbpo_envs = set()
    
    # 从core.py中获取mbpo_target_entropy_dict字典
    try:
        from synther.REDQ.redq.algos.core import mbpo_target_entropy_dict, mbpo_epoches
        mbpo_envs = set(mbpo_target_entropy_dict.keys())
        print(f"\n在mbpo_target_entropy_dict中找到 {len(mbpo_envs)} 个环境:")
        for env in sorted(mbpo_envs):
            print(f"  - {env}")
        env_names.update(mbpo_envs)
    except ImportError:
        print("无法导入mbpo_target_entropy_dict")
    
    # 搜索Python文件中的环境名称
    env_pattern_prefixes = [
        'Hopper-', 'HalfCheetah-', 'Walker2d-', 'Ant-', 'Humanoid-', 
        'quadruped-', 'cheetah-', 'reacher-'
    ]
    
    env_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') or file.endswith('.gin'):
                env_files.append(os.path.join(root, file))
    
    print(f"\n在 {len(env_files)} 个文件中搜索环境名称...")
    
    for file_path in env_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 查找常见环境名称模式
                for line in content.split('\n'):
                    for prefix in env_pattern_prefixes:
                        if prefix in line:
                            parts = line.split(prefix)
                            for i in range(1, len(parts)):
                                # 尝试提取完整环境名称
                                name_part = parts[i]
                                if name_part:
                                    # 提取环境版本号
                                    version = ""
                                    for char in name_part:
                                        if char.isdigit() or char == 'v':
                                            version += char
                                        else:
                                            break
                                    
                                    if version:
                                        env_name = prefix + version
                                        env_names.add(env_name)
        except:
            continue
    
    # 删除可能的误报和重复
    env_names = sorted(env_names)
    
    print(f"\n共找到 {len(env_names)} 个可能的环境:")
    for env in env_names:
        print(f"  - {env}")
    
    return env_names

def inspect_dmcgym_environments():
    """检查DMC环境包含哪些环境"""
    try:
        import dmcgym
        import gym
        
        # 尝试列出所有DMC环境
        all_envs = []
        for env_id in gym.envs.registry.env_specs:
            if any(domain in env_id for domain in ['cheetah', 'quadruped', 'reacher', 'walker', 'humanoid']):
                all_envs.append(env_id)
        
        if all_envs:
            print(f"\n从gym注册表中发现 {len(all_envs)} 个DMC环境:")
            for env_id in sorted(all_envs):
                print(f"  - {env_id}")
            return all_envs
        else:
            print("\n无法从gym注册表中获取DMC环境")
            return []
    except Exception as e:
        print(f"\n检查DMC环境时出错: {e}")
        return []

def check_robot_tasks():
    """检查项目中是否有机器人相关任务，特别是医疗/手术机器人"""
    robot_keywords = [
        'robot', 'surgical', 'surgery', 'medical', 'davinci', 
        'da vinci', 'manipulator', 'arm', 'gripper'
    ]
    
    matches = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') or file.endswith('.md') or file.endswith('.gin'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        for keyword in robot_keywords:
                            if keyword in content:
                                # 提取上下文
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if keyword in line.lower():
                                        context = {
                                            'file': file_path,
                                            'line_num': i + 1,
                                            'line': lines[i].strip(),
                                            'keyword': keyword
                                        }
                                        matches.append(context)
                except:
                    continue
    
    if matches:
        print(f"\n找到 {len(matches)} 处与机器人相关的关键词匹配:")
        for match in matches:
            print(f"  文件: {match['file']}, 行 {match['line_num']}, 关键词: '{match['keyword']}'")
            print(f"    {match['line']}")
    else:
        print("\n未找到与医疗/手术机器人相关的任务")
    
    return matches

def analyze_project_tasks():
    """分析项目中实现的任务类型"""
    print("=" * 50)
    print("项目环境和任务分析")
    print("=" * 50)
    
    # 1. 扫描导入的环境库
    env_libraries = scan_imports()
    
    # 2. 在代码中查找环境名称
    env_names = find_env_names_in_code()
    
    # 3. 检查DMC环境（如果可用）
    if 'dmcgym' in env_libraries:
        dmc_envs = inspect_dmcgym_environments()
    
    # 4. 检查是否有医疗/手术机器人相关任务
    robot_matches = check_robot_tasks()
    
    # 5. 总结项目支持的环境和任务
    print("\n" + "=" * 50)
    print("项目环境和任务总结")
    print("=" * 50)
    
    # 对环境进行分类
    mujoco_envs = [env for env in env_names if any(prefix in env for prefix in ['Hopper', 'HalfCheetah', 'Walker2d', 'Ant', 'Humanoid'])]
    dmc_envs_found = [env for env in env_names if any(prefix in env for prefix in ['quadruped', 'cheetah', 'reacher'])]
    
    print(f"\n1. MuJoCo环境: {len(mujoco_envs)}个")
    for env in sorted(mujoco_envs):
        print(f"  - {env}")
    
    print(f"\n2. DeepMind Control环境: {len(dmc_envs_found)}个")
    for env in sorted(dmc_envs_found):
        print(f"  - {env}")
    
    print("\n3. 医疗/手术机器人任务:")
    if robot_matches:
        print(f"  找到{len(robot_matches)}处可能与机器人相关的引用")
        robot_files = set(match['file'] for match in robot_matches)
        print(f"  涉及文件: {', '.join(robot_files)}")
    else:
        print("  未找到与医疗/手术机器人相关的任务")
    
    # 6. 结论
    print("\n" + "=" * 50)
    print("结论")
    print("=" * 50)
    
    has_surgical_robot = any('davinci' in match['keyword'].lower() or 'surgical' in match['keyword'].lower() for match in robot_matches)
    
    if has_surgical_robot:
        print("该项目包含与达芬奇手术机器人或医疗手术机器人相关的任务。")
    else:
        print("该项目主要包含标准强化学习基准环境，未发现与达芬奇手术机器人或医疗手术机器人相关的特定任务。")
        print("项目使用的环境包括:")
        print("1. 常规MuJoCo环境: Hopper, HalfCheetah, Walker2d, Ant, Humanoid等")
        print("2. DeepMind Control Suite环境: quadruped-walk, cheetah-run, reacher-hard等")
        print("\n这些环境主要关注基本的运动控制和机器人学习，而不是医疗手术机器人的精细操作任务。")

if __name__ == "__main__":
    analyze_project_tasks() 