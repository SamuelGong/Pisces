"""
Author: Zhifeng Jiang From HKUST
Home-made logic for installing, deploying and using Plato it in a cluster
where the server and clients can be placed in separate nodes.
"""

import sys
import json
import os
import yaml
from utils import ExecutionEngine
from shutil import copy as shutil_copy
import copy


def chunks_idx(l, n):
    d, r = divmod(l, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def copy_config(config_path, task_folder, config_rel):
    new_config_path = os.path.join(task_folder, config_rel)
    shutil_copy(config_path, new_config_path)
    return new_config_path


def my_insert(cur, tup, val):
    if len(tup) == 1:
        cur[tup[0]] = val
        return
    if tup[0] not in cur:
        cur[tup[0]] = {}
    my_insert(cur[tup[0]], tup[1:], val)


def edit_config(config_path, keys, value):
    with open(config_path, 'r') as fin:
        dictionary = yaml.load(fin, Loader=yaml.FullLoader)
    my_insert(dictionary, keys, value)
    with open(config_path, 'w') as fout:
        yaml.dump(dictionary, fout)


def extract_coordinator_address(launch_result):
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        if 'coordinator' in name:
            return public_ip
    return None


def get_client_launch_plan(config_path, launch_result):
    with open(config_path, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    total_clients = config['clients']['total_clients']
    num_workers = len(launch_result) - 1
    range_generator = chunks_idx(total_clients, num_workers)

    client_launch_plan = {}
    for node in launch_result:
        name = node['name']
        if 'worker' in name:
            begin, end = next(range_generator)
            num_clients = end - begin
            client_launch_plan[name] = num_clients

    return client_launch_plan, total_clients


def main(args):
    command = args[0]
    launch_result_path = args[1]
    local_private_key_path = args[2]
    last_response_path = args[3]
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    if command in ["remote_start", "remote_kill", "collect_result"]:
        last_response_rel = last_response_path
        issue_result_rel = 'pid.txt'  # TODO: avoid hard-coding
        log_rel = 'log.txt'
        config_rel = 'config.yml'
        result_rel = 'result.csv'

        if command == "remote_start":
            start_time = args[4]
            config_path = args[5]
            server_port = 8000  # TODO: avoid hard-coding
            client_port = 80
            coordinator_address = extract_coordinator_address(launch_result)

            task_parent_folder = '/'.join(config_path.split('/')[:-1])
            task_folder = os.path.join(task_parent_folder, start_time)
            os.makedirs(task_folder)
            last_response_path = os.path.join(task_folder, last_response_rel)

            idx = task_parent_folder.find("Pisces")
            remote_task_parent_folder = task_parent_folder[idx:]
            remote_task_folder = os.path.join(remote_task_parent_folder, start_time)
            client_launch_plan, total_clients = get_client_launch_plan(
                config_path, launch_result)
            new_config_path = copy_config(config_path, task_folder, config_rel)
            edit_config(new_config_path, ['results', 'results_dir'],
                        '/home/ubuntu/' + remote_task_folder + '/')
            edit_config(new_config_path, ['clients', 'total_clients'],
                        total_clients)
        else:
            task_folder = args[4]
            idx = task_folder.find("Pisces")
            remote_task_folder = task_folder[idx:]
            last_response_path = os.path.join(task_folder, last_response_rel)
        remote_task_folder_short = '/'.join(remote_task_folder.split('/')[-2:])
    elif command in ["add_dependency"]:
        dependency = args[4]

    execution_plan_list = []
    remote_template = {
        'username': 'ubuntu',
        'key_filename': local_private_key_path
    }
    client_idx = 1  # cannot start from 0 if using FEMNIST!
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = []

        if command == "standalone":
            remote_template.update({
                'commands': [
                    "cd Pisces/experiments/dev/ "
                    "&& source standalone_install.sh"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Standalone installation finished on node '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "clean_memory":
            remote_template.update({
                'commands': [
                    "sudo sh -c 'echo 3 >  /proc/sys/vm/drop_caches'"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Memory clean on node '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "deploy_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        "cd Pisces/experiments/dev/ "
                        "&& source cluster_install.sh",
                        "sudo apt install unzip -y "
                        "&& cd ~/Pisces "
                        "&& wget https://jiangzhifeng.s3.us-east-2.amazonaws.com/plato-data-server.zip "
                        "&& unzip -q plato-data-server.zip "
                        "&& rm plato-data-server.zip"
                    ]
                })
            else:
                remote_template.update({
                    'commands': [
                        "sudo apt install unzip -y "
                        "&& cd ~/Pisces "
                        "&& wget https://jiangzhifeng.s3.us-east-2.amazonaws.com/plato-data-client.zip "
                        "&& unzip -q plato-data-client.zip "
                        "&& rm plato-data-client.zip"
                    ]
                })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Cluster server deployed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "start_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        "sudo systemctl start nginx"
                    ]
                })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Cluster server started on '
                                f'{name} ({public_ip}).'])
                ]

        elif command == "add_dependency":
            remote_template.update({
                'commands': [
                    "source ~/anaconda3/etc/profile.d/conda.sh && "
                    "conda activate pisces && "  # TODO: avoid hard-coding the env name
                    f"pip install {dependency} && "
                    "conda deactivate"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Dependency {dependency} installed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "remote_start":
            if 'coordinator' in name:
                before_copy = copy.deepcopy(remote_template)
                after_copy = copy.deepcopy(remote_template)
                before_copy.update({
                    'commands': [
                        f"mkdir -p {remote_task_parent_folder} && "
                        f"cd {remote_task_parent_folder} && "
                        f"mkdir -p {start_time}"
                    ]
                })
                after_copy.update({
                    'commands': [
                        f"source ~/anaconda3/etc/profile.d/conda.sh && "
                        f"cd ~/Pisces && conda activate pisces && "
                        f"cat ~/{remote_task_folder}/{name}-{config_rel} "
                        f"> ~/{remote_task_folder}/{log_rel} && ./run "
                        f"--config=$HOME/{remote_task_folder}/{name}-{config_rel} "
                        f"--port {server_port} "
                        f">> ~/{remote_task_folder}/{log_rel} 2>&1 &"
                    ]
                })
                execution_sequence = [
                    ('remote', before_copy),
                    ('local', [
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"{new_config_path} "
                        f"ubuntu@{public_ip}:"
                        f"~/{remote_task_folder}/{name}-{config_rel}"
                    ]),
                    ('remote', after_copy),
                    ('prompt', [f'Task {start_time} started on '
                                f'{name} ({public_ip}).'])
                ]
            else:
                before_copy = copy.deepcopy(remote_template)
                after_copy = copy.deepcopy(remote_template)
                before_copy.update({
                    'commands': [
                        # "sleep 10",  # allow some time for the server to start
                        f"mkdir -p {remote_task_parent_folder} && "
                        f"cd {remote_task_parent_folder} && "
                        f"mkdir -p {start_time}"
                    ]
                })
                commands = []
                num_clients = client_launch_plan[name]
                commands.append(f"cat ~/{remote_task_folder}/{name}-{config_rel} "
                                f"> ~/{remote_task_folder}/{log_rel}")
                for i in range(num_clients):
                    actual_client_idx = client_idx + i
                    commands.append(f"source ~/anaconda3/etc/profile.d/conda.sh && "
                                    f"cd ~/Pisces && conda activate pisces && ./run_client "
                                    f"--config=$HOME/{remote_task_folder}/{name}-{config_rel} "
                                    f"-i {actual_client_idx} "
                                    f"--server {coordinator_address}:{client_port} "
                                    f">> ~/{remote_task_folder}/{log_rel} 2>&1 &")
                client_idx += num_clients

                after_copy.update({ 'commands': commands })
                execution_sequence = [
                    ('remote', before_copy),
                    ('local', [
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"{new_config_path} "
                        f"ubuntu@{public_ip}:"
                        f"~/{remote_task_folder}/{name}-{config_rel}"
                    ]),
                    ('remote', after_copy),
                    ('prompt', [f'Task {start_time} started on '
                                f'{name} ({public_ip}).'])
                ]

        elif command == "remote_kill":  # only work when there is only one task running
            remote_template.update({
                'commands': [
                    f"ps -ef | grep plato | grep python > "
                    f"{remote_task_folder}/{issue_result_rel}",
                    f"ps -ef | grep pisces | grep python >> "
                    f"{remote_task_folder}/{issue_result_rel}",
                    f"cat {remote_task_folder}/{issue_result_rel} | awk '{{print $2}}' "
                    f"| xargs kill -9 1>/dev/null 2>&1"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Task {remote_task_folder_short} killed on '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "collect_result":
            if 'coordinator' in name:
                remote_template.update({  # so that git pull does not alert of inconsistency
                    'commands': [
                        f"rm -rf ~/{remote_task_folder}/{result_rel}"
                    ]
                })
                execution_sequence = [
                    ('local', [
                        f"mkdir -p {task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{log_rel} "
                        f"{task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{result_rel} "
                        f"{task_folder}",
                    ]),
                    ('remote', remote_template),
                    ('prompt', [f'Partial results for {remote_task_folder_short} '
                                f'retrieved from {name} ({public_ip}).'])
                ]
            else:
                execution_sequence = [
                    ('local', [
                        f"mkdir -p {task_folder}/{name}",
                        f"scp -q -i {local_private_key_path} "
                        f"-o StrictHostKeyChecking=no "
                        f"-o UserKnownHostsFile=/dev/null "
                        f"ubuntu@{public_ip}:~/{remote_task_folder}/{log_rel} "
                        f"{task_folder}/{name}"
                    ]),
                    ('prompt', [f'Partial results for {remote_task_folder_short} '
                                f'retrieved from {name} ({public_ip}).'])
                ]
        else:
            execution_sequence = []

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)

    if command == "remote_start":
        print(f"\nUse the command at local host if you want to stop it:\n"
              f"\tbash run.sh kill_a_task {remote_task_folder_short}")
        print(f"Use the command at local host if you want to retrieve the log:\n"
              f"\tbash run.sh conclude_a_task {remote_task_folder_short}")
        print(f"Use the command at a computing node if you want to watch its state:\n"
              f"\tvim ~/{remote_task_folder}/{log_rel}")


if __name__ == '__main__':
    main(sys.argv[1:])