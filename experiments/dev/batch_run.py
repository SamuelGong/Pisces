import time
import json
import sys
import os
import logging
from utils import execute_remotely, execute_locally
sys.path.append('..')
from patch.utils import calc_sleep_time


def poll_if_stop(coordinator_public_ip, local_private_key_path):
    username = 'ubuntu'
    commands = ['ps -ef | grep plato | grep python']
    try:
        raw_resp = execute_remotely(commands, coordinator_public_ip,
                                    username, local_private_key_path)
        resp = raw_resp[0]['stdout']
        stopped = True
        for line in resp:
            if 'config' in line.split('\n')[0]:
                stopped = False
                break
        if stopped:  # for debugging only, should be deleted later
            logging.info(f"It seems that the program has stopped. "
                         f"The responses are {resp}")
        return stopped
    except Exception as e:  # probably SSH errors
        return str(e)


def start_tasks(launch_result_path, local_private_key_path,
                batch_plan_path, shutdown_after_completion=False):
    logging.basicConfig(
        format='[%(levelname)s][%(asctime)s]: %(message)s',
        level=logging.INFO,
        datefmt='(%m-%d) %H:%M:%S')
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.info(f'Batch jobs started. '
                 f'The pid of this coordinator is {os.getpid()}. '
                 f'Will shut down the cluster after completion: '
                 f'{shutdown_after_completion}.')

    # Step 1: find the coordinator's public IP
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)
    for node in launch_result:
        if 'coordinator' in node['name']:
            coordinator_public_ip = node['public_ip']
            break
    logging.info(f'The public IP of the coordinator is {coordinator_public_ip}')

    # Step 2: launching and polling
    with open(batch_plan_path, 'r') as fin:
        tasks = fin.readlines()
    tasks = [task.split('\n')[0] for task in tasks]

    poll_interval = 20
    for idx, task in enumerate(tasks):
        if not task:
            continue

        start_cmd = f'bash setup.sh clean_memory'
        try:
            _ = execute_locally([start_cmd])
        except Exception as e:
            logging.info(f'Failed to clean memory. '
                  f'Messages: {e} ({type(e)}))')
            continue

        start_cmd = f'bash run.sh start_a_task {task}'
        try:
            raw_resp = execute_locally([start_cmd])
        except Exception as e:
            logging.info(f'Failed to start {task}. '
                  f'Messages: {e} ({type(e)}))')
            continue

        resp_list = raw_resp[0]['resp'].split('\n')

        # highly dependent on the code in plato_related.py
        try:
            kill_cmd = resp_list[-5].replace('\t', '')
            conclude_cmd = resp_list[-3].replace('\t', '')
            vim_cmd = resp_list[-1].replace('\t', '')
        except Exception as e:  # error case 1
            logging.info(f'Failed to start {task}. '
                         f'Messages: {resp_list}')
            continue
        else:
            if 'bash' not in kill_cmd:  # error case 2
                logging.info(f'Failed to start Task {idx}: {task}. '
                             f'Responses: {resp_list}')
                continue

        logging.info(f'Task {idx}: {task} started.\n'
                     f'Can be killed halfway by \n\t{kill_cmd}\n'
                     f'Can be concluded by \n\t{conclude_cmd}\n'
                     f'Can be viewed at a computing node by \n\t{vim_cmd}')

        poll_step = 0
        start_time = time.perf_counter()
        logging.info(f'Polling its status...')
        while True:
            stopped = poll_if_stop(coordinator_public_ip, local_private_key_path)
            if isinstance(stopped, bool):
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}, "
                             f"task {task} has stopped: {stopped}.")
                if stopped:
                    break
            else:
                logging.info(f"\tElapsed time: "
                             f"{round(time.perf_counter() - start_time)}. "
                             f"Status cannot be probed due to {stopped}. "
                             f"Retry in {poll_interval} seconds.")

            sleep_time = calc_sleep_time(poll_interval,
                                         poll_step, start_time)
            time.sleep(sleep_time)
            poll_step += 1

        try:
            _ = execute_locally([conclude_cmd])
        except Exception as e:
            logging.info(f'Failed to conclude {task}. '
                         f'Messages: {e} ({type(e)}))')
            continue
        logging.info(f'Task {idx}: {task} concluded.')

    logging.info(f'Batch jobs ended.')

    if shutdown_after_completion:
        shutdown_cmd = 'bash manage_cluster.sh stop'
        try:
            _ = execute_locally([shutdown_cmd])
        except Exception as e:
            logging.info(f'Failed to shut down the cluster. '
                         f'Messages: {e} ({type(e)}))')
        else:
            logging.info(f'The cluster shut down.')

def main(args):
    command = args[0]
    if command == 'start_tasks':
        launch_result_path = args[1]
        local_private_key_path = args[2]
        batch_plan_path = args[3]
        # start_tasks(launch_result_path,
        #             local_private_key_path, batch_plan_path)
        start_tasks(launch_result_path,
                    local_private_key_path, batch_plan_path, True)


if __name__ == '__main__':
    main(sys.argv[1:])
