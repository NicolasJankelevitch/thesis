from datetime import datetime
import os
import signal
import time
from pathlib import Path
from secrets import username, password

import paramiko as paramiko
from tqdm import tqdm


def killall_on_machines(machine_ids_cores, new_machine=False):
    for machine_id, _ in machine_ids_cores:
        client = paramiko.SSHClient()
        if new_machine:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.load_host_keys("known_hosts.txt")
        hostname = "{}.cs.kuleuven.be".format(machine_id)
        print("killing processes on ", hostname)
        client.connect(
            hostname,
            timeout = 1000000,
            auth_timeout = 100000,
            banner_timeout = 1000000, 
            username=username,
            password=password
        )

        client.get_transport().set_keepalive(30)
        print("killing processes on ", hostname)
        (stdin, stdout, stderr) = client.exec_command("killall -u jonass", timeout = 100000)
        client.close()
    # return client, stdout, stderr


directory = "/cw/dtaijupiter/NoCsBack/dtai/jonass/cobras_testing/COBRAS_testing"


def run_task_files_over_ssh(task_files, machine_ids_cores, all_result_files, new_machine=False, skip_validation=False):
    # validate if we really want to run
    if not skip_validation:
        print("run remote tasks?")
        input_str = "None"
        while input_str not in "YynN":
            input_str = input("y/n:")
        if input_str in "nN":
            return

    # run all the task files keep a reference to the input, output and err streams
    machine_info = []
    for task_file, (machine_id, nb_cores) in zip(task_files, machine_ids_cores):
        print("running task {} on {}".format(task_file, machine_id))
        info_tuple = run_task_file_over_ssh(task_file, machine_id, nb_cores, new_machine)
        machine_info.append(info_tuple + (machine_id,))

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        # display a loading bar
        all_result_files = set(all_result_files)
        with tqdm(total=len(all_result_files)) as pbar:

            while len(machine_info) > 0:
                # update the progress bar based on the existing files
                finished = []
                for result_file in all_result_files:
                    if os.path.isfile(result_file):
                        finished.append(result_file)
                for finished_file in finished:
                    all_result_files.remove(finished_file)
                if len(finished) > 0:
                    pbar.update(len(finished))

                # if a client finished show message
                for client, output, error, machine_id in machine_info:
                    if output.channel.exit_status_ready():
                        out = output.readlines()
                        err = error.readlines()
                        if len(err) <= 1:
                            print("machine {} finished with no errors".format(machine_id))
                            print("\n".join(err))
                        else:
                            print("machine {} finished with error: (displaying last 8 lines)".format(machine_id))
                            print("\n".join(err))

                        machine_info.remove((client, output, error, machine_id))
                        client.close()
                # let this thread sleep (allows paramiko to do asynchronous work)
                time.sleep(5)
    except KeyboardInterrupt:
        killall_on_machines(machine_ids_cores)
        raise KeyboardInterrupt


def run_task_file_over_ssh(task_file, machine_id, nb_cores, new_machine=False):
    client = paramiko.SSHClient()
    client.load_host_keys("known_hosts.txt")
    if new_machine:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    hostname = "{}.cs.kuleuven.be".format(machine_id)
    print("connecting to", hostname)
    client.connect(
        hostname,
        username=username,
        password=password
    )


    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y_%H:%M")
    path = Path('/cw/dtaijupiter/NoCsBack/dtai/jonass/cobras_testing/logging')
    path = path / dt_string / task_file
    path.parent.mkdir(parents = True, exist_ok = True)
    client.get_transport().set_keepalive(30)
    execute_file_command = "PYTHONPATH='/cw/dtaijupiter/NoCsBack/dtai/jonass/cobras_testing/COBRAS_testing' python3 /cw/dtaijupiter/NoCsBack/dtai/jonass/cobras_testing/COBRAS_testing/run_through_ssh/run_task_file.py {} {} >{} 2>&1".format(
        task_file, nb_cores, path)
    (stdin, stdout, stderr) = client.exec_command(execute_file_command, get_pty=False)
    return client, stdout, stderr

# if __name__ == '__main__':
#     # run_task_file_on_machine("machine1.txt", "pinac21", 8)
#     run_task_files(["machine1.txt","machine2.txt", "machine3.txt","machine4.txt","machine5.txt"], [("pinac21",8),("pinac22",8),("pinac23",8),("pinac24",8),("pinac25",8)],new_machine=True)
