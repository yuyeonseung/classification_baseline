import argparse
import subprocess
import os
from getpass import getpass

def main(args):
    try :
        command_list = [f'python -m torch.distributed.launch --master_addr={args.master_addr} --master_port={args.master_port} --nnodes={args.nnodes} --node_rank=0 --nproc_per_node={args.nproc_per_node} train_ddp.py {args.argument}']
        stop_command_list = []
        worker_number = 1
        # for worker_number in range(1, args.nnodes):
        while worker_number < args.nnodes:
            user=input(f"worker-{worker_number}'s username : ")
            host=input(f"worker-{worker_number}'s host : ")
            dockername=input(f"worker-{worker_number}'s dockername : ")
            host_name = f'{user}@{host}'
            password=getpass(f"worker-{worker_number}'s password : ")
            
            check = subprocess.run(
                f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker exec {dockername} ls"'
                , shell=True
                , capture_output=False
            )

            if check.returncode != 0:
                # print('No response')
                continue
            else:
                
                nproc=input(f"Check Success \n worker-{worker_number}'s nporc : ")
                worker_number += 1

                command = f'python -m torch.distributed.launch --master_addr={args.master_addr} --master_port={args.master_port} --nnodes={args.nnodes} --node_rank={worker_number-1} --nproc_per_node={nproc} train_ddp.py {args.argument}'
                command_list.insert(0,f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker exec {dockername} sh -c \'cd {os.getcwd()} && {command}\'"')
                stop_command_list.insert(0,f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker restart {dockername}"')
                print(f'Success - worker-{worker_number} is ready')

        for cmd in reversed(command_list):
            # print(cmd)
            master_command = '&'.join(command_list)
        try :
            subprocess.run(
                master_command
                , shell=True
            )
        
        except KeyboardInterrupt:
            for stop_command in stop_command_list:
                subprocess.run(
                    stop_command
                    , shell=True
                )
            print('Process killed')
    except Exception as e:
        print('Failed', e)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_addr', type=str, required=True)
    parser.add_argument('--master_port', type=int, required=True)
    parser.add_argument('--nnodes', type=int, required=True)
    parser.add_argument('--nproc_per_node', type=str, required=True)
    parser.add_argument('--argument', type=str, required=True)

    args = parser.parse_args()
    assert args.nnodes > 1 ,'두개 이상의 노드가 필요합니다.' 
    main(args)
