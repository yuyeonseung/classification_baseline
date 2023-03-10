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
                f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker exec {dockername} echo checking..."'
                , shell=True
            )

            if check.returncode != 0:
                # print('No response')
                continue
            else:
                print("Check Success")
                while True :
                    nproc=input(f"worker-{worker_number}'s nporc : ")
                    
                    if nproc in ['1','2','3','4','5','6','7','8']:
                        break
                    else:
                        print('Please retry')
                        continue

                worker_number += 1

                command = f'python -m torch.distributed.launch --master_addr={args.master_addr} --master_port={args.master_port} --nnodes={args.nnodes} --node_rank={worker_number-1} --nproc_per_node={nproc} train_ddp.py {args.argument}'
                command_list.insert(0,f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker exec {dockername} sh -c \'cd {os.getcwd()} && {command}\'"')
                stop_command_list.insert(0,f'sshpass -p {password} ssh {host_name} "echo \'{password}\' | sudo -S docker restart {dockername}"')
                print(f'Success - worker-{worker_number} is ready (nproc - {nproc})')

        for cmd in reversed(command_list):
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
    assert args.nnodes > 1 ,'?????? ????????? ????????? ???????????????.' 
    main(args)
