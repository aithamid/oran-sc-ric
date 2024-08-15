#!/usr/bin/env python3

import time
import datetime
import argparse
import signal
import csv
import pprint
from lib.xAppBase import xAppBase

class UE:
    def __init__(self, ip, ueid, gnbid):
        self.ip = ip
        self.ueid = ueid
        self.gnbid = gnbid

# Creating instances of UE
ue1 = UE(ip="10.45.1.2", ueid="0", gnbid="gnbd_001_001_00000213_0")
ue2 = UE(ip="10.45.1.3", ueid="1", gnbid="gnbd_001_001_00000213_0")
virtual = UE(ip="10.45.1.4", ueid="0", gnbid="gnbd_001_001_00000213_1")

# Creating a dictionary to store the UE instances
ue_dict = {
    ue1.ip: ue1,
    ue2.ip: ue2,
    virtual.ip: virtual
}

OUTPUT_FOLDER = "./output/"
RC_HEADER = ['Time', 'Attack/Restore', 'IP', 'PRB']

file_path = "attackers.txt"

DECREASE = 5
INCREASE = 100

class MyXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        self.old_attackers = []
        self.attackers = []

        # CSV Output file
        self.history_file = open(OUTPUT_FOLDER + 'resourcecontrol-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        self.history_file.truncate(0)  # clean the file content (as we open the file in append mode)
        self.history_writer = csv.DictWriter(self.history_file, fieldnames=RC_HEADER)
        self.history_writer.writeheader()
        self.history_file.flush()

        super(MyXapp, self).__init__(config, http_server_port, rmr_port)
        pass

    def change_resources(self, ip, max_prb_ratio):
        min_prb_ratio = 1
        current_time = datetime.datetime.now()
        e2_node_id = ue_dict[ip].gnbid
        ue_id = ue_dict[ip].ueid
        print(f"\033[36m{current_time.strftime('%H:%M:%S')} - Sending RIC Control Request to E2 node ID: {e2_node_id} for UE ID: {ue_id}, PRB_min: {min_prb_ratio}, PRB_max: {max_prb_ratio}\033[0m")
        self.e2sm_rc.control_slice_level_prb_quota(e2_node_id, int(ue_id), min_prb_ratio=1, max_prb_ratio=max_prb_ratio, dedicated_prb_ratio=100, ack_request=1)
        time.sleep(1)

    @xAppBase.start_function
    def start(self):
        while self.running:
            print("\033[32m---- xAPP Attack Detection Active ----\033[0m")

            self.old_attackers = self.attackers[:]
            self.attackers.clear()
            
            with open(file_path, 'r') as file:
                file = file.readlines()
                if len(file) == 0:
                    print("\033[33mNo attackers detected.\033[0m")
                else:
                    for line in file:
                        ip = line.strip()
                        if ip in ue_dict:
                            self.attackers.append(ip)

            for attacker_ip in self.attackers:
                if attacker_ip not in self.old_attackers:
                    row = {
                        'Time':time.strftime("%D %T"),
                        'Attack/Restore':"Attack",
                        'IP':attacker_ip,
                        'PRB':DECREASE
                    }
                    self.history_writer.writerow(row)
                    self.history_file.flush()
                    pprint.pprint(row, sort_dicts=False)
                    print(f"\033[31mAttackers detected with IP address {attacker_ip}:\033[0m")
                    self.change_resources(attacker_ip, DECREASE)
                
            for old_attacker in self.old_attackers:
                if old_attacker not in self.attackers:
                    row = {
                        'Time':time.strftime("%D %T"),
                        'Attack/Restore':"Restore",
                        'IP':attacker_ip,
                        'PRB':INCREASE
                    }
                    self.history_writer.writerow(row)
                    self.history_file.flush()
                    print("\033[35mRestoring resources for previously detected attackers.\033[0m")
                    self.change_resources(old_attacker, INCREASE)

            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My example xApp')
    parser.add_argument("--config", type=str, default='', help="xApp config file path")
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server listen port")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port")
    parser.add_argument("--ran_func_id", type=int, default=3, help="E2SM RC RAN function ID")

    args = parser.parse_args()
    config = args.config
    ran_func_id = args.ran_func_id

    myXapp = MyXapp(config, args.http_server_port, args.rmr_port)
    myXapp.e2sm_rc.set_ran_func_id(ran_func_id)

    signal.signal(signal.SIGQUIT, myXapp.signal_handler)
    signal.signal(signal.SIGTERM, myXapp.signal_handler)
    signal.signal(signal.SIGINT, myXapp.signal_handler)

    myXapp.start()
