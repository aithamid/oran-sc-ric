#!/usr/bin/env python3

import os
import sys
import time
import argparse
import signal
import subprocess
from lib.xAppBase import xAppBase
from pathlib import Path
from threading import Thread
import csv
import pprint
from lucid_dataset_parser import *
from lucid_cnn import *


class AttackDetectionXapp(xAppBase):
    def __init__(self, config, http_server_port, rmr_port):
        super(AttackDetectionXapp, self).__init__(config, http_server_port, rmr_port)


    def lucid_live_predict (self, predict_live, model_path, attack_net, victim_net):

        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if predict_live is None:
            #container should have access to the pcap file. We should update docker-compose file to add its path.
            pcap_file = f"/opt/pcap_files/traffic.pcap"
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        elif predict_live.endswith('.pcap'):
            pcap_file = predict_live
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        else:
            cap =  pyshark.LiveCapture(interface=predict_live)
            data_source = predict_live

        print(f"Starting lucid prediction on {data_source}")
        #print ("Prediction on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(None, attack_net, victim_net)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(model_path)

        mins, maxs = static_min_max(time_window)

        while (True):
            samples, source_ips = process_live_traffic(cap, None, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X,Y_true,keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5,axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])

                # Filter attacker IPs based on prediction results
                malicious_ips = set()
                for i, pred in enumerate(Y_pred):
                    if pred > 0.5 and keys[i][0].startswith('10.45.') :  # If the prediction is positive for DDoS
                        malicious_ips.add(keys[i][0])  # Add source IP of the malicious flow

                self.report_process_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time, predict_writer, list(malicious_ips))
                predict_file.flush()    
            
            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\\nNo more packets in file ", data_source)
                break
        
        predict_file.close()    

    def write_attacker_ips_to_file(self, attacker_ips, file_path='attackers.txt'):
        with open(file_path, 'w') as f:
            for ip in attacker_ips:
                f.write(f"{ip}\n")
    
    def report_process_results(self, Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer, attacker_ips):
        ddos_rate = sum(Y_pred) / Y_pred.shape[0]
        output_file  = OUTPUT_FOLDER+'Attackers.txt'
        
        if Y_true is not None and len(Y_true.shape) > 0:  # if we have the labels, we can compute the classification accuracy
            Y_true = Y_true.reshape((Y_true.shape[0], 1))
            accuracy = accuracy_score(Y_true, Y_pred)

            f1 = f1_score(Y_true, Y_pred)
            tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0

            row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
                'Samples': Y_pred.shape[0], 'DDOS%': '{:04.3f}'.format(ddos_rate), 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1),
                'TPR': '{:05.4f}'.format(tpr), 'FPR': '{:05.4f}'.format(fpr), 'TNR': '{:05.4f}'.format(tnr), 'FNR': '{:05.4f}'.format(fnr),
                'Source': data_source, 'Attackers': ', '.join(attacker_ips)}
            
            if ddos_rate > 0.7 and accuracy > 0.7 and f1 > 0.7:
                with open(output_file, 'w') as file:
                    file.writelines(f"{ip}\n" for ip in attacker_ips)
            else:
                with open(output_file, 'w') as file:
                    pass
            
        else:
            row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
                'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
                'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source, 'Attackers': ', '.join(attacker_ips)}
           
            with open(output_file, 'w') as file:
                pass
        pprint.pprint(row, sort_dicts=False)
        writer.writerow(row)
        self.write_attacker_ips_to_file(attacker_ips)
           
    

    @xAppBase.start_function
    def start(self, predict_live, model_path, attack_net, victim_net):
        print("Xapp Start function....")
        
        try:
            self.lucid_live_predict (predict_live, model_path, attack_net, victim_net)
 
        except KeyboardInterrupt:
            print("Script stopped by user.")
        
        except Exception as e:
            print(f"Error processing provided {predict_live}: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Traffic Monitor xApp')
    parser.add_argument("--config", type=str, default='', help="xApp config file path")
    parser.add_argument("--http_server_port", type=int, default=8090, help="HTTP server listen port")
    parser.add_argument("--rmr_port", type=int, default=4560, help="RMR port")
    parser.add_argument("--e2_node_id", type=str, default='gnbd_001_001_00019b_0', help="E2 Node ID")
    parser.add_argument("--ran_func_id", type=int, default=3, help="E2SM RC RAN function ID")
    parser.add_argument("--predict_live", required=False, help="Network interface to monitor or pcap file")
    parser.add_argument("--model", type=str, required=True, help="Path to the LUCID model file")
    parser.add_argument("--attack_net", type=str, required=True, help="CIDR of the attack network")
    parser.add_argument("--victim_net", type=str, required=True, help="CIDR of the victim network")
    #parser.add_argument("--ue_ids", type=str, required=True, help="Comma-separated list of UE IDs")

    args = parser.parse_args()
    config = args.config
    e2_node_id = args.e2_node_id # TODO: get available E2 nodes from SubMgr, now the id has to be given.
    ran_func_id = args.ran_func_id # TODO: get available E2 nodes from SubMgr, now the id has to be given.
   
    if args.model is not None and args.model.endswith('.h5'):
        model_path = args.model
    else:
        print ("No valid model specified!")
        exit(-1)
   
    attack_net = args.attack_net
    victim_net = args.victim_net
    
    predict_live = args.predict_live


    # Create and start the xApp
    myXapp = AttackDetectionXapp(config, args.http_server_port, args.rmr_port)
    myXapp.e2sm_rc.set_ran_func_id(ran_func_id)
    
    # Connect exit signals.
    signal.signal(signal.SIGQUIT, myXapp.signal_handler)
    signal.signal(signal.SIGTERM, myXapp.signal_handler)
    signal.signal(signal.SIGINT, myXapp.signal_handler)

    myXapp.start(predict_live, model_path, attack_net, victim_net)

# For execution:
# docker compose exec python_xapp_runner ./attack_detection_xapp_v3.py --predict_live /opt/pcap_files/traffic.pcap --model ./lucid-ddos-master-github/output_github/10t-10n-DOS2019-LUCID.h5 --attack_net 10.45.1.0/24 --victim_net 10.53.1.2 --ue_ids 0,1
