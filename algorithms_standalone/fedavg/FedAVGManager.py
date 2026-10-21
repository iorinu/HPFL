import copy
import logging
import random
from re import M
import numpy as np
import torch
import wandb
import json
import time
import os
import signal
import subprocess

from .client import FedAVGClient
from .aggregator import FedAVGAggregator

from utils.data_utils import (
    get_avg_num_iterations,
    get_label_distribution,
    get_selected_clients_label_distribution
)
from utils.checkpoint import setup_checkpoint_config, save_checkpoint


from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer

from algorithms.fedavg.fedavg_server_timer import FedAVGServerTimer

class FedAVGManager(BasePSManager):
    def __init__(self, device, args):
        self.client_executor = None
        self.process_history = []
        self.capture_process_history = False
        if args.client_execution_backend == "process":
            from configs.experiment import validate_process_config
            validate_process_config(args)
            if str(device) not in ("cpu", "cuda:0"):
                raise ValueError("process は cpu または単一の cuda:0 のみ対応します")
            if type(args.max_concurrent_clients) is not int or not 1 <= args.max_concurrent_clients <= 2:
                raise ValueError("process 並行数は 1..2 が必要です")
            if args.client_rng_policy != "task-sha256-v1":
                raise ValueError("process の乱数ポリシーが不正です")
        super().__init__(device, args)

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        # local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

        self.server_timer = FedAVGServerTimer(
            self.args,
            self.global_num_iterations,
            None,
            self.global_epochs_per_round,
            local_num_epochs_per_comm_round_dict
        )
        self.total_train_tracker.timer = self.server_timer
        self.total_test_tracker.timer = self.server_timer



    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device, pretrained=self.args.pretrained, **self.other_params)
        num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
        init_state_kargs = self.get_init_state_kargs()
        model_trainer = create_trainer(
            self.args, self.device, model, num_iterations=num_iterations,
            train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
            train_data_global=self.train_global, test_data_global=self.test_global,
            train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
            test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
            server_index=0, role='server',
            **init_state_kargs
        )
        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedAVGAggregator(self.train_global, self.test_global, self.train_data_num_in_total,
                self.train_data_local_dict, self.test_data_local_dict,
                self.train_data_local_num_dict, self.args.client_num_in_total, self.device,
                self.args, model_trainer, perf_timer=self.perf_timer, metrics=self.metrics, traindata_cls_counts=self.traindata_cls_counts)

        # self.aggregator.traindata_cls_counts = self.traindata_cls_counts
        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        process_mode = self.args.client_execution_backend == "process"
        client_device = torch.device("cpu") if process_mode else self.device
        init_state_kargs = self.get_init_state_kargs()
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=client_device, pretrained=self.args.pretrained, **self.other_params)
            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict, self.args.batch_size)
            model_trainer = create_trainer(
                self.args, client_device, model, num_iterations=num_iterations,
                train_data_num=self.train_data_num_in_total, test_data_num=self.test_data_num_in_total,
                train_data_global=self.train_global, test_data_global=self.test_global,
                train_data_local_num_dict=self.train_data_local_num_dict, train_data_local_dict=self.train_data_local_dict,
                test_data_local_dict=self.test_data_local_dict, class_num=self.class_num, other_params=self.other_params,
                client_index=client_index, role='client',
                **init_state_kargs
            )
            # model_trainer = create_trainer(self.args, self.device, model)
            # 親では client ごとの DataLoader worker/iterator を起動しない。
            client_args = self.args.clone() if process_mode else self.args
            if process_mode:
                client_args.data_save_memory_mode = True
            c = FedAVGClient(client_index, self.train_data_local_dict, self.train_data_local_num_dict,
                    self.test_data_local_dict, self.train_data_num_in_total,
                    self.device, client_args, model_trainer,
                    perf_timer=self.perf_timer, metrics=self.metrics)
            self.client_list.append(c)
            c.args = self.args
        logging.info("############setup_clients (END)#############")

    def train(self):
        previous_handler = None
        if self.args.client_execution_backend == "process":
            previous_handler = signal.getsignal(signal.SIGTERM)

            def terminate(signum, frame):
                raise RuntimeError("終了要求により client 学習を中止しました")

            signal.signal(signal.SIGTERM, terminate)
        try:
            return super().train()
        finally:
            if self.client_executor is not None:
                self.client_executor.close()
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)

    def _execution_event(self, event):
        with open("client-execution.jsonl", "a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _observe_resources(self, worker_pids):
        pids = [os.getpid(), *worker_pids]
        rss = {}
        for pid in pids:
            try:
                with open(f"/proc/{pid}/statm") as stream:
                    rss[str(pid)] = int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, IndexError):
                continue
        event = dict(event="resources", time=time.monotonic(), rss_bytes_by_pid=rss,
                     host_current_total_rss_bytes=sum(rss.values()))
        if self.device.type == "cuda":
            try:
                sample = subprocess.run(
                    ["nvidia-smi", "--id=0", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                event["driver_sample"] = sample.stdout.strip() if sample.returncode == 0 else "取得失敗"
            except (OSError, subprocess.TimeoutExpired):
                event["driver_sample"] = "取得不可"
        self._execution_event(event)

    def _process_round(self, client_indexes, named_params, params_type,
                       global_other_params, update_state_kargs, global_time_info,
                       shared_params_for_simulation):
        from algorithms_standalone.basePS.client_executor import BoundedExecutor
        from .process_worker import (export_loaders, export_client, cpu_tree,
                                     initialize_worker, train_task, weights_digest)

        if params_type != "model" or global_other_params or shared_params_for_simulation or update_state_kargs:
            raise ValueError("process 非対応の共有状態/更新引数です")
        start = time.monotonic()
        round_index = self.server_timer.global_comm_round_idx
        try:
            if self.client_executor is None:
                bank = {}
                trains = export_loaders(self.train_data_local_dict, bank)
                tests = export_loaders(self.test_data_local_dict, bank)
                self.client_executor = BoundedExecutor(
                    self.args.max_concurrent_clients, self.args.client_timeout_seconds,
                    train_task, initialize_worker, (trains, tests, self.train_data_local_num_dict))
            weights = cpu_tree(named_params)
            digest = weights_digest(weights)
            tasks = [dict(client_id=int(index), round=round_index,
                          phase="head" if round_index == self.max_comm_round else "shared",
                          config=dict(self.args), device=str(self.device), weights=weights, weights_sha256=digest,
                          state=export_client(self.client_list[index]),
                          update=update_state_kargs, time=global_time_info) for index in client_indexes]
            results = self.client_executor.run(tasks, observer=self._observe_resources)
            if [r["client_id"] for r in results] != [int(i) for i in client_indexes]:
                raise RuntimeError("worker の client 順序/ID が不正です")
            for result in results:
                self._execution_event(result["observation"])
            self._execution_event(dict(event="round", round=round_index, status="passed",
                                       start=start, end=time.monotonic(),
                                       parent_pid=os.getpid(),
                                       client_ids=[int(i) for i in client_indexes]))
            return results
        except BaseException as exc:
            if self.client_executor is not None:
                self.client_executor.close()
            self._execution_event(dict(event="round", round=round_index, status="failed",
                                       start=start, end=time.monotonic(), error=str(exc)))
            raise



    # override
    def check_end_epoch(self):
        return True

    # override
    def check_test_frequency(self):
        return ( self.server_timer.global_comm_round_idx % self.args.frequency_of_the_test == 0 \
            or self.server_timer.global_comm_round_idx == self.max_comm_round - 1)

    # override
    def check_and_test(self):
        if self.check_test_frequency():
            self.test()
        else:
            self.reset_train_test_tracker()

    def algorithm_train(self, client_indexes, named_params, params_type,
                        global_other_params,
                        update_state_kargs, global_time_info,
                        shared_params_for_simulation):
        process_results = None
        if self.args.client_execution_backend == "process":
            process_results = self._process_round(
                client_indexes, named_params, params_type, global_other_params,
                update_state_kargs, global_time_info, shared_params_for_simulation)
        ingest_start = time.monotonic()
        for i, client_index in enumerate(client_indexes):

            copy_global_other_params = copy.deepcopy(global_other_params)
            if self.args.exchange_model == True:
                copy_named_model_params = copy.deepcopy(named_params)

            if self.args.instantiate_all:
                client = self.client_list[client_index]
            else:
                # WARNING! All instantiated clients are only used in current round.
                # The history information saved may cause BUGs in the realistic FL scenario.
                client = self.client_list[i]

            if global_time_info["global_time_info"]["global_comm_round_idx"] == 0:
                traininig_start = True
            else:
                traininig_start = False

            # client training.............
            if process_results is not None:
                from .process_worker import restore_client
                restore_client(client, process_results[i]["state"])
                trained_result = process_results[i]["result"]
            else:
                trained_result = client.train(update_state_kargs, global_time_info,
                    client_index, copy_named_model_params, params_type,
                    copy_global_other_params,
                    traininig_start=traininig_start,
                    shared_params_for_simulation=shared_params_for_simulation)
            model_params, model_indexes, local_sample_number, client_other_params, \
                local_train_tracker_info, local_time_info, shared_params_for_simulation = trained_result

            self.total_train_tracker.decode_local_info(client_index, local_train_tracker_info)
            # self.total_test_tracker.decode_local_info(client_index, local_test_tracker_info)

            self.server_timer.update_time_info(local_time_info)
            self.aggregator.add_local_trained_result(
                client_index, model_params, model_indexes, local_sample_number, client_other_params)

            if self.args.HPFL and self.server_timer.global_comm_round_idx==self.max_comm_round:
                # upload the Pluggable,and save it in aggregator
                print("upload the Pluggable,and save it in aggregator")
                self.aggregator.model_list.append(copy.deepcopy(list(client.trainer.model.children())[-self.args.freeze_layer:]))
                # upload the distribution of feature extracted with backbone
                # print("client.cal_dist(client_index)",client.cal_dist(client_index))
                marker = (process_results[i]["marker"] if process_results is not None
                          else client.cal_dist(client_index))
                self.aggregator.dist_list.append(copy.deepcopy(list(marker)))
                print("upload successfully")
                
            # show_memory()
            # torch.cuda.empty_cache()

        # update global weights and return them
        # global_model_params = self.aggregator.aggregate()
        aggregation_start = time.monotonic()
        if not self.args.HPFL or self.server_timer.global_comm_round_idx!=self.max_comm_round:
            global_model_params, global_other_params, shared_params_for_simulation = self.aggregator.aggregate(
                global_comm_round=self.server_timer.global_comm_round_idx,
                global_outer_epoch_idx=self.server_timer.global_outer_epoch_idx,
                tracker=self.total_train_tracker,
                metrics=self.metrics)

        params_type = 'model'
        aggregation_end = time.monotonic()
        if process_results is not None:
            from .process_worker import cpu_tree
            history = dict(
                round=self.server_timer.global_comm_round_idx,
                client_ids=[int(i) for i in client_indexes],
                counts=[r["result"][2] for r in process_results],
                metrics=[r["result"][4] for r in process_results])
            # 全 round の巨大なモデル履歴は合成契約テストだけで保存する。
            if self.capture_process_history:
                history["weights"] = cpu_tree(self.aggregator.get_global_model_params())
            self.process_history.append(history)
        
        self.check_and_test()
        evaluation_end = time.monotonic()

        save_checkpoints_config = setup_checkpoint_config(self.args) if self.args.checkpoint_save else None
        save_checkpoint(
            self.args, save_checkpoints_config, extra_name="server", 
            epoch=self.server_timer.global_outer_epoch_idx,
            model_state_dict=self.aggregator.get_global_model_params(),
            optimizer_state_dict=self.aggregator.trainer.optimizer.state_dict(),
            train_metric_info=self.total_train_tracker.get_metric_info(self.metrics),
            test_metric_info=self.total_test_tracker.get_metric_info(self.metrics),
            postfix=self.args.checkpoint_custom_name
        )
        if process_results is not None:
            self._execution_event(dict(event="parent_phases", round=self.server_timer.global_comm_round_idx,
                                       ingest_seconds=aggregation_start-ingest_start,
                                       aggregation_seconds=aggregation_end-aggregation_start,
                                       evaluation_seconds=evaluation_end-aggregation_end,
                                       checkpoint_seconds=time.monotonic()-evaluation_end))
        self.server_timer.past_epochs(epochs=1*self.global_epochs_per_round)
        self.server_timer.past_comm_round(comm_round=1)
        
        if not self.args.HPFL or self.server_timer.global_comm_round_idx!=self.max_comm_round+1:
            return global_model_params, params_type, global_other_params, shared_params_for_simulation
        else:
            return 


