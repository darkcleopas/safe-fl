#!/usr/bin/env python3
import argparse
import os
import logging
import tensorflow as tf
import gc

from fl_simulator import FLSimulator  # or SimpleFLSimulator

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf_num_threads = 16
tf.config.threading.set_intra_op_parallelism_threads(tf_num_threads)
tf.config.threading.set_inter_op_parallelism_threads(tf_num_threads)

def main():
    parser = argparse.ArgumentParser(description='Run federated learning simulation')
    parser.add_argument('--config', help='Path to config file or config directory or config file list', type=str, required=True, nargs='+')
    parser.add_argument('--threads', action='store_true', help='Use multi-threading')
    args = parser.parse_args()

    config_files = []
    for config in args.config:
        if not os.path.exists(config):
            raise ValueError(f"Config path does not exist: {config}")
        elif os.path.isdir(config):
            config_files.extend([os.path.join(config, f) for f in os.listdir(config) if f.endswith('.yaml')])
        elif os.path.isfile(config):
            if not config.endswith('.yaml'):
                raise ValueError(f"Provided config file is not a YAML file: {config}")
            config_files.append(config)


    for config_file in config_files:
        # Create simulator
        simulator = FLSimulator(config_file, use_threads=args.threads)

        simulator.run_simulation()

        del simulator
        gc.collect()
        tf.keras.backend.clear_session()

        print(f"Finished simulation for config: {config_file}")

    print("Simulation completed!")

if __name__ == "__main__":
    main()