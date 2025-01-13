#!/bin/bash

# Define variables
NUM_NODES=5                       # Number of nodes (1 master + NUM_NODES - 1 workers)
MASTER_IP=$(hostname -I | awk '{print $1}')
SPARK_HOME="/users/gmongaras/bin/spark"
PEX_FILE="/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/img2dataset.pex"
OUTPUT_DIR="/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/relaion2B-en-research-data"
URL_LIST="/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/laion-metadata/relaion2B-en-research/"
PORT=7077
HOST=c119.cm.cluster


# Load necessary modules
# module load python/3.8.10 spark/3.2.0

# Start Spark Master on the first node
if [[ $SLURM_NODEID -eq 0 ]]; then
  $SPARK_HOME/sbin/start-master.sh  -p $PORT --webui-port 8084 --host $HOST
  echo "Master started on $MASTER_IP:$PORT"
fi

# Start Spark Workers on all other nodes
if [[ $SLURM_NODEID -gt 0 && $SLURM_NODEID -lt $NUM_NODES ]]; then
  # Wait a little for the master to start
  sleep 10
  echo "Starting worker on $(hostname)"
  $SPARK_HOME/sbin/start-worker.sh -c 16 -m 24G "spark://$HOST:$PORT" #"spark://$MASTER_IP:$PORT"
  # srun --ntasks=$NUM_NODES $SPARK_HOME/sbin/start-worker.sh -c 16 -m 75G "spark://$HOST:$PORT"
  echo "Worker started on $(hostname)"
  # Infinite loop to keep the worker running
  while true; do
    sleep 10
  done
fi

# Wait for all nodes to initialize
sleep 30

# Run the Python script on the master node
if [[ $SLURM_NODEID -gt -1 ]]; then
  python3 - <<EOF
from img2dataset import download
from pyspark.sql import SparkSession
import os
import time

def create_spark_session():
    # os.environ['PYSPARK_PYTHON'] = "$PEX_FILE"
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client") \
        #.config("spark.files", pex_file) \ # you may choose to uncomment this option if you want spark to automatically download the pex file, but it may be slow
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        .config("spark.executor.cores", "16")
        .config("spark.cores.max", "48") # you can reduce this number if you want to use only some cores ; if you're using yarn the option name is different, check spark doc
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "172.31.44.42")
        .config("spark.driver.bindAddress", "172.31.44.42")
        .config("spark.executor.memory", "16G") # make sure to increase this if you're using more cores per executor
        .config("spark.executor.memoryOverhead", "8G")
        .config("spark.task.maxFailures", "100")
        .config("spark.driver.host", "$HOST")
        .config("spark.driver.bindAddress", "$HOST")
        .master("spark://$HOST:$PORT")
        .appName("laion_download")
        .getOrCreate()
    )
    return spark

spark = create_spark_session()

import os
# Set spark environments
# os.environ['PYSPARK_PYTHON'] = 'path/to/python'
# os.environ['PYSPARK_DRIVER_PYTHON'] = 'path/to/python'
print(os.environ['PYSPARK_PYTHON'] )
print(os.environ['PYSPARK_DRIVER_PYTHON'] )

download(
    processes_count=1,
    thread_count=64,
    url_list="$URL_LIST",
    image_size=102400,
    resize_only_if_bigger=False,
    resize_mode="no",
    skip_reencode=True,
    output_folder="$OUTPUT_DIR",
    output_format="webdataset",
    input_format="parquet",
    url_col="url",
    caption_col="caption",
    enable_wandb=False,
    encode_quality=7,
    encode_format="png",
    number_sample_per_shard=10000,
    distributor="pyspark",
    save_additional_columns=["punsafe", "pwatermark", "similarity"],
    oom_shard_count=6,
)
time.sleep(30)
print("done")
exit()
EOF
fi
