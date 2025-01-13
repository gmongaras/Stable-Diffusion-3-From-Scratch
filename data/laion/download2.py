from img2dataset import download
from pyspark.sql import SparkSession
import os

def create_spark_session():
    os.environ['PYSPARK_PYTHON'] = "/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/img2dataset.pex"
    spark = (
        SparkSession.builder
        .config("spark.submit.deployMode", "client")
        .config("spark.executorEnv.PEX_ROOT", "./.pex")
        .config("spark.executor.memory", "40G")
        .config("spark.executor.memoryOverhead", "40G")
        .config("spark.driver.port", "5678")
        .config("spark.driver.blockManager.port", "6678")
        .config("spark.driver.host", "$MASTER_IP")
        .config("spark.driver.bindAddress", "$MASTER_IP")
        .master("spark://$MASTER_IP:7077")
        .appName("laion_download")
        .getOrCreate()
    )
    return spark

spark = create_spark_session()

download(
    processes_count=1,
    thread_count=64,
    url_list="/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/laion-metadata/relaion2B-en-research/",
    image_size=102400,
    resize_only_if_bigger=False,
    resize_mode="no",
    skip_reencode=True,
    output_folder="/projects/eclarson/protein_diffusion/gmongaras_diffusion_models/Stable_Diffusion_3/data/laion/relaion2B-en-research-data",
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
