#coding:utf-8

from loguru import logger
import argparse
from utils.download_model import Downloader



model_types = {
    1: "ActionRegconition",
    2: "Classification",
    3: "Detection",
    4: "DepthPrediction",
    5: "Segmentation",
    6: "PostEstimation"
}


def main(model_name, model_type):
    dnw = Downloader(model_name, model_type)
    dnw.download()
    logger.info(f"Finish download {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-n", type=str, required=True)
    parser.add_argument("--model_type", "-t", type=int, required=True)
    args = parser.parse_args()
    
    model_name = args.model_name
    model_type = args.model_type

    main(model_name, model_type)