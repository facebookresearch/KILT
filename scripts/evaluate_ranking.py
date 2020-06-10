import json
import argparse

from kilt import ranker_evaluation
from kilt import kilt_utils as utils
from kilt.retrievers import DrQA_tfidf, Solr_BM25, DPR_connector, BLINK_connector


def execute(
    logger, test_config_json, retriever, log_directory, model_name, output_folder
):

    # run evaluation
    result = ranker_evaluation.run(
        test_config_json, retriever, model_name, logger, output_folder=output_folder
    )

    # dump results on json file
    with open("{}/{}_result.json".format(log_directory, model_name), "w") as outfile:
        json.dump(result, outfile)


def main(args):

    # load configs
    with open(args.test_config, "r") as fin:
        test_config_json = json.load(fin)

    # create a new directory to log and store results
    log_directory = utils.create_logdir_with_timestamp(args.logdir)
    logger = None

    logger = utils.init_logging(log_directory, args.model_name, logger)
    logger.info("loading {} ...".format(args.model_name))

    if args.model_name == "drqa":
        # 1. DrQA tf-idf
        if args.model_configuration:
            retriever = DrQA_tfidf.DrQA.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = DrQA_tfidf.DrQA.from_default_config(args.model_name)
    elif args.model_name == "solr":
        # 2. Solr BM25,
        if args.model_configuration:
            retriever = Solr_BM25.Solr.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = Solr_BM25.Solr.from_default_config(args.model_name)
    elif args.model_name == "dpr":
        # 3. DPR
        if args.model_configuration:
            retriever = DPR_connector.DPR.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = DPR_connector.DPR.from_default_config(args.model_name)
    elif args.model_name == "blink":
        # 4. BLINK
        if args.model_configuration:
            retriever = BLINK_connector.BLINK.from_config_file(
                args.model_name, args.model_configuration
            )
        else:
            retriever = BLINK_connector.BLINK.from_default_config(args.model_name)

    execute(
        logger,
        test_config_json,
        retriever,
        log_directory,
        args.model_name,
        args.output_folder,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_config",
        dest="test_config",
        type=str,
        default="kilt/configs/test_data.json",
        help="Test Configuration.",
    )

    parser.add_argument(
        "--logdir", dest="logdir", type=str, default="logs/ranking/", help="logdir",
    )

    parser.add_argument(
        "--model_name",
        "-m",
        dest="model_name",
        type=str,
        required=True,
        help="model name {drqa,solr,dpr,blink}",
    )

    parser.add_argument(
        "--model_configuration",
        "-c",
        dest="model_configuration",
        type=str,
        default=None,
        help="model configuration",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        dest="output_folder",
        type=str,
        default="",
        help="output folder",
    )

    args = parser.parse_args()

    main(args)
