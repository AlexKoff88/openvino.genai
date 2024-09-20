import argparse
import difflib
import os
import json
import pandas as pd
import logging
from datasets import load_dataset
from optimum.exporters import TasksManager
from optimum.intel.openvino import OVModelForCausalLM
from optimum.utils import NormalizedConfigManager, NormalizedTextConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from . import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TasksManager._SUPPORTED_MODEL_TYPE["stablelm-epoch"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
NormalizedConfigManager._conf["stablelm-epoch"] = NormalizedTextConfig.with_args(
    num_layers="num_hidden_layers",
    num_attention_heads="num_attention_heads",
)


def load_model(model_id, device="CPU", ov_config=None, use_hf=False, use_genai=False):
    if use_hf:
        logger.info("Using HF model")
        return AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=device.lower())

    if ov_config:
        with open(ov_config) as f:
            ov_options = json.load(f)
    else:
        ov_options = None
    try:
        model = OVModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device=device, ov_config=ov_options)
    except ValueError:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            use_cache=True,
            device=device,
            ov_config=ov_options
        )
    return model


def load_prompts(args):
    if args.dataset is None:
        return None
    split = "validation"
    if args.split is not None:
        split = args.split
    if "," in args.dataset:
        path_name = args.dataset.split(",")
        path = path_name[0]
        name = path_name[1]
    else:
        path = args.dataset
        name = None
    data = load_dataset(path=path, name=name, split=split)

    res = data[args.dataset_field]

    res = {"questions": list(res)}

    return res


def parse_args():
    parser = argparse.ArgumentParser(
        prog="WWB CLI",
        description="This sript generates answers for questions from csv file",
    )

    parser.add_argument(
        "--base-model",
        default=None,
        help="Model to ground truth generation.",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Model to comparison with base_model. Usually it is compressed, quantized version of base_model.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer for divergency metric. If not defined then will be load from base_model or target_model.",
    )

    parser.add_argument(
        "--gt-data",
        default=None,
        help="CSV file with base_model generation. If defined and exists then base_model will not used."
        "I defined and not exists them will be generated by base_model evaluation.",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Model for measurement of similarity between base_model and target_model."
        "By default it is sentence-transformers/all-mpnet-base-v2,"
        "but for Chinese LLMs better to use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of the dataset with prompts. The interface for dataset is load_dataset from datasets library."
        "Please provide this argument in format path,name (for example wikitext,wikitext-2-v1)."
        "If None then internal list of prompts will be used.",
    )
    parser.add_argument(
        "--dataset-field",
        type=str,
        default="questions",
        help="The name of field in dataset for prompts. For example question or context in squad."
        "Will be used only if dataset is defined.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split of prompts from dataset (for example train, validation, train[:32])."
        "Will be used only if dataset is defined.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory name for saving the per sample comparison and metrics in CSV files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of prompts to use from dataset",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print results and their difference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Device to run the model, e.g. 'CPU', 'GPU'.",
    )
    parser.add_argument(
        "--ov-config",
        type=str,
        default=None,
        help="Path to the JSON file that contains OpenVINO Runtime configuration.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "cn"],
        default=None,
        help="Used to select default prompts based on the primary model language, e.g. 'en', 'ch'.",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use AutoModelForCausalLM from transformers library to instantiate the model.",
    )

    return parser.parse_args()


def check_args(args):
    assert not (args.base_model is None and args.target_model is None)
    assert not (args.base_model is None and args.gt_data is None)


def load_tokenizer(args):
    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
    elif args.base_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, trust_remote_code=True
        )

    return tokenizer


def diff_strings(a: str, b: str, *, use_loguru_colors: bool = False) -> str:
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    if use_loguru_colors:
        green = "<GREEN><black>"
        red = "<RED><black>"
        endgreen = "</black></GREEN>"
        endred = "</black></RED>"
    else:
        green = "\x1b[38;5;16;48;5;2m"
        red = "\x1b[38;5;16;48;5;1m"
        endgreen = "\x1b[0m"
        endred = "\x1b[0m"

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            output.append(a[a0:a1])
        elif opcode == "insert":
            output.append(f"{green}{b[b0:b1]}{endgreen}")
        elif opcode == "delete":
            output.append(f"{red}{a[a0:a1]}{endred}")
        elif opcode == "replace":
            output.append(f"{green}{b[b0:b1]}{endgreen}")
            output.append(f"{red}{a[a0:a1]}{endred}")
    return "".join(output)


def main():
    args = parse_args()
    check_args(args)

    prompts = load_prompts(args)
    tokenizer = load_tokenizer(args)
    if args.gt_data and os.path.exists(args.gt_data):
        evaluator = Evaluator(
            base_model=None,
            gt_data=args.gt_data,
            test_data=prompts,
            tokenizer=tokenizer,
            similarity_model_id=args.text_encoder,
            num_samples=args.num_samples,
            language=args.language,
        )
    else:
        base_model = load_model(args.base_model, args.device, args.ov_config, args.hf)
        evaluator = Evaluator(
            base_model=base_model,
            test_data=prompts,
            tokenizer=tokenizer,
            similarity_model_id=args.text_encoder,
            num_samples=args.num_samples,
            language=args.language,
        )
        if args.gt_data:
            evaluator.dump_gt(args.gt_data)
        del base_model

    if args.target_model:
        target_model = load_model(args.target_model, args.device, args.ov_config, args.hf)
        all_metrics_per_question, all_metrics = evaluator.score(target_model)
        logger.info("Metrics for model: %s", args.target_model)
        logger.info(all_metrics)

        if args.output:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            df = pd.DataFrame(all_metrics_per_question)
            df.to_csv(os.path.join(args.output, "metrics_per_qustion.csv"))
            df = pd.DataFrame(all_metrics)
            df.to_csv(os.path.join(args.output, "metrics.csv"))

    if args.verbose:
        metric_of_interest = "similarity"
        worst_examples = evaluator.worst_examples(top_k=5, metric=metric_of_interest)
        for i, e in enumerate(worst_examples):
            ref_text = ""
            actual_text = ""
            diff = ""
            for l1, l2 in zip(e["source_model"].splitlines(), e["optimized_model"].splitlines()):
                if l1 == "" and l2 == "":
                    continue
                ref_text += l1 + "\n"
                actual_text += l2 + "\n"
                diff += diff_strings(l1, l2) + "\n"

            logger.info("--------------------------------------------------------------------------------------")
            logger.info("## Reference text %d:\n%s", i + 1, ref_text)
            logger.info("## Actual text %d:\n%s", i + 1, actual_text)
            logger.info("## Diff %d: ", i + 1)
            logger.info(diff)


if __name__ == "__main__":
    main()
