from typing import Any, Union

import os
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity

default_data = {
    "en": {
        "prompts": [
            "Who is Mark Twain?",
            "Who is William Shakespeare?",
            "Mirror, mirror on the wall who is the boldest of the world?",
            "Compute the result of the following equation: \"2 + 3 =\"?",
            "Solve the following system of equations: \"2 * x + y = 10, -x + 3 * y = 9\"?",
            "What is the capital of Great Britain? When was it founded?",
            "Explain the following proverb: \"All that glitters is not gold\"?",
            "Continue the following proverb and explain it: \"A penny saved is ...\"",
            "Who won the Nobel Prize in Physics in 1921? What he is famous for?",
            "Continue the song: \"And now, the end is near ...\"",
            "What do you know about Greco-Persian wars?",
            "What are the main principles of Daosism?",
            "Who is Leo Tolstoy?",
            "Who is the author of 'Time is money' book?",
            "Who is Stephen King?",
            "What is C++?",
            "Who is the most famous ancient Greek scientist?",
            "When and where will be the next Summer Olympic Games?",
            """Give me a numbered list of unique verbs in the following text:
            "It met with positive sales in Japan, and was praised by both Japanese and western critics.
            After release, it received downloadable content, along with an expanded edition in November of that year." """,
            "Come up with a joke in verse.",
            """Clean, but not water,
            White, but not snow,
            Sweet, but not ice cream.
            What is it?""",
            """\"The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.
            Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
            Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses
            and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.
            First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall,
            flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns
            the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.
            However, she said more preventative work could have been carried out to ensure the retaining wall did not fail.
            "It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate
            that - but it is almost like we're neglected or forgotten," she said. "That may not be true but it is perhaps
            my perspective over the last few days.\"
            Summarize the above text in one sentence.
            """,
            "What would be the set of prompts you recommend for testing the Large Language Model?",
        ],
    },
    "cn": {
        "prompts": [
            "马克吐温是谁?",
            "谁是威廉-莎士比亚?",
            "阿加莎-克里斯蒂是谁?",
            "芭芭拉-卡特兰是谁?",
            "丹妮尔-斯蒂尔是谁?",
            "谁是哈罗德-罗宾斯?",
            "乔治-西默农是谁?",
            "伊妮德-布莱顿是谁?",
            "西德尼-谢尔顿是谁?",
            "鸟山明是谁?",
            "谁是列夫-托尔斯泰?",
            "亚历山大-普希金是谁?",
            "斯蒂芬-金是谁?",
            "C++是什么?",
            "Python是什么?",
            "什么是 Java?",
            "JavaScript是什么?",
            "什么是 Perl?",
            "什么是 OpenCV?",
            "谁是最著名的作家?",
            "谁是最有名的发明家?",
            "谁是最著名的数学家?",
            "最著名的作曲家是谁?",
            "谁是最有名的程序员?",
            "谁是最著名的运动员?",
            "谁是最著名的古希腊科学家?",
            "蓝色和黄色混合会得到什么颜色?",
        ],
    },
}


def autodetect_language(model):
    model2language = {
        "chatglm": "cn",
        "qwen2": "cn",
        "qwen": "cn",
        "baichuan": "cn",
        "minicpmv": "cn",
        "internlm": "cn",
    }

    if not hasattr(model, "config"):
        return "en"
    return model2language.get(model.config.model_type, "en")


@register_evaluator(
    "text"
)
class TextEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
        language=None,
        gen_answer_fn=None,
        generation_config=None,
        generation_config_base=None,
        seqs_per_request=None,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self._crop_question = crop_question
        self.num_samples = num_samples
        self.generation_config = generation_config
        self.generation_config_base = generation_config
        self.seqs_per_request = seqs_per_request
        self.generation_fn = gen_answer_fn
        if self.generation_config is not None:
            assert self.seqs_per_request is not None

        # Take language from the base model if provided
        self.language = language
        if self.language is None:
            if base_model is not None:
                self.language = autodetect_language(base_model)

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_answer_fn, generation_config=generation_config
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        # Take language ground truth if no base model provided
        if self.language is None and "language" in self.gt_data.columns:
            self.language = self.gt_data["language"].values[0]

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, self.generation_config)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_answer_fn=None, generation_config=None):
        def default_gen_answer(model, tokenizer, prompt, max_new_tokens, crop_question):
            inputs = self.tokenizer(prompt, return_tensors="pt")

            tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

            if crop_question:
                tokens = tokens[:, inputs["input_ids"].shape[-1] :]

            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompts": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            if self.language is None:
                print(
                    "No language detecting in the base model or ground truth data. Taking language from target model."
                )
                self.language = autodetect_language(model)
            data = pd.DataFrame.from_dict(default_data[self.language])

        prompt_data = data["prompts"]

        answers = []
        prompts = (
            prompt_data.values
            if self.num_samples is None
            else prompt_data.values[: self.num_samples]
        )

        if generation_config is None:
            for p in tqdm(prompts, desc="Evaluate pipeline"):
                answers.append(
                    gen_answer_fn(
                        model,
                        self.tokenizer,
                        p,
                        self.max_new_tokens,
                        self._crop_question,
                    )
                )
        else:
            with tqdm(total=len(prompt_data.values)) as progress_bar:
                batch = []
                for p_idx, p in enumerate(prompt_data.values):
                    progress_bar.update(1)
                    batch.append(p)
                    if (
                        len(batch) == self.seqs_per_request
                        or p_idx == len(prompt_data.values) - 1
                    ):
                        ans_batch = model.generate(
                            batch, [generation_config] * len(batch)
                        )
                        for ans in ans_batch:
                            answers.append(ans.m_generation_ids[0])

                        batch.clear()

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)
        df["language"] = self.language

        return df
