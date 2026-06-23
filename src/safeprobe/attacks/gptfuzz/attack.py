
"""GPTFuzz attack — fazer a descrição inicial aqui """

import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pandas as pd
from typing import Any, Dict

#from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
#from gptfuzzer.fuzzer.mutator import (
#    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
#    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
#from gptfuzzer.fuzzer import GPTFuzzer
#from gptfuzzer.llm import OpenAILLM
#from gptfuzzer.utils.predict import RoBERTaPredictor

from safeprobe.config import Config 
from safeprobe.utils.logging import get_logger

#puxar o UnifiedBench-----------------------------------------------
from safeprobe.datasets import prompt.py

logger = get_logger(__name__)


class GPTFuzzAttack:
    def __init__(self, config=None):
        self.name = "GPTFuzz"
        self.description = "Fuzzing-based jailbreak template generation (LLM-Fuzzer)"
        self.config = config or Config()

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "openai_key":    self.config.get_api_key("openai"),
            "model_path":    "gpt-3.5-turbo",
            "target_model":  self.config.target_model,
            "seed_path":     "datasets/prompts/GPTFuzzer.csv",
            "max_query":     500,
            "max_jailbreak": 1,
            "energy":        1,
            "dataset":       "unifiedbench",
            "sample_size":   self.config.sample_size or 50,
            "category":      None,
            "output_file":   str(self.config.results_dir / "gptfuzz_results.json"),
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # carrega questions do UnifiedBench via SafeProbe
            from safeprobe.datasets.prompts import load_dataset
            raw = load_dataset(
                params.get("dataset", "unifiedbench"),
                max_samples=params.get("sample_size"),
                category=params.get("category"),
            )
            questions = [d["goal"] for d in raw if d.get("goal")]
            logger.info(f"GPTFuzz: {len(questions)} questions carregadas")

            # igual ao gptfuzz.py original
            initial_seed = pd.read_csv(params["seed_path"])["text"].tolist()

            openai_model  = OpenAILLM(params["model_path"], params["openai_key"])
            target_model  = OpenAILLM(params["target_model"], params["openai_key"])
            roberta_model = RoBERTaPredictor("hubert233/GPTFuzz", device="cpu")

            fuzzer = GPTFuzzer(
                questions=questions,
                target=target_model,
                predictor=roberta_model,
                initial_seed=initial_seed,
                mutate_policy=MutateRandomSinglePolicy([
                    OpenAIMutatorCrossOver(openai_model, temperature=1.0),
                    OpenAIMutatorExpand(openai_model, temperature=1.0),
                    OpenAIMutatorGenerateSimilar(openai_model, temperature=1.0),
                    OpenAIMutatorRephrase(openai_model, temperature=1.0),
                    OpenAIMutatorShorten(openai_model, temperature=1.0)],
                    concatentate=True,
                ),
                select_policy=MCTSExploreSelectPolicy(),
                energy=params.get("energy", 1),
                max_jailbreak=params.get("max_jailbreak", 1),
                max_query=params.get("max_query", 500),
                generate_in_batch=False,
            )

            fuzzer.run()

            out = params.get("output_file", "gptfuzz_results.json")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w") as f:
                json.dump([], f)

            logger.info("GPTFuzz concluído")
            return {
                "technique":   self.name,
                "success":     True,
                "output_file": out,
            }

        except Exception as exc:
            logger.error(f"GPTFuzz falhou: {exc}", exc_info=True)
            return {"technique": self.name, "success": False, "error": str(exc)}
