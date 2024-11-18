class ModelEvaluator:
    """Comprehensive evaluation suite"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {
            "perplexity": Perplexity(),
            "code_quality": CodeEvalMetric(),
            "instruction_following": InstructionMetric(),
            "reasoning": ReasoningMetric()
        }
    
    async def evaluate(self, eval_datasets: Dict[str, Dataset]) -> Dict[str, float]:
        results = {}
        for dataset_name, dataset in eval_datasets.items():
            # Run evaluation with proper batching
            scores = await self._evaluate_dataset(dataset)
            results[dataset_name] = scores
            
        return self._aggregate_results(results) 