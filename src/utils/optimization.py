class DeviceOptimizer:
    def optimize_for_inference(self, model):
        # Quantize to 4-bit
        model = quantize(model, bits=4)
        # Enable Metal performance shaders
        model.enable_metal_optimizations()
        # Cache compiled graphs
        model.cache_computation_graphs() 