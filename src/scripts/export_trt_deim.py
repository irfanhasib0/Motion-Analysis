import tensorrt as trt

def export_tensorrt(onnx_path, engine_path, input_shape, fp16=False, int8=False, max_batch_size=1):
        """
        Export ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            input_shape: Tuple of (C, H, W) for input shape
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            max_batch_size: Maximum batch size
        """
        
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        # Always use explicit batch for TensorRT 8.x+
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Get input tensor name
        input_name = network.get_input(0).name
        
        # Create optimization profile
        profile = builder.create_optimization_profile()
        
        if max_batch_size > 1:
            # Dynamic batch: min=1, opt=1, max=max_batch_size
            min_shape = (1, *input_shape)
            opt_shape = (1, *input_shape)
            max_shape = (max_batch_size, *input_shape)
        else:
            # Fixed batch: all shapes are the same
            min_shape = (1, *input_shape)
            opt_shape = (1, *input_shape)
            max_shape = (1, *input_shape)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build and serialize engine
        serialized_engine = builder.build_serialized_network(network, config)
        
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"TensorRT engine saved to {engine_path}")

onnx_path = '../models/deimv2_dinov3_s_coco.onnx'
trt_path  = onnx_path.replace('.onnx', '.engine')
export_tensorrt(onnx_path, trt_path, input_shape=(3, 640, 640))
