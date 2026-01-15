import torch, os, json
import numpy as np
from diffsynth import load_state_dict
from structured_noise import generate_structured_noise_batch_vectorized
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, flux_parser
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.trainers.hf_url_dataset import HuggingFaceURLImageDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class FluxTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        
        # Extra inputs
        controlnet_input = {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        input_latents = inputs["input_latents"]
        cutoff_radius = np.random.exponential(scale=1/0.1)
        input_noise = torch.randn_like(input_latents.float())
        structured_noise = generate_structured_noise_batch_vectorized(input_latents.float(), cutoff_radius=cutoff_radius, input_noise=input_noise)
        inputs["noise"] = structured_noise.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        loss = self.pipe.training_loss(**models, **inputs)
        return loss



if __name__ == "__main__":
    parser = flux_parser()
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use for debugging.")
    args = parser.parse_args()
    dataset = HuggingFaceURLImageDataset(
        dataset_name="bghira/photo-concept-bucket",
        url_field="url",
        text_field="cogvlm_caption",  # or "description", "alt", "title"
        cache_dir="./url_cache",
        max_pixels=args.max_pixels,
        height=None,  # Use dynamic resolution
        width=None,
        data_file_keys=("image", "text"),  # Return both image and text
        repeat=args.dataset_repeat,
        max_samples=args.max_samples,  # Limit to 100 samples for debugging (set to None for full dataset)
    )
    
    print(f"Dataset size: {len(dataset)}")
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=FluxLoRAConverter.align_to_opensource_format if args.align_to_opensource_format else lambda x:x,
    )
    launch_training_task(dataset, model, model_logger, args=args, num_workers=2)
