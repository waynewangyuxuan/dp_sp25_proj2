# LoRA Knowledge Distillation

This project implements knowledge distillation with LoRA (Low-Rank Adaptation) for text classification on the AG News dataset. The approach uses a larger LoRA model as the teacher and a smaller LoRA model as the student, while respecting the 1 million parameter limit for the student model.

## Overview

Knowledge distillation is a model compression technique where a smaller model (student) is trained to mimic the behavior of a larger model (teacher). In this implementation:

1. A larger LoRA model is trained first as the teacher
2. A smaller LoRA model is then trained as the student, using knowledge distillation from the teacher
3. Both models are based on the same pre-trained model (e.g., RoBERTa-base)
4. The student model is constrained to have fewer than 1 million trainable parameters

## Implementation Details

### Teacher Model

The teacher model is a LoRA-adapted model with:
- Higher LoRA rank (default: 16)
- Higher LoRA alpha (default: 32)
- More target layers (default: 8)

### Student Model

The student model is a LoRA-adapted model with:
- Lower LoRA rank (default: 8)
- Lower LoRA alpha (default: 16)
- Fewer target layers (default: 6)

### Distillation Process

The distillation process uses a combination of:
1. Hard label loss (cross-entropy with ground truth labels)
2. Soft label loss (KL divergence with teacher's logits)

The total loss is a weighted combination:
```
loss = (1 - alpha) * hard_loss + alpha * soft_loss
```

Where:
- `alpha` controls the weight of the distillation loss (default: 0.5)
- `temperature` controls the sharpness of the softmax (default: 2.0)

## Usage

### Training

To run the LoRA distillation training:

```bash
./scripts/run_lora_distillation.sh
```

This will run multiple experiments with different configurations.

### Custom Training

To run a single experiment with custom parameters:

```bash
python scripts/train_distillation.py \
    --exp_name "custom_experiment" \
    --base_model "roberta-base" \
    --teacher_lora_r 16 \
    --teacher_lora_alpha 32 \
    --teacher_target_layers 8 \
    --student_lora_r 8 \
    --student_lora_alpha 16 \
    --student_target_layers 6 \
    --alpha 0.5 \
    --temperature 2.0 \
    --learning_rate 2e-4 \
    --batch_size 32 \
    --num_epochs 10
```

### Parameters

- `--exp_name`: Experiment name (default: auto-generated)
- `--base_model`: Base model to use for both teacher and student (default: "roberta-base")
- `--teacher_lora_r`: Teacher LoRA rank (default: 16)
- `--teacher_lora_alpha`: Teacher LoRA alpha (default: 32)
- `--teacher_target_layers`: Number of layers to apply LoRA to in teacher (default: 8)
- `--student_lora_r`: Student LoRA rank (default: 8)
- `--student_lora_alpha`: Student LoRA alpha (default: 16)
- `--student_target_layers`: Number of layers to apply LoRA to in student (default: 6)
- `--alpha`: Weight for distillation loss (default: 0.5)
- `--temperature`: Temperature for distillation (default: 2.0)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 10)
- `--teacher_checkpoint`: Path to pre-trained teacher model checkpoint (optional)

## Results

The training results are saved in the `outputs/lora_distillation` directory, with each experiment in its own subdirectory. The results include:

- Trained teacher model
- Trained student model
- Training logs
- Evaluation metrics

## Advantages of This Approach

1. **Parameter Efficiency**: The student model has fewer trainable parameters than the teacher, making it more efficient for inference.
2. **Knowledge Transfer**: The student model learns from the teacher's "soft targets" in addition to the hard labels.
3. **Flexibility**: The approach can be applied to different base models and LoRA configurations.
4. **Constraint Satisfaction**: The student model respects the 1 million parameter limit while still benefiting from the teacher's knowledge.

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
- Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685. 