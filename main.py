"""Main entry point with command-line arguments for ablation studies."""
import traceback
import argparse
from pathlib import Path
import torch
from mcanet.data_utils import create_data_loaders, create_model
from mcanet.train import train_model
from mcanet.evaluate import evaluate_model, calculate_model_stats
from mcanet.visualize import plot_iteration_history, save_history_to_csv, save_full_report
from mcanet.quantize import quantize_int8_static_fx, export_to_ptl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCANet-Tiny Training and Evaluation')
    parser.add_argument('--datasets', type=str, default='all',
                        help='comma-separated dataset names (e.g., ASet,TSet). Default: all')
    parser.add_argument('--attn', type=str, default='ca', choices=['ca', 'se', 'none'],
                        help='attention type: ca (Coordinate Attention), se (Squeeze-and-Excitation), none')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'],
                        help='optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'],
                        help='learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--warmup', type=int, default=3,
                        help='warmup epochs (only for cosine scheduler)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='base learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--no_quantize', action='store_true',
                        help='skip quantization and PTL export')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Ablation config: attn={args.attn}, optimizer={args.optimizer}, '
          f'scheduler={args.scheduler}, epochs={args.epochs}, warmup={args.warmup}')

    # Dataset configuration list (modify paths as needed)
    dataset_configs = [
        {
            'name': 'ASet',
            'data_dir': Path(r"./dataset/A-SET"),
            'save_dir': Path(f"./MCANet_Tiny_{args.attn}_{args.optimizer}_{args.scheduler}_ASet_Results"),
        },
        {
            'name': 'TSet',
            'data_dir': Path(r"./dataset/T-SET"),
            'save_dir': Path(f"./MCANet_Tiny_{args.attn}_{args.optimizer}_{args.scheduler}_TSet_Results"),
        },
        {
            'name': 'SSet',
            'data_dir': Path(r"./dataset/S-SET"),
            'save_dir': Path(f"./MCANet_Tiny_{args.attn}_{args.optimizer}_{args.scheduler}_SSet_Results"),
        },
        {
            'name': 'GSet',
            'data_dir': Path(r"./dataset/G-SET"),
            'save_dir': Path(f"./MCANet_Tiny_{args.attn}_{args.optimizer}_{args.scheduler}_GSet_Results"),
        },
    ]

    if args.datasets.lower() != 'all':
        selected = args.datasets.split(',')
        filtered = [cfg for cfg in dataset_configs if cfg['name'] in selected]
        if filtered:
            dataset_configs = filtered
        else:
            print(f"Warning: No matching dataset found for {selected}, using all datasets.")

    for config in dataset_configs:
        print(f"\n{'='*60}")
        print(f"Starting processing dataset: {config['name']}")
        print(f"Data path: {config['data_dir']}")
        print(f"Results save path: {config['save_dir']}")
        print('='*60)

        save_dir = Path(config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(config['data_dir'])
        train_dir = data_dir / "train"
        val_dir = data_dir / "validation"
        test_dir = data_dir / "test"

        # 1. Data loading
        train_loader, val_loader, class_to_idx = create_data_loaders(
            train_dir, val_dir, batch_size=args.batch_size
        )

        # 2. Create model
        print(f"\nCreating model (dataset: {config['name']})...")
        model = create_model(len(class_to_idx), pretrained=False, attn_type=args.attn)

        # 3. Model statistics
        model_stats = calculate_model_stats(model, device)
        print(f"Parameters: {model_stats['Parameters']:.2f} M | "
              f"GFLOPs: {model_stats['GFLOPs']:.2f} | "
              f"Size: {model_stats['Size']:.2f} MB")

        # 4. Training
        print("\nStarting training...")
        history, trained_model = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs, device=device,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            warmup_epochs=args.warmup,
            base_lr=args.lr
        )

        # 5. Save model weights
        model_path = save_dir / 'mcanet_tiny.pth'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_to_idx': class_to_idx
        }, model_path)
        print(f"Model saved: {model_path}")

        # 6. Visualization and CSV
        plot_iteration_history(history, save_dir=save_dir)
        save_history_to_csv(history, save_dir=save_dir)

        # 7. Test evaluation (original FP32)
        print("\nEvaluating on test set (FP32)...")
        test_metrics = evaluate_model(
            trained_model, test_dir, device=device, class_to_idx=class_to_idx
        )
        print(f"Accuracy: {test_metrics['Accuracy']:.4f} | "
              f"F1: {test_metrics['F1-Score']:.4f} | "
              f"Inference Time: {test_metrics['Inference Time']:.2f} ms")

        save_full_report(test_metrics, model_stats, save_dir, class_to_idx)

        if not args.no_quantize:
            # 8. Export FP32 PTL model
            print("\nExporting FP32 PTL model...")
            fp32_ptl_path = save_dir / 'mobile_fp32.ptl'
            export_to_ptl(trained_model, fp32_ptl_path, model_name="FP32")

            # 9. INT8 quantization and evaluation
            print("\nStarting INT8 quantization...")
            try:
                int8_model = quantize_int8_static_fx(trained_model, val_loader, backend="qnnpack")

                int8_state_dict_path = save_dir / 'int8_quantized.pth'
                torch.save(int8_model.state_dict(), int8_state_dict_path)
                print(f"INT8 quantized model weights saved: {int8_state_dict_path}")

                print("\nEvaluating INT8 model (CPU)...")
                int8_test_metrics = evaluate_model(
                    int8_model, test_dir, device=torch.device('cpu'), class_to_idx=class_to_idx
                )
                print(f"INT8 Accuracy: {int8_test_metrics['Accuracy']:.4f} | "
                      f"F1: {int8_test_metrics['F1-Score']:.4f} | "
                      f"Inference Time: {int8_test_metrics['Inference Time']:.2f} ms")

                # Save INT8 simplified report
                int8_report_path = Path(save_dir) / 'int8_metrics_report.txt'
                with open(int8_report_path, 'w', encoding='utf-8') as f:
                    f.write("========== INT8 Quantized Model Performance Report ==========\n")
                    f.write(f"Accuracy:   {int8_test_metrics['Accuracy']:.4f}\n")
                    f.write(f"Precision:  {int8_test_metrics['Precision']:.4f}\n")
                    f.write(f"Recall:     {int8_test_metrics['Recall']:.4f}\n")
                    f.write(f"F1-Score:   {int8_test_metrics['F1-Score']:.4f}\n")
                    f.write(f"Inference Time: {int8_test_metrics['Inference Time']:.2f} ms\n")
                    f.write(f"Inference FPS:  {int8_test_metrics['Inference FPS']:.2f}\n")
                print(f"INT8 report saved: {int8_report_path}")

                # Export INT8 PTL model
                print("\nExporting INT8 PTL model...")
                int8_ptl_path = save_dir / 'mobile_int8.ptl'
                export_to_ptl(int8_model, int8_ptl_path, model_name="INT8")

            except Exception as e:
                print(f"Error during quantization or exporting INT8 model: {e}")
                traceback.print_exc()

        del model, trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll datasets processed!")