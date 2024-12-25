import argparse

from intent_classify.modules import IntentClassificationModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "-f",
        "--checkpoint",
        type=str,
        required=True,
        help="lightning module checkpoint path (.ckpt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="pytorch checkpoint save file name",
    )
    args = parser.parse_args()

    save_dir = args.output

    module = IntentClassificationModule.load_from_checkpoint(args.checkpoint)
    module.export(filepath=save_dir)
    print(f"Successfully saved checkpoint to {save_dir}")
