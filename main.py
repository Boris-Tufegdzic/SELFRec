from SELFRec import SELFRec
from util.conf import ModelConf
import time
import argparse

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a specified model using SELFRec.')
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        help='The name of the model to run (e.g., XSimGCL, SASRec, etc.).'
    )
    args = parser.parse_args()

    # Check if the specified model exists
    model = args.model

    s = time.time()
    all_models = sum(models.values(), [])
    if model in all_models:
        kaggle_base_dir = "/kaggle/working/SELFRec/"
        conf = ModelConf(kaggle_base_dir + f'conf/{model}.yaml')
        rec = SELFRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
