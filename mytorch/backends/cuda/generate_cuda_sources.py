import argparse
import os.path as osp
from mytorch.backends.cuda.env import CudaSourceGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate CUDA sources for template automatic instantiation"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    generator = CudaSourceGenerator()
    filenames = ["conv.cu", "batch_norm.cu"]

    if args.generate:
        folder = osp.join(osp.dirname(__file__), "../../native/cuda")
        for filename in filenames:
            if osp.isdir(osp.join(folder, filename)) and filename.endswith(".cu"):
                generator.save_templated_source(filename)

    if args.list:
        generated_paths = [
            osp.abspath(osp.join(generator.generated_src_path, filename))
            for filename in filenames
        ]
        print(";".join(generated_paths))
